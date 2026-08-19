#include "gatzk/protocol/trace.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <openssl/rand.h>

#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/psq.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/zkmap.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol
{
    // 本文件负责把 context 中的静态对象与模型前向结果物化成正式 witness 列，
    // 并按固定 transcript 顺序生成挑战、lookup、route、PSQ、zkMaP 与 dynamic commitment 载体。
    // 它是 proving 主链最重的执行层文件，但所有缓存和路由都只能复用等价对象，不能改变 formal 行为。

#if GATZK_ENABLE_CUDA_BACKEND
    std::vector<std::size_t> lookup_histogram_indices_cuda(
        const std::vector<std::size_t> &indices,
        std::size_t domain_size);
#endif

    std::vector<std::size_t> stable_dynamic_commitment_chunk_order(
        const std::vector<std::vector<std::string>> &produced_labels,
        const std::vector<std::vector<std::string>> &required_labels)
    {
        if (produced_labels.size() != required_labels.size())
        {
            throw std::runtime_error(
                "dynamic commitment chunk dependency shape mismatch");
        }
        std::unordered_map<std::string, std::size_t> producer;
        for (std::size_t chunk = 0; chunk < produced_labels.size(); ++chunk)
        {
            for (const auto &label : produced_labels[chunk])
            {
                if (label.empty() || !producer.emplace(label, chunk).second)
                {
                    throw std::runtime_error(
                        "duplicate or empty dynamic commitment chunk output: " + label);
                }
            }
        }

        std::vector<std::vector<std::size_t>> consumers(
            produced_labels.size());
        std::vector<std::size_t> indegree(
            produced_labels.size(), 0);
        for (std::size_t chunk = 0; chunk < required_labels.size(); ++chunk)
        {
            std::unordered_set<std::size_t> dependencies;
            for (const auto &label : required_labels[chunk])
            {
                const auto producer_it = producer.find(label);
                if (producer_it == producer.end() || producer_it->second == chunk || !dependencies.insert(producer_it->second).second)
                {
                    continue;
                }
                consumers[producer_it->second].push_back(chunk);
                ++indegree[chunk];
            }
        }

        std::vector<std::size_t> order;
        order.reserve(produced_labels.size());
        std::vector<bool> scheduled(produced_labels.size(), false);
        while (order.size() != produced_labels.size())
        {
            std::size_t ready = produced_labels.size();
            for (std::size_t chunk = 0; chunk < produced_labels.size(); ++chunk)
            {
                if (!scheduled[chunk] && indegree[chunk] == 0)
                {
                    ready = chunk;
                    break;
                }
            }
            if (ready == produced_labels.size())
            {
                throw std::runtime_error(
                    "cyclic deferred dynamic commitment chunk dependencies");
            }
            scheduled[ready] = true;
            order.push_back(ready);
            for (const auto consumer : consumers[ready])
            {
                if (indegree[consumer] == 0)
                {
                    throw std::runtime_error(
                        "dynamic commitment chunk dependency accounting underflow");
                }
                --indegree[consumer];
            }
        }
        return order;
    }

    namespace
    {

        using algebra::FieldElement;
        using algebra::Polynomial;
        using data::Edge;
        using model::Matrix;
        using Clock = std::chrono::steady_clock;

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        bool progress_enabled()
        {
            return std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        }

        void progress_note(const std::string &label)
        {
            if (progress_enabled())
            {
                std::cerr << "[progress] " << label << '\n';
            }
        }

        template <typename Builder>
        crypto::G1Point timed_derived_commitment_point(
            RunMetrics *metrics,
            Builder builder,
            std::string_view label = {})
        {
            const auto start = Clock::now();
            auto point = builder();
            const auto duration_ms = elapsed_ms(start, Clock::now());
            if (metrics != nullptr)
            {
                metrics->derived_commitment_point_ms +=
                    duration_ms;
            }
            if (!label.empty())
            {
                progress_note(
                    "trace:derived_point:" + std::string(label) + ":ms=" + std::to_string(duration_ms));
            }
            return point;
        }

        crypto::Commitment commit_proof_polynomial(
            const std::string &name,
            const Polynomial &polynomial,
            const crypto::KZGKeyPair &key,
            RunMetrics *metrics)
        {
            crypto::CommitObjectProfile profile;
            auto commitment = crypto::KZG::commit(
                name, polynomial, key, metrics != nullptr ? &profile : nullptr);
            if (metrics != nullptr)
            {
                metrics->derived_commitment_point_ms += profile.msm_ms;
            }
            progress_note(
                "trace:proof_commit:" + name + ":ms=" + std::to_string(profile.msm_ms));
            return commitment;
        }

        crypto::Commitment commit_reblinded_evaluation_from_anchor(
            const crypto::Commitment &anchor,
            const std::string &name,
            const Polynomial &polynomial,
            const crypto::KZGKeyPair &key,
            RunMetrics *metrics)
        {
            if (anchor.opening_polynomial == nullptr || anchor.opening_polynomial->basis != algebra::PolynomialBasis::Evaluation || polynomial.basis != algebra::PolynomialBasis::Evaluation || anchor.opening_polynomial->domain == nullptr || polynomial.domain == nullptr || anchor.opening_polynomial->domain->size != polynomial.domain->size || !(anchor.opening_polynomial->domain->omega == polynomial.domain->omega) || anchor.opening_polynomial->values() != polynomial.values())
            {
                throw std::runtime_error(
                    "proof-local model polynomial differs from its canonical anchor: " + name);
            }
            const auto start = Clock::now();
            const auto evaluation_component =
                crypto::KZG::evaluation_commitment_component(anchor, key);
            auto commitment =
                crypto::KZG::commit_evaluation_from_component(
                    name, polynomial, evaluation_component, key);
            const auto duration_ms = elapsed_ms(start, Clock::now());
            if (metrics != nullptr)
            {
                metrics->derived_commitment_point_ms +=
                    duration_ms;
            }
            progress_note(
                "trace:model_reblind_derived:" + name + ":ms=" + std::to_string(duration_ms));
            return commitment;
        }

        crypto::Commitment commit_shifted_proof_polynomial(
            const crypto::Commitment &source,
            const std::string &shifted_name,
            std::size_t configured_degree,
            const FieldElement &delta,
            const crypto::KZGKeyPair &key,
            RunMetrics *metrics)
        {
            crypto::CommitObjectProfile profile;
            auto commitment = crypto::KZG::commit_taylor_shifted(
                source,
                shifted_name,
                configured_degree,
                delta,
                key,
                metrics != nullptr ? &profile : nullptr);
            if (metrics != nullptr)
            {
                metrics->derived_commitment_point_ms += profile.msm_ms;
            }
            progress_note(
                "trace:taylor_shift_commit:" + shifted_name + ":ms=" + std::to_string(profile.msm_ms));
            return commitment;
        }

        void progress_commit_batch_profile(
            const std::string &batch_label,
            const crypto::CommitBatchProfile &profile)
        {
            if (!progress_enabled() || profile.objects.empty())
            {
                return;
            }
            std::vector<const crypto::CommitObjectProfile *> objects;
            objects.reserve(profile.objects.size());
            for (const auto &object : profile.objects)
            {
                objects.push_back(&object);
            }
            std::sort(
                objects.begin(),
                objects.end(),
                [](const auto *lhs, const auto *rhs)
                {
                    return lhs->msm_ms > rhs->msm_ms;
                });
            std::ostringstream out;
            out << batch_label << ":msm_profile:total_ms="
                << std::fixed << std::setprecision(3) << profile.msm_ms
                << ":objects=" << profile.objects.size() << ":top=";
            const auto limit = std::min<std::size_t>(objects.size(), 5);
            for (std::size_t i = 0; i < limit; ++i)
            {
                const auto &object = *objects[i];
                if (i != 0)
                {
                    out << ';';
                }
                out << object.name << '@' << object.domain
                    << '/' << object.basis
                    << "/len=" << object.vector_length
                    << "/nz=" << object.nonzero_scalar_count
                    << "/ms=" << std::fixed << std::setprecision(3) << object.msm_ms;
                if (object.sparse_compaction)
                {
                    out << "/sparse";
                }
                if (object.constant_tail_folding)
                {
                    out << "/tail";
                }
                if (object.multithreaded_msm)
                {
                    out << "/mt";
                }
                if (object.semantic_commitment_reuse)
                {
                    out << "/semantic-reuse";
                }
            }
            progress_note(out.str());
        }
        std::vector<FieldElement> padded_column(const std::vector<FieldElement> &values, std::size_t size)
        {
            if (values.size() > size)
            {
                throw std::runtime_error("column values exceed target domain size");
            }
            std::vector<FieldElement> out(size, FieldElement::zero());
            std::copy(values.begin(), values.end(), out.begin());
            return out;
        }

        Polynomial coeff_poly(const std::string &name, const std::vector<FieldElement> &values)
        {
            return Polynomial::from_coefficients(name, values);
        }

        void add_metric(double *metric, const Clock::time_point &start)
        {
            if (metric != nullptr)
            {
                *metric += elapsed_ms(start, Clock::now());
            }
        }

        std::uint64_t fnv1a64(std::string_view text)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto ch : text)
            {
                hash ^= static_cast<unsigned char>(ch);
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        std::string stable_hash(std::string_view text)
        {
            return std::to_string(fnv1a64(text));
        }

        void append_fnv1a64(
            std::uint64_t *hash,
            std::string_view text)
        {
            if (hash == nullptr)
            {
                throw std::runtime_error(
                    "cannot append to a null FNV-1a state");
            }
            for (const auto ch : text)
            {
                *hash ^= static_cast<unsigned char>(ch);
                *hash *= 1099511628211ULL;
            }
        }

        std::string hash_field_matrix(const Matrix &matrix)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            const auto append = [&](std::string_view text)
            {
                append_fnv1a64(&hash, text);
            };
            append("rows=");
            append(std::to_string(matrix.size()));
            for (const auto &row : matrix)
            {
                append("|cols=");
                append(std::to_string(row.size()));
                for (const auto &value : row)
                {
                    append(":");
                    append(value.to_string());
                }
            }
            return std::to_string(hash);
        }

        std::string hash_float_matrix(const model::FloatMatrix &matrix)
        {
            std::ostringstream stream;
            stream << std::setprecision(17);
            stream << "rows=" << matrix.size();
            for (const auto &row : matrix)
            {
                stream << "|cols=" << row.size();
                for (const auto value : row)
                {
                    stream << ':' << value;
                }
            }
            return stable_hash(stream.str());
        }

        FieldElement quantize_float(double value);

        void record_public_output_hashes(
            TraceArtifacts &trace,
            const Matrix &secure_trace_logits,
            const Matrix &reference_quant_logits,
            const model::FloatMatrix &reference_fp32_logits)
        {
            // The quantized reference is the canonical integer forward, not a
            // one-shot rounding of the final FP32 logits.  Quantization happens
            // at every registered projection, attention, lookup and
            // SignedRoundDiv boundary, so rounding only the final floating-point
            // result is a different computation.
            trace.public_logits =
                algebra::flatten_matrix_coefficients(secure_trace_logits);
            trace.public_logits_hash = hash_field_matrix(secure_trace_logits);
            // A valid formal run expects these matrices to be identical, and
            // main checks the exported hashes accordingly. Native field
            // equality is far cheaper than converting millions of identical
            // elements to decimal text a second time. Preserve mismatch
            // diagnostics by hashing the reference only when equality fails.
            trace.reference_quant_logits_hash =
                secure_trace_logits == reference_quant_logits
                    ? trace.public_logits_hash
                    : hash_field_matrix(reference_quant_logits);
            trace.secure_trace_logits_hash = trace.public_logits_hash;
            trace.proof_public_output_hash.clear();
            trace.reference_fp32_logits_hash =
                hash_float_matrix(reference_fp32_logits);
        }

        template <typename Fn>
        void parallel_for_ranges(std::size_t item_count, std::size_t min_parallel_items, Fn &&fn)
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto task_count =
                (item_count >= min_parallel_items && cpu_count > 1)
                    ? std::min<std::size_t>(cpu_count, item_count)
                    : 1;
            if (task_count <= 1)
            {
                fn(0, item_count);
                return;
            }

            const auto chunk_size = (item_count + task_count - 1) / task_count;
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                const auto begin = task * chunk_size;
                const auto end = std::min(item_count, begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(
                    std::launch::async,
                    [&fn, begin, end]()
                    {
                        fn(begin, end);
                    }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
        }

        std::vector<FieldElement> powers(const FieldElement &base, std::size_t count)
        {
            std::vector<FieldElement> out(count, FieldElement::one());
            for (std::size_t i = 1; i < count; ++i)
            {
                out[i] = out[i - 1] * base;
            }
            return out;
        }

        std::vector<FieldElement> strided_powers(const FieldElement &base, std::size_t count, std::size_t stride)
        {
            std::vector<FieldElement> out(count, FieldElement::one());
            const auto step = base.pow(static_cast<std::uint64_t>(stride));
            for (std::size_t i = 1; i < count; ++i)
            {
                out[i] = out[i - 1] * step;
            }
            return out;
        }

        const std::vector<mcl::Fr> &cached_barycentric_weights_native(
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const FieldElement &point)
        {
            const auto cache_key =
                domain->name + ":" + std::to_string(domain->size) + ":" + point.to_string();
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return *it->second;
                }
            }

            auto weights = std::make_shared<std::vector<mcl::Fr>>(domain->barycentric_weights_native(point));
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, weights);
                return *it->second;
            }
        }

        std::vector<FieldElement> weighted_column_sum_native(
            const Matrix &matrix,
            const std::vector<mcl::Fr> &row_weights,
            std::size_t row_count)
        {
            if (matrix.empty())
            {
                return {};
            }
            if (row_count > matrix.size() || row_count > row_weights.size())
            {
                throw std::runtime_error("weighted column sum native input size mismatch");
            }
            const auto column_count = matrix.front().size();
            std::vector<mcl::Fr> native_out(column_count);
            for (auto &value : native_out)
            {
                value.clear();
            }

            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            if (row_count >= 256 && column_count >= 4 && cpu_count > 1)
            {
                const auto task_count = std::min<std::size_t>(cpu_count, row_count);
                const auto chunk_size = (row_count + task_count - 1) / task_count;
                std::vector<std::future<std::vector<mcl::Fr>>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(row_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&matrix, &row_weights, begin, end, column_count]()
                        {
                            std::vector<mcl::Fr> partial(column_count);
                            for (auto &value : partial)
                            {
                                value.clear();
                            }
                            for (std::size_t row = begin; row < end; ++row)
                            {
                                const auto &weight = row_weights[row];
                                for (std::size_t column = 0; column < column_count; ++column)
                                {
                                    mcl::Fr term;
                                    mcl::Fr::mul(term, matrix[row][column].native(), weight);
                                    mcl::Fr::add(partial[column], partial[column], term);
                                }
                            }
                            return partial;
                        }));
                }
                for (auto &future : futures)
                {
                    const auto partial = future.get();
                    for (std::size_t column = 0; column < column_count; ++column)
                    {
                        mcl::Fr::add(native_out[column], native_out[column], partial[column]);
                    }
                }
            }
            else
            {
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    const auto &weight = row_weights[row];
                    for (std::size_t column = 0; column < column_count; ++column)
                    {
                        mcl::Fr term;
                        mcl::Fr::mul(term, matrix[row][column].native(), weight);
                        mcl::Fr::add(native_out[column], native_out[column], term);
                    }
                }
            }

            std::vector<FieldElement> out(column_count, FieldElement::zero());
            for (std::size_t column = 0; column < column_count; ++column)
            {
                out[column] = FieldElement::from_native(native_out[column]);
            }
            return out;
        }

        std::vector<FieldElement> linear_form_by_powers(
            const std::vector<std::vector<FieldElement>> &matrix,
            const std::vector<FieldElement> &column_powers)
        {
            if (matrix.empty())
            {
                return {};
            }
            std::vector<FieldElement> out(matrix.size(), FieldElement::zero());
            parallel_for_ranges(
                matrix.size(),
                256,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        mcl::Fr sum;
                        sum.clear();
                        for (std::size_t j = 0; j < matrix[i].size(); ++j)
                        {
                            mcl::Fr term;
                            mcl::Fr::mul(term, matrix[i][j].native(), column_powers[j].native());
                            mcl::Fr::add(sum, sum, term);
                        }
                        out[i] = FieldElement::from_native(sum);
                    }
                });
            return out;
        }

        std::vector<FieldElement> build_route_node_accumulator(
            const std::vector<FieldElement> &inverse,
            const std::vector<FieldElement> &multiplicity,
            std::size_t valid_length,
            std::size_t domain_size)
        {
            if (inverse.size() != domain_size || multiplicity.size() != domain_size)
            {
                throw std::runtime_error("route node accumulator input size mismatch");
            }
            std::vector<FieldElement> out(domain_size, FieldElement::zero());
            if (domain_size == 0 || valid_length == 0)
            {
                return out;
            }

            mcl::Fr running;
            running.clear();
            for (std::size_t i = 0; i < valid_length; ++i)
            {
                mcl::Fr term;
                mcl::Fr::mul(
                    term,
                    multiplicity[i].native(),
                    inverse[i].native());
                mcl::Fr::add(running, running, term);
                out[i + 1] = FieldElement::from_native(running);
            }
            for (std::size_t i = valid_length + 1; i < domain_size; ++i)
            {
                out[i] = out[valid_length];
            }
            return out;
        }

        std::vector<FieldElement> build_route_edge_accumulator(
            const std::vector<FieldElement> &inverse,
            std::size_t valid_length,
            std::size_t domain_size)
        {
            if (inverse.size() != domain_size)
            {
                throw std::runtime_error("route edge accumulator input size mismatch");
            }
            std::vector<FieldElement> out(domain_size, FieldElement::zero());
            if (domain_size == 0 || valid_length == 0)
            {
                return out;
            }

            mcl::Fr running;
            running.clear();
            for (std::size_t i = 0; i < valid_length; ++i)
            {
                mcl::Fr::add(running, running, inverse[i].native());
                out[i + 1] = FieldElement::from_native(running);
            }
            for (std::size_t i = valid_length + 1; i < domain_size; ++i)
            {
                out[i] = out[valid_length];
            }
            return out;
        }

        enum class DynamicCommitmentKind
        {
            Column,
            Matrix,
        };

        struct DynamicCommitmentLinearTerm
        {
            std::optional<crypto::G1Point> point;
            const crypto::Commitment *commitment = nullptr;
            std::string label;
            FieldElement scale = FieldElement::zero();
        };

        struct DynamicCommitmentSpec
        {
            std::string name;
            DynamicCommitmentKind kind = DynamicCommitmentKind::Column;
            std::vector<FieldElement> owned_column_values;
            std::shared_ptr<const std::vector<FieldElement>> shared_column_values;
            const std::vector<FieldElement> *borrowed_column_values = nullptr;
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            const Matrix *matrix = nullptr;
            std::optional<crypto::G1Point> precomputed_commitment_point;
            std::vector<DynamicCommitmentLinearTerm> deferred_linear_terms;
            bool linear_commitment_derived = false;
            bool retain_polynomial = true;
            bool compact = false;
        };

        struct PreparedDynamicCommitment
        {
            std::string name;
            DynamicCommitmentKind kind = DynamicCommitmentKind::Column;
            std::optional<std::vector<FieldElement>> column_export;
            std::optional<Matrix> matrix_export;
            std::optional<Polynomial> polynomial;
            crypto::Commitment commitment;
            std::string domain_name;
            std::size_t domain_size = 0;
            bool compact = false;
            bool shared_or_borrowed = false;
            bool reused_shared = false;
            bool commitment_ready = false;
            bool precomputed_commitment = false;
            bool deferred_linear_commitment = false;
            bool linear_commitment_derived = false;
            bool proof_local_blinded = false;
            std::vector<DynamicCommitmentLinearTerm> deferred_linear_terms;
            std::uintptr_t source_identity = 0;
            double materialization_ms = 0.0;
        };

        bool is_public_dynamic_polynomial(const std::string &name)
        {
            if (name == "P_H" || name == "P_Y" || name == "P_Y_star" || name == "P_Y_star_edge" || name == "P_Table_feat" || name == "P_Query_feat")
            {
                return true;
            }
            const auto public_table_suffix = [&](std::string_view suffix)
            {
                return name == "P_" + std::string(suffix) || name.ends_with("_" + std::string(suffix));
            };
            return public_table_suffix("Table_L") || public_table_suffix("Table_R") || public_table_suffix("Table_exp") || public_table_suffix("Table_ELU") || name.ends_with("_BiasView_m_can");
        }

        std::vector<FieldElement> sample_trace_blinder(
            const ProtocolCompilerArtifact &artifact,
            const std::string &name)
        {
            const auto ledger_it = std::find_if(
                artifact.observation_ledger.begin(),
                artifact.observation_ledger.end(),
                [&](const CompilerLedgerEntry &entry)
                { return entry.object_or_challenge == name; });
            if (ledger_it == artifact.observation_ledger.end() || ledger_it->query_bound == 0)
            {
                throw std::runtime_error(
                    "missing ObservationLedger blinder resource: " + name);
            }
            const auto kBlinderCoefficientCount = ledger_it->query_bound;
            std::vector<FieldElement> out(kBlinderCoefficientCount, FieldElement::zero());
            std::array<std::uint8_t, 64> bytes{};
            for (auto &coefficient : out)
            {
                if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
                {
                    throw std::runtime_error(
                        "CSPRNG failed while sampling trace blinder: " + name);
                }
                coefficient = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }
            while (std::all_of(out.begin(), out.end(), [](const FieldElement &value)
                               { return value.is_zero(); }))
            {
                if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
                {
                    throw std::runtime_error(
                        "CSPRNG failed while resampling trace blinder: " + name);
                }
                out.front() = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }
            return out;
        }

        void append_dynamic_commitment(
            TraceArtifacts &trace,
            const std::string &name,
            crypto::Commitment commitment)
        {
            commitment.name = name;
            trace.commitments[name] = std::move(commitment);
            trace.commitment_order.push_back(name);
        }

        DynamicCommitmentLinearTerm linear_label_term(
            std::string label,
            const FieldElement &scale)
        {
            return DynamicCommitmentLinearTerm{
                .point = std::nullopt,
                .commitment = nullptr,
                .label = std::move(label),
                .scale = scale,
            };
        }

        DynamicCommitmentLinearTerm linear_point_term(
            const crypto::G1Point &point,
            const FieldElement &scale)
        {
            return DynamicCommitmentLinearTerm{
                .point = point,
                .commitment = nullptr,
                .label = {},
                .scale = scale,
            };
        }

        DynamicCommitmentLinearTerm linear_commitment_term(
            const crypto::Commitment &commitment,
            const FieldElement &scale)
        {
            return DynamicCommitmentLinearTerm{
                .point = std::nullopt,
                .commitment = &commitment,
                .label = {},
                .scale = scale,
            };
        }

        const crypto::Commitment *find_linear_dependency_commitment(
            const std::string &label,
            const TraceArtifacts &trace,
            const std::unordered_map<std::string, const crypto::Commitment *> &batch_commitments)
        {
            if (const auto batch_it = batch_commitments.find(label);
                batch_it != batch_commitments.end())
            {
                return batch_it->second;
            }
            if (const auto trace_it = trace.commitments.find(label);
                trace_it != trace.commitments.end())
            {
                return &trace_it->second;
            }
            return nullptr;
        }

        std::shared_ptr<const Polynomial> load_spilled_evaluation_polynomial(
            const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
            const std::string &name,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            if (domain == nullptr || spilled.domain_name != domain->name || spilled.size > domain->size)
            {
                throw std::runtime_error(
                    "spilled polynomial domain mismatch: " + name);
            }
            std::ifstream stream(spilled.path, std::ios::binary);
            if (!stream)
            {
                throw std::runtime_error(
                    "failed to open spilled polynomial: " + spilled.path);
            }
            std::vector<FieldElement> values;
            try
            {
                values = algebra::read_field_elements_binary(
                    stream, spilled.size);
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(
                    "failed to read spilled polynomial: " + spilled.path + ": " + error.what());
            }
            auto polynomial = Polynomial::from_shared_evaluations(
                name,
                std::make_shared<const std::vector<FieldElement>>(
                    std::move(values)),
                domain);
            polynomial.set_zero_domain_blinder(
                spilled.zero_domain_blinder);
            return std::make_shared<const Polynomial>(
                std::move(polynomial));
        }

        std::shared_ptr<const Polynomial> linear_dependency_opening_material(
            const std::string &label,
            const crypto::Commitment &commitment,
            const TraceArtifacts &trace,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            if (commitment.opening_polynomial != nullptr)
            {
                return commitment.opening_polynomial;
            }
            const auto spilled_it =
                trace.spilled_polynomials.find(label);
            if (spilled_it == trace.spilled_polynomials.end())
            {
                throw std::runtime_error(
                    "missing polynomial dependency material: " + label);
            }
            return load_spilled_evaluation_polynomial(
                spilled_it->second, label, domain);
        }

        crypto::Commitment trace_commitment_with_opening_material(
            const TraceArtifacts &trace,
            const std::string &label,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            const auto commitment_it = trace.commitments.find(label);
            if (commitment_it == trace.commitments.end())
            {
                throw std::runtime_error(
                    "trace opening source commitment is missing: " + label);
            }
            auto resolved = commitment_it->second;
            resolved.opening_polynomial =
                resolve_trace_opening_polynomial(trace, label, domain);
            return resolved;
        }

        FieldElement evaluate_trace_polynomial(
            const TraceArtifacts &trace,
            const std::string &name,
            const FieldElement &point,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            if (const auto polynomial_it =
                    trace.polynomials.find(name);
                polynomial_it != trace.polynomials.end())
            {
                return polynomial_it->second.evaluate(point);
            }
            const auto spilled_it =
                trace.spilled_polynomials.find(name);
            if (spilled_it == trace.spilled_polynomials.end())
            {
                throw std::runtime_error(
                    "trace evaluation is missing polynomial: " + name);
            }
            return load_spilled_evaluation_polynomial(
                       spilled_it->second, name, domain)
                ->evaluate(point);
        }

        crypto::G1Point resolve_deferred_linear_evaluation_component(
            const std::vector<DynamicCommitmentLinearTerm> &terms,
            const TraceArtifacts &trace,
            const std::unordered_map<std::string, const crypto::Commitment *> &batch_commitments,
            const crypto::KZGKeyPair &key,
            const std::string &name,
            const std::shared_ptr<algebra::RootOfUnityDomain> &target_domain)
        {
            auto out = crypto::g1_zero();
            for (const auto &term : terms)
            {
                if (term.scale == FieldElement::zero())
                {
                    continue;
                }
                const crypto::G1Point *base = nullptr;
                crypto::G1Point dependency_component;
                if (term.point.has_value())
                {
                    base = &*term.point;
                }
                else if (term.commitment != nullptr)
                {
                    if (term.commitment->opening_polynomial == nullptr)
                    {
                        throw std::runtime_error(
                            "explicit deferred linear commitment is missing opening material: " + name);
                    }
                    dependency_component =
                        crypto::KZG::evaluation_commitment_component(
                            *term.commitment, key);
                    base = &dependency_component;
                }
                else
                {
                    const auto *commitment =
                        find_linear_dependency_commitment(
                            term.label, trace, batch_commitments);
                    if (commitment != nullptr)
                    {
                        auto material =
                            linear_dependency_opening_material(
                                term.label,
                                *commitment,
                                trace,
                                target_domain);
                        auto resolved_commitment = *commitment;
                        resolved_commitment.opening_polynomial =
                            std::move(material);
                        dependency_component =
                            crypto::KZG::evaluation_commitment_component(
                                resolved_commitment, key);
                        base = &dependency_component;
                    }
                }
                if (base == nullptr)
                {
                    throw std::runtime_error(
                        "missing dependency for deferred linear dynamic commitment " + name + ": " + term.label);
                }
                out = crypto::g1_add(out, crypto::g1_mul(*base, term.scale));
            }
            return out;
        }

        void validate_deferred_linear_evaluation_values(
            const std::vector<DynamicCommitmentLinearTerm> &terms,
            const Polynomial &target,
            const TraceArtifacts &trace,
            const std::unordered_map<std::string, const crypto::Commitment *> &batch_commitments,
            const std::string &name)
        {
            if (target.basis != algebra::PolynomialBasis::Evaluation || target.domain == nullptr)
            {
                throw std::runtime_error(
                    "deferred linear target is not an evaluation polynomial: " + name);
            }
            for (const auto &term : terms)
            {
                // Explicit point terms have no opening material and therefore
                // cannot participate in the element-wise source check.
                if (term.point.has_value())
                {
                    return;
                }
            }

            std::vector<const Polynomial *> dependencies;
            std::vector<std::shared_ptr<const Polynomial>>
                dependency_owners;
            dependencies.reserve(terms.size());
            dependency_owners.reserve(terms.size());
            for (const auto &term : terms)
            {
                if (term.scale == FieldElement::zero())
                {
                    dependencies.push_back(nullptr);
                    dependency_owners.push_back(nullptr);
                    continue;
                }
                const auto *commitment = term.commitment != nullptr
                                             ? term.commitment
                                             : find_linear_dependency_commitment(
                                                   term.label, trace, batch_commitments);
                if (commitment == nullptr)
                {
                    throw std::runtime_error(
                        "missing polynomial dependency for deferred linear dynamic commitment " + name + ": " + term.label);
                }
                std::shared_ptr<const Polynomial> material;
                try
                {
                    material = term.commitment != nullptr
                                   ? commitment->opening_polynomial
                                   : linear_dependency_opening_material(
                                         term.label,
                                         *commitment,
                                         trace,
                                         target.domain);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "missing polynomial dependency for deferred linear dynamic commitment " + name + ": " + term.label);
                }
                if (material == nullptr)
                {
                    throw std::runtime_error(
                        "missing polynomial dependency for deferred linear dynamic commitment " + name + ": " + term.label);
                }
                const auto *source = material.get();
                if (source->basis != algebra::PolynomialBasis::Evaluation || source->domain == nullptr || source->size() > target.size() || source->domain->size != target.domain->size || !(source->domain->omega == target.domain->omega))
                {
                    throw std::runtime_error(
                        "domain mismatch for deferred linear dynamic commitment " + name + ": " + term.label);
                }
                dependency_owners.push_back(std::move(material));
                dependencies.push_back(source);
            }

            const auto &target_values = target.values();
            for (std::size_t i = 0; i < target_values.size(); ++i)
            {
                auto expected = FieldElement::zero();
                for (std::size_t term_index = 0;
                     term_index < terms.size(); ++term_index)
                {
                    if (dependencies[term_index] == nullptr)
                    {
                        continue;
                    }
                    const auto &source_values =
                        dependencies[term_index]->values();
                    if (i >= source_values.size())
                    {
                        continue;
                    }
                    expected +=
                        source_values[i] * terms[term_index].scale;
                }
                if (expected != target_values[i])
                {
                    throw std::runtime_error(
                        "value mismatch for deferred linear dynamic commitment " + name + " at index " + std::to_string(i));
                }
            }
        }

        void trim_coefficients(std::vector<FieldElement> *coefficients)
        {
            while (coefficients->size() > 1 && coefficients->back() == FieldElement::zero())
            {
                coefficients->pop_back();
            }
        }

        Polynomial fh_evaluations_to_coefficients(
            const std::string &name,
            std::vector<FieldElement> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            if (domain == nullptr || values.size() != domain->size)
            {
                throw std::runtime_error("FH coefficient materialization domain mismatch: " + name);
            }
            auto *native_values = FieldElement::mutable_native_array_data(values.data());
            algebra::ntt_plan(values.size())->intt(native_values);
            trim_coefficients(&values);
            return Polynomial::from_coefficients(name, std::move(values));
        }

        std::shared_ptr<const Polynomial> cached_fh_coefficient_material(
            const crypto::Commitment &source,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            if (source.opening_polynomial == nullptr)
            {
                throw std::runtime_error("missing FH source material for coefficient alias: " + source.name);
            }
            if (source.opening_polynomial->basis == algebra::PolynomialBasis::Coefficient)
            {
                return source.opening_polynomial;
            }
            if (source.opening_polynomial->domain == nullptr || domain == nullptr || source.opening_polynomial->domain->size != domain->size || !(source.opening_polynomial->domain->omega == domain->omega))
            {
                throw std::runtime_error("FH coefficient alias domain mismatch: " + source.name);
            }

            static std::mutex mutex;
            static std::unordered_map<const Polynomial *, std::weak_ptr<const Polynomial>> cache;
            const auto *key = source.opening_polynomial.get();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (const auto it = cache.find(key); it != cache.end())
                {
                    if (auto cached = it->second.lock())
                    {
                        return cached;
                    }
                }
            }

            auto values = source.opening_polynomial->values();
            auto material = std::make_shared<Polynomial>(
                fh_evaluations_to_coefficients(source.opening_polynomial->name, std::move(values), domain));
            {
                std::lock_guard<std::mutex> lock(mutex);
                cache[key] = material;
            }
            return material;
        }

        crypto::Commitment fh_coefficient_renamed_commitment(
            const std::string &name,
            const ProtocolContext &context,
            const std::string &source_name,
            const crypto::Commitment &source,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            std::shared_ptr<const Polynomial> material;
            if (const auto it = context.public_fh_coefficient_materials.find(source_name);
                it != context.public_fh_coefficient_materials.end())
            {
                material = it->second;
            }
            else
            {
                material = cached_fh_coefficient_material(source, domain);
            }
            return crypto::Commitment{
                .name = name,
                .point = source.point,
                .opening_polynomial = std::move(material),
            };
        }

        std::size_t estimated_dynamic_commitment_payload_bytes(const DynamicCommitmentSpec &spec)
        {
            if (spec.kind == DynamicCommitmentKind::Column)
            {
                const auto *values = spec.shared_column_values != nullptr
                                         ? spec.shared_column_values.get()
                                         : (spec.borrowed_column_values != nullptr ? spec.borrowed_column_values : &spec.owned_column_values);
                return values->size() * sizeof(FieldElement);
            }
            if (spec.matrix == nullptr)
            {
                return 0;
            }
            std::size_t rows = spec.matrix->size();
            std::size_t cols = rows == 0 ? 0 : (*spec.matrix)[0].size();
            return rows * cols * sizeof(FieldElement);
        }

        std::size_t dynamic_commitment_prepared_budget_bytes()
        {
            static constexpr std::size_t kMinimumBudgetBytes =
                96ULL * 1024ULL * 1024ULL;

            static constexpr std::size_t kMaximumBudgetBytes =
                2ULL * 1024ULL * 1024ULL * 1024ULL;

            // The old fixed 96 MiB window fragments large same-SRS
            // commitment families before they can reach KZG::commit_batch.
            //
            // Size the execution window from the host/unified-memory class,
            // while keeping a conservative hard ceiling. This changes only
            // how many already-required commitment objects are prepared
            // together; it does not change their values, labels or order.
            static const std::size_t budget = []()
            {
                std::ifstream input("/proc/meminfo");

                if (!input)
                {
                    return kMinimumBudgetBytes;
                }

                std::string line;

                while (std::getline(input, line))
                {
                    std::istringstream row(line);

                    std::string key;
                    std::uint64_t kibibytes = 0;

                    if (!(row >> key >> kibibytes) || key != "MemTotal:")
                    {
                        continue;
                    }

                    constexpr std::uint64_t kKiB =
                        1024ULL;

                    const auto max_size =
                        static_cast<std::uint64_t>(
                            std::numeric_limits<
                                std::size_t>::max());

                    if (kibibytes >
                        max_size / kKiB)
                    {
                        return kMaximumBudgetBytes;
                    }

                    const auto total_bytes =
                        static_cast<std::size_t>(
                            kibibytes * kKiB);

                    // At most 1/64 of physical/unified memory is admitted
                    // as simultaneously prepared trace commitment payload.
                    // 128 GiB therefore admits the 2 GiB ceiling, while
                    // smaller machines automatically retain smaller windows.
                    const std::size_t adaptive_budget =
                        total_bytes / std::size_t{64};

                    return std::clamp(
                        adaptive_budget,
                        kMinimumBudgetBytes,
                        kMaximumBudgetBytes);
                }

                return kMinimumBudgetBytes;
            }();

            return budget;
        }

        std::size_t dynamic_commitment_prepared_budget_bytes(
            const std::vector<DynamicCommitmentSpec> &specs)
        {
            // A full 2^23 edge-domain FieldElement column is approximately
            // 256 MiB. For this workload class, retaining three or four such
            // objects in one prepared family creates an unnecessarily large
            // host/UMA working set before KZG execution.
            //
            // Keep at most roughly two full large-edge columns prepared
            // together. Smaller edge workloads and FH workloads retain the
            // broader host-adaptive window from the no-argument helper.
            constexpr std::size_t kLargeEdgeDomainSize =
                1ULL << 23;

            constexpr std::size_t kLargeEdgeFamilyBudgetBytes =
                512ULL * 1024ULL * 1024ULL;

            const auto host_budget =
                dynamic_commitment_prepared_budget_bytes();

            if (specs.size() <= 1)
            {
                return host_budget;
            }

            bool large_edge_family = false;
            std::size_t total_payload_bytes = 0;

            for (const auto &spec : specs)
            {
                if (spec.domain != nullptr &&
                    spec.domain->name == "edge" &&
                    spec.domain->size >= kLargeEdgeDomainSize)
                {
                    large_edge_family = true;
                }

                const auto spec_bytes =
                    estimated_dynamic_commitment_payload_bytes(
                        spec);

                if (spec_bytes >
                    std::numeric_limits<
                        std::size_t>::max() -
                        total_payload_bytes)
                {
                    total_payload_bytes =
                        std::numeric_limits<
                            std::size_t>::max();
                }
                else
                {
                    total_payload_bytes +=
                        spec_bytes;
                }
            }

            if (!large_edge_family ||
                total_payload_bytes <=
                    kLargeEdgeFamilyBudgetBytes)
            {
                return host_budget;
            }

            return std::min(
                host_budget,
                kLargeEdgeFamilyBudgetBytes);
        }
        std::size_t
        dynamic_commitment_materialization_worker_count(
            const std::vector<DynamicCommitmentSpec> &specs,
            std::size_t prepared_budget_bytes)
        {
            if (specs.size() <= 1)
            {
                return 1;
            }

            const auto hardware_workers =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            if (hardware_workers <= 1)
            {
                return 1;
            }

            std::size_t largest_payload_bytes = 0;

            for (const auto &spec : specs)
            {
                largest_payload_bytes =
                    std::max(
                        largest_payload_bytes,
                        estimated_dynamic_commitment_payload_bytes(
                            spec));
            }

            constexpr std::size_t
                kMaximumMaterializationWorkers = 8;

            const auto workers_by_memory =
                largest_payload_bytes == 0
                    ? kMaximumMaterializationWorkers
                    : std::max<std::size_t>(
                          1,
                          prepared_budget_bytes / largest_payload_bytes);

            return std::max<std::size_t>(
                1,
                std::min({
                    specs.size(),
                    hardware_workers,
                    kMaximumMaterializationWorkers,
                    workers_by_memory,
                }));
        }

        DynamicCommitmentSpec column_spec(
            const std::string &name,
            std::vector<FieldElement> &&values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            DynamicCommitmentSpec spec;
            spec.name = name;
            spec.kind = DynamicCommitmentKind::Column;
            spec.owned_column_values = std::move(values);
            spec.domain = domain;
            return spec;
        }

        DynamicCommitmentSpec column_spec(
            const std::string &name,
            std::shared_ptr<const std::vector<FieldElement>> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            DynamicCommitmentSpec spec;
            spec.name = name;
            spec.kind = DynamicCommitmentKind::Column;
            spec.shared_column_values = std::move(values);
            spec.domain = domain;
            return spec;
        }

        DynamicCommitmentSpec column_spec(
            const std::string &name,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            DynamicCommitmentSpec spec;
            spec.name = name;
            spec.kind = DynamicCommitmentKind::Column;
            spec.borrowed_column_values = &values;
            spec.domain = domain;
            return spec;
        }

        DynamicCommitmentSpec linear_column_spec(
            const std::string &name,
            std::vector<FieldElement> &&values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const crypto::G1Point &commitment_point)
        {
            auto spec = column_spec(name, std::move(values), domain);
            spec.precomputed_commitment_point = commitment_point;
            spec.linear_commitment_derived = true;
            return spec;
        }

        DynamicCommitmentSpec deferred_linear_column_spec(
            const std::string &name,
            std::vector<FieldElement> &&values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::vector<DynamicCommitmentLinearTerm> terms)
        {
            auto spec = column_spec(name, std::move(values), domain);
            spec.deferred_linear_terms = std::move(terms);
            spec.linear_commitment_derived = true;
            return spec;
        }

        DynamicCommitmentSpec deferred_linear_column_spec(
            const std::string &name,
            std::shared_ptr<const std::vector<FieldElement>> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::vector<DynamicCommitmentLinearTerm> terms)
        {
            auto spec = column_spec(name, std::move(values), domain);
            spec.deferred_linear_terms = std::move(terms);
            spec.linear_commitment_derived = true;
            return spec;
        }

        DynamicCommitmentSpec deferred_linear_column_spec(
            const std::string &name,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::vector<DynamicCommitmentLinearTerm> terms)
        {
            auto spec = column_spec(name, values, domain);
            spec.deferred_linear_terms = std::move(terms);
            spec.linear_commitment_derived = true;
            return spec;
        }

        DynamicCommitmentSpec deferred_linear_compact_column_spec(
            const std::string &name,
            std::shared_ptr<const std::vector<FieldElement>> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::vector<DynamicCommitmentLinearTerm> terms)
        {
            auto spec = deferred_linear_column_spec(name, std::move(values), domain, std::move(terms));
            spec.compact = true;
            return spec;
        }

        DynamicCommitmentSpec compact_column_spec(
            const std::string &name,
            std::vector<FieldElement> &&values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            auto spec = column_spec(name, std::move(values), domain);
            spec.compact = true;
            return spec;
        }

        DynamicCommitmentSpec compact_column_spec(
            const std::string &name,
            std::shared_ptr<const std::vector<FieldElement>> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            auto spec = column_spec(name, std::move(values), domain);
            spec.compact = true;
            return spec;
        }

        template <typename Owner>
        std::shared_ptr<const std::vector<FieldElement>> alias_column_values(
            const std::shared_ptr<const Owner> &owner,
            const std::vector<FieldElement> *values)
        {
            if (owner == nullptr || values == nullptr)
            {
                throw std::runtime_error("shared dynamic commitment column is missing its owner");
            }
            return std::shared_ptr<const std::vector<FieldElement>>(owner, values);
        }

        template <typename Owner>
        DynamicCommitmentSpec shared_column_spec(
            const std::string &name,
            const std::shared_ptr<const Owner> &owner,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return column_spec(name, alias_column_values(owner, &values), domain);
        }

        template <typename Owner>
        DynamicCommitmentSpec deferred_linear_compact_shared_column_spec(
            const std::string &name,
            const std::shared_ptr<const Owner> &owner,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::vector<DynamicCommitmentLinearTerm> terms)
        {
            return deferred_linear_compact_column_spec(
                name,
                alias_column_values(owner, &values),
                domain,
                std::move(terms));
        }

        template <typename Owner>
        DynamicCommitmentSpec shared_compact_column_spec(
            const std::string &name,
            const std::shared_ptr<const Owner> &owner,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return compact_column_spec(name, alias_column_values(owner, &values), domain);
        }

        DynamicCommitmentSpec matrix_spec(const std::string &name, const Matrix &matrix)
        {
            DynamicCommitmentSpec spec;
            spec.name = name;
            spec.kind = DynamicCommitmentKind::Matrix;
            spec.matrix = &matrix;
            return spec;
        }

        DynamicCommitmentSpec matrix_commitment_only_spec(const std::string &name, const Matrix &matrix)
        {
            auto spec = matrix_spec(name, matrix);
            return spec;
        }

        FieldElement row_polynomial_at_point(const std::vector<FieldElement> &row, const FieldElement &point)
        {
            mcl::Fr out;
            out.clear();
            for (std::size_t column = row.size(); column-- > 0;)
            {
                mcl::Fr::mul(out, out, point.native());
                mcl::Fr::add(out, out, row[column].native());
            }
            return FieldElement::from_native(out);
        }

        FieldElement matrix_row_major_evaluation(const Matrix &matrix, const FieldElement &point)
        {
            if (matrix.empty())
            {
                return FieldElement::zero();
            }
            if (matrix.front().empty())
            {
                return FieldElement::zero();
            }

            const auto &route2 = util::route2_options();
            const auto row_count = matrix.size();
            const auto column_count = matrix.front().size();
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            std::vector<FieldElement> row_evaluations(row_count, FieldElement::zero());

            // 这里保持与 row-major 扁平化完全等价的系数多项式，
            // 只是工程上改为“逐行在点评估后再按 `point^cols` 折叠”，
            // 从而避免为矩阵承诺额外构造一次临时扁平系数向量。
            if (route2.trace_layout_upgrade && row_count >= 8 && cpu_count > 1)
            {
                const auto task_count = std::min<std::size_t>(cpu_count, row_count);
                const auto chunk_size = (row_count + task_count - 1) / task_count;
                std::vector<std::future<void>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(row_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&matrix, &row_evaluations, &point, begin, end]()
                        {
                            for (std::size_t row = begin; row < end; ++row)
                            {
                                row_evaluations[row] = row_polynomial_at_point(matrix[row], point);
                            }
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            }
            else
            {
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    row_evaluations[row] = row_polynomial_at_point(matrix[row], point);
                }
            }

            const auto row_stride = point.pow(static_cast<std::uint64_t>(column_count));
            mcl::Fr out;
            out.clear();
            for (std::size_t row = row_count; row-- > 0;)
            {
                mcl::Fr::mul(out, out, row_stride.native());
                mcl::Fr::add(out, out, row_evaluations[row].native());
            }
            return FieldElement::from_native(out);
        }

        void require_flat_row_major_matches_matrix(
            const std::vector<FieldElement> &flat,
            const Matrix &matrix,
            std::size_t expected_rows,
            std::size_t expected_columns,
            std::string_view label)
        {
            if (matrix.size() != expected_rows)
            {
                throw std::runtime_error(
                    std::string(label) + " row count mismatch");
            }

            if (expected_rows != 0 && flat.size() != expected_rows * expected_columns)
            {
                throw std::runtime_error(
                    std::string(label) + " flat length mismatch");
            }

            for (std::size_t row = 0;
                 row < expected_rows;
                 ++row)
            {
                if (matrix[row].size() != expected_columns)
                {
                    throw std::runtime_error(
                        std::string(label) + " column count mismatch");
                }

                for (std::size_t column = 0;
                     column < expected_columns;
                     ++column)
                {
                    const auto flat_index =
                        row * expected_columns + column;

                    if (flat[flat_index] != matrix[row][column])
                    {
                        throw std::runtime_error(
                            std::string(label) + " flat/matrix value mismatch");
                    }
                }
            }
        }

        void attach_precomputed_commitment(
            PreparedDynamicCommitment *out,
            const DynamicCommitmentSpec &spec)
        {
            if (!spec.precomputed_commitment_point.has_value())
            {
                return;
            }
            if (out == nullptr || !out->polynomial.has_value())
            {
                throw std::runtime_error("precomputed dynamic commitment is missing polynomial material: " + spec.name);
            }
            auto material = std::make_shared<Polynomial>(*out->polynomial);
            material->name = spec.name;
            out->commitment = crypto::Commitment{
                .name = spec.name,
                .point = *spec.precomputed_commitment_point,
                .opening_polynomial = std::move(material),
            };
            out->commitment_ready = true;
            out->precomputed_commitment = true;
            out->linear_commitment_derived = spec.linear_commitment_derived;
        }

        void promote_evaluation_values_to_shared_backing(Polynomial *polynomial)
        {
            constexpr std::size_t kMinSharedValueCount = 1024;
            if (polynomial == nullptr || polynomial->basis != algebra::PolynomialBasis::Evaluation || polynomial->shared_data != nullptr || polynomial->data.size() < kMinSharedValueCount)
            {
                return;
            }
            polynomial->shared_data =
                std::make_shared<const std::vector<FieldElement>>(
                    std::move(polynomial->data));
            polynomial->data.clear();
        }

        void fill_feature_lookup_rows(
            const std::vector<FieldElement> &row_ids,
            const std::vector<FieldElement> &feature_indices,
            const Matrix &features,
            const FieldElement &eta_feature_index,
            const FieldElement &eta_feature_value,
            std::vector<FieldElement> &out)
        {
            if (features.empty())
            {
                return;
            }
            const auto row_count = features.size();
            const auto row_width = feature_indices.size();
            parallel_for_ranges(
                row_count,
                32,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        const auto base = row * row_width;
                        const auto &row_id = row_ids[row];
                        const auto &feature_row = features[row];
                        for (std::size_t column = 0; column < row_width; ++column)
                        {
                            out[base + column] =
                                row_id + eta_feature_index * feature_indices[column] + eta_feature_value * feature_row[column];
                        }
                    }
                });
        }

        void fill_repeated_row_values(
            const std::vector<FieldElement> &row_values,
            std::size_t row_width,
            std::vector<FieldElement> &out)
        {
            if (row_width == 0 || row_values.empty())
            {
                return;
            }
            parallel_for_ranges(
                row_values.size(),
                32,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        std::fill_n(
                            out.begin() + static_cast<std::ptrdiff_t>(row * row_width),
                            static_cast<std::ptrdiff_t>(row_width),
                            row_values[row]);
                    }
                });
        }

        PreparedDynamicCommitment materialize_commitment(
            DynamicCommitmentSpec spec,
            const ProtocolCompilerArtifact &compiler_artifact,
            bool keep_trace_payloads)
        {
            constexpr std::size_t kMinSharedColumnLength = 1024;
            const auto materialize_start = Clock::now();
            PreparedDynamicCommitment out;
            out.name = spec.name;
            out.kind = spec.kind;
            out.domain_name = spec.domain != nullptr ? spec.domain->name : "coef";
            out.domain_size = spec.domain != nullptr ? spec.domain->size : 0;
            out.compact = spec.compact;
            out.shared_or_borrowed = spec.shared_column_values != nullptr || spec.borrowed_column_values != nullptr;
            out.deferred_linear_terms = std::move(spec.deferred_linear_terms);
            out.deferred_linear_commitment = !out.deferred_linear_terms.empty();
            out.linear_commitment_derived = spec.linear_commitment_derived;
            if (spec.kind == DynamicCommitmentKind::Column)
            {
                const auto *source_values = spec.shared_column_values != nullptr
                                                ? spec.shared_column_values.get()
                                                : (spec.borrowed_column_values != nullptr ? spec.borrowed_column_values : &spec.owned_column_values);
                const bool use_shared_backing =
                    spec.shared_column_values != nullptr && source_values->size() >= kMinSharedColumnLength;
                out.reused_shared = use_shared_backing;
                if (use_shared_backing)
                {
                    out.source_identity = reinterpret_cast<std::uintptr_t>(source_values);
                }
                if (keep_trace_payloads)
                {
                    out.column_export = *source_values;
                }
                if (use_shared_backing)
                {
                    out.polynomial.emplace(Polynomial::from_shared_evaluations(
                        spec.name,
                        std::move(spec.shared_column_values),
                        spec.domain));
                }
                else if (spec.shared_column_values != nullptr)
                {
                    out.polynomial.emplace(Polynomial::from_evaluations(
                        spec.name,
                        std::vector<FieldElement>(source_values->begin(), source_values->end()),
                        spec.domain));
                }
                else if (spec.borrowed_column_values != nullptr)
                {
                    out.polynomial.emplace(Polynomial::from_evaluations(
                        spec.name,
                        std::vector<FieldElement>(source_values->begin(), source_values->end()),
                        spec.domain));
                }
                else
                {
                    out.polynomial.emplace(Polynomial::from_evaluations(
                        spec.name,
                        std::move(spec.owned_column_values),
                        spec.domain));
                }
                if (!is_public_dynamic_polynomial(spec.name))
                {
                    out.polynomial->set_zero_domain_blinder(
                        sample_trace_blinder(compiler_artifact, spec.name));
                    out.proof_local_blinded = true;
                }
                promote_evaluation_values_to_shared_backing(
                    &*out.polynomial);
                if (is_public_dynamic_polynomial(spec.name))
                {
                    attach_precomputed_commitment(&out, spec);
                }
                out.materialization_ms = elapsed_ms(materialize_start, Clock::now());
                return out;
            }

            if (spec.matrix == nullptr)
            {
                throw std::runtime_error("matrix commitment is missing matrix input");
            }
            if (keep_trace_payloads)
            {
                out.matrix_export = *spec.matrix;
            }
            out.polynomial.emplace(coeff_poly(spec.name, algebra::flatten_matrix_coefficients(*spec.matrix)));
            attach_precomputed_commitment(&out, spec);
            out.materialization_ms = elapsed_ms(materialize_start, Clock::now());
            return out;
        }

        std::string dynamic_commitment_category(const std::string &label)
        {
            if (label.find("PSQ") != std::string::npos || label.find("psq") != std::string::npos)
            {
                return "PSQ";
            }
            if (label.find("Table") != std::string::npos || label.find("Query") != std::string::npos || label.find("_m_") != std::string::npos || label.find("_R_") != std::string::npos || label.ends_with("_R"))
            {
                return "lookup";
            }
            if (label.find("Route") != std::string::npos || label.find("_src") != std::string::npos || label.find("_dst") != std::string::npos || label.find("edge") != std::string::npos)
            {
                return "route";
            }
            if (label.find("ELU") != std::string::npos || label.find("H_agg") != std::string::npos)
            {
                return "activation";
            }
            if (label.find("Y") != std::string::npos || label.rfind("P_out_", 0) == 0)
            {
                return "output";
            }
            if (label.find("alpha") != std::string::npos || label.find("Sum") != std::string::npos || label.find("inv") != std::string::npos)
            {
                return "attention";
            }
            return "trace";
        }

        std::string dynamic_commitment_ownership(const std::string &label)
        {
            if (label.rfind("P_out_", 0) == 0 || label == "P_Y" || label == "P_Y_lin")
            {
                return "output";
            }
            if (label.rfind("P_h", 0) == 0)
            {
                const auto marker = label.find('_');
                if (marker != std::string::npos)
                {
                    return "hidden_head/" + label.substr(1, marker - 1);
                }
            }
            if (label.find("_cat") != std::string::npos || label.find("H_cat") != std::string::npos)
            {
                return "hidden_concat";
            }
            return "global";
        }

        std::string dynamic_commitment_classification(const std::string &label, const PreparedDynamicCommitment &item)
        {
            if (label == "P_H")
            {
                return "public_input_dynamic";
            }
            if (label == "P_Y" || label == "P_Y_lin")
            {
                return "public_output_dynamic";
            }
            if (item.reused_shared)
            {
                return "private_trace_shared";
            }
            if (item.compact)
            {
                return "private_trace_compact";
            }
            return item.kind == DynamicCommitmentKind::Matrix ? "private_trace_matrix" : "private_trace_column";
        }

        std::string dynamic_commitment_opportunity(
            const PreparedDynamicCommitment &item,
            const crypto::CommitObjectProfile &object,
            double zero_scalar_ratio,
            double padding_ratio)
        {
            if (item.linear_commitment_derived)
            {
                return "linear_commitment_derivation";
            }
            if (object.constant_tail_folding)
            {
                return "constant_tail_commitment_folding";
            }
            if (object.sparse_compaction)
            {
                return "safe_sparse_srs_msm";
            }
            if (item.compact && padding_ratio > 0.0)
            {
                return "compact_padding_aware_msm";
            }
            if (zero_scalar_ratio >= 0.25)
            {
                return "zero_scalar_skipping";
            }
            if (item.reused_shared)
            {
                return "shared_column_backing";
            }
            if (object.multithreaded_msm)
            {
                return "multithreaded_srs_msm";
            }
            return "dense_srs_msm";
        }

        void record_dynamic_commit_totals(RunMetrics *metrics)
        {
            if (metrics == nullptr)
            {
                return;
            }
            metrics->commit_dynamic_ms =
                metrics->dynamic_commit_input_ms + metrics->dynamic_polynomial_materialization_ms + metrics->dynamic_commit_pack_ms + metrics->dynamic_fft_ms + metrics->dynamic_domain_convert_ms + metrics->dynamic_copy_convert_ms + metrics->dynamic_commit_msm_ms + metrics->dynamic_bundle_finalize_ms;
            metrics->dynamic_commit_finalize_ms =
                metrics->dynamic_copy_convert_ms + metrics->dynamic_bundle_finalize_ms;
        }

        bool should_spill_dynamic_polynomial(
            const std::string &name,
            const algebra::Polynomial &polynomial,
            bool keep_trace_payloads)
        {
            // A single 2^22 evaluation column is already 128 MiB. Retaining
            // every edge/FH opening column makes whole-graph traces grow with
            // the number of committed objects and can exceed host memory.
            // File backing is execution-only: values, blinders, commitments,
            // labels, and the canonical proof schema remain unchanged.
            constexpr std::size_t kMinFileBackedValueCount = 1ULL << 22;
            return !keep_trace_payloads && polynomial.basis == algebra::PolynomialBasis::Evaluation && polynomial.domain != nullptr && polynomial.domain->size >= kMinFileBackedValueCount && name.rfind("P_", 0) == 0;
        }

        std::filesystem::path spill_root_directory()
        {
            const auto root = std::filesystem::current_path() / "runs" / ".trace_spill";
            std::filesystem::create_directories(root);
            return root;
        }

        std::shared_ptr<const std::string> create_trace_spill_directory_lease()
        {
            const auto root = spill_root_directory();
            for (std::size_t attempt = 0; attempt < 16; ++attempt)
            {
                std::array<std::uint8_t, 16> random_bytes{};
                if (RAND_bytes(
                        random_bytes.data(),
                        static_cast<int>(random_bytes.size())) != 1)
                {
                    throw std::runtime_error(
                        "CSPRNG failed while creating trace spill directory");
                }
                std::ostringstream suffix;
                suffix << "proof-";
                for (const auto byte : random_bytes)
                {
                    suffix << std::hex << std::setw(2)
                           << std::setfill('0')
                           << static_cast<unsigned>(byte);
                }
                const auto directory = root / suffix.str();
                std::error_code create_error;
                if (!std::filesystem::create_directory(
                        directory, create_error))
                {
                    if (!create_error)
                    {
                        continue;
                    }
                    throw std::runtime_error(
                        "failed to create trace spill directory: " + create_error.message());
                }
                auto *owned_path =
                    new std::string(directory.string());
                return std::shared_ptr<const std::string>(
                    owned_path,
                    [](const std::string *path)
                    {
                        std::error_code cleanup_error;
                        std::filesystem::remove_all(
                            std::filesystem::path(*path),
                            cleanup_error);
                        delete path;
                    });
            }
            throw std::runtime_error(
                "failed to allocate a unique trace spill directory");
        }

        const std::filesystem::path &ensure_trace_spill_directory(
            TraceArtifacts &trace,
            std::filesystem::path *storage)
        {
            if (storage == nullptr)
            {
                throw std::runtime_error(
                    "trace spill directory storage is null");
            }
            if (trace.spill_directory_lease == nullptr)
            {
                trace.spill_directory_lease =
                    create_trace_spill_directory_lease();
            }
            *storage = std::filesystem::path(
                *trace.spill_directory_lease);
            return *storage;
        }

        std::string sanitize_spill_name(const std::string &name)
        {
            std::string out = name;
            for (auto &ch : out)
            {
                if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-'))
                {
                    ch = '_';
                }
            }
            return out;
        }

        bool is_signed_division_compact_zero_scan_candidate(
            const std::string &name)
        {
            return name.find("_Div_Limb_") != std::string::npos || (name.find("_Div_Range_") != std::string::npos && name.ends_with("_Query"));
        }

        TraceArtifacts::SpilledEvaluationPolynomial spill_evaluation_polynomial(
            TraceArtifacts &trace,
            const algebra::Polynomial &polynomial,
            std::uintptr_t semantic_source_id)
        {
            bool compact_semantic_source_all_zero = false;
            if (semantic_source_id != 0 && is_signed_division_compact_zero_scan_candidate(
                                               polynomial.name))
            {
                if (polynomial.domain == nullptr || polynomial.size() > polynomial.domain->size)
                {
                    throw std::runtime_error(
                        "signed-division compact semantic source domain mismatch: " + polynomial.name);
                }
                if (polynomial.size() < polynomial.domain->size)
                {
                    if (polynomial.size() == 0)
                    {
                        throw std::runtime_error(
                            "signed-division compact semantic source is empty: " + polynomial.name);
                    }
                    const auto domain_name = polynomial.domain->name;
                    const auto [cache_it, inserted] =
                        trace.compact_semantic_source_zero_cache.emplace(
                            semantic_source_id,
                            TraceArtifacts::CompactSemanticSourceZeroCacheEntry{
                                .domain_name = domain_name,
                                .size = polynomial.size(),
                                .all_zero = false,
                            });
                    if (!inserted && (cache_it->second.domain_name != domain_name || cache_it->second.size != polynomial.size()))
                    {
                        throw std::runtime_error(
                            "signed-division compact semantic source identity mismatch: " + polynomial.name);
                    }
                    if (inserted)
                    {
                        cache_it->second.all_zero = std::all_of(
                            polynomial.values().begin(),
                            polynomial.values().end(),
                            [](const FieldElement &value)
                            {
                                return value.is_zero();
                            });
                    }
                    compact_semantic_source_all_zero =
                        cache_it->second.all_zero;
                }
            }
            std::filesystem::path root_storage;
            const auto &root = ensure_trace_spill_directory(
                trace, &root_storage);
            const auto unique =
                sanitize_spill_name(polynomial.name) + "_" + std::to_string(std::hash<std::string>{}(polynomial.name + ":" + std::to_string(polynomial.size())));
            const auto path = root / (unique + ".bin");
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            if (!stream)
            {
                throw std::runtime_error("failed to open spill file for polynomial " + polynomial.name);
            }
            try
            {
                algebra::write_field_elements_binary(
                    stream, polynomial.values());
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(
                    "failed to write spill file for polynomial " + polynomial.name + ": " + error.what());
            }
            return TraceArtifacts::SpilledEvaluationPolynomial{
                .path = path.string(),
                .domain_name = polynomial.domain != nullptr ? polynomial.domain->name : std::string(),
                .size = polynomial.size(),
                .semantic_source_id = semantic_source_id,
                .compact_semantic_source_all_zero =
                    compact_semantic_source_all_zero,
                .zero_domain_blinder = polynomial.zero_domain_blinder,
            };
        }

        void detach_retained_aggregate_column_owner(
            PreparedDynamicCommitment &item)
        {
            if (!item.reused_shared || !item.polynomial.has_value() || item.polynomial->basis != algebra::PolynomialBasis::Evaluation || item.polynomial->shared_data == nullptr)
            {
                return;
            }
            // `shared_column_spec` uses an aliasing shared_ptr so a member
            // vector can be committed without a transient copy.  Retaining
            // that alias after the batch would retain the entire aggregate
            // owner (including unrelated edge-domain columns).  Give only
            // the non-spilled polynomial its own backing before the aggregate
            // owner leaves its natural scope.
            auto detached_values =
                std::make_shared<const std::vector<FieldElement>>(
                    item.polynomial->values());
            item.polynomial->shared_data =
                std::move(detached_values);
            item.polynomial->data.clear();
            auto opening_material =
                std::make_shared<Polynomial>(*item.polynomial);
            opening_material->name = item.name;
            item.commitment.opening_polynomial =
                std::move(opening_material);
        }

        void add_dynamic_commitment_batch(
            TraceArtifacts &trace,
            std::vector<DynamicCommitmentSpec> specs,
            const crypto::KZGKeyPair &key,
            bool keep_trace_payloads,
            RunMetrics *metrics = nullptr)
        {
            if (specs.empty())
            {
                return;
            }
            if (trace.compiler_artifact == nullptr)
            {
                throw std::runtime_error(
                    "dynamic commitment materialization requires ProtocolCompilerArtifact");
            }
            const auto compiler_artifact = trace.compiler_artifact;
            const auto batch_progress_label = [&]()
            {
                std::ostringstream out;
                out << "trace:commit_batch:size=" << specs.size() << ":labels=";
                const auto limit = std::min<std::size_t>(specs.size(), 6);
                for (std::size_t i = 0; i < limit; ++i)
                {
                    if (i != 0)
                    {
                        out << ',';
                    }
                    out << specs[i].name;
                }
                if (specs.size() > limit)
                {
                    out << ",...";
                }
                return out.str();
            }();
            progress_note(batch_progress_label + ":start");

            const auto prepared_payload_budget =
                dynamic_commitment_prepared_budget_bytes(
                    specs);

            if (specs.size() > 1)
            {
                std::size_t total_bytes = 0;

                for (const auto &spec : specs)
                {
                    const auto spec_bytes =
                        estimated_dynamic_commitment_payload_bytes(
                            spec);

                    if (spec_bytes >
                        std::numeric_limits<
                            std::size_t>::max() -
                            total_bytes)
                    {
                        total_bytes =
                            std::numeric_limits<
                                std::size_t>::max();

                        break;
                    }

                    total_bytes += spec_bytes;
                }

                if (total_bytes >
                    prepared_payload_budget)
                {
                    std::vector<std::string>
                        canonical_names;

                    canonical_names.reserve(
                        specs.size());

                    for (const auto &spec : specs)
                    {
                        canonical_names.push_back(
                            spec.name);
                    }

                    std::vector<
                        std::vector<
                            DynamicCommitmentSpec>>
                        chunks;

                    std::vector<
                        DynamicCommitmentSpec>
                        chunk;

                    std::size_t chunk_bytes = 0;

                    for (auto &spec : specs)
                    {
                        const auto spec_bytes =
                            estimated_dynamic_commitment_payload_bytes(
                                spec);

                        const bool exceeds_budget =
                            chunk_bytes >= prepared_payload_budget || spec_bytes > prepared_payload_budget - chunk_bytes;

                        if (!chunk.empty() && exceeds_budget)
                        {
                            chunks.push_back(
                                std::move(chunk));

                            chunk = {};
                            chunk_bytes = 0;
                        }

                        if (spec_bytes >
                            std::numeric_limits<
                                std::size_t>::max() -
                                chunk_bytes)
                        {
                            chunk_bytes =
                                std::numeric_limits<
                                    std::size_t>::max();
                        }
                        else
                        {
                            chunk_bytes +=
                                spec_bytes;
                        }

                        chunk.push_back(
                            std::move(spec));
                    }

                    if (!chunk.empty())
                    {
                        chunks.push_back(
                            std::move(chunk));
                    }

                    std::vector<
                        std::vector<std::string>>
                        produced_labels(
                            chunks.size());

                    std::vector<
                        std::vector<std::string>>
                        required_labels(
                            chunks.size());

                    for (std::size_t chunk_index = 0;
                         chunk_index < chunks.size();
                         ++chunk_index)
                    {
                        for (const auto &spec :
                             chunks[chunk_index])
                        {
                            produced_labels[chunk_index]
                                .push_back(
                                    spec.name);

                            for (const auto &term :
                                 spec.deferred_linear_terms)
                            {
                                if (!term.label.empty() && term.scale != FieldElement::zero())
                                {
                                    required_labels[chunk_index]
                                        .push_back(
                                            term.label);
                                }
                            }
                        }
                    }

                    const auto execution_order =
                        stable_dynamic_commitment_chunk_order(
                            produced_labels,
                            required_labels);

                    const auto
                        commitment_order_start =
                            trace.commitment_order
                                .size();

                    for (const auto chunk_index :
                         execution_order)
                    {
                        add_dynamic_commitment_batch(
                            trace,
                            std::move(
                                chunks[chunk_index]),
                            key,
                            keep_trace_payloads,
                            metrics);
                    }

                    if (trace.commitment_order.size() != commitment_order_start + canonical_names.size())
                    {
                        throw std::runtime_error(
                            "dynamic commitment chunk execution changed "
                            "the canonical object count");
                    }

                    trace.commitment_order.erase(
                        trace.commitment_order.begin() + static_cast<
                                                             std::ptrdiff_t>(
                                                             commitment_order_start),
                        trace.commitment_order.end());

                    trace.commitment_order.insert(
                        trace.commitment_order.end(),
                        canonical_names.begin(),
                        canonical_names.end());

                    return;
                }
            }

            const auto input_start =
                Clock::now();

            const auto materialization_workers =
                dynamic_commitment_materialization_worker_count(
                    specs,
                    prepared_payload_budget);

            const bool run_parallel =
                materialization_workers > 1;

            add_metric(
                metrics != nullptr
                    ? &metrics
                           ->dynamic_commit_input_ms
                    : nullptr,
                input_start);

            std::vector<
                PreparedDynamicCommitment>
                prepared;

            prepared.reserve(
                specs.size());

            const auto materialize_start =
                Clock::now();

            if (run_parallel)
            {
                const auto chunk_size =
                    (specs.size() + materialization_workers - 1) / materialization_workers;

                std::vector<
                    std::future<
                        std::vector<
                            PreparedDynamicCommitment>>>
                    futures;

                futures.reserve(
                    materialization_workers);

                for (std::size_t worker = 0;
                     worker < materialization_workers;
                     ++worker)
                {
                    const auto begin =
                        worker * chunk_size;

                    const auto end =
                        std::min(
                            specs.size(),
                            begin + chunk_size);

                    if (begin >= end)
                    {
                        break;
                    }

                    futures.push_back(
                        std::async(
                            std::launch::async,
                            [&specs,
                             begin,
                             end,
                             compiler_artifact,
                             keep_trace_payloads]() mutable
                            {
                                std::vector<
                                    PreparedDynamicCommitment>
                                    local;

                                local.reserve(
                                    end - begin);

                                for (std::size_t index =
                                         begin;
                                     index < end;
                                     ++index)
                                {
                                    local.push_back(
                                        materialize_commitment(
                                            std::move(
                                                specs[index]),
                                            *compiler_artifact,
                                            keep_trace_payloads));
                                }

                                return local;
                            }));
                }

                // Futures are consumed in the same contiguous-range order
                // in which they were created, so PreparedDynamicCommitment
                // order remains identical to the input spec order.
                for (auto &future : futures)
                {
                    auto local =
                        future.get();

                    for (auto &item : local)
                    {
                        prepared.push_back(
                            std::move(item));
                    }
                }
            }
            else
            {
                for (auto &spec : specs)
                {
                    prepared.push_back(
                        materialize_commitment(
                            std::move(spec),
                            *compiler_artifact,
                            keep_trace_payloads));
                }
            }

            progress_note(
                batch_progress_label + ":materialized");

            add_metric(metrics != nullptr ? &metrics->dynamic_polynomial_materialization_ms : nullptr, materialize_start);

            const auto pack_start = Clock::now();
            std::vector<std::pair<std::string, const algebra::Polynomial *>> named_polynomials;
            std::vector<std::size_t> polynomial_indices;
            std::unordered_map<std::string, crypto::CommitObjectProfile> precomputed_object_profiles;
            std::vector<std::size_t> deferred_linear_indices;
            named_polynomials.reserve(prepared.size());
            auto make_zero_msm_profile = [](const PreparedDynamicCommitment &item, const Polynomial &polynomial)
            {
                crypto::CommitObjectProfile object_profile;
                object_profile.name = item.name;
                object_profile.basis =
                    polynomial.basis == algebra::PolynomialBasis::Coefficient ? "coefficient" : "evaluation";
                object_profile.domain = polynomial.domain != nullptr ? polynomial.domain->name : "coef";
                object_profile.vector_length = polynomial.size();
                object_profile.nonzero_scalar_count = polynomial.size();
                object_profile.effective_scalar_count = 0;
                object_profile.zero_scalar_skipping = true;
                object_profile.sparse_compaction = false;
                object_profile.multithreaded_msm = false;
                object_profile.msm_ms = 0.0;
                return object_profile;
            };
            for (std::size_t i = 0; i < prepared.size(); ++i)
            {
                auto &item = prepared[i];
                if (item.polynomial.has_value())
                {
                    const auto &polynomial = *item.polynomial;
                    if (item.precomputed_commitment)
                    {
                        precomputed_object_profiles.emplace(item.name, make_zero_msm_profile(item, polynomial));
                        continue;
                    }
                    if (item.deferred_linear_commitment)
                    {
                        deferred_linear_indices.push_back(i);
                        precomputed_object_profiles.emplace(item.name, make_zero_msm_profile(item, polynomial));
                        continue;
                    }
                    named_polynomials.push_back({item.name, &*item.polynomial});
                    polynomial_indices.push_back(i);
                }
                else
                {
                    throw std::runtime_error("dynamic commitment is missing polynomial material");
                }
            }
            add_metric(metrics != nullptr ? &metrics->dynamic_commit_pack_ms : nullptr, pack_start);
            if (!deferred_linear_indices.empty())
            {
                progress_note(
                    batch_progress_label + ":linear_derived=" + std::to_string(deferred_linear_indices.size()) + ":msm_inputs=" + std::to_string(named_polynomials.size()));
            }

            crypto::CommitBatchProfile polynomial_profile;
            if (!named_polynomials.empty())
            {
                progress_note(batch_progress_label + ":msm:start");
                const auto commitments = crypto::KZG::commit_batch(
                    named_polynomials,
                    key,
                    &polynomial_profile,
                    trace.evaluation_commitment_cache.get());
                progress_note(batch_progress_label + ":msm:done");
                progress_commit_batch_profile(batch_progress_label, polynomial_profile);
                for (std::size_t i = 0; i < commitments.size(); ++i)
                {
                    const auto prepared_index = polynomial_indices[i];
                    prepared[prepared_index].commitment = commitments[i];
                    prepared[prepared_index].commitment_ready = true;
                }
            }
            if (!deferred_linear_indices.empty())
            {
                std::unordered_map<std::string, const crypto::Commitment *>
                    batch_commitments;
                batch_commitments.reserve(prepared.size());
                for (const auto &item : prepared)
                {
                    if (item.commitment_ready)
                    {
                        batch_commitments.emplace(
                            item.name, &item.commitment);
                    }
                }
                std::vector<std::vector<std::string>> produced_labels;
                std::vector<std::vector<std::string>> required_labels;
                produced_labels.reserve(deferred_linear_indices.size());
                required_labels.reserve(deferred_linear_indices.size());
                for (const auto prepared_index : deferred_linear_indices)
                {
                    const auto &item = prepared[prepared_index];
                    produced_labels.push_back({item.name});
                    required_labels.emplace_back();
                    for (const auto &term : item.deferred_linear_terms)
                    {
                        if (!term.label.empty() && term.scale != FieldElement::zero())
                        {
                            required_labels.back().push_back(term.label);
                        }
                    }
                }
                const auto deferred_execution_order =
                    stable_dynamic_commitment_chunk_order(
                        produced_labels, required_labels);
                for (const auto deferred_position :
                     deferred_execution_order)
                {
                    const auto prepared_index =
                        deferred_linear_indices[deferred_position];
                    auto &item = prepared[prepared_index];
                    if (!item.polynomial.has_value())
                    {
                        throw std::runtime_error(
                            "deferred linear dynamic commitment is missing polynomial material: " + item.name);
                    }
                    const auto validation_start = Clock::now();
                    validate_deferred_linear_evaluation_values(
                        item.deferred_linear_terms,
                        *item.polynomial,
                        trace,
                        batch_commitments,
                        item.name);
                    progress_note(
                        "trace:linear_value_validation:" + item.name + ":ms=" + std::to_string(elapsed_ms(validation_start, Clock::now())));
                    const auto evaluation_component =
                        timed_derived_commitment_point(
                            metrics,
                            [&]()
                            {
                                return resolve_deferred_linear_evaluation_component(
                                    item.deferred_linear_terms,
                                    trace,
                                    batch_commitments,
                                    key,
                                    item.name,
                                    item.polynomial->domain);
                            },
                            item.name);
                    item.commitment =
                        crypto::KZG::commit_evaluation_from_component(
                            item.name,
                            *item.polynomial,
                            evaluation_component,
                            key);
                    item.commitment_ready = true;
                    item.linear_commitment_derived = true;
                    batch_commitments[item.name] =
                        &item.commitment;
                }
            }
            if (metrics != nullptr)
            {
                metrics->dynamic_domain_convert_ms += polynomial_profile.evaluation_ms;
                metrics->dynamic_commit_msm_ms += polynomial_profile.msm_ms;
                std::unordered_map<std::string, crypto::CommitObjectProfile> object_profiles;
                object_profiles.reserve(polynomial_profile.objects.size());
                for (auto &object : polynomial_profile.objects)
                {
                    object_profiles.emplace(object.name, std::move(object));
                }
                for (auto &[name, object] : precomputed_object_profiles)
                {
                    object_profiles.emplace(name, std::move(object));
                }
                for (const auto &item : prepared)
                {
                    const auto profile_it = object_profiles.find(item.name);
                    if (profile_it == object_profiles.end())
                    {
                        continue;
                    }
                    const auto &object = profile_it->second;
                    const auto vector_length = object.vector_length;
                    const auto nonzero_count = object.nonzero_scalar_count;
                    const auto domain_size = item.domain_size != 0 ? item.domain_size : vector_length;
                    const auto zero_scalar_ratio = vector_length == 0
                                                       ? 0.0
                                                       : static_cast<double>(vector_length - std::min(vector_length, nonzero_count)) / static_cast<double>(vector_length);
                    const auto padding_ratio =
                        domain_size > vector_length
                            ? static_cast<double>(domain_size - vector_length) / static_cast<double>(domain_size)
                            : 0.0;
                    const auto fingerprint = stable_hash(item.commitment.point.to_string());
                    std::string duplicate_of;
                    for (const auto &previous : metrics->dynamic_commitment_profiles)
                    {
                        if (previous.commitment_fingerprint == fingerprint)
                        {
                            duplicate_of = previous.label;
                            break;
                        }
                    }
                    DynamicCommitmentObjectProfile dynamic_profile;
                    dynamic_profile.label = item.name;
                    dynamic_profile.dataset = metrics->dataset;
                    dynamic_profile.domain = object.domain.empty() ? item.domain_name : object.domain;
                    dynamic_profile.basis = object.basis;
                    dynamic_profile.classification = dynamic_commitment_classification(item.name, item);
                    dynamic_profile.ownership = dynamic_commitment_ownership(item.name);
                    dynamic_profile.category = dynamic_commitment_category(item.name);
                    dynamic_profile.optimization_opportunity =
                        dynamic_commitment_opportunity(item, object, zero_scalar_ratio, padding_ratio);
                    dynamic_profile.duplicate_of = duplicate_of;
                    dynamic_profile.commitment_fingerprint = fingerprint;
                    dynamic_profile.vector_length = vector_length;
                    dynamic_profile.domain_size = domain_size;
                    dynamic_profile.nonzero_scalar_count = nonzero_count;
                    dynamic_profile.effective_scalar_count = object.effective_scalar_count;
                    dynamic_profile.zero_scalar_ratio = zero_scalar_ratio;
                    dynamic_profile.padding_ratio = padding_ratio;
                    dynamic_profile.materialization_ms = item.materialization_ms;
                    dynamic_profile.msm_ms = object.msm_ms;
                    dynamic_profile.total_commitment_ms =
                        item.materialization_ms + object.msm_ms;
                    dynamic_profile.compact = item.compact;
                    dynamic_profile.shared_or_borrowed = item.shared_or_borrowed;
                    dynamic_profile.reused_shared = item.reused_shared;
                    dynamic_profile.duplicates_another = !duplicate_of.empty();
                    dynamic_profile.zero_scalar_skipping = object.zero_scalar_skipping;
                    dynamic_profile.sparse_compaction = object.sparse_compaction;
                    dynamic_profile.constant_tail_folding = object.constant_tail_folding;
                    dynamic_profile.multithreaded_msm = object.multithreaded_msm;
                    metrics->dynamic_commitment_profiles.push_back(std::move(dynamic_profile));
                }
            }

            const auto copy_start = Clock::now();
            for (auto &item : prepared)
            {
                if (item.column_export.has_value())
                {
                    trace.columns[item.name] = std::move(*item.column_export);
                }
                if (item.matrix_export.has_value())
                {
                    trace.matrices[item.name] = std::move(*item.matrix_export);
                }
                if (item.polynomial.has_value())
                {
                    if ((*item.polynomial).domain != nullptr)
                    {
                        trace.polynomial_domains[item.name] = (*item.polynomial).domain->name;
                    }
                    if (should_spill_dynamic_polynomial(item.name, *item.polynomial, keep_trace_payloads))
                    {
                        trace.spilled_polynomials[item.name] =
                            spill_evaluation_polynomial(
                                trace,
                                *item.polynomial,
                                item.source_identity);
                        item.commitment.opening_polynomial.reset();
                    }
                    else
                    {
                        detach_retained_aggregate_column_owner(
                            item);
                        trace.polynomials[item.name] = std::move(*item.polynomial);
                    }
                }
            }
            add_metric(metrics != nullptr ? &metrics->dynamic_copy_convert_ms : nullptr, copy_start);

            const auto finalize_start = Clock::now();
            for (auto &item : prepared)
            {
                if (!item.commitment_ready)
                {
                    throw std::runtime_error("dynamic commitment was not finalized: " + item.name);
                }
                trace.commitments[item.name] = std::move(item.commitment);
                trace.commitment_order.push_back(item.name);
            }
            add_metric(metrics != nullptr ? &metrics->dynamic_bundle_finalize_ms : nullptr, finalize_start);
            record_dynamic_commit_totals(metrics);
            progress_note(batch_progress_label + ":done");
        }

        const std::vector<FieldElement> &eval_data(const ProtocolContext &context, const std::string &name)
        {
            if (!context.public_polynomials.contains(name))
            {
                constexpr std::size_t kMinLazyLargeFhDomainSize = 1ULL << 24;
                const bool lazy_large_feature_selector =
                    context.config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features && context.domains.fh != nullptr && context.domains.fh->size >= kMinLazyLargeFhDomainSize && (name == "P_Q_tbl_feat" || name == "P_Q_qry_feat");
                if (lazy_large_feature_selector)
                {
                    const auto valid_count =
                        name == "P_Q_tbl_feat"
                            ? context.dataset.num_nodes * context.dataset.num_features
                            : context.local.num_nodes * context.local.num_features;
                    const auto cache_key =
                        context.config.checkpoint_bundle + ":" + context.config.dataset + ":selector:" + name + ":" + std::to_string(valid_count) + ":" + std::to_string(context.domains.fh->size);
                    {
                        std::lock_guard<std::mutex> lock(cache_mutex);
                        if (const auto it = cache.find(cache_key); it != cache.end())
                        {
                            return *it->second;
                        }
                    }
                    auto values = std::make_shared<std::vector<FieldElement>>(
                        build_selector(valid_count, context.domains.fh->size));
                    {
                        std::lock_guard<std::mutex> lock(cache_mutex);
                        const auto [it, _] = cache.emplace(cache_key, values);
                        return *it->second;
                    }
                }
            }
            return context.public_polynomials.at(name).values();
        }

        std::int64_t quantize_float_raw(double value)
        {
            return static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5);
        }

        FieldElement quantize_float(double value)
        {
            return FieldElement::from_signed(quantize_float_raw(value));
        }

        std::int64_t quantize_lrelu_raw(std::int64_t raw)
        {
            constexpr double kAttentionScale = 256.0;
            const auto input = static_cast<double>(raw) / kAttentionScale;
            const auto output = input >= 0.0 ? input : 0.2 * input;
            return static_cast<std::int64_t>(
                output >= 0.0 ? output * kAttentionScale + 0.5 : output * kAttentionScale - 0.5);
        }

        std::int64_t quantize_exp_neg_raw(std::int64_t raw)
        {
            constexpr double kAttentionScale = 256.0;
            const auto input = static_cast<double>(raw) / kAttentionScale;
            return std::max<std::int64_t>(0, quantize_float_raw(std::exp(-input)));
        }

        std::int64_t quantize_elu_raw(std::int64_t raw)
        {
            const auto input = static_cast<double>(raw) / 16.0;
            return quantize_float_raw(input >= 0.0 ? input : std::exp(input) - 1.0);
        }

        std::vector<FieldElement> quantize_vector(const std::vector<double> &values)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = quantize_float(values[i]);
            }
            return out;
        }

        std::vector<FieldElement> quantize_vector_at_scale(const std::vector<double> &values, double scale)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                const auto scaled = values[i] * scale;
                out[i] = FieldElement::from_signed(static_cast<std::int64_t>(
                    scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5));
            }
            return out;
        }

        Matrix quantize_matrix(const model::FloatMatrix &matrix)
        {
            Matrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_vector(matrix[row]);
            }
            return out;
        }

        using RawMatrix = std::vector<std::vector<std::int64_t>>;

        std::int64_t checked_raw_value(__int128 value, std::string_view label)
        {
            if (value > static_cast<__int128>(std::numeric_limits<std::int64_t>::max()) || value < static_cast<__int128>(std::numeric_limits<std::int64_t>::min()))
            {
                throw std::runtime_error("fixed-point raw value overflow while computing " + std::string(label));
            }
            return static_cast<std::int64_t>(value);
        }

        // Canonical SignedRoundDiv mode used by the registered fixed-point trace:
        // nearest integer, with exact half cases rounded away from zero.  Keeping
        // this operation in the signed-integer layer is essential: a field inverse
        // is only the Softmax algebra witness and is not a fixed-point division.
        std::int64_t signed_round_div_nearest(__int128 numerator, __int128 denominator, std::string_view label)
        {
            if (denominator <= 0)
            {
                throw std::runtime_error("SignedRoundDiv requires a positive denominator: " + std::string(label));
            }
            const bool negative = numerator < 0;
            const auto magnitude = negative ? -numerator : numerator;
            const auto quotient = (magnitude + denominator / 2) / denominator;
            return checked_raw_value(negative ? -quotient : quotient, label);
        }

        struct SignedRoundDivWitness
        {
            std::vector<FieldElement> active;
            std::vector<FieldElement> x;
            std::vector<FieldElement> d;
            std::vector<FieldElement> y;
            std::vector<FieldElement> sigma;
            std::vector<FieldElement> z;
            std::vector<FieldElement> a;
            std::vector<FieldElement> a_inv;
            std::vector<FieldElement> d_inv;
            std::vector<FieldElement> q;
            std::vector<FieldElement> r;
            std::vector<FieldElement> delta;
            std::unordered_map<std::string, std::vector<std::vector<FieldElement>>> limbs;
        };

        void batch_invert_active_prefix_in_place(
            std::vector<FieldElement> *values,
            std::size_t active_count,
            bool allow_zero,
            std::vector<mcl::Fr> *prefix_scratch)
        {
            if (values == nullptr || prefix_scratch == nullptr || active_count > values->size())
            {
                throw std::runtime_error(
                    "active-prefix batch inversion shape mismatch");
            }
            prefix_scratch->resize(active_count);
            mcl::Fr accumulator = 1;
            for (std::size_t i = 0; i < active_count; ++i)
            {
                (*prefix_scratch)[i] = accumulator;
                if ((*values)[i].is_zero())
                {
                    if (!allow_zero)
                    {
                        throw std::runtime_error(
                            "active-prefix batch inversion encountered zero");
                    }
                    continue;
                }
                mcl::Fr::mul(
                    accumulator, accumulator, (*values)[i].native());
            }
            mcl::Fr suffix_inverse;
            mcl::Fr::inv(suffix_inverse, accumulator);
            for (std::size_t i = active_count; i-- > 0;)
            {
                const auto original = (*values)[i].native();
                if (original.isZero())
                {
                    (*values)[i] = FieldElement::zero();
                    continue;
                }
                mcl::Fr inverse;
                mcl::Fr::mul(
                    inverse, suffix_inverse, (*prefix_scratch)[i]);
                (*values)[i] = FieldElement::from_native(inverse);
                mcl::Fr::mul(suffix_inverse, suffix_inverse, original);
            }
        }

        SignedRoundDivWitness build_signed_round_div_witness(
            const std::vector<std::int64_t> &numerators,
            const std::vector<std::int64_t> &denominators,
            std::size_t domain_size,
            std::size_t range_bits,
            std::string_view label)
        {
            if (numerators.size() != denominators.size() || numerators.size() > domain_size)
            {
                throw std::runtime_error("SignedRoundDiv witness shape mismatch: " + std::string(label));
            }
            SignedRoundDivWitness out;
            auto initialize = [&](std::vector<FieldElement> *column)
            {
                column->assign(domain_size, FieldElement::zero());
            };
            initialize(&out.active);
            initialize(&out.x);
            initialize(&out.d);
            initialize(&out.y);
            initialize(&out.sigma);
            initialize(&out.z);
            initialize(&out.a);
            initialize(&out.a_inv);
            initialize(&out.d_inv);
            initialize(&out.q);
            initialize(&out.r);
            initialize(&out.delta);
            const auto limb_count = signed_division_limb_count(range_bits);
            for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
            {
                out.limbs[value_name].assign(
                    limb_count,
                    std::vector<FieldElement>(domain_size, FieldElement::zero()));
            }
            const std::uint64_t limb_mask = (std::uint64_t{1} << range_bits) - 1U;
            auto assign_limbs = [&](std::string_view value_name, std::size_t row, std::uint64_t value)
            {
                auto &columns = out.limbs.at(std::string(value_name));
                for (std::size_t limb = 0; limb < columns.size(); ++limb)
                {
                    columns[limb][row] = FieldElement(value & limb_mask);
                    value >>= range_bits;
                }
                if (value != 0)
                {
                    throw std::runtime_error("SignedRoundDiv limb bound exceeded: " + std::string(label));
                }
            };
            for (std::size_t i = 0; i < numerators.size(); ++i)
            {
                const auto x = static_cast<__int128>(numerators[i]);
                const auto d = static_cast<__int128>(denominators[i]);
                if (d <= 0)
                {
                    throw std::runtime_error("SignedRoundDiv denominator is not positive: " + std::string(label));
                }
                const bool negative = x < 0;
                const auto magnitude = negative ? -x : x;
                const auto q = (magnitude + d / 2) / d;
                const auto r = 2 * magnitude + d - 2 * d * q;
                const auto delta = 2 * d - 1 - r;
                if (magnitude > std::numeric_limits<std::int64_t>::max() || q > std::numeric_limits<std::int64_t>::max() || r < 0 || r > std::numeric_limits<std::int64_t>::max() || delta < 0 || delta > std::numeric_limits<std::int64_t>::max())
                {
                    throw std::runtime_error("SignedRoundDiv integer bound exceeded: " + std::string(label));
                }
                const auto a_value = FieldElement::from_signed(static_cast<std::int64_t>(magnitude));
                const auto d_value = FieldElement::from_signed(denominators[i]);
                const auto q_value = FieldElement::from_signed(static_cast<std::int64_t>(q));
                out.active[i] = FieldElement::one();
                out.x[i] = FieldElement::from_signed(numerators[i]);
                out.d[i] = d_value;
                out.y[i] = FieldElement::from_signed(signed_round_div_nearest(x, d, label));
                out.sigma[i] = negative ? FieldElement::one() : FieldElement::zero();
                out.z[i] = magnitude == 0 ? FieldElement::one() : FieldElement::zero();
                out.a[i] = a_value;
                out.a_inv[i] = a_value;
                out.d_inv[i] = d_value;
                out.q[i] = q_value;
                out.r[i] = FieldElement::from_signed(static_cast<std::int64_t>(r));
                out.delta[i] = FieldElement::from_signed(static_cast<std::int64_t>(delta));
                assign_limbs("A", i, static_cast<std::uint64_t>(magnitude));
                assign_limbs("DMinus1", i, static_cast<std::uint64_t>(d - 1));
                assign_limbs("Q", i, static_cast<std::uint64_t>(q));
                assign_limbs("R", i, static_cast<std::uint64_t>(r));
                assign_limbs("Delta", i, static_cast<std::uint64_t>(delta));
            }
            std::vector<mcl::Fr> inverse_prefix_scratch;
            inverse_prefix_scratch.reserve(numerators.size());
            batch_invert_active_prefix_in_place(
                &out.a_inv,
                numerators.size(),
                true,
                &inverse_prefix_scratch);
            batch_invert_active_prefix_in_place(
                &out.d_inv,
                denominators.size(),
                false,
                &inverse_prefix_scratch);
            return out;
        }

        std::vector<std::int64_t> quantize_vector_raw(const std::vector<double> &values)
        {
            std::vector<std::int64_t> out(values.size(), 0);
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = quantize_float_raw(values[i]);
            }
            return out;
        }

        RawMatrix quantize_matrix_raw(const model::FloatMatrix &matrix)
        {
            RawMatrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_vector_raw(matrix[row]);
            }
            return out;
        }

        std::vector<std::int64_t> quantize_vector_raw_at_scale(
            const std::vector<double> &values,
            double scale)
        {
            std::vector<std::int64_t> out(values.size(), 0);
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                const auto scaled = values[i] * scale;
                out[i] = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
            }
            return out;
        }

        RawMatrix quantize_matrix_raw_at_scale(const model::FloatMatrix &matrix, double scale)
        {
            RawMatrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_vector_raw_at_scale(matrix[row], scale);
            }
            return out;
        }

        Matrix field_matrix_from_raw(const RawMatrix &matrix)
        {
            Matrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row].assign(matrix[row].size(), FieldElement::zero());
                for (std::size_t column = 0; column < matrix[row].size(); ++column)
                {
                    out[row][column] = FieldElement::from_signed(matrix[row][column]);
                }
            }
            return out;
        }

        RawMatrix project_raw_matrix(const RawMatrix &left, const RawMatrix &right, std::string_view label)
        {
            if (left.empty() || right.empty())
            {
                return {};
            }
            const auto shared = left.front().size();
            if (right.size() != shared)
            {
                throw std::runtime_error("fixed-point matrix multiply dimension mismatch: " + std::string(label));
            }
            const auto rows = left.size();
            const auto cols = right.front().size();
            RawMatrix out(rows, std::vector<std::int64_t>(cols, 0));
            parallel_for_ranges(
                rows,
                32,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        for (std::size_t col = 0; col < cols; ++col)
                        {
                            __int128 sum = 0;
                            for (std::size_t k = 0; k < shared; ++k)
                            {
                                sum += static_cast<__int128>(left[row][k]) * static_cast<__int128>(right[k][col]);
                            }
                            out[row][col] = checked_raw_value(sum, label);
                        }
                    }
                });
            return out;
        }

        std::vector<std::int64_t> raw_matvec_no_bias(
            const RawMatrix &matrix,
            const std::vector<std::int64_t> &vector,
            std::string_view label)
        {
            std::vector<std::int64_t> out(matrix.size(), 0);
            parallel_for_ranges(
                matrix.size(),
                64,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        if (matrix[row].size() != vector.size())
                        {
                            throw std::runtime_error("fixed-point matvec dimension mismatch: " + std::string(label));
                        }
                        __int128 sum = 0;
                        for (std::size_t col = 0; col < vector.size(); ++col)
                        {
                            sum += static_cast<__int128>(matrix[row][col]) * static_cast<__int128>(vector[col]);
                        }
                        out[row] = checked_raw_value(sum, label);
                    }
                });
            return out;
        }

        RawMatrix concat_raw_columns(const std::vector<RawMatrix> &matrices, std::string_view label)
        {
            if (matrices.empty())
            {
                return {};
            }
            const auto rows = matrices.front().size();
            std::size_t cols = 0;
            for (const auto &matrix : matrices)
            {
                if (matrix.size() != rows)
                {
                    throw std::runtime_error("fixed-point concat row mismatch: " + std::string(label));
                }
                cols += rows == 0 ? 0 : matrix.front().size();
            }
            RawMatrix out(rows, std::vector<std::int64_t>(cols, 0));
            for (std::size_t row = 0; row < rows; ++row)
            {
                std::size_t offset = 0;
                for (const auto &matrix : matrices)
                {
                    std::copy(matrix[row].begin(), matrix[row].end(), out[row].begin() + static_cast<std::ptrdiff_t>(offset));
                    offset += matrix[row].size();
                }
            }
            return out;
        }

        Matrix average_field_matrix_views(
            const std::vector<
                const Matrix *> &matrices)
        {
            if (matrices.empty())
            {
                return {};
            }

            if (matrices.front() == nullptr)
            {
                throw std::runtime_error(
                    "cannot average null field matrix view");
            }

            auto out =
                *matrices.front();

            const auto inv_count =
                FieldElement::one() / FieldElement(
                                          matrices.size());

            for (std::size_t matrix_index = 1;
                 matrix_index < matrices.size();
                 ++matrix_index)
            {
                const auto *matrix =
                    matrices[matrix_index];

                if (matrix == nullptr || matrix->size() != out.size())
                {
                    throw std::runtime_error(
                        "cannot average field matrix views "
                        "with different row counts");
                }

                for (std::size_t row = 0;
                     row < out.size();
                     ++row)
                {
                    if ((*matrix)[row].size() != out[row].size())
                    {
                        throw std::runtime_error(
                            "cannot average field matrix views "
                            "with different column counts");
                    }

                    for (std::size_t col = 0;
                         col < out[row].size();
                         ++col)
                    {
                        out[row][col] +=
                            (*matrix)[row][col];
                    }
                }
            }

            for (auto &row : out)
            {
                for (auto &value : row)
                {
                    value *= inv_count;
                }
            }

            return out;
        }

        struct HiddenHeadStaticQuantizedParameters
        {
            Matrix seq_kernel;
            std::vector<FieldElement> attn_src;
            std::vector<FieldElement> attn_dst;
        };

        struct OutputHeadStaticQuantizedParameters
        {
            Matrix seq_kernel;
            std::vector<FieldElement> attn_src;
            std::vector<FieldElement> attn_dst;
        };

        const HiddenHeadStaticQuantizedParameters &cached_quantized_hidden_head_parameters(
            const ProtocolContext &context,
            std::size_t head_index)
        {
            const auto &head = context.model.hidden_heads.at(head_index);
            const auto cache_key =
                context.config.checkpoint_bundle + ":" + context.config.dataset + ":" + std::to_string(context.config.seed) + ":hidden:" + std::to_string(head_index) + ":" + std::to_string(head.seq_kernel_fp.size()) + ":" + std::to_string(head.seq_kernel_fp.empty() ? 0 : head.seq_kernel_fp.front().size());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return *it->second;
                }
            }

            auto packed = std::make_shared<HiddenHeadStaticQuantizedParameters>();
            packed->seq_kernel = quantize_matrix(head.seq_kernel_fp);
            packed->attn_src = quantize_vector(head.attn_src_kernel_fp);
            packed->attn_dst = quantize_vector(head.attn_dst_kernel_fp);
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return *it->second;
            }
        }

        const OutputHeadStaticQuantizedParameters &cached_quantized_output_head_parameters(
            const ProtocolContext &context,
            std::size_t head_index)
        {
            const auto &head = context.model.output_layer.heads.at(head_index);
            const auto cache_key =
                context.config.checkpoint_bundle + ":" + context.config.dataset + ":" + std::to_string(context.config.seed) + ":output:" + std::to_string(head_index) + ":" + std::to_string(head.seq_kernel_fp.size()) + ":" + std::to_string(head.seq_kernel_fp.empty() ? 0 : head.seq_kernel_fp.front().size());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return *it->second;
                }
            }

            auto packed = std::make_shared<OutputHeadStaticQuantizedParameters>();
            packed->seq_kernel = quantize_matrix(head.seq_kernel_fp);
            packed->attn_src = quantize_vector_at_scale(head.attn_src_kernel_fp, 1.0);
            packed->attn_dst = quantize_vector_at_scale(head.attn_dst_kernel_fp, 1.0);
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return *it->second;
            }
        }

        struct HiddenQuantizedWitness
        {
            Matrix h_prime;
            Matrix h_agg_pre;
            Matrix h_agg_pre_lookup;
            Matrix h_agg;
            RawMatrix h_agg_raw;
            std::vector<FieldElement> e_src;
            std::vector<FieldElement> e_dst;
            std::vector<FieldElement> s;
            std::vector<FieldElement> z;
            std::vector<FieldElement> m;
            std::vector<FieldElement> delta;
            std::vector<FieldElement> u;
            std::vector<FieldElement> sum;
            std::vector<FieldElement> inv;
            std::vector<FieldElement> alpha;
            std::vector<FieldElement> e_src_edge;
            std::vector<FieldElement> e_dst_edge;
            std::vector<FieldElement> m_edge;
            std::vector<FieldElement> sum_edge;
            std::vector<FieldElement> inv_edge;
            SignedRoundDivWitness aggregate_div;
        };

        struct OutputQuantizedWitness
        {
            Matrix y_prime;
            Matrix y_lin;
            Matrix y;
            Matrix y_quant;
            RawMatrix y_quant_raw;
            std::vector<FieldElement> y_pre_flat;
            std::vector<FieldElement> bias_flat;
            std::vector<FieldElement> y_flat;
            std::vector<FieldElement> e_src;
            std::vector<FieldElement> e_dst;
            std::vector<FieldElement> s;
            std::vector<FieldElement> z;
            std::vector<FieldElement> m;
            std::vector<FieldElement> delta;
            std::vector<FieldElement> u;
            std::vector<FieldElement> sum;
            std::vector<FieldElement> inv;
            std::vector<FieldElement> alpha;
            std::vector<FieldElement> e_src_edge;
            std::vector<FieldElement> e_dst_edge;
            std::vector<FieldElement> m_edge;
            std::vector<FieldElement> sum_edge;
            std::vector<FieldElement> inv_edge;
            SignedRoundDivWitness aggregate_div;
        };

        struct CachedForwardArtifacts
        {
            model::ForwardProfile profile;
            std::vector<Matrix> hidden_layer_inputs_quantized;
            std::vector<Matrix> hidden_layer_concat_quantized;
            std::vector<HiddenQuantizedWitness> hidden_head_quantized;
            std::vector<OutputQuantizedWitness> output_head_quantized;
            model::FloatMatrix reference_logits_fp32;
            Matrix output_y_lin_average;
            Matrix output_y_average;
            SignedRoundDivWitness output_average_div;
        };

        std::string trace_cache_prefix(const ProtocolContext &context);

        HiddenQuantizedWitness quantize_hidden_witness_cpu(
            const model::HeadForwardTrace &fp,
            const std::vector<std::size_t> &edge_src_indices,
            const std::vector<std::size_t> &edge_dst_indices,
            const std::vector<std::pair<std::size_t, std::size_t>>
                &destination_groups,
            std::size_t edge_domain_size,
            std::size_t range_bits,
            const std::unordered_map<std::string, FieldElement> &elu_output_by_input,
            const RawMatrix *h_prime_raw_override = nullptr,
            const std::vector<std::int64_t> *e_src_raw_override = nullptr,
            const std::vector<std::int64_t> *e_dst_raw_override = nullptr)
        {
            HiddenQuantizedWitness out;
            const auto node_count = fp.H_prime.size();
            const auto d_h = node_count == 0 ? 0 : fp.H_prime.front().size();
            const auto edge_count = edge_src_indices.size();
            if (edge_count != edge_dst_indices.size() || edge_count > edge_domain_size)
            {
                throw std::runtime_error(
                    "hidden compact edge witness exceeds its registered domain");
            }
            out.h_prime.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
            out.h_agg_pre.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
            out.h_agg_pre_lookup.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
            out.h_agg.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
            out.h_agg_raw.assign(node_count, std::vector<std::int64_t>(d_h, 0));
            out.e_src.assign(fp.E_src.size(), FieldElement::zero());
            out.e_dst.assign(fp.E_dst.size(), FieldElement::zero());
            out.m.assign(fp.M.size(), FieldElement::zero());
            out.sum.assign(fp.Sum.size(), FieldElement::zero());
            out.inv.assign(fp.inv.size(), FieldElement::zero());
            out.s.assign(edge_count, FieldElement::zero());
            out.z.assign(edge_count, FieldElement::zero());
            out.delta.assign(edge_count, FieldElement::zero());
            out.u.assign(edge_count, FieldElement::zero());
            out.alpha.assign(edge_count, FieldElement::zero());
            out.e_src_edge.assign(edge_count, FieldElement::zero());
            out.e_dst_edge.assign(edge_count, FieldElement::zero());
            out.m_edge.assign(edge_count, FieldElement::zero());
            out.sum_edge.assign(edge_count, FieldElement::zero());
            out.inv_edge.assign(edge_count, FieldElement::zero());

            std::vector<std::int64_t> e_src_raw(out.e_src.size(), 0);
            std::vector<std::int64_t> e_dst_raw(out.e_dst.size(), 0);
            std::vector<std::int64_t> m_raw(out.m.size(), std::numeric_limits<std::int64_t>::min());
            std::vector<std::uint8_t> m_seen(out.m.size(), 0);
            std::vector<std::int64_t> s_raw(edge_count, 0);
            std::vector<std::int64_t> z_raw(edge_count, 0);
            std::vector<std::int64_t> u_raw(edge_count, 0);
            std::vector<std::int64_t> sum_raw(node_count, 0);
            RawMatrix h_prime_raw(node_count, std::vector<std::int64_t>(d_h, 0));

            parallel_for_ranges(
                node_count,
                32,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        e_src_raw[row] = e_src_raw_override != nullptr ? e_src_raw_override->at(row) : quantize_float_raw(fp.E_src[row]);
                        e_dst_raw[row] = e_dst_raw_override != nullptr ? e_dst_raw_override->at(row) : quantize_float_raw(fp.E_dst[row]);
                        out.e_src[row] = FieldElement::from_signed(e_src_raw[row]);
                        out.e_dst[row] = FieldElement::from_signed(e_dst_raw[row]);
                        for (std::size_t column = 0; column < d_h; ++column)
                        {
                            h_prime_raw[row][column] = h_prime_raw_override != nullptr
                                                           ? h_prime_raw_override->at(row).at(column)
                                                           : quantize_float_raw(fp.H_prime[row][column]);
                            out.h_prime[row][column] = FieldElement::from_signed(h_prime_raw[row][column]);
                            out.h_agg_pre[row][column] = quantize_float(fp.H_agg_pre_bias[row][column]);
                            out.h_agg_pre_lookup[row][column] = out.h_agg_pre[row][column];
                            out.h_agg[row][column] = quantize_float(fp.H_agg[row][column]);
                        }
                    }
                });

            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                s_raw[edge_index] = e_src_raw.at(src) + e_dst_raw.at(dst);
                z_raw[edge_index] = quantize_lrelu_raw(s_raw[edge_index]);
                if (dst >= m_raw.size())
                {
                    throw std::runtime_error("hidden edge destination index out of range");
                }
                if (m_seen[dst] == 0 || z_raw[edge_index] > m_raw[dst])
                {
                    m_raw[dst] = z_raw[edge_index];
                    m_seen[dst] = 1;
                }
            }
            for (std::size_t row = 0; row < out.m.size(); ++row)
            {
                if (m_seen[row] == 0)
                {
                    m_raw[row] = 0;
                }
                out.m[row] = FieldElement::from_signed(m_raw[row]);
            }
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                const auto delta_raw = m_raw[dst] - z_raw[edge_index];
                out.s[edge_index] = FieldElement::from_signed(s_raw[edge_index]);
                out.z[edge_index] = FieldElement::from_signed(z_raw[edge_index]);
                out.delta[edge_index] = FieldElement::from_signed(delta_raw);
                u_raw[edge_index] = quantize_exp_neg_raw(delta_raw);
                out.u[edge_index] = FieldElement::from_signed(u_raw[edge_index]);
                out.e_src_edge[edge_index] = out.e_src[src];
                out.e_dst_edge[edge_index] = out.e_dst[dst];
                out.m_edge[edge_index] = out.m[dst];
            }
            std::fill(out.sum.begin(), out.sum.end(), FieldElement::zero());
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto dst = edge_dst_indices[edge_index];
                sum_raw[dst] = checked_raw_value(
                    static_cast<__int128>(sum_raw[dst]) + u_raw[edge_index],
                    "hidden softmax denominator");
                out.sum[dst] = FieldElement::from_signed(sum_raw[dst]);
            }
            for (std::size_t row = 0; row < node_count; ++row)
            {
                if (sum_raw[row] <= 0)
                {
                    throw std::runtime_error("hidden softmax denominator must be a positive integer");
                }
            }
            std::copy(out.sum.begin(), out.sum.end(), out.inv.begin());
            {
                std::vector<mcl::Fr> softmax_inverse_prefix_scratch;
                batch_invert_active_prefix_in_place(
                    &out.inv,
                    node_count,
                    false,
                    &softmax_inverse_prefix_scratch);
            }
            for (auto &row : out.h_agg_pre)
            {
                std::fill(row.begin(), row.end(), FieldElement::zero());
            }
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                out.sum_edge[edge_index] = out.sum[dst];
                out.inv_edge[edge_index] = out.inv[dst];
                out.alpha[edge_index] = out.u[edge_index] * out.inv_edge[edge_index];
                for (std::size_t column = 0; column < d_h; ++column)
                {
                    out.h_agg_pre[dst][column] += out.alpha[edge_index] * out.h_prime[src][column];
                }
            }
            const auto aggregate_numerators =
                weighted_edge_aggregate_raw_by_destination(
                    u_raw,
                    h_prime_raw,
                    edge_src_indices,
                    edge_dst_indices,
                    destination_groups,
                    node_count,
                    "hidden aggregate numerator");
            std::vector<std::int64_t> aggregate_denominators(node_count * d_h, 0);
            for (std::size_t row = 0; row < node_count; ++row)
            {
                for (std::size_t column = 0; column < d_h; ++column)
                {
                    const auto flat_index = row * d_h + column;
                    const auto aggregate_numerator =
                        static_cast<__int128>(
                            aggregate_numerators[flat_index]);
                    aggregate_denominators[flat_index] = checked_raw_value(
                        static_cast<__int128>(sum_raw[row]) * 16,
                        "hidden aggregate denominator");
                    const auto elu_input_raw = signed_round_div_nearest(
                        aggregate_numerator,
                        static_cast<__int128>(sum_raw[row]) * 16,
                        "hidden aggregate ELU input");
                    out.h_agg_pre_lookup[row][column] = FieldElement::from_signed(elu_input_raw);
                    const auto it = elu_output_by_input.find(out.h_agg_pre_lookup[row][column].to_string());
                    if (it == elu_output_by_input.end())
                    {
                        throw std::runtime_error(
                            "hidden SignedRoundDiv output escaped the registered ELU table");
                    }
                    out.h_agg_raw[row][column] = quantize_elu_raw(elu_input_raw);
                    if (it->second != FieldElement::from_signed(out.h_agg_raw[row][column]))
                    {
                        throw std::runtime_error("registered ELU table disagrees with its deterministic generator");
                    }
                    out.h_agg[row][column] = it->second;
                }
            }
            out.aggregate_div = build_signed_round_div_witness(
                aggregate_numerators,
                aggregate_denominators,
                aggregate_numerators.size(),
                range_bits,
                "hidden aggregate ELU input");
            return out;
        }

        OutputQuantizedWitness quantize_output_witness_cpu(
            const model::HeadForwardTrace &fp,
            const std::vector<double> &output_bias_fp,
            const std::vector<std::size_t> &edge_src_indices,
            const std::vector<std::size_t> &edge_dst_indices,
            const std::vector<std::pair<std::size_t, std::size_t>>
                &destination_groups,
            std::size_t edge_domain_size,
            std::size_t range_bits,
            const RawMatrix *y_prime_raw_override = nullptr,
            const std::vector<std::int64_t> *e_src_raw_override = nullptr,
            const std::vector<std::int64_t> *e_dst_raw_override = nullptr)
        {
            OutputQuantizedWitness out;
            const auto node_count = fp.H_prime.size();
            const auto output_dim = node_count == 0 ? 0 : fp.H_prime.front().size();
            const auto edge_count = edge_src_indices.size();
            if (edge_count != edge_dst_indices.size() || edge_count > edge_domain_size)
            {
                throw std::runtime_error(
                    "output compact edge witness exceeds its registered domain");
            }
            out.y_prime.assign(node_count, std::vector<FieldElement>(output_dim, FieldElement::zero()));
            out.y_lin.assign(node_count, std::vector<FieldElement>(output_dim, FieldElement::zero()));
            out.y.assign(node_count, std::vector<FieldElement>(output_dim, FieldElement::zero()));
            out.y_quant.assign(node_count, std::vector<FieldElement>(output_dim, FieldElement::zero()));
            out.y_quant_raw.assign(node_count, std::vector<std::int64_t>(output_dim, 0));
            out.e_src.assign(fp.E_src.size(), FieldElement::zero());
            out.e_dst.assign(fp.E_dst.size(), FieldElement::zero());
            out.m.assign(fp.M.size(), FieldElement::zero());
            out.sum.assign(fp.Sum.size(), FieldElement::zero());
            out.inv.assign(fp.inv.size(), FieldElement::zero());
            out.s.assign(edge_count, FieldElement::zero());
            out.z.assign(edge_count, FieldElement::zero());
            out.delta.assign(edge_count, FieldElement::zero());
            out.u.assign(edge_count, FieldElement::zero());
            out.alpha.assign(edge_count, FieldElement::zero());
            out.e_src_edge.assign(edge_count, FieldElement::zero());
            out.e_dst_edge.assign(edge_count, FieldElement::zero());
            out.m_edge.assign(edge_count, FieldElement::zero());
            out.sum_edge.assign(edge_count, FieldElement::zero());
            out.inv_edge.assign(edge_count, FieldElement::zero());

            std::vector<std::int64_t> e_src_raw(out.e_src.size(), 0);
            std::vector<std::int64_t> e_dst_raw(out.e_dst.size(), 0);
            std::vector<std::int64_t> m_raw(out.m.size(), std::numeric_limits<std::int64_t>::min());
            std::vector<std::uint8_t> m_seen(out.m.size(), 0);
            std::vector<std::int64_t> s_raw(edge_count, 0);
            std::vector<std::int64_t> z_raw(edge_count, 0);
            std::vector<std::int64_t> u_raw(edge_count, 0);
            std::vector<std::int64_t> sum_raw(node_count, 0);

            const auto quantized_bias = quantize_vector(output_bias_fp);
            parallel_for_ranges(
                node_count,
                32,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        e_src_raw[row] = e_src_raw_override != nullptr ? e_src_raw_override->at(row) : quantize_float_raw(fp.E_src[row]);
                        e_dst_raw[row] = e_dst_raw_override != nullptr ? e_dst_raw_override->at(row) : quantize_float_raw(fp.E_dst[row]);
                        out.e_src[row] = FieldElement::from_signed(e_src_raw[row]);
                        out.e_dst[row] = FieldElement::from_signed(e_dst_raw[row]);
                        for (std::size_t column = 0; column < output_dim; ++column)
                        {
                            const auto y_prime_value = y_prime_raw_override != nullptr
                                                           ? FieldElement::from_signed(y_prime_raw_override->at(row).at(column))
                                                           : quantize_float(fp.H_prime[row][column]);
                            const auto y_lin_value = quantize_float(fp.H_agg_pre_bias[row][column]);
                            out.y_prime[row][column] = y_prime_value;
                            out.y_lin[row][column] = y_lin_value;
                            out.y[row][column] =
                                y_lin_value + (column < quantized_bias.size() ? quantized_bias[column] : FieldElement::zero());
                        }
                    }
                });

            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                s_raw[edge_index] = e_src_raw.at(src) + e_dst_raw.at(dst);
                z_raw[edge_index] = quantize_lrelu_raw(s_raw[edge_index]);
                if (dst >= m_raw.size())
                {
                    throw std::runtime_error("output edge destination index out of range");
                }
                if (m_seen[dst] == 0 || z_raw[edge_index] > m_raw[dst])
                {
                    m_raw[dst] = z_raw[edge_index];
                    m_seen[dst] = 1;
                }
            }
            for (std::size_t row = 0; row < out.m.size(); ++row)
            {
                if (m_seen[row] == 0)
                {
                    m_raw[row] = 0;
                }
                out.m[row] = FieldElement::from_signed(m_raw[row]);
            }
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                const auto delta_raw = m_raw[dst] - z_raw[edge_index];
                out.s[edge_index] = FieldElement::from_signed(s_raw[edge_index]);
                out.z[edge_index] = FieldElement::from_signed(z_raw[edge_index]);
                out.delta[edge_index] = FieldElement::from_signed(delta_raw);
                u_raw[edge_index] = quantize_exp_neg_raw(delta_raw);
                out.u[edge_index] = FieldElement::from_signed(u_raw[edge_index]);
                out.e_src_edge[edge_index] = out.e_src[src];
                out.e_dst_edge[edge_index] = out.e_dst[dst];
                out.m_edge[edge_index] = out.m[dst];
            }
            std::fill(out.sum.begin(), out.sum.end(), FieldElement::zero());
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto dst = edge_dst_indices[edge_index];
                sum_raw[dst] = checked_raw_value(
                    static_cast<__int128>(sum_raw[dst]) + u_raw[edge_index],
                    "output softmax denominator");
                out.sum[dst] = FieldElement::from_signed(sum_raw[dst]);
            }
            for (std::size_t row = 0; row < node_count; ++row)
            {
                if (sum_raw[row] <= 0)
                {
                    throw std::runtime_error("output softmax denominator must be a positive integer");
                }
            }
            std::copy(out.sum.begin(), out.sum.end(), out.inv.begin());
            {
                std::vector<mcl::Fr> softmax_inverse_prefix_scratch;
                batch_invert_active_prefix_in_place(
                    &out.inv,
                    node_count,
                    false,
                    &softmax_inverse_prefix_scratch);
            }
            for (auto &row : out.y_lin)
            {
                std::fill(row.begin(), row.end(), FieldElement::zero());
            }
            for (std::size_t edge_index = 0; edge_index < edge_src_indices.size(); ++edge_index)
            {
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                out.sum_edge[edge_index] = out.sum[dst];
                out.inv_edge[edge_index] = out.inv[dst];
                out.alpha[edge_index] = out.u[edge_index] * out.inv_edge[edge_index];
                for (std::size_t column = 0; column < output_dim; ++column)
                {
                    out.y_lin[dst][column] += out.alpha[edge_index] * out.y_prime[src][column];
                }
            }
            if (y_prime_raw_override == nullptr)
            {
                throw std::runtime_error(
                    "output integer aggregation is missing projected rows");
            }
            const auto aggregate_numerators =
                weighted_edge_aggregate_raw_by_destination(
                    u_raw,
                    *y_prime_raw_override,
                    edge_src_indices,
                    edge_dst_indices,
                    destination_groups,
                    node_count,
                    "output aggregate numerator");
            std::vector<std::int64_t> aggregate_denominators(node_count * output_dim, 0);
            const auto quantized_bias_raw = quantize_vector_raw(output_bias_fp);
            for (std::size_t row = 0; row < node_count; ++row)
            {
                for (std::size_t column = 0; column < output_dim; ++column)
                {
                    const auto flat_index = row * output_dim + column;
                    const auto aggregate_numerator =
                        static_cast<__int128>(
                            aggregate_numerators[flat_index]);
                    aggregate_denominators[flat_index] = checked_raw_value(
                        static_cast<__int128>(sum_raw[row]) * 16,
                        "output aggregate denominator");
                    const auto y_quantized = signed_round_div_nearest(
                        aggregate_numerator,
                        static_cast<__int128>(sum_raw[row]) * 16,
                        "output aggregate");
                    out.y_quant_raw[row][column] = checked_raw_value(
                        static_cast<__int128>(y_quantized) + (column < quantized_bias_raw.size() ? quantized_bias_raw[column] : 0),
                        "output bias addition");
                    out.y_quant[row][column] = FieldElement::from_signed(out.y_quant_raw[row][column]);
                    out.y[row][column] =
                        out.y_lin[row][column] + (column < quantized_bias.size() ? quantized_bias[column] : FieldElement::zero());
                }
            }

            out.aggregate_div = build_signed_round_div_witness(
                aggregate_numerators,
                aggregate_denominators,
                aggregate_numerators.size(),
                range_bits,
                "output aggregate");
            out.y_pre_flat = out.aggregate_div.y;
            out.bias_flat.assign(
                node_count * output_dim, FieldElement::zero());
            out.y_flat.assign(
                node_count * output_dim, FieldElement::zero());
            for (std::size_t row = 0; row < node_count; ++row)
            {
                for (std::size_t column = 0; column < output_dim; ++column)
                {
                    const auto flat_index = row * output_dim + column;
                    out.bias_flat[flat_index] =
                        column < quantized_bias.size() ? quantized_bias[column] : FieldElement::zero();
                    out.y_flat[flat_index] = out.y_quant[row][column];
                }
            }

            return out;
        }

        std::shared_ptr<const CachedForwardArtifacts> cached_forward_artifacts(
            const ProtocolContext &context,
            const std::vector<Edge> &edges,
            const std::vector<std::size_t> &edge_src_indices,
            const std::vector<std::size_t> &edge_dst_indices,
            const std::vector<std::pair<std::size_t, std::size_t>>
                &destination_groups,
            std::size_t edge_domain_size)
        {
            auto artifacts = std::make_shared<CachedForwardArtifacts>();
            auto forward = model::forward_note_style(context.local.features_fp, edges, context.model, &artifacts->profile);
            artifacts->reference_logits_fp32 = forward.Y;
            artifacts->hidden_layer_inputs_quantized.reserve(forward.hidden_layer_traces.size());
            artifacts->hidden_layer_concat_quantized.reserve(forward.hidden_layer_traces.size());
            std::size_t hidden_head_count = 0;
            for (const auto &layer_trace : forward.hidden_layer_traces)
            {
                hidden_head_count += layer_trace.head_traces.size();
            }
            artifacts->hidden_head_quantized.reserve(hidden_head_count);
            std::unordered_map<std::string, FieldElement> elu_output_by_input;
            elu_output_by_input.reserve(context.tables.elu.size());
            for (const auto &[input, output] : context.tables.elu)
            {
                const auto [_, inserted] = elu_output_by_input.emplace(input.to_string(), output);
                if (!inserted)
                {
                    throw std::runtime_error("ELU table contains a duplicate input");
                }
            }
            std::vector<RawMatrix> hidden_layer_concat_raw;
            hidden_layer_concat_raw.reserve(forward.hidden_layer_traces.size());
            std::size_t global_hidden_head_index = 0;
            for (std::size_t layer_index = 0; layer_index < forward.hidden_layer_traces.size(); ++layer_index)
            {
                const auto &layer_trace = forward.hidden_layer_traces[layer_index];
                const RawMatrix input_raw =
                    layer_index == 0 ? quantize_matrix_raw_at_scale(layer_trace.input, 1.0) : hidden_layer_concat_raw.back();
                artifacts->hidden_layer_inputs_quantized.push_back(field_matrix_from_raw(input_raw));

                std::vector<RawMatrix> layer_head_outputs_raw;
                layer_head_outputs_raw.reserve(layer_trace.head_traces.size());
                for (const auto &head_trace : layer_trace.head_traces)
                {
                    const auto &head = context.model.hidden_heads.at(global_hidden_head_index);
                    const auto seq_kernel_raw = quantize_matrix_raw(head.seq_kernel_fp);
                    const auto attn_src_raw = quantize_vector_raw(head.attn_src_kernel_fp);
                    const auto attn_dst_raw = quantize_vector_raw(head.attn_dst_kernel_fp);
                    auto h_prime_raw = project_raw_matrix(
                        input_raw,
                        seq_kernel_raw,
                        "hidden head projection " + std::to_string(global_hidden_head_index));
                    auto e_src_raw = raw_matvec_no_bias(
                        h_prime_raw,
                        attn_src_raw,
                        "hidden source attention " + std::to_string(global_hidden_head_index));
                    auto e_dst_raw = raw_matvec_no_bias(
                        h_prime_raw,
                        attn_dst_raw,
                        "hidden destination attention " + std::to_string(global_hidden_head_index));
                    auto quantized_head = quantize_hidden_witness_cpu(
                        head_trace,
                        edge_src_indices,
                        edge_dst_indices,
                        destination_groups,
                        edge_domain_size,
                        context.config.range_bits,
                        elu_output_by_input,
                        &h_prime_raw,
                        &e_src_raw,
                        &e_dst_raw);
                    layer_head_outputs_raw.push_back(quantized_head.h_agg_raw);
                    artifacts->hidden_head_quantized.push_back(std::move(quantized_head));
                    ++global_hidden_head_index;
                }
                hidden_layer_concat_raw.push_back(concat_raw_columns(
                    layer_head_outputs_raw,
                    "hidden layer concat " + std::to_string(layer_index)));
                artifacts->hidden_layer_concat_quantized.push_back(field_matrix_from_raw(hidden_layer_concat_raw.back()));
            }
            artifacts->output_head_quantized.reserve(
                forward.output_head_traces.size());

            std::vector<const Matrix *>
                per_head_y_lin_matrices;

            std::vector<const RawMatrix *>
                per_head_y_raw_matrices;

            per_head_y_lin_matrices.reserve(
                forward.output_head_traces.size());

            per_head_y_raw_matrices.reserve(
                forward.output_head_traces.size());
            const RawMatrix output_input_raw =
                hidden_layer_concat_raw.empty() ? RawMatrix{} : hidden_layer_concat_raw.back();
            for (std::size_t head_index = 0; head_index < forward.output_head_traces.size(); ++head_index)
            {
                const auto &head = context.model.output_layer.heads.at(head_index);
                const auto seq_kernel_raw = quantize_matrix_raw(head.seq_kernel_fp);
                const auto attn_src_raw = quantize_vector_raw_at_scale(head.attn_src_kernel_fp, 1.0);
                const auto attn_dst_raw = quantize_vector_raw_at_scale(head.attn_dst_kernel_fp, 1.0);
                auto y_prime_raw = project_raw_matrix(
                    output_input_raw,
                    seq_kernel_raw,
                    "output head projection " + std::to_string(head_index));
                auto e_src_raw = raw_matvec_no_bias(
                    y_prime_raw,
                    attn_src_raw,
                    "output source attention " + std::to_string(head_index));
                auto e_dst_raw = raw_matvec_no_bias(
                    y_prime_raw,
                    attn_dst_raw,
                    "output destination attention " + std::to_string(head_index));
                auto quantized = quantize_output_witness_cpu(
                    forward.output_head_traces[head_index],
                    context.model.output_layer.heads[head_index].output_bias_fp,
                    edge_src_indices,
                    edge_dst_indices,
                    destination_groups,
                    edge_domain_size,
                    context.config.range_bits,
                    &y_prime_raw,
                    &e_src_raw,
                    &e_dst_raw);
                artifacts->output_head_quantized.push_back(
                    std::move(quantized));

                const auto &stored_quantized =
                    artifacts
                        ->output_head_quantized
                        .back();

                per_head_y_lin_matrices.push_back(
                    &stored_quantized.y_lin);

                per_head_y_raw_matrices.push_back(
                    &stored_quantized.y_quant_raw);
            }
            if (!per_head_y_lin_matrices.empty())
            {
                artifacts->output_y_lin_average =
                    average_field_matrix_views(
                        per_head_y_lin_matrices);
                const auto row_count =
                    per_head_y_raw_matrices
                        .front()
                        ->size();

                const auto column_count =
                    row_count == 0
                        ? 0
                        : per_head_y_raw_matrices
                              .front()
                              ->front()
                              .size();
                std::vector<std::int64_t> sum_values(row_count * column_count, 0);
                std::vector<std::int64_t> denominators(
                    row_count * column_count,
                    static_cast<std::int64_t>(per_head_y_raw_matrices.size()));
                artifacts->output_y_average.assign(
                    row_count,
                    std::vector<FieldElement>(column_count, FieldElement::zero()));
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    for (std::size_t column = 0; column < column_count; ++column)
                    {
                        __int128 sum = 0;
                        for (const auto *head_matrix :
                             per_head_y_raw_matrices)
                        {
                            if (head_matrix == nullptr)
                            {
                                throw std::runtime_error(
                                    "null output raw matrix view");
                            }

                            sum +=
                                head_matrix
                                    ->at(row)
                                    .at(column);
                        }

                        const auto flat_index = row * column_count + column;
                        sum_values[flat_index] = checked_raw_value(sum, "final output head sum");
                        artifacts->output_y_average[row][column] = FieldElement::from_signed(
                            signed_round_div_nearest(
                                sum,
                                denominators[flat_index],
                                "final output head average"));
                    }
                }
                artifacts->output_average_div = build_signed_round_div_witness(
                    sum_values,
                    denominators,
                    sum_values.size(),
                    context.config.range_bits,
                    "final output head average");
            }
            return artifacts;
        }

        std::size_t lookup_active_transition_limit(std::size_t valid_count, std::size_t domain_size)
        {
            if (domain_size <= 1)
            {
                return 0;
            }
            return std::min(valid_count, domain_size - 1U);
        }

        std::size_t lookup_active_transition_limit(
            std::size_t table_valid_count,
            std::size_t query_valid_count,
            std::size_t domain_size)
        {
            return lookup_active_transition_limit(std::max(table_valid_count, query_valid_count), domain_size);
        }

        std::vector<FieldElement> compress_rows_with_challenge(const Matrix &matrix, const FieldElement &challenge)
        {
            if (matrix.empty())
            {
                return {};
            }
            const auto column_powers = powers(challenge, matrix.front().size());
            return linear_form_by_powers(matrix, column_powers);
        }

        std::vector<FieldElement> broadcast_node_values(
            const std::vector<FieldElement> &node_values,
            const std::vector<Edge> &edges,
            bool use_src,
            std::size_t domain_size)
        {
            if (edges.size() > domain_size)
            {
                throw std::runtime_error(
                    "broadcast edge view exceeds its registered domain");
            }
            // Rows after the real edge block are canonical zeros. Store only
            // the non-padding prefix; evaluation-basis KZG and every opening
            // path interpret the omitted suffix as zero on the same domain.
            std::vector<FieldElement> out(
                edges.size(), FieldElement::zero());
            parallel_for_ranges(
                edges.size(),
                256,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t edge_index = begin; edge_index < end; ++edge_index)
                    {
                        out[edge_index] = node_values[use_src ? edges[edge_index].src : edges[edge_index].dst];
                    }
                });
            return out;
        }

        std::vector<FieldElement> count_by_src(const std::vector<Edge> &edges, std::size_t node_count, std::size_t domain_size)
        {
            std::vector<FieldElement> out(domain_size, FieldElement::zero());
            std::vector<std::size_t> counts(node_count, 0);
            for (const auto &edge : edges)
            {
                counts[edge.src] += 1;
            }
            for (std::size_t node = 0; node < node_count; ++node)
            {
                out[node] = FieldElement(counts[node]);
            }
            return out;
        }

        std::vector<FieldElement> count_by_dst(const std::vector<Edge> &edges, std::size_t node_count, std::size_t domain_size)
        {
            std::vector<FieldElement> out(domain_size, FieldElement::zero());
            std::vector<std::size_t> counts(node_count, 0);
            for (const auto &edge : edges)
            {
                counts[edge.dst] += 1;
            }
            for (std::size_t node = 0; node < node_count; ++node)
            {
                out[node] = FieldElement(counts[node]);
            }
            return out;
        }

        std::vector<std::pair<std::size_t, std::size_t>> dst_edge_groups(const std::vector<Edge> &edges)
        {
            std::vector<std::pair<std::size_t, std::size_t>> groups;
            if (edges.empty())
            {
                return groups;
            }
            std::size_t group_begin = 0;
            while (group_begin < edges.size())
            {
                std::size_t group_end = group_begin;
                while (group_end + 1 < edges.size() && edges[group_end + 1].dst == edges[group_begin].dst)
                {
                    ++group_end;
                }
                groups.emplace_back(group_begin, group_end + 1);
                group_begin = group_end + 1;
            }
            return groups;
        }

        std::vector<FieldElement> build_group_target(
            const std::vector<FieldElement> &node_values,
            const std::vector<Edge> &edges,
            std::size_t domain_size)
        {
            return broadcast_node_values(node_values, edges, false, domain_size);
        }

        std::vector<FieldElement> build_first_zero_group_selector(
            const std::vector<FieldElement> &values,
            const std::vector<std::pair<std::size_t, std::size_t>> &groups,
            std::size_t domain_size)
        {
            std::vector<FieldElement> out(domain_size, FieldElement::zero());
            if (groups.empty())
            {
                return out;
            }
            parallel_for_ranges(
                groups.size(),
                256,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t group_index = begin; group_index < end; ++group_index)
                    {
                        const auto [group_begin, group_end] = groups[group_index];
                        if (group_end > values.size() || group_end > out.size())
                        {
                            throw std::runtime_error("group selector boundary exceeds edge values");
                        }
                        for (std::size_t k = group_begin; k < group_end; ++k)
                        {
                            if (values[k].is_zero())
                            {
                                out[k] = FieldElement::one();
                                break;
                            }
                        }
                    }
                });
            return out;
        }

        struct BindingTrace
        {
            std::vector<FieldElement> a;
            std::vector<FieldElement> b;
            std::vector<FieldElement> acc;
            FieldElement mu = FieldElement::zero();
        };

        BindingTrace build_binding_trace_from_folded_columns(
            const std::vector<FieldElement> &folded_columns,
            const std::vector<FieldElement> &parameter_values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &dim_domain,
            std::size_t real_dim_count,
            const std::optional<FieldElement> &target_mu = std::nullopt)
        {
            if (dim_domain == nullptr)
            {
                throw std::runtime_error("binding trace is missing dimension domain");
            }
            if (real_dim_count > dim_domain->size || real_dim_count > folded_columns.size() || real_dim_count > parameter_values.size())
            {
                throw std::runtime_error("binding trace dimension mismatch");
            }
            BindingTrace out;
            out.a.assign(dim_domain->size, FieldElement::zero());
            out.b.assign(dim_domain->size, FieldElement::zero());
            out.acc.assign(dim_domain->size, FieldElement::zero());
            FieldElement raw_mu = FieldElement::zero();
            for (std::size_t i = 0; i < real_dim_count; ++i)
            {
                out.a[i] = folded_columns[i];
                out.b[i] = parameter_values[i];
                raw_mu += out.a[i] * out.b[i];
            }
            if (target_mu.has_value() && raw_mu != *target_mu)
            {
                throw std::runtime_error(
                    "binding trace target differs from its canonical inner product");
            }
            for (std::size_t i = 0; i < real_dim_count; ++i)
            {
                out.mu += out.a[i] * out.b[i];
                if (i + 1 < dim_domain->size)
                {
                    out.acc[i + 1] = out.mu;
                }
            }
            for (std::size_t i = real_dim_count + 1; i < dim_domain->size; ++i)
            {
                out.acc[i] = out.mu;
            }
            return out;
        }

        std::string trace_cache_prefix(const ProtocolContext &context)
        {
            constexpr std::size_t kMinVolatileLargeTraceFhDomainSize = 1ULL << 24;
            const bool volatile_large_trace = context.config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features && context.domains.fh != nullptr && context.domains.fh->size > kMinVolatileLargeTraceFhDomainSize;

            // trace cache key 只区分会影响执行层对象复用的正式维度，避免把实验标签写进主线 key。
            return context.config.checkpoint_bundle + ":" + context.config.dataset + ":" + std::to_string(context.local.num_nodes) + ":" + std::to_string(context.local.edges.size()) + (volatile_large_trace ? ":volatile_large_trace" : "") + ":" + std::to_string(context.config.seed);
        }

        struct TraceStaticArtifacts
        {
            std::vector<FieldElement> feature_index_fields;
            std::vector<FieldElement> absolute_id_fields;
            std::vector<FieldElement> dataset_index_fields;
            std::vector<FieldElement> edge_src_fields;
            std::vector<FieldElement> edge_dst_fields;
            std::vector<std::size_t> edge_src_indices;
            std::vector<std::size_t> edge_dst_indices;
            std::vector<FieldElement> src_multiplicity_template;
            std::vector<FieldElement> dst_multiplicity_template;
            std::vector<std::pair<std::size_t, std::size_t>> edge_groups_by_dst;
            bool full_feature_identity = false;
        };

        const TraceStaticArtifacts &cached_trace_static_artifacts(const ProtocolContext &context)
        {
            const auto cache_key =
                trace_cache_prefix(context) + ":trace_static:" + std::to_string(context.local.num_features) + ":" + std::to_string(context.dataset.num_nodes) + ":" + std::to_string(context.domains.n->size) + ":" + std::to_string(context.domains.edge->size);
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return *it->second;
                }
            }

            auto artifacts = std::make_shared<TraceStaticArtifacts>();
            artifacts->feature_index_fields.assign(context.local.num_features, FieldElement::zero());
            for (std::size_t j = 0; j < context.local.num_features; ++j)
            {
                artifacts->feature_index_fields[j] = FieldElement(j);
            }

            artifacts->absolute_id_fields.assign(context.local.num_nodes, FieldElement::zero());
            artifacts->full_feature_identity =
                context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features;
            for (std::size_t i = 0; i < context.local.num_nodes; ++i)
            {
                artifacts->absolute_id_fields[i] = FieldElement(context.local.absolute_ids[i]);
                artifacts->full_feature_identity =
                    artifacts->full_feature_identity && context.local.absolute_ids[i] == i;
            }

            artifacts->dataset_index_fields.assign(context.dataset.num_nodes, FieldElement::zero());
            for (std::size_t i = 0; i < context.dataset.num_nodes; ++i)
            {
                artifacts->dataset_index_fields[i] = FieldElement(i);
            }

            const auto edge_count = context.local.edges.size();
            artifacts->edge_src_fields.assign(edge_count, FieldElement::zero());
            artifacts->edge_dst_fields.assign(edge_count, FieldElement::zero());
            artifacts->edge_src_indices.assign(edge_count, 0);
            artifacts->edge_dst_indices.assign(edge_count, 0);
            for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index)
            {
                artifacts->edge_src_indices[edge_index] = context.local.edges[edge_index].src;
                artifacts->edge_dst_indices[edge_index] = context.local.edges[edge_index].dst;
                artifacts->edge_src_fields[edge_index] = FieldElement(artifacts->edge_src_indices[edge_index]);
                artifacts->edge_dst_fields[edge_index] = FieldElement(artifacts->edge_dst_indices[edge_index]);
            }

            artifacts->src_multiplicity_template =
                count_by_src(context.local.edges, context.local.num_nodes, context.domains.n->size);
            artifacts->dst_multiplicity_template =
                count_by_dst(context.local.edges, context.local.num_nodes, context.domains.n->size);
            artifacts->edge_groups_by_dst = dst_edge_groups(context.local.edges);

            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, artifacts);
                return *it->second;
            }
        }

        BindingTrace build_binding_trace(
            const Matrix &matrix,
            const std::shared_ptr<algebra::RootOfUnityDomain> &row_domain,
            std::size_t real_row_count,
            const FieldElement &point,
            const std::vector<FieldElement> &parameter_values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &dim_domain,
            std::size_t real_dim_count,
            const std::optional<FieldElement> &target_mu = std::nullopt)
        {
            const auto &row_weights = cached_barycentric_weights_native(row_domain, point);
            const auto folded = weighted_column_sum_native(matrix, row_weights, real_row_count);
            return build_binding_trace_from_folded_columns(folded, parameter_values, dim_domain, real_dim_count, target_mu);
        }

        BindingTrace build_row_major_projection_binding_trace(
            const Matrix &input_matrix,
            std::size_t real_row_count,
            const FieldElement &point,
            std::size_t output_column_count,
            const std::vector<FieldElement> &parameter_values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &dim_domain,
            std::size_t real_dim_count,
            const std::optional<FieldElement> &target_mu = std::nullopt)
        {
            if (input_matrix.empty())
            {
                return build_binding_trace_from_folded_columns({}, parameter_values, dim_domain, real_dim_count, target_mu);
            }
            if (real_row_count > input_matrix.size())
            {
                throw std::runtime_error("row-major projection binding row count mismatch");
            }
            const auto row_weights = strided_powers(point, real_row_count, output_column_count);
            std::vector<mcl::Fr> native_out(input_matrix.front().size());
            for (auto &value : native_out)
            {
                value.clear();
            }
            for (std::size_t row = 0; row < real_row_count; ++row)
            {
                const auto &weight = row_weights[row].native();
                for (std::size_t column = 0; column < input_matrix[row].size(); ++column)
                {
                    mcl::Fr term;
                    mcl::Fr::mul(term, input_matrix[row][column].native(), weight);
                    mcl::Fr::add(native_out[column], native_out[column], term);
                }
            }
            std::vector<FieldElement> folded(input_matrix.front().size(), FieldElement::zero());
            for (std::size_t column = 0; column < folded.size(); ++column)
            {
                folded[column] = FieldElement::from_native(native_out[column]);
            }
            return build_binding_trace_from_folded_columns(folded, parameter_values, dim_domain, real_dim_count, target_mu);
        }

        template <typename Builder>
        std::shared_ptr<const BindingTrace>
        cached_binding_trace(
            Builder &&builder)
        {
            return std::make_shared<BindingTrace>(
                builder());
        }
        struct HiddenCompressedTrace
        {
            std::vector<FieldElement> h_star;
            std::vector<FieldElement> h_star_edge;
            std::vector<FieldElement> h_agg_pre_star;
            std::vector<FieldElement> h_agg_pre_star_edge;
            std::vector<FieldElement> h_agg_star;
            std::vector<FieldElement> h_agg_star_edge;
        };

        std::shared_ptr<const HiddenCompressedTrace>
        cached_hidden_compressed_trace(
            const Matrix &h_prime,
            const Matrix &h_agg_pre,
            const Matrix &h_agg,
            const std::vector<Edge> &edges,
            std::size_t edge_domain_size,
            const FieldElement &xi)
        {
            auto packed =
                std::make_shared<HiddenCompressedTrace>();

            packed->h_star =
                compress_rows_with_challenge(
                    h_prime,
                    xi);

            packed->h_star_edge =
                broadcast_node_values(
                    packed->h_star,
                    edges,
                    true,
                    edge_domain_size);

            packed->h_agg_pre_star =
                compress_rows_with_challenge(
                    h_agg_pre,
                    xi);

            packed->h_agg_pre_star_edge =
                broadcast_node_values(
                    packed->h_agg_pre_star,
                    edges,
                    false,
                    edge_domain_size);

            packed->h_agg_star =
                compress_rows_with_challenge(
                    h_agg,
                    xi);

            packed->h_agg_star_edge =
                broadcast_node_values(
                    packed->h_agg_star,
                    edges,
                    false,
                    edge_domain_size);

            return packed;
        }

        struct OutputCompressedTrace
        {
            std::vector<FieldElement> y_prime_star;
            std::vector<FieldElement> y_prime_star_edge;
            std::vector<FieldElement> y_star;
            std::vector<FieldElement> y_star_edge;
        };

        std::shared_ptr<const OutputCompressedTrace>
        cached_output_compressed_trace(
            const Matrix &y_prime,
            const Matrix &y_lin,
            const std::vector<Edge> &edges,
            std::size_t edge_domain_size,
            const FieldElement &xi)
        {
            auto packed =
                std::make_shared<OutputCompressedTrace>();

            packed->y_prime_star =
                compress_rows_with_challenge(
                    y_prime,
                    xi);

            packed->y_prime_star_edge =
                broadcast_node_values(
                    packed->y_prime_star,
                    edges,
                    true,
                    edge_domain_size);

            packed->y_star =
                compress_rows_with_challenge(
                    y_lin,
                    xi);

            packed->y_star_edge =
                broadcast_node_values(
                    packed->y_star,
                    edges,
                    false,
                    edge_domain_size);

            return packed;
        }

        struct RouteTrace
        {
            std::vector<FieldElement> table;
            std::vector<FieldElement> query;
            std::vector<FieldElement> multiplicity;
            std::vector<FieldElement> node_inverse;
            std::vector<FieldElement> edge_inverse;
            std::vector<FieldElement> node_acc;
            std::vector<FieldElement> edge_acc;
            FieldElement total = FieldElement::zero();
        };

        template <typename Builder>
        std::shared_ptr<const RouteTrace>
        cached_route_trace(
            Builder &&builder)
        {
            return std::make_shared<RouteTrace>(
                builder());
        }

        template <typename Builder>
        std::shared_ptr<const std::vector<FieldElement>>
        cached_state_vector(
            Builder &&builder)
        {
            return std::make_shared<
                std::vector<FieldElement>>(
                builder());
        }

        RouteTrace build_route_trace_from_padded(
            std::vector<FieldElement> table,
            std::vector<FieldElement> query,
            std::vector<FieldElement> multiplicity,
            std::size_t real_node_count,
            std::size_t real_edge_count,
            const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
            const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
            const FieldElement &beta)
        {
            if (node_domain == nullptr || edge_domain == nullptr || table.size() != node_domain->size || multiplicity.size() != node_domain->size || query.size() != edge_domain->size)
            {
                throw std::runtime_error(
                    "padded route trace domain shape mismatch");
            }
            // Route accumulators store the active terminal at index real_count.
            if (real_node_count >= node_domain->size || real_edge_count >= edge_domain->size)
            {
                throw std::runtime_error(
                    "padded route trace requires a terminal row");
            }

            RouteTrace out;
            out.table = std::move(table);
            out.query = std::move(query);
            out.multiplicity = std::move(multiplicity);
            out.node_inverse = build_active_prefix_inverse_column(
                out.table, real_node_count, beta);
            out.edge_inverse = build_active_prefix_inverse_column(
                out.query, real_edge_count, beta);
            out.node_acc = build_route_node_accumulator(
                out.node_inverse,
                out.multiplicity,
                real_node_count,
                node_domain->size);
            out.edge_acc = build_route_edge_accumulator(
                out.edge_inverse,
                real_edge_count,
                edge_domain->size);
            out.total = out.node_acc[real_node_count];
            return out;
        }

        std::uint64_t load_u64_le(const std::uint8_t *bytes)
        {
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < 8; ++i)
            {
                value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return value;
        }

        struct FieldKey
        {
            std::array<std::uint64_t, 4> limbs = {0, 0, 0, 0};

            bool operator==(const FieldKey &other) const
            {
                return limbs == other.limbs;
            }
        };

        FieldKey pack_field_key(const FieldElement &value)
        {
            std::array<std::uint8_t, 32> bytes{};
            const auto written = value.native().getLittleEndian(bytes.data(), bytes.size());
            if (written > bytes.size())
            {
                throw std::runtime_error("field key little-endian serialization overflow");
            }
            FieldKey out;
            for (std::size_t limb = 0; limb < out.limbs.size(); ++limb)
            {
                out.limbs[limb] = load_u64_le(bytes.data() + limb * 8U);
            }
            return out;
        }

        struct FieldPairKey
        {
            FieldKey first;
            FieldKey second;

            bool operator==(const FieldPairKey &other) const
            {
                return first == other.first && second == other.second;
            }
        };

        struct FieldPairKeyHash
        {
            static std::size_t mix_key(const FieldKey &key)
            {
                std::size_t out = 0;
                for (const auto limb : key.limbs)
                {
                    out ^= std::hash<std::uint64_t>{}(limb) + 0x9e3779b97f4a7c15ULL + (out << 6U) + (out >> 2U);
                }
                return out;
            }

            std::size_t operator()(const FieldPairKey &key) const
            {
                return mix_key(key.first) ^ (mix_key(key.second) << 1U);
            }
        };

        std::unordered_map<std::uint64_t, std::size_t> build_value_index_map(
            const std::vector<FieldElement> &values,
            std::size_t valid_count)
        {
            std::unordered_map<std::uint64_t, std::size_t> index;
            index.reserve(valid_count);
            for (std::size_t i = 0; i < valid_count; ++i)
            {
                index.emplace(values[i].value(), i);
            }
            return index;
        }

        std::shared_ptr<const std::unordered_map<std::uint64_t, std::size_t>> cached_value_index_map(
            const std::string &label,
            const std::vector<FieldElement> &values,
            std::size_t valid_count)
        {
            const auto cache_key =
                label + ":" + std::to_string(valid_count) + ":" + std::to_string(values.size());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return it->second;
                }
            }
            auto packed = std::make_shared<std::unordered_map<std::uint64_t, std::size_t>>(
                build_value_index_map(values, valid_count));
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return it->second;
            }
        }

        std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash> build_pair_index_map(
            const std::vector<std::pair<FieldElement, FieldElement>> &values)
        {
            std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash> index;
            index.reserve(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                index.emplace(FieldPairKey{pack_field_key(values[i].first), pack_field_key(values[i].second)}, i);
            }
            return index;
        }

        std::shared_ptr<const std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>> cached_pair_index_map(
            const std::string &label,
            const std::vector<std::pair<FieldElement, FieldElement>> &values)
        {
            const auto cache_key =
                label + ":" + std::to_string(values.size()) + ":first=" + (values.empty() ? std::string("empty") : values.front().first.to_string() + "," + values.front().second.to_string()) + ":last=" + (values.empty() ? std::string("empty") : values.back().first.to_string() + "," + values.back().second.to_string());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return it->second;
                }
            }
            auto packed = std::make_shared<std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>>(
                build_pair_index_map(values));
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return it->second;
            }
        }

        std::vector<FieldElement> multiplicity_from_indices(
            const std::vector<std::size_t> &indices,
            std::size_t domain_size)
        {
            std::vector<FieldElement> multiplicity(domain_size, FieldElement::zero());
            if (algebra::should_use_cuda_lookup_histogram(indices.size(), domain_size))
            {
#if GATZK_ENABLE_CUDA_BACKEND
                const auto counts = lookup_histogram_indices_cuda(indices, domain_size);
                if (!counts.empty())
                {
                    for (std::size_t index = 0; index < domain_size; ++index)
                    {
                        multiplicity[index] = FieldElement(counts[index]);
                    }
                    return multiplicity;
                }
                throw std::runtime_error("CUDA lookup histogram failed; refusing silent CPU fallback");
#else
                throw std::runtime_error("CUDA lookup histogram requested in a non-CUDA build");
#endif
            }
            for (const auto index : indices)
            {
                multiplicity[index] += FieldElement::one();
            }
            return multiplicity;
        }

        std::shared_ptr<const std::vector<FieldElement>>
        cached_single_lookup_table_shared(
            const std::string &label,
            const std::vector<FieldElement> &values,
            std::size_t domain_size)
        {
            const auto cache_key =
                label + ":" + std::to_string(domain_size) + ":" + std::to_string(values.size()) + ":first=" + (values.empty() ? std::string("empty") : values.front().to_string()) + ":last=" + (values.empty() ? std::string("empty") : values.back().to_string());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return it->second;
                }
            }
            auto packed = std::make_shared<std::vector<FieldElement>>(domain_size, FieldElement::zero());
            std::copy(values.begin(), values.end(), packed->begin());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return it->second;
            }
        }

        const std::vector<FieldElement> &cached_single_lookup_table(
            const std::string &label,
            const std::vector<FieldElement> &values,
            std::size_t domain_size)
        {
            return *cached_single_lookup_table_shared(
                label, values, domain_size);
        }

        const std::vector<FieldElement> &cached_pair_lookup_table(
            const std::string &label,
            const std::vector<std::pair<FieldElement, FieldElement>> &values,
            const FieldElement &eta,
            std::size_t domain_size)
        {
            const auto cache_key =
                label + ":" + std::to_string(domain_size) + ":" + std::to_string(values.size()) + ":" + eta.to_string() + ":first=" + (values.empty() ? std::string("empty") : values.front().first.to_string() + "," + values.front().second.to_string()) + ":last=" + (values.empty() ? std::string("empty") : values.back().first.to_string() + "," + values.back().second.to_string());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return *it->second;
                }
            }
            auto packed = std::make_shared<std::vector<FieldElement>>(domain_size, FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                (*packed)[i] = values[i].first + eta * values[i].second;
            }
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, packed);
                return *it->second;
            }
        }

        struct SingleLookupArtifacts
        {
            std::vector<FieldElement> query;
            std::vector<FieldElement> multiplicity;
        };

        template <typename Builder>
        std::shared_ptr<const SingleLookupArtifacts>
        cached_single_lookup_artifacts(
            Builder &&builder)
        {
            return std::make_shared<SingleLookupArtifacts>(
                builder());
        }

        SingleLookupArtifacts build_single_lookup_artifacts(
            const std::unordered_map<std::uint64_t, std::size_t> &index_by_value,
            const std::vector<FieldElement> &query_values,
            std::size_t domain_size,
            std::size_t valid_count,
            const std::string &label,
            RunMetrics *metrics)
        {
            SingleLookupArtifacts out;
            out.query.assign(domain_size, FieldElement::zero());
            auto stage_start = Clock::now();
            std::vector<std::size_t> indices(valid_count, 0);
            for (std::size_t i = 0; i < valid_count; ++i)
            {
                out.query[i] = query_values[i];
                const auto it = index_by_value.find(query_values[i].value());
                if (it == index_by_value.end())
                {
                    throw std::runtime_error(
                        "lookup query escaped static table for " + label + " value=" + query_values[i].to_string() + " table_size=" + std::to_string(index_by_value.size()));
                }
                indices[i] = it->second;
            }
            add_metric(metrics != nullptr ? &metrics->lookup_query_pack_ms : nullptr, stage_start);
            stage_start = Clock::now();
            out.multiplicity = multiplicity_from_indices(indices, domain_size);
            add_metric(metrics != nullptr ? &metrics->lookup_multiplicity_ms : nullptr, stage_start);
            return out;
        }

        std::vector<FieldElement> build_single_lookup_multiplicity_for_packed_query(
            const std::unordered_map<std::uint64_t, std::size_t> &index_by_value,
            const std::vector<FieldElement> &query_values,
            std::size_t domain_size,
            std::size_t valid_count,
            const std::string &label,
            RunMetrics *metrics)
        {
            if (query_values.size() != domain_size || valid_count > domain_size)
            {
                throw std::runtime_error(
                    "packed lookup query shape mismatch for " + label);
            }
            auto stage_start = Clock::now();
            std::vector<std::size_t> indices(valid_count, 0);
            for (std::size_t i = 0; i < valid_count; ++i)
            {
                const auto it = index_by_value.find(query_values[i].value());
                if (it == index_by_value.end())
                {
                    throw std::runtime_error(
                        "lookup query escaped static table for " + label + " value=" + query_values[i].to_string() + " table_size=" + std::to_string(index_by_value.size()));
                }
                indices[i] = it->second;
            }
            for (std::size_t i = valid_count; i < query_values.size(); ++i)
            {
                if (!query_values[i].is_zero())
                {
                    throw std::runtime_error(
                        "packed lookup query has a nonzero inactive row for " + label);
                }
            }
            add_metric(metrics != nullptr ? &metrics->lookup_query_pack_ms : nullptr, stage_start);
            stage_start = Clock::now();
            auto multiplicity =
                multiplicity_from_indices(indices, domain_size);
            add_metric(metrics != nullptr ? &metrics->lookup_multiplicity_ms : nullptr, stage_start);
            return multiplicity;
        }

        struct PairLookupArtifacts
        {
            std::vector<FieldElement> query;
            std::vector<FieldElement> multiplicity;
        };

        template <typename Builder>
        std::shared_ptr<const PairLookupArtifacts>
        cached_pair_lookup_artifacts(
            Builder &&builder)
        {
            return std::make_shared<PairLookupArtifacts>(
                builder());
        }

        PairLookupArtifacts build_pair_lookup_artifacts(
            const std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash> &index_by_value,
            const std::vector<FieldElement> &query_left,
            const std::vector<FieldElement> &query_right,
            const FieldElement &eta,
            std::size_t domain_size,
            std::size_t valid_count,
            const std::string &label,
            RunMetrics *metrics)
        {
            PairLookupArtifacts out;
            out.query.assign(domain_size, FieldElement::zero());
            auto stage_start = Clock::now();
            std::vector<std::size_t> indices(valid_count, 0);
            for (std::size_t i = 0; i < valid_count; ++i)
            {
                out.query[i] = query_left[i] + eta * query_right[i];
                const auto it = index_by_value.find(FieldPairKey{pack_field_key(query_left[i]), pack_field_key(query_right[i])});
                if (it == index_by_value.end())
                {
                    throw std::runtime_error(
                        "pair lookup query escaped static table for " + label + " left=" + query_left[i].to_string() + " right=" + query_right[i].to_string());
                }
                indices[i] = it->second;
            }
            add_metric(metrics != nullptr ? &metrics->lookup_query_pack_ms : nullptr, stage_start);
            stage_start = Clock::now();
            out.multiplicity = multiplicity_from_indices(indices, domain_size);
            add_metric(metrics != nullptr ? &metrics->lookup_multiplicity_ms : nullptr, stage_start);
            return out;
        }

        std::vector<FieldElement> build_weighted_sum(
            const std::vector<FieldElement> &left,
            const FieldElement &lambda,
            const std::vector<FieldElement> &right,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < left.size(); ++i)
            {
                out[i] = left[i] + lambda * right[i];
            }
            return out;
        }

        void finalize_trace_registry(
            const ProtocolContext &context,
            crypto::Transcript &transcript,
            TraceArtifacts &trace,
            RunMetrics *metrics)
        {
            const auto stage_start = Clock::now();
            trace.challenges["alpha_quot"] = transcript.challenge("alpha_quot");
            for (const auto &[label, point] :
                 derive_domain_opening_challenges(context, transcript))
                trace.challenges[label] = point;
            add_metric(metrics != nullptr ? &metrics->trace_finalize_ms : nullptr, stage_start);

            const auto canonical_commitment_order = dynamic_commitment_labels(context);
            std::set<std::string> expected_commitments(
                canonical_commitment_order.begin(),
                canonical_commitment_order.end());
            for (const auto &subproof : trace.pi_bind->subproofs)
            {
                for (const auto &reblind : subproof.model_reblinds)
                {
                    expected_commitments.insert(reblind.proof_local_commitment.name);
                    expected_commitments.insert(reblind.blinder_commitment.name);
                }
                for (const auto *group : {
                         &subproof.view_commitments,
                         &subproof.view_logup_commitments,
                         &subproof.aggregation_commitments,
                         &subproof.basis_commitments,
                         &subproof.crpc_commitments,
                         &subproof.relation_commitments})
                {
                    for (const auto &[label, commitment] : *group)
                    {
                        if (label != commitment.name)
                            throw std::runtime_error(
                                "Pi_bind commitment key/name mismatch: " + label);
                        expected_commitments.insert(label);
                    }
                }
            }
            for (std::size_t head = 0;
                 head < context.model.output_layer.heads.size(); ++head)
            {
                for (const auto &label : {
                         output_bias_view_label(head, "F1"),
                         output_bias_view_label(head, "F2"),
                         output_bias_view_label(head, "terminal_relation_virtual"),
                         output_bias_basis_label(head, "relation_virtual")})
                    expected_commitments.insert(label);
            }
            for (const auto &subproof : trace.pi_bind->subproofs)
            {
                for (const auto &route :
                     subproof.relation_meta.route_terminal_relations)
                    expected_commitments.insert(route.relation_virtual_object);
                for (const auto &aggregation :
                     subproof.relation_meta.aggregation_relations)
                    for (const auto *route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        expected_commitments.insert(route->relation_virtual_object);
            }
            if (trace.commitments.size() != expected_commitments.size())
            {
                std::ostringstream detail;
                detail << "dynamic commitment set size diverged from ProofSchema"
                       << " (actual=" << trace.commitments.size()
                       << ", expected=" << expected_commitments.size()
                       << ", unexpected=";
                bool first = true;
                for (const auto &[label, _] : trace.commitments)
                {
                    if (expected_commitments.contains(label))
                        continue;
                    if (!first)
                        detail << ',';
                    detail << label;
                    first = false;
                }
                detail << ')';
                throw std::runtime_error(detail.str());
            }
            for (const auto &label : expected_commitments)
            {
                if (!trace.commitments.contains(label))
                    throw std::runtime_error(
                        "dynamic commitment is missing from ProofSchema: " + label);
            }
            trace.commitment_order = canonical_commitment_order;
            const auto replayed_pre_quotient =
                replay_challenges(context, trace.commitments, {}, &trace.public_metadata);
            for (const auto &[label, value] : trace.challenges)
            {
                if (label == "alpha_quot" || !replayed_pre_quotient.contains(label))
                    continue;
                if (replayed_pre_quotient.at(label) != value)
                    throw std::runtime_error(
                        "trace challenge diverged from canonical verifier replay: " + label + " trace=" + value.to_string() + " replay=" + replayed_pre_quotient.at(label).to_string());
            }
            trace.challenges["alpha_quot"] = replayed_pre_quotient.at("alpha_quot");
        }

        // 多层多头正式路径：按层序物化隐藏头、输出头与跨层拼接相关的全部 witness 列。
        TraceArtifacts build_multihead_trace(const ProtocolContext &context, RunMetrics *metrics)
        {
            std::string shape_reason;
            if (!model::hidden_family_dimension_chain_is_valid(context.model, &shape_reason))
            {
                throw std::runtime_error("formal multi-head proof shape is invalid: " + shape_reason);
            }
            // 多层多头正式 trace 入口：按层序生成公开列、lookup 列和状态机列，不改 transcript 顺序。
            const auto trace_start = Clock::now();
            const auto initial_forward_ms = metrics != nullptr ? metrics->forward_ms : 0.0;
            const auto initial_commit_dynamic_ms = metrics != nullptr ? metrics->commit_dynamic_ms : 0.0;
            TraceArtifacts trace;
            trace.compiler_artifact = context.compiler_artifact;
            trace.evaluation_commitment_cache =
                std::make_shared<crypto::EvaluationCommitmentCache>();
            if (trace.compiler_artifact == nullptr)
            {
                throw std::runtime_error(
                    "formal trace requires authenticated ProtocolCompilerArtifact");
            }
            trace.pi_bind = std::make_shared<PiBindProof>();
            for (auto &meta : build_binding_relation_metadata(context))
            {
                trace.pi_bind->subproofs.push_back(BindingSubproof{.relation_meta = std::move(meta)});
            }
            crypto::Transcript transcript("gatzkml");
            const bool keep_trace_payloads = context.config.dump_trace;
            const auto &local = context.local;
            const auto &domains = context.domains;
            const auto &edges = local.edges;
            const std::size_t n_nodes = local.num_nodes;
            const std::size_t n_edges = edges.size();
            const std::size_t d_in = local.num_features;
            const std::size_t d_h = model::max_hidden_head_dim(context.model);
            const std::size_t d_cat = model::max_hidden_concat_width(context.model);
            const std::size_t n_classes = local.num_classes;
            const bool enable_single_graph_selector_shortcut = local.graph_count == 1;
            auto profiled_logup_inverse = [&](
                                              const std::vector<FieldElement> &table,
                                              const std::vector<FieldElement> &query,
                                              const std::vector<FieldElement> &q_table,
                                              const std::vector<FieldElement> &q_query,
                                              const FieldElement &beta)
            {
                const auto start = Clock::now();
                auto out = build_logup_inverse_columns(
                    table, query, q_table, q_query, beta);
                add_metric(
                    metrics != nullptr
                        ? &metrics->lookup_inverse_ms
                        : nullptr,
                    start);
                return out;
            };
            auto profiled_logup_accumulator = [&](
                                                  const std::vector<FieldElement> &table_inverse,
                                                  const std::vector<FieldElement> &query_inverse,
                                                  const std::vector<FieldElement> &multiplicity,
                                                  const std::vector<FieldElement> &q_table,
                                                  const std::vector<FieldElement> &q_query,
                                                  std::size_t active_count)
            {
                const auto start = Clock::now();
                auto out =
                    build_logup_accumulator_from_inverse_columns_with_active_count(
                        table_inverse,
                        query_inverse,
                        multiplicity,
                        q_table,
                        q_query,
                        active_count);
                add_metric(
                    metrics != nullptr
                        ? &metrics->lookup_accumulator_ms
                        : nullptr,
                    start);
                return out;
            };

            auto stage_start = Clock::now();
            const auto &static_artifacts = cached_trace_static_artifacts(context);
            const auto &feature_index_fields = static_artifacts.feature_index_fields;
            const auto &absolute_id_fields = static_artifacts.absolute_id_fields;
            const auto &dataset_index_fields = static_artifacts.dataset_index_fields;
            const auto &edge_src_fields = static_artifacts.edge_src_fields;
            const auto &edge_dst_fields = static_artifacts.edge_dst_fields;
            const auto &edge_src_indices = static_artifacts.edge_src_indices;
            const auto &edge_dst_indices = static_artifacts.edge_dst_indices;
            const auto &src_multiplicity_template = static_artifacts.src_multiplicity_template;
            const auto &dst_multiplicity_template = static_artifacts.dst_multiplicity_template;
            const auto &edge_groups_by_dst = static_artifacts.edge_groups_by_dst;
            add_metric(metrics != nullptr ? &metrics->shared_helper_build_ms : nullptr, stage_start);

            // The complete public output is part of M_pub and must be fixed before
            // the first Fiat--Shamir challenge.  Compute the registered integer
            // forward once here; all later witness construction consumes this same
            // immutable artifact.
            const auto forward_start = Clock::now();
            const auto forward_artifacts = cached_forward_artifacts(
                context,
                edges,
                edge_src_indices,
                edge_dst_indices,
                edge_groups_by_dst,
                domains.edge->size);
            if (metrics != nullptr)
            {
                metrics->forward_ms += elapsed_ms(forward_start, Clock::now());
                metrics->feature_projection_ms +=
                    forward_artifacts->profile.hidden_projection_ms + forward_artifacts->profile.output_projection_ms;
                metrics->hidden_forward_projection_ms += forward_artifacts->profile.hidden_projection_ms;
                metrics->hidden_forward_attention_ms += forward_artifacts->profile.hidden_attention_ms;
                metrics->hidden_forward_activation_ms += forward_artifacts->profile.hidden_activation_ms;
                metrics->hidden_concat_ms += forward_artifacts->profile.hidden_concat_ms;
                metrics->output_forward_projection_ms += forward_artifacts->profile.output_projection_ms;
                metrics->output_forward_attention_ms += forward_artifacts->profile.output_attention_ms;
                metrics->output_forward_activation_ms += forward_artifacts->profile.output_activation_ms;
            }
            record_public_output_hashes(
                trace,
                forward_artifacts->output_y_average,
                forward_artifacts->output_y_average,
                forward_artifacts->reference_logits_fp32);
            trace.public_metadata = canonical_public_metadata(context);
            trace.public_metadata.proof_nonce = fresh_proof_nonce();
            trace.public_metadata.public_output = canonical_public_output_descriptor(
                trace.public_logits,
                n_nodes,
                n_classes);

            stage_start = Clock::now();
            // 这里先把公开选择器列取出来备用；真正的动态 witness 还没开始物化。
            const auto q_new = eval_data(context, "P_Q_grp_new_edge");
            const auto q_tbl_feat = eval_data(context, "P_Q_tbl_feat");
            const auto q_qry_feat = eval_data(context, "P_Q_qry_feat");
            const auto q_tbl_l = eval_data(context, "P_Q_tbl_L");
            const auto q_qry_l = eval_data(context, "P_Q_qry_L");
            const auto q_tbl_r = eval_data(context, "P_Q_tbl_R");
            const auto q_qry_r = eval_data(context, "P_Q_qry_R");
            const auto q_tbl_exp = eval_data(context, "P_Q_tbl_exp");
            const auto q_qry_exp = eval_data(context, "P_Q_qry_exp");
            const auto q_tbl_elu = eval_data(context, "P_Q_tbl_ELU");
            add_metric(metrics != nullptr ? &metrics->public_poly_residual_ms : nullptr, stage_start);

            if (static_artifacts.full_feature_identity)
            {
                append_dynamic_commitment(
                    trace,
                    "P_H",
                    fh_coefficient_renamed_commitment("P_H", context, "P_T_H", context.public_commitments.at("P_T_H"), domains.fh));
            }
            else
            {
                auto p_h_fh_values = padded_column(algebra::flatten_matrix_coefficients(local.features), domains.fh->size);
                add_dynamic_commitment_batch(
                    trace,
                    {column_spec("P_H", std::move(p_h_fh_values), domains.fh)},
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
            }

            // 第一轮 transcript 先吸收公开元数据、图规模和输入特征承诺，
            // 由此固定后续特征 lookup 的挑战标签。
            absorb_public_metadata(transcript, trace.public_metadata);
            transcript.absorb_scalar("N", FieldElement(n_nodes));
            transcript.absorb_scalar("E", FieldElement(n_edges));
            transcript.absorb_scalar("d_in", FieldElement(d_in));
            transcript.absorb_scalar("d_h", FieldElement(d_h));
            transcript.absorb_scalar("d_cat", FieldElement(d_cat));
            transcript.absorb_scalar("C", FieldElement(n_classes));
            transcript.absorb_scalar("B", FieldElement(context.config.range_bits));
            transcript.absorb_commitment("P_I", context.public_commitments.at("P_I").point);
            transcript.absorb_commitment("P_src", context.public_commitments.at("P_src").point);
            transcript.absorb_commitment("P_dst", context.public_commitments.at("P_dst").point);
            transcript.absorb_commitment("P_node_gid", context.public_commitments.at("P_node_gid").point);
            transcript.absorb_commitment("P_edge_gid", context.public_commitments.at("P_edge_gid").point);
            transcript.absorb_commitment("P_Q_graph_new_edge", context.public_commitments.at("P_Q_graph_new_edge").point);
            transcript.absorb_commitment("P_Q_graph_end_edge", context.public_commitments.at("P_Q_graph_end_edge").point);
            transcript.absorb_commitment("P_Q_grp_new_edge", context.public_commitments.at("P_Q_grp_new_edge").point);
            transcript.absorb_commitment("P_Q_grp_end_edge", context.public_commitments.at("P_Q_grp_end_edge").point);
            transcript.absorb_commitment("P_Q_edge_valid", context.public_commitments.at("P_Q_edge_valid").point);
            transcript.absorb_commitment("P_Q_N", context.public_commitments.at("P_Q_N").point);
            transcript.absorb_commitment(
                "P_AggBind_m_src",
                context.public_commitments.at("P_AggBind_m_src").point);
            transcript.absorb_commitment(
                "P_AggBind_m_dst",
                context.public_commitments.at("P_AggBind_m_dst").point);
            transcript.absorb_commitment(
                "P_AggBind_index_N",
                context.public_commitments.at("P_AggBind_index_N").point);
            transcript.absorb_commitment("P_Q_proj_valid", context.public_commitments.at("P_Q_proj_valid").point);
            transcript.absorb_commitment("P_Q_d_valid", context.public_commitments.at("P_Q_d_valid").point);
            transcript.absorb_commitment("P_Q_cat_valid", context.public_commitments.at("P_Q_cat_valid").point);
            transcript.absorb_commitment("P_Q_C_valid", context.public_commitments.at("P_Q_C_valid").point);
            transcript.absorb_commitment("P_BiasIndex_C", context.public_commitments.at("P_BiasIndex_C").point);
            transcript.absorb_commitment("P_BiasMultiplicity_C", context.public_commitments.at("P_BiasMultiplicity_C").point);
            transcript.absorb_commitment("P_BiasIndex_edge", context.public_commitments.at("P_BiasIndex_edge").point);
            constexpr std::size_t kCoefficientTailBaseObservationCount = 3;
            const auto bias_basis_chunks = coefficient_chunk_layout(
                n_classes + kCoefficientTailBaseObservationCount,
                domains.edge->size);
            for (std::size_t head_index = 0;
                 head_index < context.model.output_layer.heads.size(); ++head_index)
                for (const auto &chunk : bias_basis_chunks)
                    for (const auto *region : {
                             "active", "semantic", "gap", "tail"})
                    {
                        const auto label =
                            output_bias_basis_region_selector_label(
                                head_index, chunk.chunk_index, region);
                        transcript.absorb_commitment(
                            label, context.public_commitments.at(label).point);
                    }
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto label = "P_Q_div_h" + std::to_string(head_index);
                transcript.absorb_commitment(label, context.public_commitments.at(label).point);
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto label = "P_Q_div_out" + std::to_string(head_index);
                transcript.absorb_commitment(label, context.public_commitments.at(label).point);
            }
            transcript.absorb_commitment("P_Q_div_avg", context.public_commitments.at("P_Q_div_avg").point);
            transcript.absorb_commitment("P_H", trace.commitments.at("P_H").point);
            transcript.absorb_commitment("V_T_H", context.static_commitments.at("V_T_H").point);
            trace.challenges["eta_feat"] = transcript.challenge("eta_feat");

            {
                // 特征 lookup 是全链路最先出现的查表约束，用来把原始输入特征绑定到 formal 主线。
                const auto eta_feat_powers = powers(trace.challenges.at("eta_feat"), 3);
                const auto eta_feature_index = eta_feat_powers[1];
                const auto eta_feature_value = eta_feat_powers[2];
                // The formal lookup always follows the same two-round schedule:
                // Table/Query/m precede beta, while J/R are beta-dependent.
                std::vector<FieldElement> table_feat(domains.fh->size, FieldElement::zero());
                std::vector<FieldElement> query_feat(domains.fh->size, FieldElement::zero());
                std::vector<FieldElement> multiplicity_feat(domains.fh->size, FieldElement::zero());
                std::vector<std::size_t> feat_hits(context.dataset.num_nodes, 0);
                auto fh_stage_start = Clock::now();
                for (const auto absolute_id : local.absolute_ids)
                {
                    if (absolute_id < feat_hits.size())
                    {
                        feat_hits[absolute_id] += 1;
                    }
                }
                std::vector<FieldElement> feat_hit_fields(context.dataset.num_nodes, FieldElement::zero());
                for (std::size_t v = 0; v < context.dataset.num_nodes; ++v)
                {
                    feat_hit_fields[v] = FieldElement(feat_hits[v]);
                }
                add_metric(metrics != nullptr ? &metrics->fh_multiplicity_build_ms : nullptr, fh_stage_start);
                fh_stage_start = Clock::now();
                const auto &dataset_feature_rows =
                    !context.dataset.features.empty()
                        ? context.dataset.features
                        : local.features;
                if (context.dataset.features.empty() && !static_artifacts.full_feature_identity)
                {
                    throw std::runtime_error(
                        "released dataset feature table lacks a full canonical local view");
                }
                fill_feature_lookup_rows(
                    dataset_index_fields,
                    feature_index_fields,
                    dataset_feature_rows,
                    eta_feature_index,
                    eta_feature_value,
                    table_feat);
                add_metric(metrics != nullptr ? &metrics->fh_table_materialization_ms : nullptr, fh_stage_start);
                fh_stage_start = Clock::now();
                fill_feature_lookup_rows(
                    absolute_id_fields,
                    feature_index_fields,
                    local.features,
                    eta_feature_index,
                    eta_feature_value,
                    query_feat);
                add_metric(metrics != nullptr ? &metrics->fh_query_materialization_ms : nullptr, fh_stage_start);
                fh_stage_start = Clock::now();
                fill_repeated_row_values(feat_hit_fields, d_in, multiplicity_feat);
                add_metric(metrics != nullptr ? &metrics->fh_multiplicity_build_ms : nullptr, fh_stage_start);

                const auto lookup_stage_start = Clock::now();
                std::vector<DynamicCommitmentSpec> feature_specs;
                feature_specs.reserve(3);
                feature_specs.push_back(deferred_linear_column_spec(
                    "P_Table_feat",
                    table_feat,
                    domains.fh,
                    {
                        linear_point_term(
                            context.public_commitments.at(
                                                          "P_Row_feat_tbl")
                                .point,
                            FieldElement::one()),
                        linear_point_term(
                            context.public_commitments.at(
                                                          "P_Col_feat_tbl")
                                .point,
                            eta_feature_index),
                        linear_point_term(
                            context.public_commitments.at("P_T_H").point,
                            eta_feature_value),
                    }));
                if (static_artifacts.full_feature_identity)
                {
                    feature_specs.push_back(
                        deferred_linear_column_spec(
                            "P_Query_feat",
                            query_feat,
                            domains.fh,
                            {
                                linear_label_term(
                                    "P_Table_feat",
                                    FieldElement::one()),
                            }));
                }
                else
                {
                    feature_specs.push_back(column_spec(
                        "P_Query_feat", query_feat, domains.fh));
                }
                feature_specs.push_back(column_spec(
                    "P_m_feat", multiplicity_feat, domains.fh));
                add_dynamic_commitment_batch(
                    trace,
                    std::move(feature_specs),
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                transcript.absorb_commitment(
                    "P_Table_feat", trace.commitments.at("P_Table_feat").point);
                transcript.absorb_commitment(
                    "P_Query_feat", trace.commitments.at("P_Query_feat").point);
                transcript.absorb_commitment(
                    "P_m_feat", trace.commitments.at("P_m_feat").point);
                trace.challenges["beta_feat"] = transcript.challenge("beta_feat");

                const auto fh_accumulator_start = Clock::now();
                auto j_feat = profiled_logup_inverse(
                    table_feat,
                    query_feat,
                    q_tbl_feat,
                    q_qry_feat,
                    trace.challenges.at("beta_feat"));
                auto r_feat =
                    profiled_logup_accumulator(
                        j_feat.table,
                        j_feat.query,
                        multiplicity_feat,
                        q_tbl_feat,
                        q_qry_feat,
                        lookup_active_transition_limit(context.dataset.num_nodes * d_in, n_nodes * d_in, domains.fh->size));
                add_metric(metrics != nullptr ? &metrics->fh_accumulator_build_ms : nullptr, fh_accumulator_start);
                add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, lookup_stage_start);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec("P_J_tbl_feat", std::move(j_feat.table), domains.fh),
                        static_artifacts.full_feature_identity
                            ? deferred_linear_column_spec(
                                  "P_J_qry_feat",
                                  std::move(j_feat.query),
                                  domains.fh,
                                  {
                                      linear_label_term(
                                          "P_J_tbl_feat",
                                          FieldElement::one()),
                                  })
                            : column_spec(
                                  "P_J_qry_feat",
                                  std::move(j_feat.query),
                                  domains.fh),
                        column_spec("P_R_feat", std::move(r_feat), domains.fh),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                transcript.absorb_commitment(
                    "P_J_tbl_feat", trace.commitments.at("P_J_tbl_feat").point);
                transcript.absorb_commitment(
                    "P_J_qry_feat", trace.commitments.at("P_J_qry_feat").point);
                transcript.absorb_commitment(
                    "P_R_feat", trace.commitments.at("P_R_feat").point);
            }

            // 从这里开始进入“量化入口 / witness 物化入口”：先跑参考前向，再把中间量转成域元素列。
            const auto &hidden_layer_inputs_quantized = forward_artifacts->hidden_layer_inputs_quantized;
            const auto &hidden_layer_concat_quantized = forward_artifacts->hidden_layer_concat_quantized;

            stage_start = Clock::now();
            const auto lrelu_index = cached_pair_index_map("lrelu", context.tables.lrelu);
            const auto range_index = cached_value_index_map("range", context.tables.range, context.tables.range.size());
            const auto exp_index = cached_pair_index_map("exp", context.tables.exp);
            const auto elu_index = cached_pair_index_map("elu", context.tables.elu);
            add_metric(metrics != nullptr ? &metrics->lookup_key_build_ms : nullptr, stage_start);

            auto commit_signed_division = [&](
                                              const std::string &division_prefix,
                                              const SignedRoundDivWitness &division,
                                              const std::string &selector_label)
            {
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(division_prefix + "Active", forward_artifacts, division.active, domains.edge),
                        shared_column_spec(division_prefix + "X", forward_artifacts, division.x, domains.edge),
                        shared_column_spec(division_prefix + "D", forward_artifacts, division.d, domains.edge),
                        shared_column_spec(division_prefix + "Y", forward_artifacts, division.y, domains.edge),
                        shared_column_spec(division_prefix + "Sigma", forward_artifacts, division.sigma, domains.edge),
                        shared_column_spec(division_prefix + "Zero", forward_artifacts, division.z, domains.edge),
                        shared_column_spec(division_prefix + "A", forward_artifacts, division.a, domains.edge),
                        shared_column_spec(division_prefix + "A_inv", forward_artifacts, division.a_inv, domains.edge),
                        shared_column_spec(division_prefix + "D_inv", forward_artifacts, division.d_inv, domains.edge),
                        shared_column_spec(division_prefix + "Q", forward_artifacts, division.q, domains.edge),
                        shared_column_spec(division_prefix + "R", forward_artifacts, division.r, domains.edge),
                        shared_column_spec(division_prefix + "Delta", forward_artifacts, division.delta, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {
                         "Active", "X", "D", "Y", "Sigma", "Zero",
                         "A", "A_inv", "D_inv", "Q", "R", "Delta"})
                {
                    const auto label = division_prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                const auto &q_query = eval_data(context, selector_label);
                const auto valid_count = static_cast<std::size_t>(std::count_if(
                    q_query.begin(), q_query.end(), [](const FieldElement &value)
                    { return !value.is_zero(); }));
                const auto table = cached_single_lookup_table_shared(
                    "signed-div-range", context.tables.range, domains.edge->size);
                const auto active_count = std::max(context.tables.range.size(), valid_count);
                for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
                {
                    const auto &limb_columns = division.limbs.at(value_name);
                    for (std::size_t limb = 0; limb < limb_columns.size(); ++limb)
                    {
                        const auto limb_label = signed_division_limb_label(
                            division_prefix, value_name, limb);
                        const auto table_label = signed_division_range_label(
                            division_prefix, value_name, limb, "Table");
                        const auto query_label = signed_division_range_label(
                            division_prefix, value_name, limb, "Query");
                        const auto multiplicity_label = signed_division_range_label(
                            division_prefix, value_name, limb, "m");
                        const auto padded_limb_query = padded_column(
                            limb_columns[limb],
                            domains.edge->size);
                        const auto multiplicity =
                            std::make_shared<const std::vector<FieldElement>>(
                                build_single_lookup_multiplicity_for_packed_query(
                                    *range_index,
                                    padded_limb_query,
                                    domains.edge->size,
                                    valid_count,
                                    limb_label,
                                    metrics));
                        add_dynamic_commitment_batch(
                            trace,
                            {
                                shared_column_spec(
                                    limb_label,
                                    forward_artifacts,
                                    limb_columns[limb],
                                    domains.edge),
                                column_spec(
                                    table_label,
                                    table,
                                    domains.edge),
                                shared_column_spec(
                                    query_label,
                                    forward_artifacts,
                                    limb_columns[limb],
                                    domains.edge),
                                column_spec(
                                    multiplicity_label,
                                    multiplicity,
                                    domains.edge),
                            },
                            context.kzg,
                            keep_trace_payloads,
                            metrics);
                        for (const auto &label : {
                                 limb_label, table_label, query_label, multiplicity_label})
                        {
                            transcript.absorb_commitment(label, trace.commitments.at(label).point);
                        }
                        const auto beta_name = signed_division_beta_name(
                            division_prefix, value_name, limb);
                        const auto beta = transcript.challenge(beta_name);
                        trace.challenges[beta_name] = beta;
                        auto inverses = profiled_logup_inverse(
                            *table,
                            padded_limb_query,
                            q_tbl_r,
                            q_query,
                            beta);
                        auto accumulator =
                            profiled_logup_accumulator(
                                inverses.table,
                                inverses.query,
                                *multiplicity,
                                q_tbl_r,
                                q_query,
                                active_count);
                        const auto j_table_label = signed_division_range_label(
                            division_prefix, value_name, limb, "J_tbl");
                        const auto j_query_label = signed_division_range_label(
                            division_prefix, value_name, limb, "J_qry");
                        const auto accumulator_label = signed_division_range_label(
                            division_prefix, value_name, limb, "R");
                        add_dynamic_commitment_batch(
                            trace,
                            {
                                column_spec(j_table_label, std::move(inverses.table), domains.edge),
                                column_spec(j_query_label, std::move(inverses.query), domains.edge),
                                column_spec(accumulator_label, std::move(accumulator), domains.edge),
                            },
                            context.kzg,
                            keep_trace_payloads,
                            metrics);
                        for (const auto &label : {
                                 j_table_label, j_query_label, accumulator_label})
                        {
                            transcript.absorb_commitment(label, trace.commitments.at(label).point);
                        }
                    }
                }
            };

            auto append_model_binding_blinder = [&](
                                                    const std::string &source_label,
                                                    const std::string &blinder_label,
                                                    const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                                    const std::vector<std::pair<std::string, FieldElement>> &proof_local_terms)
            {
                const auto source_it = trace.polynomials.find(source_label);
                if (source_it == trace.polynomials.end() || source_it->second.zero_domain_blinder.empty())
                {
                    throw std::runtime_error(
                        "model binding source is missing proof-local blinding: " + source_label);
                }
                if (proof_local_terms.empty())
                {
                    throw std::runtime_error("model binding has no proof-local canonical source: " + source_label);
                }
                std::vector<FieldElement> relation_blinder =
                    source_it->second.zero_domain_blinder;
                auto subtract_blinder = [&](const std::vector<FieldElement> &other, const FieldElement &coefficient)
                {
                    if (relation_blinder.size() < other.size())
                        relation_blinder.resize(other.size(), FieldElement::zero());
                    for (std::size_t i = 0; i < other.size(); ++i)
                        relation_blinder[i] -= coefficient * other[i];
                };
                for (std::size_t row = 0; row < domain->size; ++row)
                {
                    auto expected = FieldElement::zero();
                    for (const auto &[anchor_label, coefficient] : proof_local_terms)
                    {
                        const auto proof_label = model_proof_local_label(anchor_label);
                        const auto proof_it = trace.polynomials.find(proof_label);
                        if (proof_it == trace.polynomials.end() || proof_it->second.domain == nullptr || proof_it->second.domain->name != domain->name)
                        {
                            throw std::runtime_error(
                                "model binding proof-local source domain mismatch: " + proof_label);
                        }
                        expected += coefficient * proof_it->second.value_at(row);
                        if (row == 0)
                            subtract_blinder(proof_it->second.zero_domain_blinder, coefficient);
                    }
                    if (source_it->second.value_at(row) != expected)
                    {
                        throw std::runtime_error(
                            "model binding semantic view is not derived from proof-local canonical sources: " + source_label);
                    }
                }
                auto polynomial = Polynomial::from_coefficients(
                    blinder_label,
                    std::move(relation_blinder));
                auto commitment = commit_proof_polynomial(
                    blinder_label, polynomial, context.kzg, metrics);
                trace.polynomials.emplace(blinder_label, polynomial);
                trace.polynomial_domains[blinder_label] = domain->name;
                append_dynamic_commitment(trace, blinder_label, std::move(commitment));
            };

            auto append_external_mask = [&](
                                            const std::string &source_label,
                                            const std::string &mask_label)
            {
                const auto source_it = trace.polynomials.find(source_label);
                if (source_it == trace.polynomials.end() || source_it->second.domain == nullptr || source_it->second.domain->name != domains.n->name || source_it->second.zero_domain_blinder.empty())
                {
                    throw std::runtime_error(
                        "external evaluation object is missing its registered N-domain mask: " + source_label);
                }
                auto mask = Polynomial::from_coefficients(
                    mask_label, source_it->second.zero_domain_blinder);
                auto commitment = commit_proof_polynomial(
                    mask_label, mask, context.kzg, metrics);
                trace.polynomials.emplace(mask_label, mask);
                trace.polynomial_domains[mask_label] = domains.n->name;
                append_dynamic_commitment(
                    trace, mask_label, std::move(commitment));
            };

            auto require_binding_subproof = [&](const std::string &relation_id) -> BindingSubproof &
            {
                const auto it = std::find_if(
                    trace.pi_bind->subproofs.begin(),
                    trace.pi_bind->subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (it == trace.pi_bind->subproofs.end())
                {
                    throw std::runtime_error("missing Pi_bind subproof registry entry: " + relation_id);
                }
                return *it;
            };

            auto append_single_model_reblind = [&](
                                                   BindingSubproof &subproof,
                                                   const std::string &anchor_label,
                                                   const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
            {
                const auto anchor_it = context.static_commitments.find(anchor_label);
                if (anchor_it == context.static_commitments.end() || anchor_it->second.opening_polynomial == nullptr)
                {
                    throw std::runtime_error(
                        "model anchor lacks canonical opening material: " + anchor_label);
                }
                const auto &anchor_polynomial =
                    *anchor_it->second.opening_polynomial;
                if (anchor_polynomial.basis != algebra::PolynomialBasis::Evaluation || anchor_polynomial.domain == nullptr || anchor_polynomial.domain->name != domain->name)
                {
                    throw std::runtime_error(
                        "model anchor canonical domain mismatch: " + anchor_label);
                }

                const auto proof_label =
                    model_proof_local_label(anchor_label);
                const auto witness_label =
                    model_reblind_witness_label(anchor_label);
                if (trace.commitments.contains(proof_label) || trace.commitments.contains(witness_label))
                {
                    throw std::runtime_error(
                        "duplicate proof-local model object: " + anchor_label);
                }

                auto proof_polynomial = anchor_polynomial;
                proof_polynomial.name = proof_label;
                crypto::Commitment witness_commitment;
                algebra::Polynomial witness_polynomial;
                do
                {
                    const auto blinder = sample_trace_blinder(
                        *context.compiler_artifact, proof_label);
                    proof_polynomial.set_zero_domain_blinder(blinder);
                    witness_polynomial = Polynomial::from_coefficients(
                        witness_label, blinder);
                    witness_commitment = commit_proof_polynomial(
                        witness_label,
                        witness_polynomial,
                        context.kzg,
                        metrics);
                } while (witness_commitment.point == crypto::g1_zero());
                auto proof_commitment =
                    commit_reblinded_evaluation_from_anchor(
                        anchor_it->second,
                        proof_label, proof_polynomial,
                        context.kzg,
                        metrics);

                trace.polynomial_domains[proof_label] = domain->name;
                trace.polynomial_domains[witness_label] = domain->name;
                trace.polynomials.emplace(
                    proof_label, proof_polynomial);
                trace.polynomials.emplace(
                    witness_label, witness_polynomial);
                trace.commitments.emplace(
                    proof_label, proof_commitment);
                trace.commitments.emplace(
                    witness_label, witness_commitment);
                subproof.model_reblinds.push_back(ModelReblindProof{
                    .object_id = anchor_label,
                    .domain = domain->name,
                    .vk_model_anchor = MetadataReference{anchor_label},
                    .proof_local_commitment =
                        std::move(proof_commitment),
                    .blinder_commitment =
                        std::move(witness_commitment),
                });
            };

            auto append_model_reblinds = [&](
                                             BindingSubproof &subproof,
                                             const std::vector<std::string> &anchor_labels,
                                             const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
            {
                constexpr std::size_t kMinAmortizedBatchObjects = 4;
                if (anchor_labels.size() < kMinAmortizedBatchObjects)
                {
                    for (const auto &anchor_label : anchor_labels)
                    {
                        append_single_model_reblind(
                            subproof, anchor_label, domain);
                    }
                    return;
                }

                struct PendingModelReblind
                {
                    std::string anchor_label;
                    std::string proof_label;
                    std::string witness_label;
                    Polynomial proof_polynomial;
                    Polynomial witness_polynomial;
                    crypto::Commitment proof_commitment;
                    crypto::Commitment witness_commitment;
                };

                std::vector<PendingModelReblind> pending;
                pending.reserve(anchor_labels.size());
                for (const auto &anchor_label : anchor_labels)
                {
                    const auto anchor_it = context.static_commitments.find(anchor_label);
                    if (anchor_it == context.static_commitments.end() || anchor_it->second.opening_polynomial == nullptr)
                    {
                        throw std::runtime_error("model anchor lacks canonical opening material: " + anchor_label);
                    }
                    const auto &anchor_polynomial = *anchor_it->second.opening_polynomial;
                    if (anchor_polynomial.basis != algebra::PolynomialBasis::Evaluation || anchor_polynomial.domain == nullptr || anchor_polynomial.domain->name != domain->name)
                    {
                        throw std::runtime_error("model anchor canonical domain mismatch: " + anchor_label);
                    }

                    const auto proof_label = model_proof_local_label(anchor_label);
                    const auto witness_label = model_reblind_witness_label(anchor_label);
                    if (trace.commitments.contains(proof_label) || trace.commitments.contains(witness_label))
                    {
                        throw std::runtime_error("duplicate proof-local model object: " + anchor_label);
                    }

                    auto proof_polynomial = anchor_polynomial;
                    proof_polynomial.name = proof_label;
                    crypto::Commitment witness_commitment;
                    algebra::Polynomial witness_polynomial;
                    do
                    {
                        const auto blinder = sample_trace_blinder(
                            *context.compiler_artifact, proof_label);
                        proof_polynomial.set_zero_domain_blinder(blinder);
                        witness_polynomial = Polynomial::from_coefficients(witness_label, blinder);
                        witness_commitment = commit_proof_polynomial(
                            witness_label, witness_polynomial, context.kzg, metrics);
                    } while (witness_commitment.point == crypto::g1_zero());
                    auto proof_commitment =
                        commit_reblinded_evaluation_from_anchor(
                            anchor_it->second,
                            proof_label, proof_polynomial,
                            context.kzg, metrics);

                    pending.push_back(PendingModelReblind{
                        .anchor_label = anchor_label,
                        .proof_label = proof_label,
                        .witness_label = witness_label,
                        .proof_polynomial = std::move(proof_polynomial),
                        .witness_polynomial = std::move(witness_polynomial),
                        .proof_commitment = std::move(proof_commitment),
                        .witness_commitment = std::move(witness_commitment),
                    });
                }

                for (std::size_t index = 0; index < pending.size(); ++index)
                {
                    auto &entry = pending[index];
                    auto &proof_commitment = entry.proof_commitment;
                    trace.polynomial_domains[entry.proof_label] = domain->name;
                    trace.polynomial_domains[entry.witness_label] = domain->name;
                    trace.polynomials.emplace(
                        entry.proof_label, std::move(entry.proof_polynomial));
                    trace.polynomials.emplace(
                        entry.witness_label, std::move(entry.witness_polynomial));
                    trace.commitments.emplace(entry.proof_label, proof_commitment);
                    trace.commitments.emplace(
                        entry.witness_label, entry.witness_commitment);
                    subproof.model_reblinds.push_back(ModelReblindProof{
                        .object_id = entry.anchor_label,
                        .domain = domain->name,
                        .vk_model_anchor = MetadataReference{entry.anchor_label},
                        .proof_local_commitment = std::move(proof_commitment),
                        .blinder_commitment = std::move(entry.witness_commitment),
                    });
                }
            };

            // 每个隐藏头都独立展开一套 route / lookup / PSQ / binding 约束，
            // 但最终仍共享同一 transcript 顺序与 proof block 契约。
            auto commit_hidden_head = [&](std::size_t head_index)
            {
                std::size_t layer_index = 0;
                std::size_t local_head_index = head_index;
                while (layer_index < context.model.hidden_layers.size() && local_head_index >= context.model.hidden_layers[layer_index].heads.size())
                {
                    local_head_index -= context.model.hidden_layers[layer_index].heads.size();
                    ++layer_index;
                }
                if (layer_index >= context.model.hidden_layers.size())
                {
                    throw std::runtime_error("hidden head index escaped hidden layer family");
                }
                const auto &layer = context.model.hidden_layers[layer_index];
                const auto &input_matrix = hidden_layer_inputs_quantized.at(layer_index);
                const auto input_dim = layer.input_dim;
                const auto head_dim = layer.shape.head_dim;
                const auto q_qry_elu = eval_data(
                    context,
                    head_dim == d_h
                        ? "P_Q_qry_ELU"
                        : "P_Q_div_h" + std::to_string(head_index));
                const auto input_commitment_label =
                    layer_index == 0
                        ? std::string("P_H")
                        : hidden_layer_concat_label(layer_index - 1, false);
                const auto prefix = hidden_head_prefix(head_index) + "_";
                auto stage_start = Clock::now();
                // 这一组量已经对应笔记里的单头前向主线：特征投影、源点打分、目标点打分、
                // 稳定化最大值、分母和、分母倒数、注意力权重，以及按边聚合后的隐藏表示。
                const auto &quantized = forward_artifacts->hidden_head_quantized.at(head_index);
                const auto &h_prime = quantized.h_prime;
                const auto &e_src = quantized.e_src;
                const auto &e_dst = quantized.e_dst;
                const auto &s = quantized.s;
                const auto &z = quantized.z;
                const auto &m = quantized.m;
                const auto &delta = quantized.delta;
                const auto &u = quantized.u;
                const auto &sum = quantized.sum;
                const auto &inv = quantized.inv;
                const auto &alpha = quantized.alpha;
                const auto &h_agg_pre = quantized.h_agg_pre;
                const auto &h_agg_pre_lookup = quantized.h_agg_pre_lookup;
                const auto &h_agg = quantized.h_agg;
                const auto &e_src_edge = quantized.e_src_edge;
                const auto &e_dst_edge = quantized.e_dst_edge;
                const auto &m_edge = quantized.m_edge;
                const auto &sum_edge = quantized.sum_edge;
                const auto &inv_edge = quantized.inv_edge;
                add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);
                add_metric(metrics != nullptr ? &metrics->hidden_edge_score_trace_ms : nullptr, stage_start);

                auto &binding_subproof = require_binding_subproof(
                    "bind:hidden:" + std::to_string(layer_index + 1) + ":" + std::to_string(local_head_index));
                std::vector<std::string> model_reblind_labels;
                model_reblind_labels.reserve(head_dim + 2);
                for (std::size_t column = 0; column < head_dim; ++column)
                {
                    model_reblind_labels.push_back(
                        hidden_weight_column_label(head_index, column));
                }
                append_model_reblinds(
                    binding_subproof, model_reblind_labels, domains.in);
                append_model_reblinds(
                    binding_subproof,
                    {
                        hidden_src_label(head_index),
                        hidden_dst_label(head_index),
                    },
                    domains.d);

                add_dynamic_commitment_batch(
                    trace,
                    {
                        matrix_commitment_only_spec(prefix + "H_prime", h_prime),
                        shared_compact_column_spec(prefix + "E_src", forward_artifacts, e_src, domains.n),
                        shared_compact_column_spec(prefix + "E_dst", forward_artifacts, e_dst, domains.n),
                        shared_compact_column_spec(prefix + "E_src_edge", forward_artifacts, e_src_edge, domains.edge),
                        shared_compact_column_spec(prefix + "E_dst_edge", forward_artifacts, e_dst_edge, domains.edge),
                        deferred_linear_compact_shared_column_spec(
                            prefix + "S",
                            forward_artifacts,
                            s,
                            domains.edge,
                            {
                                linear_label_term(prefix + "E_src_edge", FieldElement::one()),
                                linear_label_term(prefix + "E_dst_edge", FieldElement::one()),
                            }),
                        shared_compact_column_spec(prefix + "Z", forward_artifacts, z, domains.edge),
                        shared_compact_column_spec(prefix + "M", forward_artifacts, m, domains.n),
                        shared_compact_column_spec(prefix + "M_edge", forward_artifacts, m_edge, domains.edge),
                        deferred_linear_compact_shared_column_spec(
                            prefix + "Delta",
                            forward_artifacts,
                            delta,
                            domains.edge,
                            {
                                linear_label_term(prefix + "M_edge", FieldElement::one()),
                                linear_label_term(prefix + "Z", -FieldElement::one()),
                            }),
                        shared_compact_column_spec(prefix + "U", forward_artifacts, u, domains.edge),
                        shared_compact_column_spec(prefix + "Sum", forward_artifacts, sum, domains.n),
                        shared_compact_column_spec(prefix + "Sum_edge", forward_artifacts, sum_edge, domains.edge),
                        shared_compact_column_spec(prefix + "inv", forward_artifacts, inv, domains.n),
                        shared_compact_column_spec(prefix + "inv_edge", forward_artifacts, inv_edge, domains.edge),
                        shared_compact_column_spec(prefix + "alpha", forward_artifacts, alpha, domains.edge),
                        matrix_commitment_only_spec(prefix + "H_agg_pre", h_agg_pre),
                        matrix_commitment_only_spec(prefix + "H_agg", h_agg),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);

                append_external_mask(
                    prefix + "E_src", hidden_score_mask_label(head_index, "src"));
                append_external_mask(
                    prefix + "E_dst", hidden_score_mask_label(head_index, "dst"));

                transcript.absorb_commitment(input_commitment_label, trace.commitments.at(input_commitment_label).point);
                transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
                for (std::size_t column = 0; column < head_dim; ++column)
                {
                    const auto label = hidden_weight_column_label(head_index, column);
                    const auto proof_label = model_proof_local_label(label);
                    const auto witness_label = model_reblind_witness_label(label);
                    transcript.absorb_commitment(proof_label, trace.commitments.at(proof_label).point);
                    transcript.absorb_commitment(witness_label, trace.commitments.at(witness_label).point);
                }
                trace.challenges["y_proj_h" + std::to_string(head_index)] = transcript.challenge("y_proj_h" + std::to_string(head_index));
                trace.challenges["xi_h" + std::to_string(head_index)] = transcript.challenge("xi_h" + std::to_string(head_index));
                transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
                transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
                transcript.absorb_commitment(
                    hidden_score_mask_label(head_index, "src"),
                    trace.commitments.at(hidden_score_mask_label(head_index, "src")).point);
                transcript.absorb_commitment(
                    model_proof_local_label(hidden_src_label(head_index)),
                    trace.commitments.at(model_proof_local_label(hidden_src_label(head_index))).point);
                transcript.absorb_commitment(
                    model_reblind_witness_label(hidden_src_label(head_index)),
                    trace.commitments.at(model_reblind_witness_label(hidden_src_label(head_index))).point);
                trace.challenges["y_src_h" + std::to_string(head_index)] = transcript.challenge("y_src_h" + std::to_string(head_index));
                transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
                transcript.absorb_commitment(prefix + "E_dst", trace.commitments.at(prefix + "E_dst").point);
                transcript.absorb_commitment(
                    hidden_score_mask_label(head_index, "dst"),
                    trace.commitments.at(hidden_score_mask_label(head_index, "dst")).point);
                transcript.absorb_commitment(
                    model_proof_local_label(hidden_dst_label(head_index)),
                    trace.commitments.at(model_proof_local_label(hidden_dst_label(head_index))).point);
                transcript.absorb_commitment(
                    model_reblind_witness_label(hidden_dst_label(head_index)),
                    trace.commitments.at(model_reblind_witness_label(hidden_dst_label(head_index))).point);
                trace.challenges["y_dst_h" + std::to_string(head_index)] = transcript.challenge("y_dst_h" + std::to_string(head_index));

                stage_start = Clock::now();
                const auto xi = trace.challenges.at("xi_h" + std::to_string(head_index));
                const auto compressed = cached_hidden_compressed_trace(
                    h_prime,
                    h_agg_pre,
                    h_agg,
                    edges,
                    domains.edge->size,
                    xi);
                const auto &h_star = compressed->h_star;
                const auto &h_star_edge = compressed->h_star_edge;
                add_metric(metrics != nullptr ? &metrics->hidden_h_star_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto &h_agg_pre_star = compressed->h_agg_pre_star;
                const auto &h_agg_pre_star_edge = compressed->h_agg_pre_star_edge;
                add_metric(metrics != nullptr ? &metrics->hidden_h_agg_pre_star_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto &h_agg_star = compressed->h_agg_star;
                const auto &h_agg_star_edge = compressed->h_agg_star_edge;
                add_metric(metrics != nullptr ? &metrics->hidden_h_agg_star_trace_ms : nullptr, stage_start);

                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_compact_column_spec(prefix + "H_star", compressed, h_star, domains.n),
                        shared_compact_column_spec(prefix + "H_src_star_edge", compressed, h_star_edge, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                append_external_mask(
                    prefix + "H_star",
                    external_mask_label(prefix + "H_star"));

                stage_start = Clock::now();
                const auto &hidden_parameters = cached_quantized_hidden_head_parameters(context, head_index);
                const auto &head_w = hidden_parameters.seq_kernel;
                const auto y_proj_h = trace.challenges.at("y_proj_h" + std::to_string(head_index));
                const auto y_proj_powers = powers(y_proj_h, head_dim);
                const auto proj_b = linear_form_by_powers(head_w, y_proj_powers);
                const auto proj_binding =
                    cached_binding_trace(
                        [&]()
                        {
                            const auto target_mu = matrix_row_major_evaluation(h_prime, y_proj_h);
                            return build_row_major_projection_binding_trace(
                                input_matrix,
                                n_nodes,
                                y_proj_h,
                                head_dim,
                                proj_b,
                                domains.in,
                                input_dim,
                                target_mu);
                        });
                add_metric(metrics != nullptr ? &metrics->hidden_projection_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto y_src_h = trace.challenges.at("y_src_h" + std::to_string(head_index));
                const auto src_binding = cached_binding_trace(
                    [&]()
                    {
                        return build_binding_trace(
                            h_prime,
                            domains.n,
                            n_nodes,
                            y_src_h,
                            hidden_parameters.attn_src,
                            domains.d,
                            head_dim);
                    });
                add_metric(metrics != nullptr ? &metrics->hidden_src_attention_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto y_dst_h = trace.challenges.at("y_dst_h" + std::to_string(head_index));
                const auto dst_binding = cached_binding_trace(
                    head_cache_prefix + ":dst:" + y_dst_h.to_string(),
                    [&]()
                    {
                        return build_binding_trace(
                            h_prime,
                            domains.n,
                            n_nodes,
                            y_dst_h,
                            hidden_parameters.attn_dst,
                            domains.d,
                            head_dim);
                    });
                transcript.absorb_commitment(prefix + "H_star", trace.commitments.at(prefix + "H_star").point);
                transcript.absorb_commitment(
                    external_mask_label(prefix + "H_star"),
                    trace.commitments.at(external_mask_label(prefix + "H_star")).point);
                trace.challenges["y_star_h" + std::to_string(head_index)] = transcript.challenge("y_star_h" + std::to_string(head_index));
                add_metric(metrics != nullptr ? &metrics->hidden_dst_attention_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto xi_powers = powers(xi, head_dim);
                const auto y_star_h = trace.challenges.at("y_star_h" + std::to_string(head_index));
                const auto star_binding = cached_binding_trace(
                    head_cache_prefix + ":star:" + y_star_h.to_string(),
                    [&]()
                    {
                        return build_binding_trace(
                            h_prime,
                            domains.n,
                            n_nodes,
                            y_star_h,
                            xi_powers,
                            domains.d,
                            head_dim);
                    });
                add_metric(metrics != nullptr ? &metrics->hidden_h_star_trace_ms : nullptr, stage_start);

                transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
                transcript.absorb_commitment(prefix + "H_star", trace.commitments.at(prefix + "H_star").point);
                transcript.absorb_commitment(prefix + "H_src_star_edge", trace.commitments.at(prefix + "H_src_star_edge").point);
                const auto eta_src = transcript.challenge("eta_src_h" + std::to_string(head_index));
                trace.challenges["eta_src_h" + std::to_string(head_index)] = eta_src;
                const auto eta_src_powers = powers(eta_src, 3);

                std::vector<FieldElement> src_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> src_query(n_edges, FieldElement::zero());
                parallel_for_ranges(
                    n_nodes,
                    64,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            src_table[i] = absolute_id_fields[i] + eta_src_powers[1] * e_src[i] + eta_src_powers[2] * h_star[i];
                        }
                    });
                parallel_for_ranges(
                    n_edges,
                    128,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t k = begin; k < end; ++k)
                        {
                            src_query[k] = edge_src_fields[k] + eta_src_powers[1] * e_src_edge[k] + eta_src_powers[2] * h_star_edge[k];
                        }
                    });
                auto src_table_padded =
                    padded_column(src_table, domains.n->size);
                auto src_query_padded =
                    padded_column(src_query, domains.edge->size);
                auto src_multiplicity_padded =
                    padded_column(
                        src_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_src",
                            src_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "E_src",
                                    eta_src_powers[1]),
                                linear_label_term(
                                    prefix + "H_star",
                                    eta_src_powers[2]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_src",
                            src_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_src"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "E_src_edge",
                                    eta_src_powers[1]),
                                linear_label_term(
                                    prefix + "H_src_star_edge",
                                    eta_src_powers[2]),
                            }),
                        column_spec(
                            prefix + "m_src",
                            src_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_src", "Query_src", "m_src"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_src = transcript.challenge("beta_src_h" + std::to_string(head_index));
                trace.challenges["beta_src_h" + std::to_string(head_index)] = beta_src;
                const auto src_route = cached_route_trace(
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(src_table_padded),
                            std::move(src_query_padded),
                            std::move(src_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_src);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_src_node", src_route, src_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_src_edge", src_route, src_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_src_node", src_route, src_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_src", src_route, src_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_src_node", "J_src_edge", "R_src_node", "R_src"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                transcript.absorb_commitment(prefix + "S", trace.commitments.at(prefix + "S").point);
                transcript.absorb_commitment(prefix + "Z", trace.commitments.at(prefix + "Z").point);
                transcript.absorb_commitment("V_T_L_x", context.static_commitments.at("V_T_L_x").point);
                transcript.absorb_commitment("V_T_L_y", context.static_commitments.at("V_T_L_y").point);
                const auto eta_l = transcript.challenge("eta_L_h" + std::to_string(head_index));
                trace.challenges["eta_L_h" + std::to_string(head_index)] = eta_l;
                std::vector<FieldElement> table_l =
                    cached_pair_lookup_table(prefix + "L", context.tables.lrelu, eta_l, domains.edge->size);
                const auto lrelu_artifacts = cached_pair_lookup_artifacts(
                    [&]()
                    {
                        return build_pair_lookup_artifacts(
                            *lrelu_index,
                            s,
                            z,
                            eta_l,
                            domains.edge->size,
                            n_edges,
                            prefix + "L",
                            metrics);
                    });
                const auto &query_l = lrelu_artifacts->query;
                const auto &m_l = lrelu_artifacts->multiplicity;
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_L",
                            table_l,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_L_x"),
                                    FieldElement::one()),
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_L_y"),
                                    eta_l),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_L",
                            query_l,
                            domains.edge,
                            {
                                linear_label_term(
                                    prefix + "S",
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "Z",
                                    eta_l),
                            }),
                        column_spec(prefix + "m_L", m_l, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_L", "Query_L", "m_L"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_l = transcript.challenge("beta_L_h" + std::to_string(head_index));
                trace.challenges["beta_L_h" + std::to_string(head_index)] = beta_l;
                auto j_l = profiled_logup_inverse(table_l, query_l, q_tbl_l, q_qry_l, beta_l);
                auto r_l =
                    profiled_logup_accumulator(
                        j_l.table,
                        j_l.query,
                        m_l,
                        q_tbl_l,
                        q_qry_l,
                        lookup_active_transition_limit(context.tables.lrelu.size(), n_edges, domains.edge->size));
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "J_tbl_L", std::move(j_l.table), domains.edge),
                        column_spec(prefix + "J_qry_L", std::move(j_l.query), domains.edge),
                        column_spec(prefix + "R_L", std::move(r_l), domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_tbl_L", "J_qry_L", "R_L"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                transcript.absorb_commitment(prefix + "M", trace.commitments.at(prefix + "M").point);
                transcript.absorb_commitment(prefix + "M_edge", trace.commitments.at(prefix + "M_edge").point);
                transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
                transcript.absorb_commitment("V_T_range", context.static_commitments.at("V_T_range").point);
                std::vector<FieldElement> table_r =
                    cached_single_lookup_table(prefix + "R", context.tables.range, domains.edge->size);
                const auto range_artifacts = cached_single_lookup_artifacts(
                    [&]()
                    {
                        return build_single_lookup_artifacts(
                            *range_index,
                            delta,
                            domains.edge->size,
                            n_edges,
                            prefix + "R",
                            metrics);
                    });
                const auto &query_r = range_artifacts->query;
                const auto &m_r = range_artifacts->multiplicity;
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "Table_R", table_r, domains.edge),
                        column_spec(prefix + "Query_R", query_r, domains.edge),
                        column_spec(prefix + "m_R", m_r, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_R", "Query_R", "m_R"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_r = transcript.challenge("beta_R_h" + std::to_string(head_index));
                trace.challenges["beta_R_h" + std::to_string(head_index)] = beta_r;
                auto j_r = profiled_logup_inverse(table_r, query_r, q_tbl_r, q_qry_r, beta_r);
                auto r_r =
                    profiled_logup_accumulator(
                        j_r.table,
                        j_r.query,
                        m_r,
                        q_tbl_r,
                        q_qry_r,
                        lookup_active_transition_limit(context.tables.range.size(), n_edges, domains.edge->size));
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "J_tbl_R", std::move(j_r.table), domains.edge),
                        column_spec(prefix + "J_qry_R", std::move(j_r.query), domains.edge),
                        column_spec(prefix + "R_R", std::move(r_r), domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_tbl_R", "J_qry_R", "R_R"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
                transcript.absorb_commitment(prefix + "U", trace.commitments.at(prefix + "U").point);
                transcript.absorb_commitment("V_T_exp_x", context.static_commitments.at("V_T_exp_x").point);
                transcript.absorb_commitment("V_T_exp_y", context.static_commitments.at("V_T_exp_y").point);
                const auto eta_exp = transcript.challenge("eta_exp_h" + std::to_string(head_index));
                trace.challenges["eta_exp_h" + std::to_string(head_index)] = eta_exp;
                std::vector<FieldElement> table_exp =
                    cached_pair_lookup_table(prefix + "exp", context.tables.exp, eta_exp, domains.edge->size);
                const auto exp_artifacts = cached_pair_lookup_artifacts(
                    head_cache_prefix + ":exp_artifacts:" + eta_exp.to_string(),
                    [&]()
                    {
                        return build_pair_lookup_artifacts(
                            *exp_index, delta, u, eta_exp, domains.edge->size, n_edges, prefix + "exp", metrics);
                    });
                const auto &query_exp = exp_artifacts->query;
                const auto &m_exp = exp_artifacts->multiplicity;
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_exp",
                            table_exp,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_exp_x"),
                                    FieldElement::one()),
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_exp_y"),
                                    eta_exp),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_exp",
                            query_exp,
                            domains.edge,
                            {
                                linear_label_term(
                                    prefix + "Delta",
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "U",
                                    eta_exp),
                            }),
                        column_spec(prefix + "m_exp", m_exp, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_exp", "Query_exp", "m_exp"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_exp = transcript.challenge("beta_exp_h" + std::to_string(head_index));
                trace.challenges["beta_exp_h" + std::to_string(head_index)] = beta_exp;
                auto j_exp = profiled_logup_inverse(table_exp, query_exp, q_tbl_exp, q_qry_exp, beta_exp);
                auto r_exp =
                    profiled_logup_accumulator(
                        j_exp.table,
                        j_exp.query,
                        m_exp,
                        q_tbl_exp,
                        q_qry_exp,
                        lookup_active_transition_limit(context.tables.exp.size(), n_edges, domains.edge->size));
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "J_tbl_exp", std::move(j_exp.table), domains.edge),
                        column_spec(prefix + "J_qry_exp", std::move(j_exp.query), domains.edge),
                        column_spec(prefix + "R_exp", std::move(r_exp), domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_tbl_exp", "J_qry_exp", "R_exp"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                stage_start = Clock::now();
                auto s_max = build_first_zero_group_selector(delta, edge_groups_by_dst, domains.edge->size);
                add_metric(metrics != nullptr ? &metrics->hidden_edge_score_trace_ms : nullptr, stage_start);
                stage_start = Clock::now();
                const auto c_max = cached_state_vector(
                    [&]()
                    {
                        return build_max_counter_state(
                            s_max,
                            q_new,
                            enable_single_graph_selector_shortcut);
                    });
                add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);
                // 从这里开始不再只是普通前向，而是在把 softmax 聚合链改写成 PSQ / state-machine 轨迹。
                const auto lambda_psq = transcript.challenge("lambda_psq_h" + std::to_string(head_index));
                trace.challenges["lambda_psq_h" + std::to_string(head_index)] = lambda_psq;
                stage_start = Clock::now();
                std::vector<FieldElement> widehat_v_pre_star(domains.edge->size, FieldElement::zero());
                for (std::size_t k = 0; k < n_edges; ++k)
                {
                    widehat_v_pre_star[k] = alpha[k] * h_star_edge[k];
                }
                auto w_psq = build_weighted_sum(u, lambda_psq, widehat_v_pre_star, domains.edge->size);
                auto t_psq = std::make_shared<std::vector<FieldElement>>(n_nodes, FieldElement::zero());
                for (std::size_t i = 0; i < n_nodes; ++i)
                {
                    (*t_psq)[i] = sum[i] + lambda_psq * h_agg_pre_star[i];
                }
                const auto t_psq_edge = std::make_shared<std::vector<FieldElement>>(
                    build_group_target(*t_psq, edges, domains.edge->size));
                add_metric(metrics != nullptr ? &metrics->hidden_softmax_chain_trace_ms : nullptr, stage_start);
                stage_start = Clock::now();
                const auto psq = cached_state_vector(
                    head_cache_prefix + ":psq:" + lambda_psq.to_string(),
                    [&]()
                    {
                        return build_group_prefix_state(w_psq, q_new, enable_single_graph_selector_shortcut);
                    });
                add_metric(metrics != nullptr ? &metrics->psq_trace_ms : nullptr, stage_start);
                add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);

                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "a_proj", proj_binding, proj_binding->a, domains.in),
                        shared_column_spec(prefix + "b_proj", proj_binding, proj_binding->b, domains.in),
                        shared_column_spec(prefix + "Acc_proj", proj_binding, proj_binding->acc, domains.in),
                        shared_column_spec(prefix + "a_src", src_binding, src_binding->a, domains.d),
                        shared_column_spec(prefix + "b_src", src_binding, src_binding->b, domains.d),
                        shared_column_spec(prefix + "Acc_src", src_binding, src_binding->acc, domains.d),
                        shared_column_spec(prefix + "a_dst", dst_binding, dst_binding->a, domains.d),
                        shared_column_spec(prefix + "b_dst", dst_binding, dst_binding->b, domains.d),
                        shared_column_spec(prefix + "Acc_dst", dst_binding, dst_binding->acc, domains.d),
                        shared_column_spec(prefix + "a_star", star_binding, star_binding->a, domains.d),
                        shared_column_spec(prefix + "b_star", star_binding, star_binding->b, domains.d),
                        shared_column_spec(prefix + "Acc_star", star_binding, star_binding->acc, domains.d),
                        column_spec(prefix + "s_max", std::move(s_max), domains.edge),
                        column_spec(prefix + "C_max", c_max, domains.edge),
                        shared_compact_column_spec(prefix + "H_agg_pre_star", compressed, h_agg_pre_star, domains.n),
                        shared_compact_column_spec(prefix + "H_agg_pre_star_edge", compressed, h_agg_pre_star_edge, domains.edge),
                        column_spec(prefix + "widehat_v_pre_star", std::move(widehat_v_pre_star), domains.edge),
                        deferred_linear_column_spec(
                            prefix + "w_psq",
                            std::move(w_psq),
                            domains.edge,
                            {
                                linear_label_term(prefix + "U", FieldElement::one()),
                                linear_label_term(prefix + "widehat_v_pre_star", lambda_psq),
                            }),
                        deferred_linear_compact_column_spec(
                            prefix + "T_psq",
                            t_psq,
                            domains.n,
                            {
                                linear_label_term(prefix + "Sum", FieldElement::one()),
                                linear_label_term(prefix + "H_agg_pre_star", lambda_psq),
                            }),
                        deferred_linear_compact_column_spec(
                            prefix + "T_psq_edge",
                            t_psq_edge,
                            domains.edge,
                            {
                                linear_label_term(prefix + "Sum_edge", FieldElement::one()),
                                linear_label_term(prefix + "H_agg_pre_star_edge", lambda_psq),
                            }),
                        column_spec(prefix + "PSQ", psq, domains.edge),
                        shared_compact_column_spec(prefix + "H_agg_star", compressed, h_agg_star, domains.n),
                        shared_compact_column_spec(prefix + "H_agg_star_edge", compressed, h_agg_star_edge, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                append_external_mask(
                    prefix + "H_agg_pre_star",
                    external_mask_label(prefix + "H_agg_pre_star"));
                append_external_mask(
                    prefix + "H_agg_star",
                    external_mask_label(prefix + "H_agg_star"));

                transcript.absorb_commitment(prefix + "T_psq", trace.commitments.at(prefix + "T_psq").point);
                transcript.absorb_commitment(prefix + "T_psq_edge", trace.commitments.at(prefix + "T_psq_edge").point);
                const auto eta_t = transcript.challenge("eta_t_h" + std::to_string(head_index));
                trace.challenges["eta_t_h" + std::to_string(head_index)] = eta_t;
                const auto eta_t_powers = powers(eta_t, 2);
                std::vector<FieldElement> t_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> t_query(n_edges, FieldElement::zero());
                for (std::size_t i = 0; i < n_nodes; ++i)
                {
                    t_table[i] = absolute_id_fields[i] + eta_t_powers[1] * (*t_psq)[i];
                }
                for (std::size_t k = 0; k < n_edges; ++k)
                {
                    t_query[k] = edge_dst_fields[k] + eta_t_powers[1] * (*t_psq_edge)[k];
                }
                auto t_table_padded =
                    padded_column(t_table, domains.n->size);
                auto t_query_padded =
                    padded_column(t_query, domains.edge->size);
                auto t_multiplicity_padded =
                    padded_column(
                        dst_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_t",
                            t_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "T_psq",
                                    eta_t_powers[1]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_t",
                            t_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_dst"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "T_psq_edge",
                                    eta_t_powers[1]),
                            }),
                        column_spec(
                            prefix + "m_t",
                            t_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_t", "Query_t", "m_t"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_t = transcript.challenge("beta_t_h" + std::to_string(head_index));
                trace.challenges["beta_t_h" + std::to_string(head_index)] = beta_t;
                const auto t_route = cached_route_trace(
                    head_cache_prefix + ":t_route:" + eta_t.to_string() + ":" + beta_t.to_string(),
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(t_table_padded),
                            std::move(t_query_padded),
                            std::move(t_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_t);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_t_node", t_route, t_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_t_edge", t_route, t_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_t_node", t_route, t_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_t", t_route, t_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_t_node", "J_t_edge", "R_t_node", "R_t"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                std::vector<FieldElement> sum_flat(domains.edge->size, FieldElement::zero());
                std::vector<FieldElement> agg_pre_flat(domains.edge->size, FieldElement::zero());
                std::vector<FieldElement> agg_num_flat(domains.edge->size, FieldElement::zero());
                std::vector<FieldElement> elu_in_flat(domains.edge->size, FieldElement::zero());
                std::vector<FieldElement> agg_flat(domains.edge->size, FieldElement::zero());
                for (std::size_t node = 0; node < n_nodes; ++node)
                {
                    for (std::size_t col = 0; col < head_dim; ++col)
                    {
                        const auto flat_index = node * head_dim + col;
                        sum_flat[flat_index] = sum[node];
                        agg_pre_flat[flat_index] = h_agg_pre[node][col];
                        agg_num_flat[flat_index] = sum[node] * h_agg_pre[node][col];
                        elu_in_flat[flat_index] = h_agg_pre_lookup[node][col];
                        agg_flat[flat_index] = h_agg[node][col];
                    }
                }
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "Sum_flat", sum_flat, domains.edge),
                        column_spec(prefix + "H_agg_pre_flat", agg_pre_flat, domains.edge),
                        column_spec(prefix + "H_agg_num_flat", agg_num_flat, domains.edge),
                        column_spec(prefix + "H_ELU_in_flat", elu_in_flat, domains.edge),
                        column_spec(prefix + "H_agg_flat", agg_flat, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);

                transcript.absorb_commitment(prefix + "H_agg_pre", trace.commitments.at(prefix + "H_agg_pre").point);
                transcript.absorb_commitment(prefix + "H_agg_pre_star", trace.commitments.at(prefix + "H_agg_pre_star").point);
                transcript.absorb_commitment(
                    external_mask_label(prefix + "H_agg_pre_star"),
                    trace.commitments.at(external_mask_label(prefix + "H_agg_pre_star")).point);
                trace.challenges["y_agg_pre_h" + std::to_string(head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(head_index));
                transcript.absorb_commitment(prefix + "H_agg_pre", trace.commitments.at(prefix + "H_agg_pre").point);
                transcript.absorb_commitment(prefix + "H_agg", trace.commitments.at(prefix + "H_agg").point);
                transcript.absorb_commitment("V_T_ELU_x", context.static_commitments.at("V_T_ELU_x").point);
                transcript.absorb_commitment("V_T_ELU_y", context.static_commitments.at("V_T_ELU_y").point);
                transcript.absorb_commitment(prefix + "H_ELU_in_flat", trace.commitments.at(prefix + "H_ELU_in_flat").point);
                transcript.absorb_commitment(prefix + "H_agg_flat", trace.commitments.at(prefix + "H_agg_flat").point);
                const auto eta_elu = transcript.challenge("eta_ELU_h" + std::to_string(head_index));
                trace.challenges["eta_ELU_h" + std::to_string(head_index)] = eta_elu;
                std::vector<FieldElement> table_elu =
                    cached_pair_lookup_table(prefix + "ELU", context.tables.elu, eta_elu, domains.edge->size);
                const auto elu_artifacts = cached_pair_lookup_artifacts(
                    head_cache_prefix + ":elu_artifacts:" + eta_elu.to_string(),
                    [&]()
                    {
                        return build_pair_lookup_artifacts(
                            *elu_index,
                            elu_in_flat,
                            agg_flat,
                            eta_elu,
                            domains.edge->size,
                            n_nodes * head_dim,
                            prefix + "ELU",
                            metrics);
                    });
                const auto &query_elu = elu_artifacts->query;
                const auto &m_elu = elu_artifacts->multiplicity;
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_ELU",
                            table_elu,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_ELU_x"),
                                    FieldElement::one()),
                                linear_commitment_term(
                                    context.public_commitments.at(
                                        "P_T_ELU_y"),
                                    eta_elu),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_ELU",
                            query_elu,
                            domains.edge,
                            {
                                linear_label_term(
                                    prefix + "H_ELU_in_flat",
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "H_agg_flat",
                                    eta_elu),
                            }),
                        column_spec(prefix + "m_ELU", m_elu, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_ELU", "Query_ELU", "m_ELU"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_elu = transcript.challenge("beta_ELU_h" + std::to_string(head_index));
                trace.challenges["beta_ELU_h" + std::to_string(head_index)] = beta_elu;
                auto j_elu = profiled_logup_inverse(table_elu, query_elu, q_tbl_elu, q_qry_elu, beta_elu);
                auto r_elu =
                    profiled_logup_accumulator(
                        j_elu.table,
                        j_elu.query,
                        m_elu,
                        q_tbl_elu,
                        q_qry_elu,
                        lookup_active_transition_limit(context.tables.elu.size(), n_nodes * head_dim, domains.edge->size));
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "J_tbl_ELU", std::move(j_elu.table), domains.edge),
                        column_spec(prefix + "J_qry_ELU", std::move(j_elu.query), domains.edge),
                        column_spec(prefix + "R_ELU", std::move(r_elu), domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_tbl_ELU", "J_qry_ELU", "R_ELU"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                transcript.absorb_commitment(prefix + "H_agg", trace.commitments.at(prefix + "H_agg").point);
                transcript.absorb_commitment(prefix + "H_agg_star", trace.commitments.at(prefix + "H_agg_star").point);
                transcript.absorb_commitment(
                    external_mask_label(prefix + "H_agg_star"),
                    trace.commitments.at(external_mask_label(prefix + "H_agg_star")).point);
                trace.challenges["y_agg_h" + std::to_string(head_index)] = transcript.challenge("y_agg_h" + std::to_string(head_index));
                for (const auto *suffix : {"E_dst", "M", "Sum", "inv", "H_agg_star", "E_dst_edge", "M_edge", "Sum_edge", "inv_edge", "H_agg_star_edge"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto eta_dst = transcript.challenge("eta_dst_h" + std::to_string(head_index));
                trace.challenges["eta_dst_h" + std::to_string(head_index)] = eta_dst;
                const auto eta_dst_powers = powers(eta_dst, 6);
                std::vector<FieldElement> dst_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> dst_query(n_edges, FieldElement::zero());
                for (std::size_t i = 0; i < n_nodes; ++i)
                {
                    dst_table[i] =
                        absolute_id_fields[i] + eta_dst_powers[1] * e_dst[i] + eta_dst_powers[2] * m[i] + eta_dst_powers[3] * sum[i] + eta_dst_powers[4] * inv[i] + eta_dst_powers[5] * h_agg_star[i];
                }
                for (std::size_t k = 0; k < n_edges; ++k)
                {
                    dst_query[k] =
                        edge_dst_fields[k] + eta_dst_powers[1] * e_dst_edge[k] + eta_dst_powers[2] * m_edge[k] + eta_dst_powers[3] * sum_edge[k] + eta_dst_powers[4] * inv_edge[k] + eta_dst_powers[5] * h_agg_star_edge[k];
                }
                auto dst_table_padded =
                    padded_column(dst_table, domains.n->size);
                auto dst_query_padded =
                    padded_column(dst_query, domains.edge->size);
                auto dst_multiplicity_padded =
                    padded_column(
                        dst_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_dst",
                            dst_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(prefix + "E_dst", eta_dst_powers[1]),
                                linear_label_term(prefix + "M", eta_dst_powers[2]),
                                linear_label_term(prefix + "Sum", eta_dst_powers[3]),
                                linear_label_term(prefix + "inv", eta_dst_powers[4]),
                                linear_label_term(prefix + "H_agg_star", eta_dst_powers[5]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_dst",
                            dst_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_dst"),
                                    FieldElement::one()),
                                linear_label_term(prefix + "E_dst_edge", eta_dst_powers[1]),
                                linear_label_term(prefix + "M_edge", eta_dst_powers[2]),
                                linear_label_term(prefix + "Sum_edge", eta_dst_powers[3]),
                                linear_label_term(prefix + "inv_edge", eta_dst_powers[4]),
                                linear_label_term(prefix + "H_agg_star_edge", eta_dst_powers[5]),
                            }),
                        column_spec(
                            prefix + "m_dst",
                            dst_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_dst", "Query_dst", "m_dst"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_dst = transcript.challenge("beta_dst_h" + std::to_string(head_index));
                trace.challenges["beta_dst_h" + std::to_string(head_index)] = beta_dst;
                const auto dst_route = cached_route_trace(
                    head_cache_prefix + ":dst_route:" + eta_dst.to_string() + ":" + beta_dst.to_string(),
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(dst_table_padded),
                            std::move(dst_query_padded),
                            std::move(dst_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_dst);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_dst_node", dst_route, dst_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_dst_edge", dst_route, dst_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_dst_node", dst_route, dst_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_dst", dst_route, dst_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_dst_node", "J_dst_edge", "R_dst_node", "R_dst"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                stage_start = Clock::now();
                const auto y_agg_pre_h = trace.challenges.at("y_agg_pre_h" + std::to_string(head_index));
                const auto agg_pre_binding = cached_binding_trace(
                    head_cache_prefix + ":agg_pre:" + y_agg_pre_h.to_string(),
                    [&]()
                    {
                        return build_binding_trace(
                            h_agg_pre,
                            domains.n,
                            n_nodes,
                            y_agg_pre_h,
                            xi_powers,
                            domains.d,
                            head_dim);
                    });
                add_metric(metrics != nullptr ? &metrics->hidden_h_agg_pre_star_trace_ms : nullptr, stage_start);

                stage_start = Clock::now();
                const auto y_agg_h = trace.challenges.at("y_agg_h" + std::to_string(head_index));
                const auto agg_binding = cached_binding_trace(
                    head_cache_prefix + ":agg:" + y_agg_h.to_string(),
                    [&]()
                    {
                        return build_binding_trace(
                            h_agg,
                            domains.n,
                            n_nodes,
                            y_agg_h,
                            xi_powers,
                            domains.d,
                            head_dim);
                    });
                add_metric(metrics != nullptr ? &metrics->hidden_h_agg_star_trace_ms : nullptr, stage_start);

                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "a_agg_pre", agg_pre_binding, agg_pre_binding->a, domains.d),
                        shared_column_spec(prefix + "b_agg_pre", agg_pre_binding, agg_pre_binding->b, domains.d),
                        shared_column_spec(prefix + "Acc_agg_pre", agg_pre_binding, agg_pre_binding->acc, domains.d),
                        shared_column_spec(prefix + "a_agg", agg_binding, agg_binding->a, domains.d),
                        shared_column_spec(prefix + "b_agg", agg_binding, agg_binding->b, domains.d),
                        shared_column_spec(prefix + "Acc_agg", agg_binding, agg_binding->acc, domains.d),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                const auto &div = quantized.aggregate_div;
                commit_signed_division(
                    prefix + "Div_",
                    div,
                    "P_Q_div_h" + std::to_string(head_index));
                std::vector<std::pair<std::string, FieldElement>> hidden_weight_terms;
                hidden_weight_terms.reserve(head_dim);
                auto hidden_weight_power = FieldElement::one();
                for (std::size_t column = 0; column < head_dim; ++column)
                {
                    hidden_weight_terms.push_back({hidden_weight_column_label(head_index, column),
                                                   hidden_weight_power});
                    hidden_weight_power *= trace.challenges.at(
                        "y_proj_h" + std::to_string(head_index));
                }
                append_model_binding_blinder(
                    prefix + "b_proj",
                    hidden_model_binding_blinder_label(head_index, "proj"),
                    domains.in,
                    hidden_weight_terms);
                append_model_binding_blinder(
                    prefix + "b_src",
                    hidden_model_binding_blinder_label(head_index, "src"),
                    domains.d,
                    {{hidden_src_label(head_index), FieldElement::one()}});
                append_model_binding_blinder(
                    prefix + "b_dst",
                    hidden_model_binding_blinder_label(head_index, "dst"),
                    domains.d,
                    {{hidden_dst_label(head_index), FieldElement::one()}});
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_proj"] =
                    matrix_row_major_evaluation(h_prime, trace.challenges.at("y_proj_h" + std::to_string(head_index)));
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_src"] =
                    evaluate_trace_polynomial(
                        trace, prefix + "E_src",
                        trace.challenges.at("y_src_h" + std::to_string(head_index)),
                        domains.edge);
                trace.external_evaluations[hidden_score_mask_eval_name(head_index, "src")] =
                    evaluate_trace_polynomial(
                        trace, hidden_score_mask_label(head_index, "src"),
                        trace.challenges.at("y_src_h" + std::to_string(head_index)),
                        domains.edge);
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_dst"] =
                    evaluate_trace_polynomial(
                        trace, prefix + "E_dst",
                        trace.challenges.at("y_dst_h" + std::to_string(head_index)),
                        domains.edge);
                trace.external_evaluations[hidden_score_mask_eval_name(head_index, "dst")] =
                    evaluate_trace_polynomial(
                        trace, hidden_score_mask_label(head_index, "dst"),
                        trace.challenges.at("y_dst_h" + std::to_string(head_index)),
                        domains.edge);
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_star"] =
                    evaluate_trace_polynomial(
                        trace, prefix + "H_star",
                        trace.challenges.at("y_star_h" + std::to_string(head_index)),
                        domains.n);
                trace.external_evaluations[external_mask_eval_name(
                    "mu_h" + std::to_string(head_index) + "_star")] =
                    evaluate_trace_polynomial(
                        trace, external_mask_label(prefix + "H_star"),
                        trace.challenges.at("y_star_h" + std::to_string(head_index)),
                        domains.n);
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_agg_pre"] =
                    evaluate_trace_polynomial(
                        trace, prefix + "H_agg_pre_star",
                        trace.challenges.at("y_agg_pre_h" + std::to_string(head_index)),
                        domains.n);
                trace.external_evaluations[external_mask_eval_name(
                    "mu_h" + std::to_string(head_index) + "_agg_pre")] =
                    evaluate_trace_polynomial(
                        trace, external_mask_label(prefix + "H_agg_pre_star"),
                        trace.challenges.at("y_agg_pre_h" + std::to_string(head_index)),
                        domains.n);
                trace.external_evaluations["mu_h" + std::to_string(head_index) + "_agg"] =
                    evaluate_trace_polynomial(
                        trace, prefix + "H_agg_star",
                        trace.challenges.at("y_agg_h" + std::to_string(head_index)),
                        domains.n);
                trace.external_evaluations[external_mask_eval_name(
                    "mu_h" + std::to_string(head_index) + "_agg")] =
                    evaluate_trace_polynomial(
                        trace, external_mask_label(prefix + "H_agg_star"),
                        trace.challenges.at("y_agg_h" + std::to_string(head_index)),
                        domains.n);
            };

            auto commit_hidden_layer_concat = [&](std::size_t layer_index)
            {
                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                const auto concat_label = hidden_layer_concat_label(layer_index, is_final_layer);
                const auto concat_star_label = hidden_layer_concat_star_label(layer_index, is_final_layer);
                const auto cat_prefix = hidden_layer_cat_prefix(layer_index, is_final_layer);
                const auto xi_name = hidden_concat_xi_name(layer_index, is_final_layer);
                const auto y_name = hidden_concat_y_name(layer_index, is_final_layer);
                const auto concat_width =
                    context.model.hidden_layers[layer_index].shape.head_count * context.model.hidden_layers[layer_index].shape.head_dim;
                const auto &h_cat_layer = hidden_layer_concat_quantized.at(layer_index);
                // 隐藏层这里走的是多头拼接，把每个头的结果收成下一层唯一输入表示。
                add_dynamic_commitment_batch(
                    trace,
                    {matrix_commitment_only_spec(concat_label, h_cat_layer)},
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                transcript.absorb_commitment(concat_label, trace.commitments.at(concat_label).point);
                trace.challenges[xi_name] = transcript.challenge(xi_name);
                const auto object_stage_start = Clock::now();
                auto h_cat_star = compress_rows_with_challenge(h_cat_layer, trace.challenges.at(xi_name));
                add_metric(metrics != nullptr ? &metrics->hidden_output_object_residual_ms : nullptr, object_stage_start);
                add_dynamic_commitment_batch(
                    trace,
                    {compact_column_spec(concat_star_label, std::move(h_cat_star), domains.n)},
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                append_external_mask(
                    concat_star_label, external_mask_label(concat_star_label));
                transcript.absorb_commitment(concat_star_label, trace.commitments.at(concat_star_label).point);
                transcript.absorb_commitment(
                    external_mask_label(concat_star_label),
                    trace.commitments.at(external_mask_label(concat_star_label)).point);
                trace.challenges[y_name] = transcript.challenge(y_name);
                stage_start = Clock::now();
                const auto cat_binding = cached_binding_trace(
                    cache_prefix + ":cat:" + std::to_string(layer_index) + ":" + trace.challenges.at(y_name).to_string(),
                    [&]()
                    {
                        return build_binding_trace(
                            h_cat_layer,
                            domains.n,
                            n_nodes,
                            trace.challenges.at(y_name),
                            powers(trace.challenges.at(xi_name), concat_width),
                            domains.cat,
                            concat_width);
                    });
                add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(cat_prefix + "_a", cat_binding, cat_binding->a, domains.cat),
                        shared_column_spec(cat_prefix + "_b", cat_binding, cat_binding->b, domains.cat),
                        shared_column_spec(cat_prefix + "_Acc", cat_binding, cat_binding->acc, domains.cat),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                const auto concat_eval_name =
                    is_final_layer ? std::string("mu_cat")
                                   : "mu_cat_l" + std::to_string(layer_index);
                trace.external_evaluations[concat_eval_name] =
                    evaluate_trace_polynomial(
                        trace, concat_star_label,
                        trace.challenges.at(y_name), domains.n);
                trace.external_evaluations[external_mask_eval_name(concat_eval_name)] =
                    evaluate_trace_polynomial(
                        trace, external_mask_label(concat_star_label),
                        trace.challenges.at(y_name), domains.n);
            };

            stage_start = Clock::now();
            const auto hidden_stack_start = stage_start;
            const auto hidden_stack_initial_commit_dynamic_ms =
                metrics != nullptr ? metrics->commit_dynamic_ms : 0.0;
            std::size_t global_hidden_head_index = 0;
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                for (std::size_t local_head_index = 0;
                     local_head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++local_head_index, ++global_hidden_head_index)
                {
                    commit_hidden_head(global_hidden_head_index);
                }
                commit_hidden_layer_concat(layer_index);
            }
            const auto &h_cat = hidden_layer_concat_quantized.back();
            if (metrics != nullptr)
            {
                metrics->hidden_head_trace_ms += std::max(
                    0.0,
                    elapsed_ms(hidden_stack_start, Clock::now()) - (metrics->commit_dynamic_ms - hidden_stack_initial_commit_dynamic_ms));
            }

            stage_start = Clock::now();
            const auto output_trace_start = stage_start;
            const auto output_initial_commit_dynamic_ms =
                metrics != nullptr ? metrics->commit_dynamic_ms : 0.0;
            // 输出层保持“每头先各算各的”，最后再单独物化平均关系和 bias 后关系。

            std::vector<
                DynamicCommitmentSpec>
                output_y_prime_specs;

            std::vector<std::string>
                output_y_prime_labels;

            output_y_prime_specs.reserve(
                context.model
                    .output_layer
                    .heads.size());

            output_y_prime_labels.reserve(
                context.model
                    .output_layer
                    .heads.size());

            for (std::size_t head_index = 0;
                 head_index < context.model
                                  .output_layer
                                  .heads.size();
                 ++head_index)
            {
                const auto label =
                    output_head_prefix(
                        head_index) +
                    "_Y_prime";

                output_y_prime_labels.push_back(
                    label);

                output_y_prime_specs.push_back(
                    matrix_commitment_only_spec(
                        label,
                        forward_artifacts
                            ->output_head_quantized
                            .at(head_index)
                            .y_prime));
            }

            add_dynamic_commitment_batch(
                trace,
                std::move(
                    output_y_prime_specs),
                context.kzg,
                keep_trace_payloads,
                metrics);

            // Transcript absorption remains in exactly the old
            // canonical head order.
            for (const auto &label :
                 output_y_prime_labels)
            {
                transcript.absorb_commitment(
                    label,
                    trace.commitments
                        .at(label)
                        .point);
            }

            trace.challenges["xi_out"] =
                transcript.challenge(
                    "xi_out");

            auto commit_output_head = [&](std::size_t head_index)
            {
                const auto prefix_base = output_head_prefix(head_index);
                const auto prefix = prefix_base + "_";
                const auto y_lin_label = output_y_lin_label(head_index);
                const auto y_label = output_y_label(head_index);
                const auto &quantized = forward_artifacts->output_head_quantized.at(head_index);
                const auto &y_prime = quantized.y_prime;
                const auto &out_e_src = quantized.e_src;
                const auto &out_e_dst = quantized.e_dst;
                const auto &out_s = quantized.s;
                const auto &out_z = quantized.z;
                const auto &out_m = quantized.m;
                const auto &out_delta = quantized.delta;
                const auto &out_u = quantized.u;
                const auto &out_sum = quantized.sum;
                const auto &out_inv = quantized.inv;
                const auto &out_alpha = quantized.alpha;
                const auto &y_lin_matrix =
                    quantized.y_lin;

                const auto &y_matrix =
                    quantized.y_quant;
                const auto &out_e_src_edge = quantized.e_src_edge;
                const auto &out_m_edge = quantized.m_edge;
                const auto &out_sum_edge = quantized.sum_edge;
                const auto &out_inv_edge = quantized.inv_edge;
                auto &binding_subproof = require_binding_subproof(
                    "bind:output:" + std::to_string(head_index));
                std::vector<std::string> model_reblind_labels;
                model_reblind_labels.reserve(n_classes + 3);
                for (std::size_t column = 0; column < n_classes; ++column)
                {
                    model_reblind_labels.push_back(
                        output_weight_column_label(head_index, column));
                }
                append_model_reblinds(
                    binding_subproof, model_reblind_labels, domains.cat);
                append_model_reblinds(
                    binding_subproof,
                    {
                        output_src_label(head_index),
                        output_dst_label(head_index),
                        output_bias_label(head_index),
                    },
                    domains.c);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_compact_column_spec(prefix + "E_src", forward_artifacts, out_e_src, domains.n),
                        shared_compact_column_spec(prefix + "E_dst", forward_artifacts, out_e_dst, domains.n),
                        shared_compact_column_spec(prefix + "E_src_edge", forward_artifacts, quantized.e_src_edge, domains.edge),
                        shared_compact_column_spec(prefix + "E_dst_edge", forward_artifacts, quantized.e_dst_edge, domains.edge),
                        deferred_linear_compact_shared_column_spec(
                            prefix + "S",
                            forward_artifacts,
                            out_s,
                            domains.edge,
                            {
                                linear_label_term(prefix + "E_src_edge", FieldElement::one()),
                                linear_label_term(prefix + "E_dst_edge", FieldElement::one()),
                            }),
                        shared_compact_column_spec(prefix + "Z", forward_artifacts, out_z, domains.edge),
                        shared_compact_column_spec(prefix + "M", forward_artifacts, out_m, domains.n),
                        shared_compact_column_spec(prefix + "M_edge", forward_artifacts, out_m_edge, domains.edge),
                        deferred_linear_compact_shared_column_spec(
                            prefix + "Delta",
                            forward_artifacts,
                            out_delta,
                            domains.edge,
                            {
                                linear_label_term(prefix + "M_edge", FieldElement::one()),
                                linear_label_term(prefix + "Z", -FieldElement::one()),
                            }),
                        shared_compact_column_spec(prefix + "U", forward_artifacts, out_u, domains.edge),
                        shared_compact_column_spec(prefix + "Sum", forward_artifacts, out_sum, domains.n),
                        shared_compact_column_spec(prefix + "Sum_edge", forward_artifacts, out_sum_edge, domains.edge),
                        shared_compact_column_spec(prefix + "inv", forward_artifacts, out_inv, domains.n),
                        shared_compact_column_spec(prefix + "inv_edge", forward_artifacts, out_inv_edge, domains.edge),
                        shared_compact_column_spec(prefix + "alpha", forward_artifacts, out_alpha, domains.edge),
                        matrix_commitment_only_spec(y_lin_label, y_lin_matrix),
                        matrix_commitment_only_spec(y_label, y_matrix),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);

                append_external_mask(
                    prefix + "E_src", output_score_mask_label(head_index, "src"));
                append_external_mask(
                    prefix + "E_dst", output_score_mask_label(head_index, "dst"));

                transcript.absorb_commitment("P_H_cat", trace.commitments.at("P_H_cat").point);
                transcript.absorb_commitment(prefix + "Y_prime", trace.commitments.at(prefix + "Y_prime").point);
                for (std::size_t column = 0; column < n_classes; ++column)
                {
                    const auto label = output_weight_column_label(head_index, column);
                    transcript.absorb_commitment(
                        model_proof_local_label(label),
                        trace.commitments.at(model_proof_local_label(label)).point);
                    transcript.absorb_commitment(
                        model_reblind_witness_label(label),
                        trace.commitments.at(model_reblind_witness_label(label)).point);
                }
                trace.challenges[output_challenge_name("y_proj_out", head_index)] =
                    transcript.challenge(output_challenge_name("y_proj_out", head_index));
                transcript.absorb_commitment(prefix + "Y_prime", trace.commitments.at(prefix + "Y_prime").point);
                transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
                transcript.absorb_commitment(
                    output_score_mask_label(head_index, "src"),
                    trace.commitments.at(output_score_mask_label(head_index, "src")).point);
                transcript.absorb_commitment(
                    model_proof_local_label(output_src_label(head_index)),
                    trace.commitments.at(model_proof_local_label(output_src_label(head_index))).point);
                transcript.absorb_commitment(
                    model_reblind_witness_label(output_src_label(head_index)),
                    trace.commitments.at(model_reblind_witness_label(output_src_label(head_index))).point);
                trace.challenges[output_challenge_name("y_src_out", head_index)] =
                    transcript.challenge(output_challenge_name("y_src_out", head_index));
                transcript.absorb_commitment(prefix + "Y_prime", trace.commitments.at(prefix + "Y_prime").point);
                transcript.absorb_commitment(prefix + "E_dst", trace.commitments.at(prefix + "E_dst").point);
                transcript.absorb_commitment(
                    output_score_mask_label(head_index, "dst"),
                    trace.commitments.at(output_score_mask_label(head_index, "dst")).point);
                transcript.absorb_commitment(
                    model_proof_local_label(output_dst_label(head_index)),
                    trace.commitments.at(model_proof_local_label(output_dst_label(head_index))).point);
                transcript.absorb_commitment(
                    model_reblind_witness_label(output_dst_label(head_index)),
                    trace.commitments.at(model_reblind_witness_label(output_dst_label(head_index))).point);
                trace.challenges[output_challenge_name("y_dst_out", head_index)] =
                    transcript.challenge(output_challenge_name("y_dst_out", head_index));
                const auto xi_out = trace.challenges.at("xi_out");
                const auto compressed = cached_output_compressed_trace(
                    y_prime,
                    y_lin_matrix,
                    edges,
                    domains.edge->size,
                    xi_out);
                const auto &y_prime_star = compressed->y_prime_star;
                const auto &y_prime_star_edge = compressed->y_prime_star_edge;
                const auto &y_star = compressed->y_star;
                const auto &y_star_edge = compressed->y_star_edge;
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_compact_column_spec(prefix + "Y_prime_star", compressed, y_prime_star, domains.n),
                        shared_compact_column_spec(prefix + "Y_prime_star_edge", compressed, y_prime_star_edge, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                transcript.absorb_commitment(prefix + "Y_prime_star", trace.commitments.at(prefix + "Y_prime_star").point);
                transcript.absorb_commitment(prefix + "Y_prime_star_edge", trace.commitments.at(prefix + "Y_prime_star_edge").point);
                const auto eta_src_out = transcript.challenge(output_challenge_name("eta_src_out", head_index));
                trace.challenges[output_challenge_name("eta_src_out", head_index)] = eta_src_out;
                const auto eta_src_out_powers = powers(eta_src_out, 2);
                std::vector<FieldElement> out_src_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> out_src_query(n_edges, FieldElement::zero());
                parallel_for_ranges(
                    n_nodes,
                    64,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            out_src_table[i] = absolute_id_fields[i] + eta_src_out_powers[1] * out_e_src[i];
                        }
                    });
                parallel_for_ranges(
                    n_edges,
                    128,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t k = begin; k < end; ++k)
                        {
                            out_src_query[k] = edge_src_fields[k] + eta_src_out_powers[1] * out_e_src_edge[k];
                        }
                    });
                auto out_src_table_padded =
                    padded_column(out_src_table, domains.n->size);
                auto out_src_query_padded =
                    padded_column(out_src_query, domains.edge->size);
                auto out_src_multiplicity_padded =
                    padded_column(
                        src_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_src",
                            out_src_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "E_src",
                                    eta_src_out_powers[1]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_src",
                            out_src_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_src"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "E_src_edge",
                                    eta_src_out_powers[1]),
                            }),
                        column_spec(
                            prefix + "m_src",
                            out_src_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_src", "Query_src", "m_src"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_src_out = transcript.challenge(output_challenge_name("beta_src_out", head_index));
                trace.challenges[output_challenge_name("beta_src_out", head_index)] = beta_src_out;
                const auto out_src_route = cached_route_trace(
                    head_cache_prefix + ":src_route:" + eta_src_out.to_string() + ":" + beta_src_out.to_string(),
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(out_src_table_padded),
                            std::move(out_src_query_padded),
                            std::move(out_src_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_src_out);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_src_node", out_src_route, out_src_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_src_edge", out_src_route, out_src_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_src_node", out_src_route, out_src_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_src", out_src_route, out_src_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_src_node", "J_src_edge", "R_src_node", "R_src"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                auto commit_output_logup_round = [&](
                                                     const std::string &suffix,
                                                     const std::string &beta_name,
                                                     const std::vector<FieldElement> &table,
                                                     const std::vector<FieldElement> &query,
                                                     const std::vector<FieldElement> &multiplicity,
                                                     const std::vector<FieldElement> &q_table,
                                                     const std::vector<FieldElement> &q_query,
                                                     std::vector<DynamicCommitmentLinearTerm> table_terms,
                                                     std::vector<DynamicCommitmentLinearTerm> query_terms,
                                                     std::size_t active_count)
                {
                    auto table_spec = table_terms.empty()
                                          ? column_spec(
                                                prefix + "Table_" + suffix,
                                                table,
                                                domains.edge)
                                          : deferred_linear_column_spec(
                                                prefix + "Table_" + suffix,
                                                table,
                                                domains.edge,
                                                std::move(table_terms));
                    auto query_spec = query_terms.empty()
                                          ? column_spec(
                                                prefix + "Query_" + suffix,
                                                query,
                                                domains.edge)
                                          : deferred_linear_column_spec(
                                                prefix + "Query_" + suffix,
                                                query,
                                                domains.edge,
                                                std::move(query_terms));
                    add_dynamic_commitment_batch(
                        trace,
                        {
                            std::move(table_spec),
                            std::move(query_spec),
                            column_spec(prefix + "m_" + suffix, multiplicity, domains.edge),
                        },
                        context.kzg,
                        keep_trace_payloads,
                        metrics);
                    for (const auto &stem : {std::string("Table_"), std::string("Query_"), std::string("m_")})
                    {
                        const auto label = prefix + stem + suffix;
                        transcript.absorb_commitment(label, trace.commitments.at(label).point);
                    }
                    const auto beta = transcript.challenge(beta_name);
                    trace.challenges[beta_name] = beta;
                    auto inverses = profiled_logup_inverse(table, query, q_table, q_query, beta);
                    auto accumulator =
                        profiled_logup_accumulator(
                            inverses.table,
                            inverses.query,
                            multiplicity,
                            q_table,
                            q_query,
                            active_count);
                    add_dynamic_commitment_batch(
                        trace,
                        {
                            column_spec(prefix + "J_tbl_" + suffix, std::move(inverses.table), domains.edge),
                            column_spec(prefix + "J_qry_" + suffix, std::move(inverses.query), domains.edge),
                            column_spec(prefix + "R_" + suffix, std::move(accumulator), domains.edge),
                        },
                        context.kzg,
                        keep_trace_payloads,
                        metrics);
                    for (const auto &stem : {std::string("J_tbl_"), std::string("J_qry_"), std::string("R_")})
                    {
                        const auto label = prefix + stem + suffix;
                        transcript.absorb_commitment(label, trace.commitments.at(label).point);
                    }
                    return beta;
                };

                transcript.absorb_commitment(prefix + "S", trace.commitments.at(prefix + "S").point);
                transcript.absorb_commitment(prefix + "Z", trace.commitments.at(prefix + "Z").point);
                transcript.absorb_commitment("V_T_L_x", context.static_commitments.at("V_T_L_x").point);
                transcript.absorb_commitment("V_T_L_y", context.static_commitments.at("V_T_L_y").point);
                const auto eta_l_out_name = output_challenge_name("eta_L_out", head_index);
                const auto eta_l_out = transcript.challenge(eta_l_out_name);
                trace.challenges[eta_l_out_name] = eta_l_out;
                const std::vector<FieldElement> out_table_l =
                    cached_pair_lookup_table(prefix_base + "_L", context.tables.lrelu, eta_l_out, domains.edge->size);
                const auto out_l_artifacts = cached_pair_lookup_artifacts(
                    head_cache_prefix + ":lrelu_artifacts:" + eta_l_out.to_string(),
                    [&]()
                    {
                        return build_pair_lookup_artifacts(
                            *lrelu_index,
                            out_s,
                            out_z,
                            eta_l_out,
                            domains.edge->size,
                            n_edges,
                            prefix_base + "_L",
                            metrics);
                    });
                commit_output_logup_round(
                    "L",
                    output_challenge_name("beta_L_out", head_index),
                    out_table_l,
                    out_l_artifacts->query,
                    out_l_artifacts->multiplicity,
                    q_tbl_l,
                    q_qry_l,
                    {
                        linear_commitment_term(
                            context.public_commitments.at("P_T_L_x"),
                            FieldElement::one()),
                        linear_commitment_term(
                            context.public_commitments.at("P_T_L_y"),
                            eta_l_out),
                    },
                    {
                        linear_label_term(
                            prefix + "S",
                            FieldElement::one()),
                        linear_label_term(
                            prefix + "Z",
                            eta_l_out),
                    },
                    lookup_active_transition_limit(context.tables.lrelu.size(), n_edges, domains.edge->size));

                auto out_s_max = build_first_zero_group_selector(out_delta, edge_groups_by_dst, domains.edge->size);
                const auto out_c_max = cached_state_vector(
                    head_cache_prefix + ":cmax",
                    [&]()
                    {
                        return build_max_counter_state(out_s_max, q_new, enable_single_graph_selector_shortcut);
                    });

                transcript.absorb_commitment(prefix + "M", trace.commitments.at(prefix + "M").point);
                transcript.absorb_commitment(prefix + "M_edge", trace.commitments.at(prefix + "M_edge").point);
                transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
                transcript.absorb_commitment("V_T_range", context.static_commitments.at("V_T_range").point);
                const std::vector<FieldElement> out_table_r =
                    cached_single_lookup_table(prefix_base + "_R", context.tables.range, domains.edge->size);
                const auto out_r_artifacts = cached_single_lookup_artifacts(
                    head_cache_prefix + ":range_artifacts",
                    [&]()
                    {
                        return build_single_lookup_artifacts(
                            *range_index,
                            out_delta,
                            domains.edge->size,
                            n_edges,
                            prefix_base + "_R",
                            metrics);
                    });
                commit_output_logup_round(
                    "R",
                    output_challenge_name("beta_R_out", head_index),
                    out_table_r,
                    out_r_artifacts->query,
                    out_r_artifacts->multiplicity,
                    q_tbl_r,
                    q_qry_r,
                    {},
                    {},
                    lookup_active_transition_limit(context.tables.range.size(), n_edges, domains.edge->size));

                transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
                transcript.absorb_commitment(prefix + "U", trace.commitments.at(prefix + "U").point);
                transcript.absorb_commitment("V_T_exp_x", context.static_commitments.at("V_T_exp_x").point);
                transcript.absorb_commitment("V_T_exp_y", context.static_commitments.at("V_T_exp_y").point);
                const auto eta_exp_out_name = output_challenge_name("eta_exp_out", head_index);
                const auto eta_exp_out = transcript.challenge(eta_exp_out_name);
                trace.challenges[eta_exp_out_name] = eta_exp_out;
                const std::vector<FieldElement> out_table_exp =
                    cached_pair_lookup_table(prefix_base + "_exp", context.tables.exp, eta_exp_out, domains.edge->size);
                const auto out_exp_artifacts = cached_pair_lookup_artifacts(
                    head_cache_prefix + ":exp_artifacts:" + eta_exp_out.to_string(),
                    [&]()
                    {
                        return build_pair_lookup_artifacts(
                            *exp_index,
                            out_delta,
                            out_u,
                            eta_exp_out,
                            domains.edge->size,
                            n_edges,
                            prefix_base + "_exp",
                            metrics);
                    });
                commit_output_logup_round(
                    "exp",
                    output_challenge_name("beta_exp_out", head_index),
                    out_table_exp,
                    out_exp_artifacts->query,
                    out_exp_artifacts->multiplicity,
                    q_tbl_exp,
                    q_qry_exp,
                    {
                        linear_commitment_term(
                            context.public_commitments.at("P_T_exp_x"),
                            FieldElement::one()),
                        linear_commitment_term(
                            context.public_commitments.at("P_T_exp_y"),
                            eta_exp_out),
                    },
                    {
                        linear_label_term(
                            prefix + "Delta",
                            FieldElement::one()),
                        linear_label_term(
                            prefix + "U",
                            eta_exp_out),
                    },
                    lookup_active_transition_limit(context.tables.exp.size(), n_edges, domains.edge->size));

                const auto lambda_out = transcript.challenge(output_challenge_name("lambda_out", head_index));
                trace.challenges[output_challenge_name("lambda_out", head_index)] = lambda_out;
                std::vector<FieldElement> widehat_y_star(domains.edge->size, FieldElement::zero());
                parallel_for_ranges(
                    n_edges,
                    128,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t k = begin; k < end; ++k)
                        {
                            widehat_y_star[k] = out_alpha[k] * y_prime_star_edge[k];
                        }
                    });
                auto w_out = build_weighted_sum(out_u, lambda_out, widehat_y_star, domains.edge->size);
                auto t_out = std::make_shared<std::vector<FieldElement>>(n_nodes, FieldElement::zero());
                parallel_for_ranges(
                    n_nodes,
                    64,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            (*t_out)[i] = out_sum[i] + lambda_out * y_star[i];
                        }
                    });
                const auto t_out_edge = std::make_shared<std::vector<FieldElement>>(
                    build_group_target(*t_out, edges, domains.edge->size));
                const auto psq_out = cached_state_vector(
                    head_cache_prefix + ":psq:" + lambda_out.to_string(),
                    [&]()
                    {
                        return build_group_prefix_state(w_out, q_new, enable_single_graph_selector_shortcut);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(prefix + "s_max", std::move(out_s_max), domains.edge),
                        column_spec(prefix + "C_max", out_c_max, domains.edge),
                        column_spec(prefix + "widehat_y_star", std::move(widehat_y_star), domains.edge),
                        deferred_linear_column_spec(
                            prefix + "w",
                            std::move(w_out),
                            domains.edge,
                            {
                                linear_label_term(prefix + "U", FieldElement::one()),
                                linear_label_term(prefix + "widehat_y_star", lambda_out),
                            }),
                        deferred_linear_compact_column_spec(
                            prefix + "T",
                            t_out,
                            domains.n,
                            {
                                linear_label_term(prefix + "Sum", FieldElement::one()),
                                linear_label_term(prefix + "Y_star", lambda_out),
                            }),
                        deferred_linear_compact_column_spec(
                            prefix + "T_edge",
                            t_out_edge,
                            domains.edge,
                            {
                                linear_label_term(prefix + "Sum_edge", FieldElement::one()),
                                linear_label_term(prefix + "Y_star_edge", lambda_out),
                            }),
                        column_spec(prefix + "PSQ", psq_out, domains.edge),
                        shared_compact_column_spec(prefix + "Y_star", compressed, y_star, domains.n),
                        shared_compact_column_spec(prefix + "Y_star_edge", compressed, y_star_edge, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                append_external_mask(
                    prefix + "Y_star",
                    external_mask_label(prefix + "Y_star"));

                transcript.absorb_commitment(prefix + "T", trace.commitments.at(prefix + "T").point);
                transcript.absorb_commitment(prefix + "T_edge", trace.commitments.at(prefix + "T_edge").point);
                const auto eta_t_out = transcript.challenge(output_challenge_name("eta_t_out", head_index));
                trace.challenges[output_challenge_name("eta_t_out", head_index)] = eta_t_out;
                const auto eta_t_out_powers = powers(eta_t_out, 2);
                std::vector<FieldElement> out_t_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> out_t_query(n_edges, FieldElement::zero());
                parallel_for_ranges(
                    n_nodes,
                    64,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            out_t_table[i] = absolute_id_fields[i] + eta_t_out_powers[1] * (*t_out)[i];
                        }
                    });
                parallel_for_ranges(
                    n_edges,
                    128,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t k = begin; k < end; ++k)
                        {
                            out_t_query[k] = edge_dst_fields[k] + eta_t_out_powers[1] * (*t_out_edge)[k];
                        }
                    });
                auto out_t_table_padded =
                    padded_column(out_t_table, domains.n->size);
                auto out_t_query_padded =
                    padded_column(out_t_query, domains.edge->size);
                auto out_t_multiplicity_padded =
                    padded_column(
                        dst_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_t",
                            out_t_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "T",
                                    eta_t_out_powers[1]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_t",
                            out_t_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_dst"),
                                    FieldElement::one()),
                                linear_label_term(
                                    prefix + "T_edge",
                                    eta_t_out_powers[1]),
                            }),
                        column_spec(
                            prefix + "m_t",
                            out_t_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_t", "Query_t", "m_t"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_t_out = transcript.challenge(output_challenge_name("beta_t_out", head_index));
                trace.challenges[output_challenge_name("beta_t_out", head_index)] = beta_t_out;
                const auto out_t_route = cached_route_trace(
                    head_cache_prefix + ":t_route:" + eta_t_out.to_string() + ":" + beta_t_out.to_string(),
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(out_t_table_padded),
                            std::move(out_t_query_padded),
                            std::move(out_t_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_t_out);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_t_node", out_t_route, out_t_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_t_edge", out_t_route, out_t_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_t_node", out_t_route, out_t_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_t", out_t_route, out_t_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_t_node", "J_t_edge", "R_t_node", "R_t"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                transcript.absorb_commitment(y_label, trace.commitments.at(y_label).point);
                transcript.absorb_commitment(prefix + "Y_star", trace.commitments.at(prefix + "Y_star").point);
                transcript.absorb_commitment(
                    external_mask_label(prefix + "Y_star"),
                    trace.commitments.at(external_mask_label(prefix + "Y_star")).point);
                trace.challenges[output_challenge_name("y_out_star", head_index)] =
                    transcript.challenge(output_challenge_name("y_out_star", head_index));

                const auto y_proj_out =
                    trace.challenges.at(
                        output_challenge_name(
                            "y_proj_out",
                            head_index));

                const auto y_src_out =
                    trace.challenges.at(
                        output_challenge_name(
                            "y_src_out",
                            head_index));

                const auto y_dst_out =
                    trace.challenges.at(
                        output_challenge_name(
                            "y_dst_out",
                            head_index));

                const auto y_out_star =
                    trace.challenges.at(
                        output_challenge_name(
                            "y_out_star",
                            head_index));

                const auto &output_parameters =
                    cached_quantized_output_head_parameters(
                        context,
                        head_index);

                const auto output_proj_b =
                    linear_form_by_powers(
                        output_parameters.seq_kernel,
                        powers(
                            y_proj_out,
                            context.local.num_classes));

                const auto xi_out_powers =
                    powers(
                        xi_out,
                        context.local.num_classes);

                // The same Y_prime evaluation is consumed both by the
                // projection binding and by the external opening registry.
                // Compute it once and reuse the exact value.
                const auto output_projection_evaluation =
                    matrix_row_major_evaluation(
                        y_prime,
                        y_proj_out);

                const auto out_proj_binding =
                    cached_binding_trace(
                        head_cache_prefix + ":proj:" + y_proj_out.to_string(),
                        [&]()
                        {
                            return build_row_major_projection_binding_trace(
                                h_cat,
                                n_nodes,
                                y_proj_out,
                                context.local.num_classes,
                                output_proj_b,
                                domains.cat,
                                context.model.output_layer.input_dim,
                                output_projection_evaluation);
                        });

                const auto out_src_binding =
                    cached_binding_trace(
                        head_cache_prefix + ":src:" + y_src_out.to_string(),
                        [&]()
                        {
                            return build_binding_trace(
                                y_prime,
                                domains.n,
                                n_nodes,
                                y_src_out,
                                output_parameters.attn_src,
                                domains.c,
                                context.local.num_classes);
                        });

                const auto out_dst_binding =
                    cached_binding_trace(
                        head_cache_prefix + ":dst:" + y_dst_out.to_string(),
                        [&]()
                        {
                            return build_binding_trace(
                                y_prime,
                                domains.n,
                                n_nodes,
                                y_dst_out,
                                output_parameters.attn_dst,
                                domains.c,
                                context.local.num_classes);
                        });

                const auto out_y_binding =
                    cached_binding_trace(
                        head_cache_prefix + ":y:" + y_out_star.to_string(),
                        [&]()
                        {
                            return build_binding_trace(
                                y_lin_matrix,
                                domains.n,
                                n_nodes,
                                y_out_star,
                                xi_out_powers,
                                domains.c,
                                context.local.num_classes);
                        });
                for (const auto *suffix : {"E_dst", "M", "Sum", "inv", "Y_star", "E_dst_edge", "M_edge", "Sum_edge", "inv_edge", "Y_star_edge"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto eta_dst_out = transcript.challenge(output_challenge_name("eta_dst_out", head_index));
                trace.challenges[output_challenge_name("eta_dst_out", head_index)] = eta_dst_out;
                const auto eta_dst_out_powers = powers(eta_dst_out, 6);
                std::vector<FieldElement> out_dst_table(n_nodes, FieldElement::zero());
                std::vector<FieldElement> out_dst_query(n_edges, FieldElement::zero());
                const auto &out_e_dst_edge = quantized.e_dst_edge;
                parallel_for_ranges(
                    n_nodes,
                    64,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            out_dst_table[i] = absolute_id_fields[i] + eta_dst_out_powers[1] * out_e_dst[i] + eta_dst_out_powers[2] * out_m[i] + eta_dst_out_powers[3] * out_sum[i] + eta_dst_out_powers[4] * out_inv[i] + eta_dst_out_powers[5] * y_star[i];
                        }
                    });
                parallel_for_ranges(
                    n_edges,
                    128,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t k = begin; k < end; ++k)
                        {
                            out_dst_query[k] = edge_dst_fields[k] + eta_dst_out_powers[1] * out_e_dst_edge[k] + eta_dst_out_powers[2] * out_m_edge[k] + eta_dst_out_powers[3] * out_sum_edge[k] + eta_dst_out_powers[4] * out_inv_edge[k] + eta_dst_out_powers[5] * y_star_edge[k];
                        }
                    });
                auto out_dst_table_padded =
                    padded_column(out_dst_table, domains.n->size);
                auto out_dst_query_padded =
                    padded_column(out_dst_query, domains.edge->size);
                auto out_dst_multiplicity_padded =
                    padded_column(
                        dst_multiplicity_template, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        deferred_linear_column_spec(
                            prefix + "Table_dst",
                            out_dst_table_padded,
                            domains.n,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_I"),
                                    FieldElement::one()),
                                linear_label_term(prefix + "E_dst", eta_dst_out_powers[1]),
                                linear_label_term(prefix + "M", eta_dst_out_powers[2]),
                                linear_label_term(prefix + "Sum", eta_dst_out_powers[3]),
                                linear_label_term(prefix + "inv", eta_dst_out_powers[4]),
                                linear_label_term(prefix + "Y_star", eta_dst_out_powers[5]),
                            }),
                        deferred_linear_column_spec(
                            prefix + "Query_dst",
                            out_dst_query_padded,
                            domains.edge,
                            {
                                linear_commitment_term(
                                    context.public_commitments.at("P_dst"),
                                    FieldElement::one()),
                                linear_label_term(prefix + "E_dst_edge", eta_dst_out_powers[1]),
                                linear_label_term(prefix + "M_edge", eta_dst_out_powers[2]),
                                linear_label_term(prefix + "Sum_edge", eta_dst_out_powers[3]),
                                linear_label_term(prefix + "inv_edge", eta_dst_out_powers[4]),
                                linear_label_term(prefix + "Y_star_edge", eta_dst_out_powers[5]),
                            }),
                        column_spec(
                            prefix + "m_dst",
                            out_dst_multiplicity_padded,
                            domains.n),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_dst", "Query_dst", "m_dst"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_dst_out = transcript.challenge(output_challenge_name("beta_dst_out", head_index));
                trace.challenges[output_challenge_name("beta_dst_out", head_index)] = beta_dst_out;
                const auto out_dst_route = cached_route_trace(
                    head_cache_prefix + ":dst_route:" + eta_dst_out.to_string() + ":" + beta_dst_out.to_string(),
                    [&]()
                    {
                        return build_route_trace_from_padded(
                            std::move(out_dst_table_padded),
                            std::move(out_dst_query_padded),
                            std::move(out_dst_multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n,
                            domains.edge,
                            beta_dst_out);
                    });
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "J_dst_node", out_dst_route, out_dst_route->node_inverse, domains.n),
                        shared_column_spec(prefix + "J_dst_edge", out_dst_route, out_dst_route->edge_inverse, domains.edge),
                        shared_column_spec(prefix + "R_dst_node", out_dst_route, out_dst_route->node_acc, domains.n),
                        shared_column_spec(prefix + "R_dst", out_dst_route, out_dst_route->edge_acc, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_dst_node", "J_dst_edge", "R_dst_node", "R_dst"})
                {
                    const auto label = prefix + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(prefix + "a_proj", out_proj_binding, out_proj_binding->a, domains.cat),
                        shared_column_spec(prefix + "b_proj", out_proj_binding, out_proj_binding->b, domains.cat),
                        shared_column_spec(prefix + "Acc_proj", out_proj_binding, out_proj_binding->acc, domains.cat),
                        shared_column_spec(prefix + "a_src", out_src_binding, out_src_binding->a, domains.c),
                        shared_column_spec(prefix + "b_src", out_src_binding, out_src_binding->b, domains.c),
                        shared_column_spec(prefix + "Acc_src", out_src_binding, out_src_binding->acc, domains.c),
                        shared_column_spec(prefix + "a_dst", out_dst_binding, out_dst_binding->a, domains.c),
                        shared_column_spec(prefix + "b_dst", out_dst_binding, out_dst_binding->b, domains.c),
                        shared_column_spec(prefix + "Acc_dst", out_dst_binding, out_dst_binding->acc, domains.c),
                        shared_column_spec(prefix + "a_y", out_y_binding, out_y_binding->a, domains.c),
                        shared_column_spec(prefix + "b_y", out_y_binding, out_y_binding->b, domains.c),
                        shared_column_spec(prefix + "Acc_y", out_y_binding, out_y_binding->acc, domains.c),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                const auto &div = quantized.aggregate_div;
                std::vector<FieldElement> output_sum_flat(
                    domains.edge->size,
                    FieldElement::zero());
                std::vector<FieldElement> output_pre_field_flat(
                    domains.edge->size,
                    FieldElement::zero());
                std::vector<FieldElement> output_agg_num_flat(
                    domains.edge->size,
                    FieldElement::zero());
                for (std::size_t node = 0; node < n_nodes; ++node)
                {
                    for (std::size_t column = 0; column < n_classes; ++column)
                    {
                        const auto flat_index = node * n_classes + column;
                        output_sum_flat[flat_index] = out_sum[node];
                        output_pre_field_flat[flat_index] = y_lin_matrix[node][column];
                        output_agg_num_flat[flat_index] =
                            out_sum[node] * y_lin_matrix[node][column];
                    }
                }
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_compact_column_spec(prefix + "Y_pre_flat", forward_artifacts, quantized.y_pre_flat, domains.edge),
                        column_spec(prefix + "Sum_flat", std::move(output_sum_flat), domains.edge),
                        column_spec(prefix + "Y_pre_field_flat", std::move(output_pre_field_flat), domains.edge),
                        column_spec(prefix + "Y_agg_num_flat", std::move(output_agg_num_flat), domains.edge),
                        shared_compact_column_spec(prefix + "Bias_flat", forward_artifacts, quantized.bias_flat, domains.edge),
                        shared_compact_column_spec(prefix + "Y_flat", forward_artifacts, quantized.y_flat, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                commit_signed_division(
                    prefix + "Div_",
                    div,
                    "P_Q_div_out" + std::to_string(head_index));
                std::vector<std::pair<std::string, FieldElement>> output_weight_terms;
                output_weight_terms.reserve(n_classes);
                auto output_weight_power = FieldElement::one();
                for (std::size_t column = 0; column < n_classes; ++column)
                {
                    output_weight_terms.push_back({output_weight_column_label(head_index, column),
                                                   output_weight_power});
                    output_weight_power *= trace.challenges.at(
                        output_challenge_name("y_proj_out", head_index));
                }

                append_model_binding_blinder(
                    prefix + "b_proj",
                    output_model_binding_blinder_label(
                        head_index,
                        "proj"),
                    domains.cat,
                    output_weight_terms);

                append_model_binding_blinder(
                    prefix + "b_src",
                    output_model_binding_blinder_label(
                        head_index,
                        "src"),
                    domains.c,
                    {
                        {
                            output_src_label(head_index),
                            FieldElement::one(),
                        },
                    });

                append_model_binding_blinder(
                    prefix + "b_dst",
                    output_model_binding_blinder_label(
                        head_index,
                        "dst"),
                    domains.c,
                    {
                        {
                            output_dst_label(head_index),
                            FieldElement::one(),
                        },
                    });

                trace.external_evaluations[output_external_eval_name(
                    "proj",
                    head_index)] =
                    output_projection_evaluation;

                trace.external_evaluations[output_external_eval_name(
                    "src",
                    head_index)] =
                    evaluate_trace_polynomial(
                        trace,
                        prefix + "E_src",
                        trace.challenges.at(
                            output_challenge_name(
                                "y_src_out",
                                head_index)),
                        domains.edge);

                trace.external_evaluations[output_score_mask_eval_name(
                    head_index,
                    "src")] =
                    evaluate_trace_polynomial(
                        trace,
                        output_score_mask_label(
                            head_index,
                            "src"),
                        trace.challenges.at(
                            output_challenge_name(
                                "y_src_out",
                                head_index)),
                        domains.edge);

                trace.external_evaluations[output_external_eval_name(
                    "dst",
                    head_index)] =
                    evaluate_trace_polynomial(
                        trace,
                        prefix + "E_dst",
                        trace.challenges.at(
                            output_challenge_name(
                                "y_dst_out",
                                head_index)),
                        domains.edge);

                trace.external_evaluations[output_score_mask_eval_name(
                    head_index,
                    "dst")] =
                    evaluate_trace_polynomial(
                        trace,
                        output_score_mask_label(
                            head_index,
                            "dst"),
                        trace.challenges.at(
                            output_challenge_name(
                                "y_dst_out",
                                head_index)),
                        domains.edge);

                trace.external_evaluations[output_external_eval_name(
                    "star",
                    head_index)] =
                    evaluate_trace_polynomial(
                        trace,
                        prefix + "Y_star",
                        trace.challenges.at(
                            output_challenge_name(
                                "y_out_star",
                                head_index)),
                        domains.n);

                trace.external_evaluations[external_mask_eval_name(
                    output_external_eval_name(
                        "star",
                        head_index))] =
                    evaluate_trace_polynomial(
                        trace,
                        external_mask_label(
                            prefix + "Y_star"),
                        trace.challenges.at(
                            output_challenge_name(
                                "y_out_star",
                                head_index)),
                        domains.n);
            };

            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                commit_output_head(head_index);
            }

            const auto &y_lin_average =
                forward_artifacts
                    ->output_y_lin_average;

            const auto &average_div =
                forward_artifacts
                    ->output_average_div;

            const auto &y_average =
                forward_artifacts
                    ->output_y_average;

            // Preserve the old independent consistency check:
            // average_div.y is the signed-division witness, while
            // output_y_average was independently produced by the
            // quantized forward path.
            require_flat_row_major_matches_matrix(
                average_div.y,
                y_average,
                n_nodes,
                n_classes,
                "output average witness");
            add_dynamic_commitment_batch(
                trace,
                {
                    matrix_commitment_only_spec("P_Y_lin", y_lin_average),
                    matrix_commitment_only_spec("P_Y", y_average),
                },
                context.kzg,
                keep_trace_payloads,
                metrics);
            commit_signed_division("P_avg_Div_", average_div, "P_Q_div_avg");

            std::unordered_map<std::string, const Matrix *>
                canonical_matrix_sources;
            const auto commit_canonical_matrix = [&](
                                                     const CanonicalMatrixRelationMeta &meta,
                                                     const Matrix &matrix)
            {
                if (meta.canonical_domain != domains.n->name || meta.row_count != n_nodes || meta.column_count == 0 || meta.column_objects.size() != meta.column_count || matrix.size() != meta.row_count)
                    throw std::runtime_error(
                        "canonical matrix trace shape mismatch: " + meta.relation_id);
                std::vector<
                    std::vector<FieldElement>>
                    column_values(
                        meta.column_count);

                for (auto &values :
                     column_values)
                {
                    values.assign(
                        domains.n->size,
                        FieldElement::zero());
                }

                // Matrix storage is row-major. Scan every source row once
                // and fan its values into the already-required canonical
                // column buffers. Workers own disjoint row ranges.
                parallel_for_ranges(
                    meta.row_count,
                    256,
                    [&](std::size_t begin,
                        std::size_t end)
                    {
                        for (std::size_t row =
                                 begin;
                             row < end;
                             ++row)
                        {
                            if (matrix[row].size() != meta.column_count)
                            {
                                throw std::runtime_error(
                                    "canonical matrix row width mismatch: " + meta.relation_id);
                            }

                            for (std::size_t column = 0;
                                 column < meta.column_count;
                                 ++column)
                            {
                                column_values[column][row] =
                                    matrix[row][column];
                            }
                        }
                    });

                std::vector<
                    DynamicCommitmentSpec>
                    specs;

                specs.reserve(
                    meta.column_count);

                for (std::size_t column = 0;
                     column < meta.column_count;
                     ++column)
                {
                    specs.push_back(
                        column_spec(
                            meta.column_objects[column],
                            std::move(
                                column_values[column]),
                            domains.n));
                }

                add_dynamic_commitment_batch(
                    trace,
                    std::move(specs),
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                if (!canonical_matrix_sources.emplace(
                                                 meta.matrix_object, &matrix)
                         .second)
                    throw std::runtime_error(
                        "duplicate canonical matrix source: " + meta.matrix_object);
            };

            std::size_t canonical_hidden_head = 0;
            for (std::size_t layer = 0;
                 layer < context.model.hidden_layers.size(); ++layer)
            {
                for (std::size_t local_head = 0;
                     local_head < context.model.hidden_layers[layer].heads.size();
                     ++local_head, ++canonical_hidden_head)
                {
                    auto &subproof = require_binding_subproof(
                        "bind:hidden:" + std::to_string(layer + 1) + ":" + std::to_string(local_head));
                    const auto &matrices = subproof.relation_meta
                                               .canonical_matrices;
                    if (matrices.size() != 3)
                        throw std::runtime_error(
                            "hidden canonical matrix registry mismatch");
                    const auto &quantized = forward_artifacts
                                                ->hidden_head_quantized.at(canonical_hidden_head);
                    commit_canonical_matrix(
                        matrices[0], quantized.h_prime);
                    commit_canonical_matrix(
                        matrices[1], quantized.h_agg_pre);
                    commit_canonical_matrix(
                        matrices[2], quantized.h_agg);
                }
                auto &concat_subproof = require_binding_subproof(
                    "bind:concat:" + std::to_string(layer + 1));
                if (concat_subproof.relation_meta
                        .canonical_matrices.size() != 1)
                    throw std::runtime_error(
                        "concat canonical matrix registry mismatch");
                commit_canonical_matrix(
                    concat_subproof.relation_meta.canonical_matrices.front(),
                    hidden_layer_concat_quantized.at(layer));
            }
            for (std::size_t head = 0;
                 head < context.model.output_layer.heads.size(); ++head)
            {
                auto &subproof = require_binding_subproof(
                    "bind:output:" + std::to_string(head));
                const auto &matrices =
                    subproof.relation_meta.canonical_matrices;
                if (matrices.size() != 3)
                    throw std::runtime_error(
                        "output canonical matrix registry mismatch");
                const auto &quantized =
                    forward_artifacts->output_head_quantized.at(head);
                commit_canonical_matrix(
                    matrices[0], quantized.y_prime);
                commit_canonical_matrix(
                    matrices[1], quantized.y_lin);
                commit_canonical_matrix(
                    matrices[2], quantized.y_quant);
            }
            auto &average_subproof = require_binding_subproof(
                "bind:output:avg");
            if (average_subproof.relation_meta.canonical_matrices.size() != 2)
                throw std::runtime_error(
                    "output average canonical matrix registry mismatch");
            commit_canonical_matrix(
                average_subproof.relation_meta.canonical_matrices[0],
                y_lin_average);
            commit_canonical_matrix(
                average_subproof.relation_meta.canonical_matrices[1],
                y_average);
            const auto append_aggregation_commitment = [&](
                                                           BindingSubproof &subproof,
                                                           const std::string &object)
            {
                const auto &commitment = trace.commitments.at(object);
                subproof.aggregation_commitments.push_back(
                    {object, commitment});
                transcript.absorb_commitment(object, commitment.point);
            };
            const auto build_aggregation_route = [&](
                                                     BindingSubproof &subproof,
                                                     const AggregationRouteRelationMeta &route,
                                                     const std::vector<FieldElement> &node_values,
                                                     const std::vector<FieldElement> &edge_values,
                                                     bool use_source_index)
            {
                transcript.absorb_commitment(
                    route.node_value_object,
                    trace.commitments.at(route.node_value_object).point);
                transcript.absorb_commitment(
                    route.edge_value_object,
                    trace.commitments.at(route.edge_value_object).point);
                const auto eta = transcript.challenge(
                    route.eta_challenge_label);
                trace.challenges[route.eta_challenge_label] = eta;
                std::vector<FieldElement> table(
                    n_nodes, FieldElement::zero());
                std::vector<FieldElement> query(
                    n_edges, FieldElement::zero());
                for (std::size_t node = 0; node < n_nodes; ++node)
                    table[node] = FieldElement(node) + eta * node_values[node];
                const auto &indices = use_source_index
                                          ? edge_src_fields
                                          : edge_dst_fields;
                for (std::size_t edge = 0; edge < n_edges; ++edge)
                    query[edge] = indices[edge] + eta * edge_values[edge];
                const auto &multiplicity = use_source_index
                                               ? src_multiplicity_template
                                               : dst_multiplicity_template;
                auto table_padded =
                    padded_column(table, domains.n->size);
                auto query_padded =
                    padded_column(query, domains.edge->size);
                auto multiplicity_padded =
                    padded_column(multiplicity, domains.n->size);
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(
                            route.table_object,
                            table_padded,
                            domains.n),
                        column_spec(
                            route.query_object,
                            query_padded,
                            domains.edge),
                    },
                    context.kzg, keep_trace_payloads, metrics);
                append_aggregation_commitment(
                    subproof, route.table_object);
                append_aggregation_commitment(
                    subproof, route.query_object);
                transcript.absorb_commitment(
                    route.multiplicity_object,
                    context.public_commitments.at(
                                                  route.multiplicity_object)
                        .point);
                const auto beta = transcript.challenge(
                    route.beta_challenge_label);
                trace.challenges[route.beta_challenge_label] = beta;
                std::shared_ptr<const RouteTrace> route_trace =
                    std::make_shared<RouteTrace>(
                        build_route_trace_from_padded(
                            std::move(table_padded),
                            std::move(query_padded),
                            std::move(multiplicity_padded),
                            n_nodes,
                            n_edges,
                            domains.n, domains.edge, beta));
                add_dynamic_commitment_batch(
                    trace,
                    {
                        shared_column_spec(
                            route.node_inverse_object, route_trace,
                            route_trace->node_inverse, domains.n),
                        shared_column_spec(
                            route.edge_inverse_object, route_trace,
                            route_trace->edge_inverse, domains.edge),
                        shared_column_spec(
                            route.node_accumulator_object, route_trace,
                            route_trace->node_acc, domains.n),
                        shared_column_spec(
                            route.edge_accumulator_object, route_trace,
                            route_trace->edge_acc, domains.edge),
                    },
                    context.kzg, keep_trace_payloads, metrics);
                for (const auto *object : {
                         &route.node_inverse_object,
                         &route.edge_inverse_object,
                         &route.node_accumulator_object,
                         &route.edge_accumulator_object})
                    append_aggregation_commitment(subproof, *object);

                const auto edge_accumulator =
                    trace_commitment_with_opening_material(
                        trace,
                        route.edge_accumulator_object,
                        domains.edge);
                const auto node_accumulator =
                    resolve_trace_opening_polynomial(
                        trace,
                        route.node_accumulator_object,
                        domains.n);
                auto shifted = commit_shifted_proof_polynomial(
                    edge_accumulator,
                    route.shifted_accumulator_object,
                    route.configured_shift_degree,
                    route.terminal_delta,
                    context.kzg,
                    metrics);
                trace.commitments.emplace(
                    route.shifted_accumulator_object, shifted);
                trace.polynomials.emplace(
                    route.shifted_accumulator_object,
                    *shifted.opening_polynomial);
                trace.polynomial_domains[route.shifted_accumulator_object] = "coef";
                append_aggregation_commitment(
                    subproof, route.shifted_accumulator_object);
                const auto y_rel = transcript.challenge_excluding(
                    route.terminal_challenge_label,
                    [&](const FieldElement &candidate)
                    {
                        return domains.edge->zero_polynomial_eval(
                                               candidate + route.terminal_delta)
                            .is_zero();
                    });
                trace.challenges[route.terminal_challenge_label] = y_rel;
                const auto u_rel =
                    shifted.opening_polynomial->evaluate(y_rel)
                        subproof.evaluations.push_back(BindingEvaluation{
                            .domain = "relation",
                            .object_id = route.shifted_accumulator_object + ":u_rel",
                            .values = {u_rel},
                            .kind = PayloadValueKind::MaskedPrivateValue,
                        });

                const auto node_degree = domains.n->size + registered_blinder_coefficient_count(context, route.node_accumulator_object) - 1;
                const auto relation_degree = std::max(
                    node_degree, route.configured_shift_degree);
                auto relation_coefficients =
                    materialize_blinded_coefficients(
                        *node_accumulator,
                        relation_degree);
                const auto shifted_coefficients =
                    materialize_blinded_coefficients(
                        *shifted.opening_polynomial, relation_degree);
                for (std::size_t coefficient = 0;
                     coefficient < relation_coefficients.size();
                     ++coefficient)
                    relation_coefficients[coefficient] -=
                        shifted_coefficients[coefficient];
                auto relation_polynomial = std::make_shared<Polynomial>(
                    Polynomial::from_coefficients(
                        route.relation_virtual_object,
                        std::move(relation_coefficients)));
                trace.commitments.emplace(
                    route.relation_virtual_object,
                    crypto::Commitment{
                        .name = route.relation_virtual_object,
                        .point = timed_derived_commitment_point(
                            metrics,
                            [&]()
                            {
                                return crypto::g1_add(
                                    trace.commitments.at(
                                                         route.node_accumulator_object)
                                        .point,
                                    crypto::g1_neg(shifted.point));
                            }),
                        .opening_polynomial =
                            std::move(relation_polynomial),
                    });
                for (const auto &point : {
                         y_rel,
                         y_rel + route.terminal_delta,
                         route.node_terminal_point})
                {
                    const auto group_id = "W_ext:" + point.to_string();
                    if (std::none_of(
                            subproof.opening_references.begin(),
                            subproof.opening_references.end(),
                            [&](const MetadataReference &reference)
                            { return reference.object_id == group_id; }))
                        subproof.opening_references.push_back({group_id});
                }
            };
            const auto build_route_terminal_relation = [&](
                                                           BindingSubproof &subproof,
                                                           const RouteTerminalRelationMeta &relation)
            {
                if (relation.node_domain != domains.n->name || relation.edge_domain != domains.edge->name)
                    throw std::runtime_error(
                        "route terminal relation domain mismatch: " + relation.relation_id);
                const auto edge_accumulator =
                    trace_commitment_with_opening_material(
                        trace,
                        relation.edge_accumulator_object,
                        domains.edge);
                const auto node_accumulator =
                    resolve_trace_opening_polynomial(
                        trace,
                        relation.node_accumulator_object,
                        domains.n);
                auto shifted = commit_shifted_proof_polynomial(
                    edge_accumulator,
                    relation.shifted_accumulator_object,
                    relation.configured_shift_degree,
                    relation.terminal_delta,
                    context.kzg,
                    metrics);
                trace.commitments.emplace(
                    relation.shifted_accumulator_object, shifted);
                trace.polynomials.emplace(
                    relation.shifted_accumulator_object,
                    *shifted.opening_polynomial);
                trace.polynomial_domains[relation.shifted_accumulator_object] = "coef";
                subproof.relation_commitments.push_back(
                    {relation.shifted_accumulator_object, shifted});
                transcript.absorb_commitment(
                    relation.shifted_accumulator_object, shifted.point);
                const auto y_rel = transcript.challenge_excluding(
                    relation.challenge_label,
                    [&](const FieldElement &candidate)
                    {
                        return domains.edge->zero_polynomial_eval(
                                               candidate + relation.terminal_delta)
                            .is_zero();
                    });
                trace.challenges[relation.challenge_label] = y_rel;
                const auto u_rel =
                    shifted.opening_polynomial->evaluate(y_rel);
                subproof.evaluations.push_back(BindingEvaluation{
                    .domain = "relation",
                    .object_id = relation.shifted_accumulator_object + ":u_rel",
                    .values = {u_rel},
                    .kind = PayloadValueKind::MaskedPrivateValue,
                });

                const auto node_degree = domains.n->size + registered_blinder_coefficient_count(context, relation.node_accumulator_object) - 1;
                const auto relation_degree = std::max(
                    node_degree, relation.configured_shift_degree);
                auto relation_coefficients =
                    materialize_blinded_coefficients(
                        *node_accumulator,
                        relation_degree);
                const auto shifted_coefficients =
                    materialize_blinded_coefficients(
                        *shifted.opening_polynomial, relation_degree);
                for (std::size_t coefficient = 0;
                     coefficient < relation_coefficients.size();
                     ++coefficient)
                    relation_coefficients[coefficient] -=
                        shifted_coefficients[coefficient];
                auto relation_polynomial = std::make_shared<Polynomial>(
                    Polynomial::from_coefficients(
                        relation.relation_virtual_object,
                        std::move(relation_coefficients)));
                trace.commitments.emplace(
                    relation.relation_virtual_object,
                    crypto::Commitment{
                        .name = relation.relation_virtual_object,
                        .point = timed_derived_commitment_point(
                            metrics,
                            [&]()
                            {
                                return crypto::g1_add(
                                    trace.commitments.at(
                                                         relation.node_accumulator_object)
                                        .point,
                                    crypto::g1_neg(shifted.point));
                            }),
                        .opening_polynomial =
                            std::move(relation_polynomial),
                    });
                for (const auto &point : {
                         y_rel,
                         y_rel + relation.terminal_delta,
                         relation.node_terminal_point})
                {
                    const auto group_id = "W_ext:" + point.to_string();
                    if (std::none_of(
                            subproof.opening_references.begin(),
                            subproof.opening_references.end(),
                            [&](const MetadataReference &reference)
                            { return reference.object_id == group_id; }))
                        subproof.opening_references.push_back({group_id});
                }
            };
            for (auto &subproof : trace.pi_bind->subproofs)
            {
                absorb_binding_relation_metadata(
                    transcript, subproof.relation_meta);
                for (const auto &relation :
                     subproof.relation_meta.route_terminal_relations)
                    build_route_terminal_relation(subproof, relation);
                for (const auto &matrix :
                     subproof.relation_meta.canonical_matrices)
                {
                    subproof.view_references.push_back(
                        {matrix.matrix_object});
                    const auto it = trace.commitments.find(
                        matrix.matrix_object);
                    if (it == trace.commitments.end())
                        throw std::runtime_error(
                            "canonical matrix reference commitment missing: " + matrix.matrix_object);
                    transcript.absorb_commitment(
                        matrix.matrix_object, it->second.point);
                    for (const auto &column : matrix.column_objects)
                    {
                        const auto &commitment =
                            trace.commitments.at(column);
                        subproof.view_commitments.push_back(
                            {column, commitment});
                        transcript.absorb_commitment(
                            column, commitment.point);
                    }
                    const auto compression = transcript.challenge(
                        matrix.compression_challenge_label);
                    trace.challenges[matrix.compression_challenge_label] = compression;
                    const auto source_it = canonical_matrix_sources.find(
                        matrix.matrix_object);
                    if (source_it == canonical_matrix_sources.end())
                        throw std::runtime_error(
                            "canonical matrix source material missing: " + matrix.matrix_object);
                    auto folded = compress_rows_with_challenge(
                        *source_it->second, compression);
                    folded.resize(
                        domains.n->size, FieldElement::zero());
                    add_dynamic_commitment_batch(
                        trace,
                        {column_spec(
                            matrix.compressed_object,
                            std::move(folded), domains.n)},
                        context.kzg, keep_trace_payloads, metrics);
                    const auto &fold_commitment =
                        trace.commitments.at(matrix.compressed_object);
                    subproof.view_commitments.push_back(
                        {matrix.compressed_object, fold_commitment});
                    transcript.absorb_commitment(
                        matrix.compressed_object,
                        fold_commitment.point);
                }
                for (const auto &relation :
                     subproof.relation_meta.aggregation_relations)
                {
                    if (relation.node_domain != domains.n->name || relation.edge_domain != domains.edge->name || relation.row_count != n_nodes)
                        throw std::runtime_error(
                            "aggregation relation domain mismatch: " + relation.relation_id);
                    const auto source_it = canonical_matrix_sources.find(
                        relation.source_matrix_object);
                    const auto target_it = canonical_matrix_sources.find(
                        relation.target_matrix_object);
                    if (source_it == canonical_matrix_sources.end() || target_it == canonical_matrix_sources.end())
                        throw std::runtime_error(
                            "aggregation relation canonical source missing: " + relation.relation_id);
                    transcript.absorb_commitment(
                        relation.alpha_object,
                        trace.commitments.at(relation.alpha_object).point);
                    const auto compression = transcript.challenge(
                        relation.compression_challenge_label);
                    trace.challenges[relation.compression_challenge_label] = compression;
                    auto source_node = compress_rows_with_challenge(
                        *source_it->second, compression);
                    auto target_node = compress_rows_with_challenge(
                        *target_it->second, compression);
                    auto source_edge = broadcast_node_values(
                        source_node, edges, true, domains.edge->size);
                    auto target_edge = broadcast_node_values(
                        target_node, edges, false, domains.edge->size);
                    source_node.resize(
                        domains.n->size, FieldElement::zero());
                    target_node.resize(
                        domains.n->size, FieldElement::zero());
                    std::vector<FieldElement> product(
                        domains.edge->size, FieldElement::zero());
                    const auto alpha_polynomial =
                        resolve_trace_opening_polynomial(
                            trace,
                            relation.alpha_object,
                            domains.edge);
                    for (std::size_t edge = 0; edge < n_edges; ++edge)
                        product[edge] = alpha_polynomial->value_at(edge) * source_edge[edge];
                    auto psq = build_group_prefix_state(
                        product, q_new,
                        enable_single_graph_selector_shortcut);
                    add_dynamic_commitment_batch(
                        trace,
                        {
                            column_spec(
                                relation.source_node_object, source_node,
                                domains.n),
                            column_spec(
                                relation.target_node_object, target_node,
                                domains.n),
                            column_spec(
                                relation.source_edge_object, source_edge,
                                domains.edge),
                            column_spec(
                                relation.target_edge_object, target_edge,
                                domains.edge),
                            column_spec(
                                relation.product_object, product,
                                domains.edge),
                            column_spec(
                                relation.psq_object, psq, domains.edge),
                        },
                        context.kzg, keep_trace_payloads, metrics);
                    for (const auto *object : {
                             &relation.source_node_object,
                             &relation.target_node_object,
                             &relation.source_edge_object,
                             &relation.target_edge_object,
                             &relation.product_object,
                             &relation.psq_object})
                        append_aggregation_commitment(
                            subproof, *object);
                    build_aggregation_route(
                        subproof, relation.source_route,
                        source_node, source_edge, true);
                    build_aggregation_route(
                        subproof, relation.target_route,
                        target_node, target_edge, false);
                    if (subproof.aggregation_commitments.size() != relation.commitment_order.size() || !std::equal(
                                                                                                           subproof.aggregation_commitments.begin(),
                                                                                                           subproof.aggregation_commitments.end(),
                                                                                                           relation.commitment_order.begin(),
                                                                                                           [](const auto &actual,
                                                                                                              const std::string &expected)
                                                                                                           { return actual.first == expected; }))
                        throw std::runtime_error(
                            "aggregation commitment order mismatch: " + relation.relation_id);
                }
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index) + "_";
                transcript.absorb_commitment(output_y_lin_label(head_index), trace.commitments.at(output_y_lin_label(head_index)).point);
                transcript.absorb_commitment(output_y_label(head_index), trace.commitments.at(output_y_label(head_index)).point);
                transcript.absorb_commitment(
                    model_proof_local_label(output_bias_label(head_index)),
                    trace.commitments.at(model_proof_local_label(output_bias_label(head_index))).point);
                transcript.absorb_commitment(
                    model_reblind_witness_label(output_bias_label(head_index)),
                    trace.commitments.at(model_reblind_witness_label(output_bias_label(head_index))).point);
                transcript.absorb_commitment(prefix + "Y_star", trace.commitments.at(prefix + "Y_star").point);
                transcript.absorb_commitment(prefix + "Table_dst", trace.commitments.at(prefix + "Table_dst").point);
                transcript.absorb_commitment(prefix + "Query_dst", trace.commitments.at(prefix + "Query_dst").point);
                transcript.absorb_commitment(prefix + "Y_pre_flat", trace.commitments.at(prefix + "Y_pre_flat").point);
                transcript.absorb_commitment(prefix + "Bias_flat", trace.commitments.at(prefix + "Bias_flat").point);
                transcript.absorb_commitment(prefix + "Y_flat", trace.commitments.at(prefix + "Y_flat").point);
                for (const auto *suffix : {
                         "Active", "X", "D", "Y", "Sigma", "Zero",
                         "A", "A_inv", "D_inv", "Q", "R", "Delta"})
                {
                    const auto label = prefix + "Div_" + suffix;
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                auto &binding_subproof = require_binding_subproof(
                    "bind:output:" + std::to_string(head_index));
                if (binding_subproof.relation_meta.indexed_views.size() != 1)
                {
                    throw std::runtime_error(
                        "output bias binding metadata is not uniquely registered");
                }
                const auto &view_meta = binding_subproof.relation_meta.indexed_views.front();
                const auto *shift_meta_ptr = find_unique_shift_relation_by_kind(
                    binding_subproof.relation_meta,
                    "output-bias-terminal");
                if (shift_meta_ptr == nullptr)
                    throw std::runtime_error(
                        "output bias terminal relation is not uniquely registered");
                const auto &shift_meta = *shift_meta_ptr;
                binding_subproof.view_references.insert(
                    binding_subproof.view_references.end(),
                    {
                        {view_meta.canonical_object},
                        {view_meta.view_object},
                        {"public:" + view_meta.canonical_selector_id},
                        {"public:" + view_meta.query_selector_id},
                    });
                const auto eta_view = transcript.challenge(
                    view_meta.eta_challenge_label);
                trace.challenges[view_meta.eta_challenge_label] = eta_view;

                if (n_classes >= domains.c->size)
                {
                    throw std::runtime_error(
                        "output bias canonical view leaves no terminal row");
                }
                std::vector<FieldElement> canonical_tuples(
                    domains.c->size, FieldElement::zero());
                std::vector<FieldElement> canonical_multiplicity(
                    domains.c->size, FieldElement::zero());
                const auto &proof_local_bias = trace.polynomials.at(
                    view_meta.canonical_object);
                for (std::size_t column = 0; column < n_classes; ++column)
                {
                    canonical_tuples[column] = FieldElement(column) + eta_view * proof_local_bias.value_at(column);
                    canonical_multiplicity[column] = FieldElement(n_nodes);
                }
                std::vector<FieldElement> query_tuples(
                    domains.edge->size, FieldElement::zero());
                const auto &bias_flat_values =
                    forward_artifacts->output_head_quantized[head_index].bias_flat;
                const auto view_length = n_nodes * n_classes;
                if (bias_flat_values.size() < view_length || bias_flat_values.size() > domains.edge->size || view_length >= domains.edge->size)
                {
                    throw std::runtime_error(
                        "output bias view does not cover its registered active rows and terminal");
                }
                if (std::any_of(
                        bias_flat_values.begin() + static_cast<std::ptrdiff_t>(view_length),
                        bias_flat_values.end(),
                        [](const FieldElement &value)
                        { return !value.is_zero(); }))
                {
                    throw std::runtime_error(
                        "output bias view has a nonzero padding row");
                }
                for (std::size_t index = 0; index < view_length; ++index)
                {
                    query_tuples[index] = FieldElement(index % n_classes) + eta_view * bias_flat_values[index];
                }
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(
                            output_bias_view_label(head_index, "Table_can"),
                            canonical_tuples,
                            domains.c),
                        column_spec(
                            output_bias_view_label(head_index, "Query"),
                            query_tuples,
                            domains.edge),
                        column_spec(
                            output_bias_view_label(head_index, "m_can"),
                            canonical_multiplicity,
                            domains.c),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"Table_can", "Query", "m_can"})
                {
                    const auto label = output_bias_view_label(head_index, suffix);
                    binding_subproof.view_logup_commitments.push_back(
                        {label, trace.commitments.at(label)});
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }
                const auto beta_view = transcript.challenge(
                    view_meta.beta_challenge_label);
                trace.challenges[view_meta.beta_challenge_label] = beta_view;
                const auto canonical_side =
                    build_indexed_view_active_prefix_witness(
                        canonical_tuples,
                        n_classes,
                        FieldElement(n_nodes),
                        beta_view);
                const auto query_side =
                    build_indexed_view_active_prefix_witness(
                        query_tuples,
                        view_length,
                        FieldElement::one(),
                        beta_view);
                if (canonical_side.accumulator.at(n_classes) != query_side.accumulator.at(view_length))
                {
                    throw std::runtime_error(
                        "output bias indexed-view terminals disagree");
                }
                add_dynamic_commitment_batch(
                    trace,
                    {
                        column_spec(
                            output_bias_view_label(head_index, "J_can"),
                            canonical_side.inverses, domains.c),
                        column_spec(
                            output_bias_view_label(head_index, "J_qry"),
                            query_side.inverses, domains.edge),
                        column_spec(
                            output_bias_view_label(head_index, "R_can"),
                            canonical_side.accumulator, domains.c),
                        column_spec(
                            output_bias_view_label(head_index, "R_qry"),
                            query_side.accumulator, domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
                for (const auto *suffix : {"J_can", "J_qry", "R_can", "R_qry"})
                {
                    const auto label = output_bias_view_label(head_index, suffix);
                    binding_subproof.view_logup_commitments.push_back(
                        {label, trace.commitments.at(label)});
                    transcript.absorb_commitment(label, trace.commitments.at(label).point);
                }

                const auto build_coefficient_basis_relation = [&]()
                {
                    if (binding_subproof.relation_meta
                            .coefficient_basis_relations.size() != 1)
                        throw std::runtime_error(
                            "output bias coefficient basis relation is not uniquely registered");
                    const auto &basis_meta = binding_subproof.relation_meta
                                                 .coefficient_basis_relations.front();
                    const auto region = coefficient_region_layout(
                        basis_meta.low_degree_length,
                        basis_meta.semantic_positions,
                        basis_meta.tail_length,
                        domains.edge->size);
                    if (region.total_length != basis_meta.total_length || region.chunks.size() != basis_meta.chunks.size() || basis_meta.low_degree_length != n_classes)
                        throw std::runtime_error(
                            "output bias coefficient basis layout mismatch");

                    std::vector<FieldElement> low_coefficients(
                        basis_meta.low_degree_length, FieldElement::zero());
                    for (std::size_t column = 0;
                         column < low_coefficients.size(); ++column)
                        low_coefficients[column] =
                            proof_local_bias.value_at(column);
                    const auto tail_randomness = sample_trace_blinder(
                        *trace.compiler_artifact,
                        basis_meta.coefficient_object);
                    const auto coefficients = materialize_coefficient_tail(
                        region, low_coefficients, tail_randomness);
                    Matrix coefficient_matrix{coefficients};
                    std::vector<std::vector<FieldElement>> chunk_values;
                    std::vector<DynamicCommitmentSpec> basis_specs;
                    chunk_values.reserve(basis_meta.chunks.size());
                    basis_specs.reserve(basis_meta.chunks.size() + 1);
                    basis_specs.push_back(matrix_commitment_only_spec(
                        basis_meta.coefficient_object, coefficient_matrix));
                    for (std::size_t chunk_index = 0;
                         chunk_index < basis_meta.chunks.size(); ++chunk_index)
                    {
                        const auto &chunk = basis_meta.chunks[chunk_index];
                        const auto &layout = region.chunks[chunk_index];
                        if (chunk.chunk_index != layout.chunk_index || chunk.offset != layout.offset || chunk.length != layout.length)
                            throw std::runtime_error(
                                "output bias coefficient chunk metadata mismatch");
                        auto &values = chunk_values.emplace_back(
                            domains.edge->size, FieldElement::zero());
                        std::copy_n(
                            coefficients.begin() + static_cast<std::ptrdiff_t>(chunk.offset),
                            chunk.length, values.begin());
                        basis_specs.push_back(column_spec(
                            chunk.object_id, values, domains.edge));
                    }
                    add_dynamic_commitment_batch(
                        trace, std::move(basis_specs), context.kzg,
                        keep_trace_payloads, metrics);
                    const auto pre_gamma_commitment_count =
                        basis_meta.chunks.size() + 1;
                    if (basis_meta.commitment_order.size() != pre_gamma_commitment_count + basis_meta.chunks.size())
                        throw std::runtime_error(
                            "output bias coefficient basis commitment order mismatch");
                    for (std::size_t index = 0;
                         index < basis_meta.chunks.size(); ++index)
                        if (basis_meta.commitment_order[pre_gamma_commitment_count + index] != basis_meta.chunks[index].accumulator_object)
                            throw std::runtime_error(
                                "output bias coefficient basis accumulator order mismatch");
                    for (std::size_t index = 0;
                         index < pre_gamma_commitment_count; ++index)
                    {
                        const auto &label = basis_meta.commitment_order[index];
                        binding_subproof.basis_commitments.push_back(
                            {label, trace.commitments.at(label)});
                        transcript.absorb_commitment(
                            label, trace.commitments.at(label).point);
                    }

                    const auto gamma = transcript.challenge_excluding(
                        basis_meta.gamma_challenge_label,
                        [&](const FieldElement &candidate)
                        {
                            return candidate.is_zero() || std::any_of(
                                                              basis_meta.chunks.begin(),
                                                              basis_meta.chunks.end(),
                                                              [&](const CoefficientBasisChunkMeta &chunk)
                                                              { return candidate == chunk.terminal_point; });
                        });
                    trace.challenges[basis_meta.gamma_challenge_label] = gamma;
                    std::vector<FieldElement> power_column(
                        domains.edge->size, FieldElement::one());
                    for (std::size_t row = 1;
                         row < power_column.size(); ++row)
                        power_column[row] = power_column[row - 1] * gamma;
                    auto power_polynomial = Polynomial::from_evaluations(
                        basis_meta.power_object,
                        std::move(power_column), domains.edge);
                    trace.polynomial_domains[basis_meta.power_object] =
                        domains.edge->name;
                    trace.polynomials.emplace(
                        basis_meta.power_object,
                        std::move(power_polynomial));
                    std::vector<DynamicCommitmentSpec> accumulator_specs;
                    accumulator_specs.reserve(basis_meta.chunks.size());
                    for (const auto &chunk : basis_meta.chunks)
                    {
                        std::vector<FieldElement> accumulator(
                            domains.edge->size, FieldElement::zero());
                        auto gamma_power = FieldElement::one();
                        for (std::size_t row = 0;
                             row < chunk.length; ++row)
                        {
                            const auto global_index = chunk.offset + row;
                            accumulator[row + 1] = accumulator[row] + coefficients[global_index] * gamma_power;
                            gamma_power *= gamma;
                        }
                        for (std::size_t row = chunk.length + 1;
                             row < accumulator.size(); ++row)
                            accumulator[row] = accumulator[chunk.length];
                        std::vector<FieldElement> chunk_coefficients(
                            coefficients.begin() + static_cast<std::ptrdiff_t>(chunk.offset),
                            coefficients.begin() + static_cast<std::ptrdiff_t>(
                                                       chunk.offset + chunk.length));
                        if (accumulator[chunk.length] != algebra::horner(chunk_coefficients, gamma))
                            throw std::runtime_error(
                                "output bias coefficient chunk accumulator terminal mismatch");
                        accumulator_specs.push_back(column_spec(
                            chunk.accumulator_object,
                            std::move(accumulator), domains.edge));
                    }
                    add_dynamic_commitment_batch(
                        trace, std::move(accumulator_specs),
                        context.kzg, keep_trace_payloads, metrics);
                    for (const auto &chunk : basis_meta.chunks)
                    {
                        binding_subproof.basis_commitments.push_back({chunk.accumulator_object,
                                                                      trace.commitments.at(chunk.accumulator_object)});
                        transcript.absorb_commitment(
                            chunk.accumulator_object,
                            trace.commitments.at(
                                                 chunk.accumulator_object)
                                .point);
                    }

                    std::vector<crypto::Commitment> shifted_accumulators;
                    shifted_accumulators.reserve(basis_meta.chunks.size());
                    for (const auto &chunk : basis_meta.chunks)
                    {
                        const auto delta = chunk.terminal_point - gamma;
                        const auto accumulator =
                            trace_commitment_with_opening_material(
                                trace,
                                chunk.accumulator_object,
                                domains.edge);
                        auto shifted = commit_shifted_proof_polynomial(
                            accumulator,
                            chunk.shifted_accumulator_object,
                            chunk.configured_shift_degree,
                            delta,
                            context.kzg,
                            metrics);
                        trace.commitments.emplace(
                            chunk.shifted_accumulator_object, shifted);
                        trace.polynomials.emplace(
                            chunk.shifted_accumulator_object,
                            *shifted.opening_polynomial);
                        trace.polynomial_domains[chunk.shifted_accumulator_object] = "coef";
                        binding_subproof.relation_commitments.push_back({chunk.shifted_accumulator_object, shifted});
                        transcript.absorb_commitment(
                            chunk.shifted_accumulator_object,
                            shifted.point);
                        shifted_accumulators.push_back(std::move(shifted));
                    }
                    const auto y_rel = transcript.challenge_excluding(
                        basis_meta.relation_challenge_label,
                        [&](const FieldElement &candidate)
                        {
                            return std::any_of(
                                basis_meta.chunks.begin(),
                                basis_meta.chunks.end(),
                                [&](const CoefficientBasisChunkMeta &chunk)
                                {
                                    return domains.edge->zero_polynomial_eval(
                                                           candidate + chunk.terminal_point - gamma)
                                        .is_zero();
                                });
                        });
                    trace.challenges[basis_meta.relation_challenge_label] = y_rel;
                    for (std::size_t index = 0;
                         index < basis_meta.chunks.size(); ++index)
                    {
                        const auto &chunk = basis_meta.chunks[index];
                        const auto delta = chunk.terminal_point - gamma;
                        const auto u_rel = shifted_accumulators[index]
                                               .opening_polynomial->evaluate(y_rel);
                        if (debug_taylor_shift_self_check && u_rel != evaluate_trace_polynomial(
                                                                          trace,
                                                                          chunk.accumulator_object,
                                                                          y_rel + delta,
                                                                          domains.edge))
                            throw std::runtime_error(
                                "output bias coefficient chunk Taylor shift value mismatch");
                        binding_subproof.evaluations.push_back(
                            BindingEvaluation{
                                .domain = "relation",
                                .object_id = chunk.shifted_accumulator_object + ":u_rel",
                                .values = {u_rel},
                                .kind = PayloadValueKind::MaskedPrivateValue,
                            });
                    }

                    auto relation_degree = basis_meta.total_length - 1;
                    for (const auto &chunk : basis_meta.chunks)
                        relation_degree = std::max(
                            relation_degree,
                            chunk.configured_shift_degree);
                    auto relation_coefficients =
                        materialize_blinded_coefficients(
                            trace.polynomials.at(
                                basis_meta.coefficient_object),
                            relation_degree);
                    auto relation_point = trace.commitments.at(
                                                               basis_meta.coefficient_object)
                                              .point;
                    for (std::size_t chunk_index = 0;
                         chunk_index < basis_meta.chunks.size();
                         ++chunk_index)
                    {
                        const auto scale = gamma.pow(
                            basis_meta.chunks[chunk_index].offset);
                        const auto &shifted = shifted_accumulators[chunk_index];
                        const auto &shifted_coefficients =
                            shifted.opening_polynomial->data;
                        for (std::size_t index = 0;
                             index < shifted_coefficients.size(); ++index)
                            relation_coefficients[index] -=
                                scale * shifted_coefficients[index];
                        relation_point = timed_derived_commitment_point(
                            metrics,
                            [&]()
                            {
                                return crypto::g1_add(
                                    relation_point,
                                    crypto::g1_neg(crypto::g1_mul(
                                        shifted.point, scale)));
                            });
                    }
                    const auto relation_label = output_bias_basis_label(
                        head_index, "relation_virtual");
                    auto relation_polynomial = std::make_shared<Polynomial>(
                        Polynomial::from_coefficients(
                            relation_label,
                            std::move(relation_coefficients)));
                    trace.commitments.emplace(
                        relation_label,
                        crypto::Commitment{
                            .name = relation_label,
                            .point = relation_point,
                            .opening_polynomial =
                                std::move(relation_polynomial),
                        });
                    std::vector<FieldElement> basis_points = {y_rel};
                    for (const auto &chunk : basis_meta.chunks)
                        basis_points.push_back(
                            y_rel + chunk.terminal_point - gamma);
                    basis_points.push_back(gamma);
                    for (const auto &point : basis_points)
                    {
                        const auto group_id = "W_ext:" + point.to_string();
                        if (std::none_of(
                                binding_subproof.opening_references.begin(),
                                binding_subproof.opening_references.end(),
                                [&](const MetadataReference &reference)
                                { return reference.object_id == group_id; }))
                            binding_subproof.opening_references.push_back(
                                {group_id});
                    }
                };

                if (shift_meta.configured_degrees.empty())
                {
                    if (shift_meta.canonical_points.size() != 1 || shift_meta.deltas.size() != 1 || !(shift_meta.deltas.front() == FieldElement::zero()))
                        throw std::runtime_error(
                            "invalid same-point output bias RelationMeta");
                    const auto r_can_label = output_bias_view_label(
                        head_index, "R_can");
                    const auto r_qry_label = output_bias_view_label(
                        head_index, "R_qry");
                    const auto relation_degree = std::max(
                        domains.c->size + registered_blinder_coefficient_count(context, r_can_label) - 1,
                        domains.edge->size + registered_blinder_coefficient_count(context, r_qry_label) - 1);
                    const auto r_can =
                        resolve_trace_opening_polynomial(
                            trace, r_can_label, domains.c);
                    const auto r_qry =
                        resolve_trace_opening_polynomial(
                            trace, r_qry_label, domains.edge);
                    auto relation_coefficients = materialize_blinded_coefficients(
                        *r_can, relation_degree);
                    const auto query_coefficients = materialize_blinded_coefficients(
                        *r_qry, relation_degree);
                    for (std::size_t i = 0; i < relation_coefficients.size(); ++i)
                        relation_coefficients[i] -= query_coefficients[i];
                    const auto relation_label = output_bias_view_label(
                        head_index, "terminal_relation_virtual");
                    auto relation_polynomial = std::make_shared<Polynomial>(
                        Polynomial::from_coefficients(
                            relation_label, std::move(relation_coefficients)));
                    trace.commitments.emplace(
                        relation_label,
                        crypto::Commitment{
                            .name = relation_label,
                            .point = timed_derived_commitment_point(
                                metrics,
                                [&]()
                                {
                                    return crypto::g1_add(
                                        trace.commitments.at(r_can_label).point,
                                        crypto::g1_neg(
                                            trace.commitments.at(r_qry_label).point));
                                }),
                            .opening_polynomial = std::move(relation_polynomial),
                        });
                    const auto y_public = transcript.challenge(
                        shift_meta.challenge_label);
                    trace.challenges[shift_meta.challenge_label] = y_public;
                    const auto public_m_value = trace.polynomials.at(
                                                                     output_bias_view_label(head_index, "m_can"))
                                                    .evaluate(y_public);
                    binding_subproof.evaluations.push_back(BindingEvaluation{
                        .domain = "relation",
                        .object_id = output_bias_view_label(head_index, "m_can") + ":public_bind",
                        .values = {public_m_value},
                        .kind = PayloadValueKind::PublicValue,
                    });
                    for (const auto &point : {
                             y_public, shift_meta.canonical_points.front()})
                    {
                        const auto group_id = "W_ext:" + point.to_string();
                        if (std::none_of(
                                binding_subproof.opening_references.begin(),
                                binding_subproof.opening_references.end(),
                                [&](const MetadataReference &reference)
                                { return reference.object_id == group_id; }))
                            binding_subproof.opening_references.push_back(
                                {group_id});
                    }
                    build_coefficient_basis_relation();
                    continue;
                }

                const bool canonical_is_first =
                    shift_meta.source_domains.at(0) == domains.c->name;
                const auto first_label = output_bias_view_label(
                    head_index, canonical_is_first ? "R_can" : "R_qry");
                const auto second_label = output_bias_view_label(
                    head_index, canonical_is_first ? "R_qry" : "R_can");
                auto signed_source_commitment = [&](
                                                    const std::string &source_label,
                                                    const std::shared_ptr<algebra::RootOfUnityDomain>
                                                        &source_domain,
                                                    bool negate,
                                                    const std::string &virtual_label)
                {
                    const auto source =
                        trace_commitment_with_opening_material(
                            trace, source_label, source_domain);
                    if (!negate)
                    {
                        return crypto::Commitment{
                            .name = virtual_label,
                            .point = source.point,
                            .opening_polynomial = source.opening_polynomial,
                        };
                    }
                    if (source.opening_polynomial == nullptr || source.opening_polynomial->domain == nullptr)
                    {
                        throw std::runtime_error(
                            "output bias terminal source has no evaluation polynomial");
                    }
                    const auto &source_poly = *source.opening_polynomial;
                    std::vector<FieldElement> negated_values(
                        source_poly.domain->size, FieldElement::zero());
                    for (std::size_t i = 0; i < negated_values.size(); ++i)
                        negated_values[i] = -source_poly.value_at(i);
                    auto negated = Polynomial::from_evaluations(
                        virtual_label, std::move(negated_values), source_poly.domain);
                    auto negated_blinder = source_poly.zero_domain_blinder;
                    for (auto &coefficient : negated_blinder)
                        coefficient = -coefficient;
                    negated.set_zero_domain_blinder(std::move(negated_blinder));
                    return crypto::Commitment{
                        .name = virtual_label,
                        .point = timed_derived_commitment_point(
                            metrics,
                            [&]()
                            { return crypto::g1_neg(source.point); }),
                        .opening_polynomial = std::make_shared<Polynomial>(std::move(negated)),
                    };
                };
                const auto first_source = signed_source_commitment(
                    first_label,
                    canonical_is_first ? domains.c : domains.edge,
                    !canonical_is_first,
                    prefix + "BiasView_F1");
                const auto second_source = signed_source_commitment(
                    second_label,
                    canonical_is_first ? domains.edge : domains.c,
                    canonical_is_first,
                    prefix + "BiasView_F2");
                trace.commitments.emplace(first_source.name, first_source);
                trace.commitments.emplace(second_source.name, second_source);
                const auto shifted_label = output_bias_view_label(
                    head_index, "R_shifted");
                auto shifted = commit_shifted_proof_polynomial(
                    second_source,
                    shifted_label,
                    shift_meta.configured_degrees.at(0),
                    shift_meta.deltas.at(1),
                    context.kzg,
                    metrics);
                trace.commitments.emplace(shifted_label, shifted);
                trace.polynomials.emplace(
                    shifted_label, *shifted.opening_polynomial);
                trace.polynomial_domains[shifted_label] = "coef";
                binding_subproof.relation_commitments.push_back(
                    {shifted_label, shifted});
                transcript.absorb_commitment(shifted_label, shifted.point);
                const auto &second_source_domain = *second_source.opening_polynomial->domain;
                const auto y_rel = transcript.challenge_excluding(
                    shift_meta.challenge_label,
                    [&](const FieldElement &candidate)
                    {
                        return second_source_domain.zero_polynomial_eval(
                                                       candidate + shift_meta.deltas.at(1))
                            .is_zero();
                    });
                trace.challenges[shift_meta.challenge_label] = y_rel;
                const auto u_rel =
                    shifted.opening_polynomial->evaluate(y_rel);
                binding_subproof.evaluations.push_back(BindingEvaluation{
                    .domain = "relation",
                    .object_id = shifted_label + ":u_rel",
                    .values = {u_rel},
                    .kind = PayloadValueKind::MaskedPrivateValue,
                });

                const auto first_source_degree =
                    first_source.opening_polynomial->domain->size + registered_blinder_coefficient_count(context, first_label) - 1;
                const auto relation_degree = std::max(
                    first_source_degree,
                    shift_meta.configured_degrees.at(0));
                auto relation_coefficients = materialize_blinded_coefficients(
                    *first_source.opening_polynomial, relation_degree);
                const auto &shifted_coefficients = shifted.opening_polynomial->data;
                for (std::size_t i = 0; i < shifted_coefficients.size(); ++i)
                    relation_coefficients[i] += shifted_coefficients[i];
                const auto relation_label = output_bias_view_label(
                    head_index, "terminal_relation_virtual");
                auto relation_polynomial = std::make_shared<Polynomial>(
                    Polynomial::from_coefficients(
                        relation_label, std::move(relation_coefficients)));
                trace.commitments.emplace(
                    relation_label,
                    crypto::Commitment{
                        .name = relation_label,
                        .point = timed_derived_commitment_point(
                            metrics,
                            [&]()
                            {
                                return crypto::g1_add(
                                    first_source.point, shifted.point);
                            }),
                        .opening_polynomial = std::move(relation_polynomial),
                    });
                const auto public_m_value = trace.polynomials.at(
                                                                 output_bias_view_label(head_index, "m_can"))
                                                .evaluate(y_rel);
                binding_subproof.evaluations.push_back(BindingEvaluation{
                    .domain = "relation",
                    .object_id = output_bias_view_label(head_index, "m_can") + ":public_bind",
                    .values = {public_m_value},
                    .kind = PayloadValueKind::PublicValue,
                });
                for (const auto &point : std::vector<FieldElement>{
                         y_rel,
                         y_rel + shift_meta.deltas.at(1),
                         shift_meta.canonical_points.at(0)})
                {
                    const auto group_id = "W_ext:" + point.to_string();
                    if (std::none_of(
                            binding_subproof.opening_references.begin(),
                            binding_subproof.opening_references.end(),
                            [&](const MetadataReference &reference)
                            { return reference.object_id == group_id; }))
                        binding_subproof.opening_references.push_back({group_id});
                }
                build_coefficient_basis_relation();
            }
            transcript.absorb_commitment("P_Y_lin", trace.commitments.at("P_Y_lin").point);
            transcript.absorb_commitment("P_Y", trace.commitments.at("P_Y").point);
            for (const auto *suffix : {
                     "Active", "X", "D", "Y", "Sigma", "Zero",
                     "A", "A_inv", "D_inv", "Q", "R", "Delta"})
            {
                const auto label = "P_avg_Div_" + std::string(suffix);
                transcript.absorb_commitment(label, trace.commitments.at(label).point);
            }
            trace.challenges["y_out"] =
                transcript.challenge(
                    "y_out");

            const auto &y_out_challenge =
                trace.challenges.at(
                    "y_out");

            const auto output_flat_count =
                n_nodes * n_classes;

            const auto output_powers =
                powers(
                    y_out_challenge,
                    output_flat_count);

            // y_quant is evaluated at the same y_out point both for the
            // folded-column binding target and for the external opening.
            // Retain one value per output head and reuse it below.
            std::vector<FieldElement>
                per_head_y_evaluations(
                    context.model
                        .output_layer
                        .heads.size(),
                    FieldElement::zero());

            for (std::size_t head_index = 0;
                 head_index < context.model
                                  .output_layer
                                  .heads.size();
                 ++head_index)
            {
                const auto prefix =
                    output_head_prefix(
                        head_index) +
                    "_";

                const auto &head_y_matrix =
                    forward_artifacts
                        ->output_head_quantized
                        .at(head_index)
                        .y_quant;

                const auto y_evaluation =
                    matrix_row_major_evaluation(
                        head_y_matrix,
                        y_out_challenge);

                per_head_y_evaluations[head_index] =
                    y_evaluation;

                auto binding =
                    build_binding_trace_from_folded_columns(
                        forward_artifacts
                            ->output_head_quantized
                            .at(head_index)
                            .y_flat,
                        output_powers,
                        domains.edge,
                        output_flat_count,
                        y_evaluation);

                add_dynamic_commitment_batch(
                    trace,
                    {
                        linear_column_spec(
                            prefix + "Bind_a",
                            std::move(binding.a),
                            domains.edge,
                            trace.commitments
                                .at(
                                    prefix + "Y_flat")
                                .point),
                        column_spec(
                            prefix + "Bind_b",
                            std::move(binding.b),
                            domains.edge),
                        column_spec(
                            prefix + "Bind_Acc",
                            std::move(binding.acc),
                            domains.edge),
                    },
                    context.kzg,
                    keep_trace_payloads,
                    metrics);
            }

            // The average output is likewise evaluated once and reused by
            // both the folded-column binding and mu_out.
            const auto average_y_evaluation =
                matrix_row_major_evaluation(
                    y_average,
                    y_out_challenge);

            auto average_binding =
                build_binding_trace_from_folded_columns(
                    forward_artifacts
                        ->output_average_div
                        .y,
                    output_powers,
                    domains.edge,
                    output_flat_count,
                    average_y_evaluation);

            add_dynamic_commitment_batch(
                trace,
                {
                    linear_column_spec(
                        "P_avg_Bind_a",
                        std::move(
                            average_binding.a),
                        domains.edge,
                        trace.commitments
                            .at(
                                "P_avg_Div_Y")
                            .point),
                    column_spec(
                        "P_avg_Bind_b",
                        std::move(
                            average_binding.b),
                        domains.edge),
                    column_spec(
                        "P_avg_Bind_Acc",
                        std::move(
                            average_binding.acc),
                        domains.edge),
                },
                context.kzg,
                keep_trace_payloads,
                metrics);

            for (std::size_t head_index = 0;
                 head_index < context.model
                                  .output_layer
                                  .heads.size();
                 ++head_index)
            {
                const auto &quantized =
                    forward_artifacts
                        ->output_head_quantized
                        .at(head_index);

                trace.external_evaluations[output_external_eval_name(
                    "y_lin",
                    head_index)] =
                    matrix_row_major_evaluation(
                        quantized.y_lin,
                        y_out_challenge);

                trace.external_evaluations[output_external_eval_name(
                    "y",
                    head_index)] =
                    per_head_y_evaluations[head_index];
            }

            trace.external_evaluations["mu_Y_lin"] =
                matrix_row_major_evaluation(
                    y_lin_average,
                    y_out_challenge);

            trace.external_evaluations["mu_out"] =
                average_y_evaluation;

            record_public_output_hashes(
                trace,
                y_average,
                forward_artifacts
                    ->output_y_average,
                forward_artifacts
                    ->reference_logits_fp32);

            if (metrics != nullptr)
            {
                metrics->output_head_trace_ms += std::max(
                    0.0,
                    elapsed_ms(output_trace_start, Clock::now()) - (metrics->commit_dynamic_ms - output_initial_commit_dynamic_ms));
            }

            finalize_trace_registry(context, transcript, trace, metrics);
            // Semantic commitment reuse is needed only while dynamic
            // commitments are being produced.  Do not carry its pointer/hash
            // registry into quotient and opening phases.
            trace.evaluation_commitment_cache.reset();
            if (metrics != nullptr)
            {
                const auto total_trace_ms = elapsed_ms(trace_start, Clock::now());
                const auto forward_delta = metrics->forward_ms - initial_forward_ms;
                const auto commit_dynamic_delta = metrics->commit_dynamic_ms - initial_commit_dynamic_ms;
                metrics->trace_generation_ms += std::max(0.0, total_trace_ms - forward_delta - commit_dynamic_delta);
            }
            return trace;
        }

    } // namespace

    std::vector<std::int64_t> weighted_edge_aggregate_raw_by_destination(
        const std::vector<std::int64_t> &edge_weights,
        const std::vector<std::vector<std::int64_t>> &source_rows,
        const std::vector<std::size_t> &edge_src_indices,
        const std::vector<std::size_t> &edge_dst_indices,
        const std::vector<std::pair<std::size_t, std::size_t>> &
            destination_groups,
        std::size_t node_count,
        std::string_view label)
    {
        if (edge_weights.size() != edge_src_indices.size() || edge_weights.size() != edge_dst_indices.size())
        {
            throw std::runtime_error(
                "weighted edge aggregation shape mismatch: " + std::string(label));
        }
        const auto width = source_rows.empty()
                               ? 0
                               : source_rows.front().size();
        for (const auto &row : source_rows)
        {
            if (row.size() != width)
            {
                throw std::runtime_error(
                    "weighted edge aggregation source width mismatch: " + std::string(label));
            }
        }
        std::vector<std::uint8_t> destination_seen(node_count, 0);
        std::size_t next_edge = 0;
        for (const auto &[begin, end] : destination_groups)
        {
            if (begin != next_edge || begin >= end || end > edge_weights.size())
            {
                throw std::runtime_error(
                    "weighted edge aggregation group layout mismatch: " + std::string(label));
            }
            const auto destination = edge_dst_indices[begin];
            if (destination >= node_count)
            {
                throw std::runtime_error(
                    "weighted edge aggregation destination is out of range: " + std::string(label));
            }
            if (destination_seen[destination] != 0)
            {
                throw std::runtime_error(
                    "weighted edge aggregation has duplicate destination groups: " + std::string(label));
            }
            destination_seen[destination] = 1;
            for (std::size_t edge = begin; edge < end; ++edge)
            {
                if (edge_dst_indices[edge] != destination || edge_src_indices[edge] >= source_rows.size())
                {
                    throw std::runtime_error(
                        "weighted edge aggregation group content mismatch: " + std::string(label));
                }
            }
            next_edge = end;
        }
        if (next_edge != edge_weights.size())
        {
            throw std::runtime_error(
                "weighted edge aggregation groups do not cover every edge: " + std::string(label));
        }

        std::vector<std::int64_t> out(node_count * width, 0);
        parallel_for_ranges(
            destination_groups.size(),
            256,
            [&](std::size_t group_begin, std::size_t group_end)
            {
                std::vector<__int128> sums(width, 0);
                for (std::size_t group_index = group_begin;
                     group_index < group_end; ++group_index)
                {
                    std::fill(sums.begin(), sums.end(), 0);
                    const auto [begin, end] =
                        destination_groups[group_index];
                    const auto destination = edge_dst_indices[begin];
                    for (std::size_t edge = begin; edge < end; ++edge)
                    {
                        const auto &source =
                            source_rows[edge_src_indices[edge]];
                        const auto weight =
                            static_cast<__int128>(edge_weights[edge]);
                        for (std::size_t column = 0;
                             column < width; ++column)
                        {
                            sums[column] += weight * static_cast<__int128>(source[column]);
                        }
                    }
                    for (std::size_t column = 0;
                         column < width; ++column)
                    {
                        out[destination * width + column] =
                            checked_raw_value(sums[column], label);
                    }
                }
            });
        return out;
    }

    std::shared_ptr<const algebra::Polynomial>
    resolve_trace_opening_polynomial(
        const TraceArtifacts &trace,
        const std::string &label,
        const std::shared_ptr<algebra::RootOfUnityDomain> &expected_domain)
    {
        if (expected_domain == nullptr)
        {
            throw std::runtime_error(
                "trace opening source has no expected domain: " + label);
        }
        const auto domain_it = trace.polynomial_domains.find(label);
        if (domain_it == trace.polynomial_domains.end() || domain_it->second != expected_domain->name)
        {
            throw std::runtime_error(
                "trace opening source domain registration mismatch: " + label);
        }
        const auto commitment_it = trace.commitments.find(label);
        if (commitment_it == trace.commitments.end())
        {
            throw std::runtime_error(
                "trace opening source commitment is missing: " + label);
        }

        std::shared_ptr<const algebra::Polynomial> material =
            commitment_it->second.opening_polynomial;
        if (material == nullptr)
        {
            if (const auto polynomial_it = trace.polynomials.find(label);
                polynomial_it != trace.polynomials.end())
            {
                material = std::make_shared<const algebra::Polynomial>(
                    polynomial_it->second);
            }
            else if (const auto spilled_it =
                         trace.spilled_polynomials.find(label);
                     spilled_it != trace.spilled_polynomials.end())
            {
                material = load_spilled_evaluation_polynomial(
                    spilled_it->second, label, expected_domain);
            }
        }
        if (material == nullptr)
        {
            throw std::runtime_error(
                "trace opening source material is missing: " + label);
        }
        if (material->basis != algebra::PolynomialBasis::Evaluation || material->domain == nullptr || material->domain->name != expected_domain->name || material->domain->size != expected_domain->size || !(material->domain->omega == expected_domain->omega) || material->size() > expected_domain->size)

        {
            throw std::runtime_error(
                "trace opening source polynomial/domain mismatch: " + label);
        }
        return material;
    }

    // trace 总入口：根据模型形状在多头正式路径与单头兼容路径之间分派。
    TraceArtifacts build_trace(const ProtocolContext &context, RunMetrics *metrics)
    {
        if (!context.model.has_real_multihead)
        {
            throw std::runtime_error(
                "formal trace builder rejects the removed single-output compatibility protocol");
        }
        return build_multihead_trace(context, metrics);
    }

} // namespace gatzk::protocol
