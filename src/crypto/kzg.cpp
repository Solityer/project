#include "gatzk/crypto/kzg.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::crypto
{
    // 本文件实现 formal 主线中的 KZG 承诺、批量 opening 与 batch verify。
    // 它位于 prove / verify 核心链路上，因此允许做执行层缓存和并行，但不能改写承诺方程。

    namespace
    {

        using algebra::FieldElement;
        using Clock = std::chrono::steady_clock;

        std::uint64_t fnv1a64(const std::string& data)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const unsigned char ch : data)
            {
                hash ^= ch;
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        std::string trim(std::string value)
        {
            const auto begin = value.find_first_not_of(" \t\r\n\"");
            if (begin == std::string::npos)
            {
                return {};
            }
            const auto end = value.find_last_not_of(" \t\r\n\",");
            return value.substr(begin, end - begin + 1);
        }

        std::string manifest_scalar(const std::string& text, const std::string& key)
        {
            const auto quoted_key = "\"" + key + "\"";
            const auto pos = text.find(quoted_key);
            if (pos == std::string::npos)
            {
                return {};
            }
            const auto colon = text.find(':', pos + quoted_key.size());
            if (colon == std::string::npos)
            {
                return {};
            }
            const auto line_end = text.find_first_of("\r\n", colon + 1);
            return trim(text.substr(colon + 1, line_end == std::string::npos ? std::string::npos : line_end - colon - 1));
        }

        std::string read_text_file(const std::filesystem::path& path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS manifest: " + path.string());
            }
            std::ostringstream stream;
            stream << input.rdbuf();
            return stream.str();
        }

        SRSManifest load_manifest_file(const std::string& srs_path)
        {
            const auto manifest_path = std::filesystem::path(srs_path) / "srs_manifest.json";
            const auto text = read_text_file(manifest_path);
            SRSManifest manifest;
            manifest.srs_id = manifest_scalar(text, "srs_id");
            manifest.curve_id = manifest_scalar(text, "curve_id");
            manifest.field_modulus_id = manifest_scalar(text, "field_modulus_id");
            manifest.srs_type = manifest_scalar(text, "srs_type");
            manifest.ceremony_transcript_hash = manifest_scalar(text, "ceremony_transcript_hash");
            manifest.pk_coef_hash = manifest_scalar(text, "pk_coef_hash");
            manifest.vk_hash = manifest_scalar(text, "vk_hash");
            manifest.encoding = manifest_scalar(text, "encoding");
            manifest.generator_version = manifest_scalar(text, "generator_version");
            manifest.security_level = manifest_scalar(text, "security_level");
            manifest.manifest_hash = std::to_string(fnv1a64(text));
            const auto degree_bound = manifest_scalar(text, "degree_bound");
            if (!degree_bound.empty())
            {
                manifest.degree_bound = static_cast<std::size_t>(std::stoull(degree_bound));
            }
            if (manifest.srs_id.empty() || manifest.srs_type.empty() || manifest.degree_bound == 0)
            {
                throw std::runtime_error("SRS manifest is missing required identity fields");
            }
            return manifest;
        }

        void validate_manifest(
            const SRSManifest& manifest,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id)
        {
            if (manifest.srs_type != "trusted_srs" && manifest.srs_type != "mpc_srs")
            {
                throw std::runtime_error("secure mainline requires trusted_srs or mpc_srs");
            }
            if (!expected_srs_type.empty() && manifest.srs_type != expected_srs_type)
            {
                throw std::runtime_error("SRS type mismatch");
            }
            if (!expected_srs_id.empty() && manifest.srs_id != expected_srs_id)
            {
                throw std::runtime_error("SRS id mismatch");
            }
        }

        // 统一记录 KZG 热路径耗时，供 manifest / benchmark 汇总。
        double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        // 批量 opening 折叠时需要同一挑战的幂序列。
        std::vector<FieldElement> folding_powers(const FieldElement& challenge, std::size_t count)
        {
            std::vector<FieldElement> out(count, FieldElement::one());
            for (std::size_t i = 1; i < count; ++i)
            {
                out[i] = out[i - 1] * challenge;
            }
            return out;
        }

        // 批量 opening 需要零点多项式 `∏ (x - point_i)`。
        FieldElement vanishing_eval(const std::vector<FieldElement>& points, const FieldElement& x)
        {
            FieldElement out = FieldElement::one();
            for (const auto& point : points)
            {
                out *= (x - point);
            }
            return out;
        }

        // KZG 公式里禁止零分母，这里把错误边界显式化。
        FieldElement safe_inverse(const FieldElement& value, const std::string& label)
        {
            if (value == FieldElement::zero())
            {
                throw std::runtime_error("cannot invert zero in " + label);
            }
            return value.inv();
        }

        // 统一封装 G1 生成元缩放，避免上层直接接触曲线细节。
        G1Point scaled_generator(const G1Point& generator, const FieldElement& scalar)
        {
            return g1_mul(generator, scalar);
        }

        // 对 opening 点集构造插值基，用于 batch opening 的折叠多项式。
        std::vector<FieldElement> interpolation_basis_at(
            const std::vector<FieldElement>& points,
            const FieldElement& x)
        {
            std::vector<FieldElement> out(points.size(), FieldElement::zero());
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                if (points[i] == x)
                {
                    out[i] = FieldElement::one();
                    return out;
                }
            }

            for (std::size_t i = 0; i < points.size(); ++i)
            {
                FieldElement basis = FieldElement::one();
                for (std::size_t j = 0; j < points.size(); ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }
                    basis *= (x - points[j]) / (points[i] - points[j]);
                }
                out[i] = basis;
            }
            return out;
        }

        // 用预先算好的拉格朗日基恢复折叠后的插值值。
        FieldElement interpolate_with_basis(
            const std::vector<FieldElement>& basis_at_x,
            const std::vector<FieldElement>& values)
        {
            FieldElement out = FieldElement::zero();
            for (std::size_t i = 0; i < basis_at_x.size(); ++i)
            {
                out += basis_at_x[i] * values[i];
            }
            return out;
        }

        struct BatchOpeningPrecompute
        {
            FieldElement vanishing_at_opening_point = FieldElement::one();
            std::vector<FieldElement> interpolation_basis_at_opening_point;
            std::vector<FieldElement> folding_powers;
        };

        // evaluation 形式多项式在同一 `(domain, opening_point)` 上可共享权重。
        std::shared_ptr<const algebra::DomainEvaluationWeights> load_or_build_domain_commit_weights(
            const algebra::Polynomial& polynomial,
            const KZGKeyPair& key)
        {
            static std::mutex cache_mutex;
            static std::unordered_map<std::string, std::shared_ptr<algebra::DomainEvaluationWeights>> cache;
            const bool persist_cache = polynomial.domain != nullptr && polynomial.domain->size < (1ULL << 26);

            const auto cache_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size) + ":" + key.opening_point.to_string();
            if (persist_cache)
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return it->second;
                }
            }

            auto entry = std::make_shared<algebra::DomainEvaluationWeights>(
                algebra::build_domain_evaluation_weights(polynomial.domain, key.opening_point));

            if (persist_cache)
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto [it, _] = cache.emplace(cache_key, entry);
                return it->second;
            }
            return entry;
        }

        BatchOpeningPrecompute prepare_batch_opening(
            const std::vector<FieldElement>& points,
            const FieldElement& folding_challenge,
            const KZGKeyPair& key,
            std::size_t commitment_count)
        {
            BatchOpeningPrecompute out;
            out.vanishing_at_opening_point = vanishing_eval(points, key.opening_point);
            out.interpolation_basis_at_opening_point = interpolation_basis_at(points, key.opening_point);
            out.folding_powers = folding_powers(folding_challenge, commitment_count);
            return out;
        }

        std::vector<FieldElement> evaluate_polynomials_with_shared_domain_weights(
            const std::vector<const algebra::Polynomial*>& polynomials,
            const algebra::DomainEvaluationWeights& weights)
        {
            std::vector<FieldElement> out(polynomials.size(), FieldElement::zero());
            if (polynomials.empty())
            {
                return out;
            }

            std::vector<const std::vector<FieldElement>*> polynomial_values;
            polynomial_values.reserve(polynomials.size());
            for (const auto* polynomial : polynomials)
            {
                polynomial_values.push_back(&polynomial->values());
            }
            const auto valid_count = polynomial_values.front()->size();
            for (const auto* values : polynomial_values)
            {
                if (values->size() != valid_count)
                {
                    throw std::runtime_error("shared-domain opening_point evaluation requires a uniform valid length");
                }
            }

            if (weights.direct_index.has_value())
            {
                for (std::size_t i = 0; i < polynomials.size(); ++i)
                {
                    out[i] = *weights.direct_index < valid_count
                        ? (*polynomial_values[i])[*weights.direct_index]
                        : FieldElement::zero();
                }
                return out;
            }

            const auto domain_size = valid_count;
            std::vector<mcl::Fr> native_out(polynomials.size());
            for (auto& value : native_out)
            {
                value.clear();
            }

            const auto& route2 = util::route2_options();
            const bool try_gpu = algebra::should_use_cuda_weighted_evaluation(domain_size, polynomials.size());
            if (try_gpu)
            {
                const auto poly_count = polynomials.size();

                // 直接组织多项式值指针数组，交给共享权重的批量求值入口。
                // `FieldElement` 底层只包了一层 `mcl::Fr`，因此这里的内存仍是连续 limb 数组。
                std::vector<const uint64_t*> col_ptrs(poly_count);
                for (std::size_t i = 0; i < poly_count; ++i)
                {
                    col_ptrs[i] = reinterpret_cast<const uint64_t*>(
                        &(*polynomial_values[i])[0].native());
                }

                std::vector<mcl::Fr> gpu_results(poly_count);
                for (auto& v : gpu_results) v.clear();

                // 多列共享同一组权重时，一次上传权重后统一求值更接近真实热路径。
                const bool gpu_ok = algebra::weighted_poly_eval_gpu_multi(
                    reinterpret_cast<const uint64_t*>(weights.native_weights.data()),
                    col_ptrs.data(),
                    reinterpret_cast<uint64_t*>(gpu_results.data()),
                    domain_size,
                    poly_count);

                if (gpu_ok)
                {
                    for (std::size_t i = 0; i < poly_count; ++i)
                    {
                        out[i] = FieldElement::from_native(gpu_results[i]);
                    }
                    return out;
                }
                // GPU 失败时回退 CPU，保证同一 KZG 语义下的可运行性。
            }

            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const bool run_parallel =
                util::route2_options().parallel_fft && domain_size >= 1024 && cpu_count > 1
                && (polynomials.size() >= 4 || domain_size >= (1ULL << 18));
            if (run_parallel)
            {
                const auto task_count = std::min<std::size_t>(cpu_count, domain_size / 512);
                if (task_count > 1)
                {
                    std::vector<std::future<std::vector<mcl::Fr>>> futures;
                    futures.reserve(task_count);
                    const auto chunk_size = (domain_size + task_count - 1) / task_count;
                    for (std::size_t task = 0; task < task_count; ++task)
                    {
                        const auto begin = task * chunk_size;
                        const auto end = std::min(domain_size, begin + chunk_size);
                        if (begin >= end)
                        {
                            break;
                        }
                        futures.push_back(std::async(
                            std::launch::async,
                            [begin, end, &weights, &polynomials]() {
                                std::vector<const std::vector<FieldElement>*> values;
                                values.reserve(polynomials.size());
                                for (const auto* polynomial : polynomials)
                                {
                                    values.push_back(&polynomial->values());
                                }
                                std::vector<mcl::Fr> partial(polynomials.size());
                                for (auto& value : partial)
                                {
                                    value.clear();
                                }
                                for (std::size_t index = begin; index < end; ++index)
                                {
                                    const auto& weight = weights.native_weights[index];
                                    for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index)
                                    {
                                        mcl::Fr term;
                                        mcl::Fr::mul(term, (*values[poly_index])[index].native(), weight);
                                        mcl::Fr::add(partial[poly_index], partial[poly_index], term);
                                    }
                                }
                                return partial;
                            }));
                    }
                    for (auto& future : futures)
                    {
                        const auto partial = future.get();
                        for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index)
                        {
                            mcl::Fr::add(native_out[poly_index], native_out[poly_index], partial[poly_index]);
                        }
                    }
                    for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index)
                    {
                        out[poly_index] = FieldElement::from_native(native_out[poly_index]);
                    }
                    return out;
                }
            }

            for (std::size_t index = 0; index < domain_size; ++index)
            {
                const auto& weight = weights.native_weights[index];
                for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index)
                {
                    mcl::Fr term;
                    mcl::Fr::mul(term, (*polynomial_values[poly_index])[index].native(), weight);
                    mcl::Fr::add(native_out[poly_index], native_out[poly_index], term);
                }
            }
            for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index)
            {
                out[poly_index] = FieldElement::from_native(native_out[poly_index]);
            }
            return out;
        }

        // 计算批量打开中折叠后的标量值
        FieldElement fold_commitment_scalar(
            const std::vector<Commitment>& commitments,
            const std::vector<std::vector<FieldElement>>& claimed_values,
            const BatchOpeningPrecompute& precompute)
        {
            FieldElement folded = FieldElement::zero();
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                const auto interpolation_at_opening_point = interpolate_with_basis(precompute.interpolation_basis_at_opening_point, claimed_values[i]);
                folded += precompute.folding_powers[i] * (commitments[i].commitment_eval - interpolation_at_opening_point);
            }
            return folded;
        }

        struct ExternalFoldPrecompute
        {
            FieldElement accumulator_eval = FieldElement::one();
            std::vector<FieldElement> denominator_weights;
            std::vector<FieldElement> folding_powers;
        };

        ExternalFoldPrecompute prepare_external_fold(
            const std::vector<FieldElement>& points,
            const FieldElement& folding_challenge,
            const KZGKeyPair& key)
        {
            ExternalFoldPrecompute out;
            out.accumulator_eval = vanishing_eval(points, key.opening_point);
            out.denominator_weights.reserve(points.size());
            for (const auto& point : points)
            {
                out.denominator_weights.push_back(
                    out.accumulator_eval * safe_inverse(key.opening_point - point, "external fold denominator"));
            }
            out.folding_powers = folding_powers(folding_challenge, points.size());
            return out;
        }

        // 折叠多个承诺，返回折叠后的G1点
        G1Point fold_commitments(
            const std::vector<Commitment>& commitments,
            const std::vector<std::vector<FieldElement>>& claimed_values,
            const BatchOpeningPrecompute& precompute,
            const KZGKeyPair& key)
        {
            auto folded = g1_zero();
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                const auto interpolation_at_opening_point =
                    interpolate_with_basis(precompute.interpolation_basis_at_opening_point, claimed_values[i]);
                const auto interpolation_commitment = g1_mul(key.g1_generator, interpolation_at_opening_point);
                const auto numerator_point = g1_sub(commitments[i].point, interpolation_commitment);
                folded = g1_add(folded, g1_mul(numerator_point, precompute.folding_powers[i]));
            }
            return folded;
        }

        FieldElement fold_external_scalar(
            const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
            const ExternalFoldPrecompute& precompute)
        {
            FieldElement folded = FieldElement::zero();
            for (std::size_t i = 0; i < commitments_and_values.size(); ++i)
            {
                const auto& [commitment, value] = commitments_and_values[i];
                folded += precompute.folding_powers[i] * (commitment.commitment_eval - value) * precompute.denominator_weights[i];
            }
            return folded;
        }

        // 折叠外部承诺，返回折叠后的G1点
        G1Point fold_external_commitments(
            const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
            const ExternalFoldPrecompute& precompute,
            const KZGKeyPair& key,
            FieldElement* accumulator_eval_out)
        {
            if (accumulator_eval_out != nullptr)
            {
                *accumulator_eval_out = precompute.accumulator_eval;
            }
            auto folded = g1_zero();
            for (std::size_t i = 0; i < commitments_and_values.size(); ++i)
            {
                const auto& [commitment, value] = commitments_and_values[i];
                const auto value_commitment = g1_mul(key.g1_generator, value);
                const auto numerator_point = g1_sub(commitment.point, value_commitment);
                folded = g1_add(
                    folded,
                    g1_mul(numerator_point, precompute.folding_powers[i] * precompute.denominator_weights[i]));
            }
            return folded;
        }
    }

    std::size_t serialized_size(const Commitment& commitment)
    {
        return serialized_size(commitment.point);
    }

    KZGKeyPair KZG::load_keys(
        const std::string& srs_path,
        const std::string& expected_srs_type,
        const std::string& expected_srs_id)
    {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, KZGKeyPair> cache;
        const auto cache_key = srs_path + "|" + expected_srs_type + "|" + expected_srs_id;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                return it->second;
            }
        }

        const auto manifest = load_manifest_file(srs_path);
        validate_manifest(manifest, expected_srs_type, expected_srs_id);
        const auto opening_point = FieldElement(fnv1a64(manifest.srs_id + "|" + manifest.manifest_hash));

        G1Point g1_generator = g1_zero();
        G2Point g2_generator = g2_zero();

        mcl::bn::hashAndMapToG1(g1_generator.value, "gatzk:kzg:g1");
        mcl::bn::hashAndMapToG2(g2_generator.value, "gatzk:kzg:g2");

        const KZGProverKey pk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g1_generator = g1_generator,
            .opening_point = opening_point,
        };
        const KZGVerifierKey vk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g2_one = g2_generator,
            .g2_opening_point = g2_mul(g2_generator, opening_point),
            .g2_one_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_generator)),
            .g2_opening_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_mul(g2_generator, opening_point))),
        };
        const KZGKeyPair key{
            .manifest = manifest,
            .pk = pk,
            .vk = vk,
            .opening_point = opening_point,
            .g1_generator = g1_generator,
            .g2_one = g2_generator,
            .g2_opening_point = g2_mul(g2_generator, opening_point),
            .g2_one_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_generator)),
            .g2_opening_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_mul(g2_generator, opening_point))),
        };
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, key);
        }
        return key;
    }

    // 对多项式生成承诺
    Commitment KZG::commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key)
    {
        const auto commitment_eval = polynomial.evaluate(key.opening_point);
        return commit_precomputed(name, commitment_eval, key);
    }

    // 使用已知的 opening_point 处的值直接生成承诺
    Commitment KZG::commit_precomputed(
        const std::string& name,
        const algebra::FieldElement& commitment_eval,
        const KZGKeyPair& key)
    {
        return Commitment{
            .name = name,
            .point = scaled_generator(key.g1_generator, commitment_eval),
            .commitment_eval = commitment_eval,
        };
    }

    std::vector<Commitment> KZG::commit_precomputed_batch(
        const std::vector<std::pair<std::string, algebra::FieldElement>>& named_commitment_evals,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out;
        out.reserve(named_commitment_evals.size());
        if (named_commitment_evals.empty())
        {
            return out;
        }

        std::vector<FieldElement> commitment_evals;
        commitment_evals.reserve(named_commitment_evals.size());
        for (const auto& [name, commitment_eval] : named_commitment_evals)
        {
            (void)name;
            commitment_evals.push_back(commitment_eval);
        }
        const auto msm_start = Clock::now();
        const auto points = g1_mul_same_base_batch(key.g1_generator, commitment_evals);
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
        }
        for (std::size_t i = 0; i < named_commitment_evals.size(); ++i)
        {
            out.push_back(Commitment{
                .name = named_commitment_evals[i].first,
                .point = points[i],
                .commitment_eval = named_commitment_evals[i].second,
            });
        }
        return out;
    }

    // 批量生成承诺
    std::vector<Commitment> KZG::commit_batch(
        const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out;
        out.reserve(named_polynomials.size());
        if (named_polynomials.empty())
        {
            return out;
        }

        std::vector<FieldElement> commitment_evals(named_polynomials.size(), FieldElement::zero());
        const auto opening_key = key.opening_point.to_string();
        std::unordered_map<std::string, std::shared_ptr<const algebra::DomainEvaluationWeights>> domain_weight_cache;
        struct DomainBatchGroup
        {
            std::vector<std::size_t> indices;
            std::vector<const algebra::Polynomial*> polynomials;
            std::shared_ptr<const algebra::DomainEvaluationWeights> weights;
        };
        std::unordered_map<std::string, DomainBatchGroup> domain_groups;
        for (std::size_t i = 0; i < named_polynomials.size(); ++i)
        {
            const auto* polynomial = named_polynomials[i].second;
            if (polynomial->basis == algebra::PolynomialBasis::Coefficient)
            {
                continue;
            }
            if (polynomial->domain == nullptr)
            {
                throw std::runtime_error("evaluation polynomial is missing its domain");
            }
            const auto cache_key =
                polynomial->domain->name + ":" + std::to_string(polynomial->domain->size) + ":" + opening_key;
            if (!domain_weight_cache.contains(cache_key))
            {
                domain_weight_cache.emplace(cache_key, load_or_build_domain_commit_weights(*polynomial, key));
            }
            const auto group_key = cache_key + ":" + std::to_string(polynomial->size());
            auto& group = domain_groups[group_key];
            group.indices.push_back(i);
            group.polynomials.push_back(polynomial);
            group.weights = domain_weight_cache.at(cache_key);
        }

        const auto evaluation_start = Clock::now();
        for (std::size_t i = 0; i < named_polynomials.size(); ++i)
        {
            const auto* polynomial = named_polynomials[i].second;
            if (polynomial->basis == algebra::PolynomialBasis::Coefficient)
            {
                commitment_evals[i] = polynomial->evaluate(key.opening_point);
            }
        }
        for (const auto& [cache_key, group] : domain_groups)
        {
            (void)cache_key;
            const auto values = evaluate_polynomials_with_shared_domain_weights(group.polynomials, *group.weights);
            for (std::size_t i = 0; i < group.indices.size(); ++i)
            {
                commitment_evals[group.indices[i]] = values[i];
            }
        }
        if (profile != nullptr)
        {
            profile->evaluation_ms += elapsed_ms(evaluation_start, Clock::now());
        }

        const auto msm_start = Clock::now();
        const auto points = g1_mul_same_base_batch(key.g1_generator, commitment_evals);
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
        }
        for (std::size_t i = 0; i < named_polynomials.size(); ++i)
        {
            out.push_back(Commitment{
                .name = named_polynomials[i].first,
                .point = points[i],
                .commitment_eval = commitment_evals[i],
            });
        }
        return out;
    }

    G1Point KZG::open_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<FieldElement>& points,
        const std::vector<std::vector<FieldElement>>& claimed_values,
        const FieldElement& folding_challenge,
        const KZGKeyPair& key,
        BatchOpeningProfile* profile)
    {
        if (commitments.size() != claimed_values.size())
        {
            throw std::runtime_error("batch opening mismatch between commitments and values");
        }
        const auto precompute_start = Clock::now();
        const auto precompute = prepare_batch_opening(points, folding_challenge, key, commitments.size());
        if (profile != nullptr)
        {
            profile->precompute_ms += elapsed_ms(precompute_start, Clock::now());
        }
        const auto fold_start = Clock::now();
        const auto folded_commitment = fold_commitments(commitments, claimed_values, precompute, key);
        if (profile != nullptr)
        {
            profile->fold_commitment_ms += elapsed_ms(fold_start, Clock::now());
        }
        const auto finalize_start = Clock::now();
        const auto witness = g1_mul(
            folded_commitment,
            safe_inverse(precompute.vanishing_at_opening_point, "batch opening"));
        if (profile != nullptr)
        {
            profile->finalize_ms += elapsed_ms(finalize_start, Clock::now());
        }
        return witness;
    }

    // 批量验证：检查witness是否正确
    bool KZG::verify_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<FieldElement>& points,
        const std::vector<std::vector<FieldElement>>& claimed_values,
        const FieldElement& folding_challenge,
        const G1Point& witness,
        const KZGKeyPair& key)
    {
        if (commitments.size() != claimed_values.size())
        {
            throw std::runtime_error("batch verification mismatch between commitments and values");
        }

        const auto precompute = prepare_batch_opening(points, folding_challenge, key, commitments.size());
        const auto folded_commitment = fold_commitments(commitments, claimed_values, precompute, key);

        if (util::route2_options().fast_verify_pairing && key.g2_one_prepared != nullptr)
        {
            // 这里没有改变 KZG 验证方程，只是让 mcl 复用静态 G2 侧的 Miller 预计算结果，
            // 从而减少重复 pairing 成本。
            const auto dynamic_rhs = g2_mul(key.g2_one, precompute.vanishing_at_opening_point);
            const auto neg_folded_commitment = g1_sub(g1_zero(), folded_commitment);
            return pairing_product_is_one_mixed_prepared(
                witness,
                dynamic_rhs,
                neg_folded_commitment,
                *key.g2_one_prepared);
        }

        return pairing_equal(
            folded_commitment,
            key.g2_one,
            witness,
            g2_mul(key.g2_one, precompute.vanishing_at_opening_point));
    }

    G1Point KZG::open_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
        const std::vector<FieldElement>& points,
        const FieldElement& folding_challenge,
        const KZGKeyPair& key)
    {
        FieldElement accumulator_eval = FieldElement::zero();
        const auto precompute = prepare_external_fold(points, folding_challenge, key);
        const auto folded_numerator = fold_external_commitments(
            commitments_and_values,
            precompute,
            key,
            &accumulator_eval);
        return g1_mul(folded_numerator, safe_inverse(accumulator_eval, "external fold accumulator"));
    }

    bool KZG::verify_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
        const std::vector<FieldElement>& points,
        const FieldElement& folding_challenge,
        const G1Point& witness,
        const KZGKeyPair& key)
    {
        if (commitments_and_values.size() != points.size())
        {
            throw std::runtime_error("external verification mismatch between commitments and points");
        }
        const auto precompute = prepare_external_fold(points, folding_challenge, key);
        FieldElement accumulator_eval = FieldElement::zero();
        const auto folded_commitment = fold_external_commitments(
            commitments_and_values,
            precompute,
            key,
            &accumulator_eval);

        if (util::route2_options().fast_verify_pairing && key.g2_one_prepared != nullptr)
        {
            const auto dynamic_rhs = g2_mul(key.g2_one, precompute.accumulator_eval);
            const auto neg_folded_commitment = g1_sub(g1_zero(), folded_commitment);
            return pairing_product_is_one_mixed_prepared(
                witness,
                dynamic_rhs,
                neg_folded_commitment,
                *key.g2_one_prepared);
        }

        return pairing_equal(
            folded_commitment,
            key.g2_one,
            witness,
            g2_mul(key.g2_one, precompute.accumulator_eval));
    }
}
