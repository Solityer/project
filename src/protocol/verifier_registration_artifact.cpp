#include "gatzk/protocol/verifier_context.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <openssl/sha.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "gatzk/algebra/polynomial.hpp"

namespace gatzk::protocol
{
    namespace
    {
        constexpr std::string_view kMagic = "ZKGATVRA";
        constexpr std::uint32_t kSectionCount = 12;
        constexpr std::size_t kFieldBytes = 32;
        constexpr std::size_t kMaxArtifactBytes = 2ULL << 30;
        constexpr std::size_t kMaxSectionBytes = 1ULL << 30;
        constexpr std::size_t kMaxVectorEntries = 1ULL << 27;
        constexpr std::size_t kMaxStringBytes = 1ULL << 20;
        constexpr std::size_t kMaxPointBytes = 256;

        enum class Section : std::uint32_t
        {
            Config = 1,
            Dataset = 2,
            LocalGraph = 3,
            ModelShape = 4,
            StaticTables = 5,
            Domains = 6,
            SrsAndVk = 7,
            PublicPolynomials = 8,
            PublicCommitments = 9,
            StaticCommitments = 10,
            CompilerArtifact = 11,
            Registry = 12,
        };

        void append_u8(std::vector<std::uint8_t> *out, std::uint8_t value)
        {
            out->push_back(value);
        }

        void append_u32(std::vector<std::uint8_t> *out, std::uint32_t value)
        {
            for (std::size_t i = 0; i < sizeof(value); ++i)
                out->push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
        }

        void append_u64(std::vector<std::uint8_t> *out, std::uint64_t value)
        {
            for (std::size_t i = 0; i < sizeof(value); ++i)
                out->push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
        }

        void append_bytes(
            std::vector<std::uint8_t> *out,
            std::span<const std::uint8_t> bytes)
        {
            out->insert(out->end(), bytes.begin(), bytes.end());
        }

        void append_string(
            std::vector<std::uint8_t> *out,
            std::string_view value)
        {
            if (value.size() > kMaxStringBytes)
                throw std::runtime_error("verifier registration string exceeds cap");
            append_u64(out, value.size());
            out->insert(out->end(), value.begin(), value.end());
        }

        void append_blob(
            std::vector<std::uint8_t> *out,
            std::string_view value)
        {
            append_u64(out, value.size());
            out->insert(out->end(), value.begin(), value.end());
        }

        void append_field(
            std::vector<std::uint8_t> *out,
            const algebra::FieldElement &value)
        {
            std::array<std::uint8_t, kFieldBytes> bytes{};
            // mcl reports the number of significant bytes written, which may
            // be smaller than the fixed field width for values with leading
            // zero limbs.  The buffer is zero-filled before writing, so the
            // complete 32-byte representation remains canonical.
            if (value.write_little_endian(bytes.data(), bytes.size()) > bytes.size())
                throw std::runtime_error("failed to encode canonical field element");
            append_bytes(out, bytes);
        }

        void append_point(
            std::vector<std::uint8_t> *out,
            const crypto::G1Point &point)
        {
            const auto bytes = crypto::serialize(point);
            if (bytes.empty() || bytes.size() > kMaxPointBytes)
                throw std::runtime_error("invalid G1 point encoding length");
            append_u32(out, static_cast<std::uint32_t>(bytes.size()));
            append_bytes(out, bytes);
        }

        void append_point(
            std::vector<std::uint8_t> *out,
            const crypto::G2Point &point)
        {
            const auto bytes = crypto::serialize(point);
            if (bytes.empty() || bytes.size() > kMaxPointBytes)
                throw std::runtime_error("invalid G2 point encoding length");
            append_u32(out, static_cast<std::uint32_t>(bytes.size()));
            append_bytes(out, bytes);
        }

        void append_bool(std::vector<std::uint8_t> *out, bool value)
        {
            append_u8(out, value ? 1 : 0);
        }

        void append_size(
            std::vector<std::uint8_t> *out,
            std::size_t value)
        {
            if (value > std::numeric_limits<std::uint64_t>::max())
                throw std::runtime_error("size does not fit canonical u64");
            append_u64(out, static_cast<std::uint64_t>(value));
        }

        template <typename T, typename Fn>
        void append_vector(
            std::vector<std::uint8_t> *out,
            const std::vector<T> &values,
            Fn &&append_value)
        {
            if (values.size() > kMaxVectorEntries)
                throw std::runtime_error("verifier registration vector exceeds cap");
            append_size(out, values.size());
            for (const auto &value : values)
                append_value(value);
        }

        void append_string_vector(
            std::vector<std::uint8_t> *out,
            const std::vector<std::string> &values)
        {
            append_vector(out, values, [&](const std::string &value)
                          { append_string(out, value); });
        }

        template <typename T, typename Fn>
        std::vector<T> read_vector(
            class Reader &reader,
            std::size_t maximum,
            Fn &&read_value);

        class Reader
        {
        public:
            explicit Reader(std::span<const std::uint8_t> bytes)
                : bytes_(bytes)
            {
            }

            std::size_t remaining() const
            {
                return bytes_.size() - position_;
            }

            bool empty() const
            {
                return position_ == bytes_.size();
            }

            std::uint8_t u8(std::string_view what)
            {
                require(sizeof(std::uint8_t), what);
                return bytes_[position_++];
            }

            std::uint32_t u32(std::string_view what)
            {
                require(sizeof(std::uint32_t), what);
                std::uint32_t value = 0;
                for (std::size_t i = 0; i < sizeof(value); ++i)
                    value |= static_cast<std::uint32_t>(bytes_[position_++]) << (8U * i);
                return value;
            }

            std::uint64_t u64(std::string_view what)
            {
                require(sizeof(std::uint64_t), what);
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < sizeof(value); ++i)
                    value |= static_cast<std::uint64_t>(bytes_[position_++]) << (8U * i);
                return value;
            }

            std::size_t size(std::string_view what)
            {
                const auto value = u64(what);
                if (value > std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("verifier registration size overflow: " + std::string(what));
                return static_cast<std::size_t>(value);
            }

            std::string string(std::string_view what)
            {
                const auto length = size(std::string(what) + ".length");
                if (length > kMaxStringBytes)
                    throw std::runtime_error("verifier registration string exceeds cap: " + std::string(what));
                const auto bytes = take(length, what);
                return std::string(
                    reinterpret_cast<const char *>(bytes.data()), bytes.size());
            }

            std::string blob(std::string_view what)
            {
                const auto length = size(std::string(what) + ".length");
                const auto bytes = take(length, what);
                return std::string(
                    reinterpret_cast<const char *>(bytes.data()), bytes.size());
            }

            std::vector<std::uint8_t> bytes(
                std::size_t length,
                std::string_view what)
            {
                const auto value = take(length, what);
                return std::vector<std::uint8_t>(value.begin(), value.end());
            }

            std::span<const std::uint8_t> take(
                std::size_t length,
                std::string_view what)
            {
                if (length > remaining())
                    throw std::runtime_error("truncated verifier registration " + std::string(what));
                const auto begin = position_;
                position_ += length;
                return bytes_.subspan(begin, length);
            }

            void require_empty(std::string_view what) const
            {
                if (!empty())
                    throw std::runtime_error("trailing bytes in verifier registration " + std::string(what));
            }

        private:
            void require(std::size_t length, std::string_view what) const
            {
                if (length > remaining())
                    throw std::runtime_error("truncated verifier registration " + std::string(what));
            }

            std::span<const std::uint8_t> bytes_;
            std::size_t position_ = 0;
        };

        template <typename T, typename Fn>
        std::vector<T> read_vector(
            Reader &reader,
            std::size_t maximum,
            Fn &&read_value)
        {
            const auto count = reader.size("vector.count");
            if (count > maximum || count > kMaxVectorEntries)
                throw std::runtime_error("verifier registration vector exceeds cap");
            std::vector<T> out;
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(read_value(reader));
            return out;
        }

        std::string sha256_hex(std::span<const std::uint8_t> bytes)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(bytes.data(), bytes.size(), digest.data());
            static constexpr char hex[] = "0123456789abcdef";
            std::string out;
            out.reserve(64);
            for (const auto byte : digest)
            {
                out.push_back(hex[(byte >> 4U) & 0xfU]);
                out.push_back(hex[byte & 0xfU]);
            }
            return out;
        }

        std::array<std::uint8_t, SHA256_DIGEST_LENGTH> sha256_raw(
            std::span<const std::uint8_t> bytes)
        {
            std::array<std::uint8_t, SHA256_DIGEST_LENGTH> digest{};
            SHA256(bytes.data(), bytes.size(), digest.data());
            return digest;
        }

        algebra::FieldElement read_field(Reader &reader, std::string_view what)
        {
            const auto bytes = reader.take(kFieldBytes, what);
            const auto value = algebra::FieldElement::from_little_endian_mod(
                bytes.data(), bytes.size());
            std::array<std::uint8_t, kFieldBytes> canonical{};
            if (value.write_little_endian(canonical.data(), canonical.size()) > canonical.size()
                || !std::equal(bytes.begin(), bytes.end(), canonical.begin()))
                throw std::runtime_error("non-canonical field element in verifier registration");
            return value;
        }

        crypto::G1Point read_g1(Reader &reader, std::string_view what)
        {
            const auto length = reader.u32(std::string(what) + ".length");
            if (length == 0 || length > kMaxPointBytes)
                throw std::runtime_error("invalid G1 point length in verifier registration");
            const auto bytes = reader.bytes(length, what);
            const auto point = crypto::deserialize_g1(bytes);
            if (crypto::serialize(point) != bytes)
                throw std::runtime_error("non-canonical G1 point in verifier registration");
            return point;
        }

        crypto::G2Point read_g2(Reader &reader, std::string_view what)
        {
            const auto length = reader.u32(std::string(what) + ".length");
            if (length == 0 || length > kMaxPointBytes)
                throw std::runtime_error("invalid G2 point length in verifier registration");
            const auto bytes = reader.bytes(length, what);
            const auto point = crypto::deserialize_g2(bytes);
            if (crypto::serialize(point) != bytes)
                throw std::runtime_error("non-canonical G2 point in verifier registration");
            return point;
        }

        void append_size_vector(
            std::vector<std::uint8_t> *out,
            const std::vector<std::size_t> &values)
        {
            append_vector(out, values, [&](std::size_t value)
                          { append_size(out, value); });
        }

        std::vector<std::size_t> read_size_vector(Reader &reader)
        {
            return read_vector<std::size_t>(
                reader, kMaxVectorEntries,
                [](Reader &inner)
                {
                    return inner.size("size-vector.value");
                });
        }

        void append_string_map(
            std::vector<std::uint8_t> *out,
            const std::unordered_map<std::string, std::string> &values)
        {
            std::vector<std::string> keys;
            keys.reserve(values.size());
            for (const auto &[key, _] : values)
            {
                (void)_;
                if (key.empty())
                    throw std::runtime_error("verifier registration map has an empty key");
                keys.push_back(key);
            }
            std::sort(keys.begin(), keys.end());
            append_size(out, keys.size());
            for (const auto &key : keys)
            {
                append_string(out, key);
                append_string(out, values.at(key));
            }
        }

        std::unordered_map<std::string, std::string> read_string_map(
            Reader &reader,
            std::string_view what)
        {
            const auto count = reader.size(std::string(what) + ".count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("verifier registration map exceeds cap");
            std::unordered_map<std::string, std::string> out;
            out.reserve(count);
            std::string previous;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto key = reader.string(std::string(what) + ".key");
                if (key.empty() || (!previous.empty() && key <= previous))
                    throw std::runtime_error("verifier registration map order is not canonical");
                previous = key;
                const auto value = reader.string(std::string(what) + ".value");
                if (!out.emplace(key, value).second)
                    throw std::runtime_error("duplicate verifier registration map key");
            }
            return out;
        }

        void append_size_map(
            std::vector<std::uint8_t> *out,
            const std::unordered_map<std::string, std::size_t> &values)
        {
            std::vector<std::string> keys;
            keys.reserve(values.size());
            for (const auto &[key, _] : values)
            {
                (void)_;
                if (key.empty())
                    throw std::runtime_error("verifier registration map has an empty key");
                keys.push_back(key);
            }
            std::sort(keys.begin(), keys.end());
            append_size(out, keys.size());
            for (const auto &key : keys)
            {
                append_string(out, key);
                append_size(out, values.at(key));
            }
        }

        std::unordered_map<std::string, std::size_t> read_size_map(
            Reader &reader,
            std::string_view what)
        {
            const auto count = reader.size(std::string(what) + ".count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("verifier registration map exceeds cap");
            std::unordered_map<std::string, std::size_t> out;
            out.reserve(count);
            std::string previous;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto key = reader.string(std::string(what) + ".key");
                if (key.empty() || (!previous.empty() && key <= previous))
                    throw std::runtime_error("verifier registration map order is not canonical");
                previous = key;
                if (!out.emplace(key, reader.size(std::string(what) + ".value")).second)
                    throw std::runtime_error("duplicate verifier registration map key");
            }
            return out;
        }

        void append_field_vector(
            std::vector<std::uint8_t> *out,
            const std::vector<algebra::FieldElement> &values)
        {
            append_vector(out, values, [&](const auto &value)
                          { append_field(out, value); });
        }

        std::vector<algebra::FieldElement> read_field_vector(Reader &reader)
        {
            // A 32-byte lower bound makes the allocation cap independent of
            // malformed short payloads.
            const auto count = reader.size("field-vector.count");
            if (count > kMaxVectorEntries || count > reader.remaining() / kFieldBytes)
                throw std::runtime_error("verifier registration field vector exceeds cap");
            std::vector<algebra::FieldElement> out;
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(read_field(reader, "field-vector.value"));
            return out;
        }

        void append_field_matrix(
            std::vector<std::uint8_t> *out,
            const std::vector<std::vector<algebra::FieldElement>> &rows)
        {
            append_vector(out, rows, [&](const auto &row)
                          { append_field_vector(out, row); });
        }

        std::vector<std::vector<algebra::FieldElement>> read_field_matrix(Reader &reader)
        {
            const auto count = reader.size("field-matrix.row-count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("verifier registration matrix has too many rows");
            std::vector<std::vector<algebra::FieldElement>> out;
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(read_field_vector(reader));
            return out;
        }

        void append_edge(
            std::vector<std::uint8_t> *out,
            const data::Edge &edge)
        {
            append_size(out, edge.src);
            append_size(out, edge.dst);
            append_size(out, edge.graph_id);
            append_size(out, edge.stable_index);
        }

        data::Edge read_edge(Reader &reader)
        {
            return data::Edge{
                .src = reader.size("edge.src"),
                .dst = reader.size("edge.dst"),
                .graph_id = reader.size("edge.graph_id"),
                .stable_index = reader.size("edge.stable_index")};
        }

        void append_normalized_input(
            std::vector<std::uint8_t> *out,
            const data::NormalizedGraphInput &input)
        {
            append_size(out, input.N_total);
            append_size(out, input.N);
            append_size(out, input.E);
            append_size(out, input.G_batch);
            append_size_vector(out, input.I);
            append_size_vector(out, input.src);
            append_size_vector(out, input.dst);
            append_size_vector(out, input.node_ptr);
            append_size_vector(out, input.edge_ptr);
        }

        data::NormalizedGraphInput read_normalized_input(Reader &reader)
        {
            return data::NormalizedGraphInput{
                .N_total = reader.size("public-input.N_total"),
                .N = reader.size("public-input.N"),
                .E = reader.size("public-input.E"),
                .G_batch = reader.size("public-input.G_batch"),
                .I = read_size_vector(reader),
                .src = read_size_vector(reader),
                .dst = read_size_vector(reader),
                .node_ptr = read_size_vector(reader),
                .edge_ptr = read_size_vector(reader)};
        }

        void append_config(
            std::vector<std::uint8_t> *out,
            const VerifierConfig &config)
        {
            append_string(out, config.dataset);
            append_string(out, config.task_type);
            append_string(out, config.report_unit);
            append_string(out, config.batching_rule);
            append_string(out, config.subgraph_rule);
            append_string(out, config.self_loop_rule);
            append_string(out, config.edge_sort_rule);
            append_string(out, config.chunking_rule);
            append_string(out, config.quant_cfg_id);
            const auto &q = config.quantization;
            append_u64(out, q.s_input);
            append_u64(out, q.s_delta);
            append_u64(out, q.s_exp);
            append_vector(out, q.s_h, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_w_hidden, [&](const auto &row) { append_vector(out, row, [&](auto value) { append_u64(out, value); }); });
            append_vector(out, q.s_h_prime, [&](const auto &row) { append_vector(out, row, [&](auto value) { append_u64(out, value); }); });
            append_vector(out, q.s_a_src_hidden, [&](const auto &row) { append_vector(out, row, [&](auto value) { append_u64(out, value); }); });
            append_vector(out, q.s_a_dst_hidden, [&](const auto &row) { append_vector(out, row, [&](auto value) { append_u64(out, value); }); });
            append_vector(out, q.s_agg_pre, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_agg, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_cat, [&](auto value) { append_u64(out, value); });
            append_u64(out, q.s_out_pre);
            append_u64(out, q.s_out_head);
            append_u64(out, q.s_out);
            append_vector(out, q.s_w_output, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_y_prime, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_a_src_output, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_a_dst_output, [&](auto value) { append_u64(out, value); });
            append_vector(out, q.s_b_output, [&](auto value) { append_u64(out, value); });
            append_string(out, config.model_arch_id);
            append_string(out, config.model_param_id);
            append_string(out, config.input_preprocess_id);
            append_string(out, config.static_table_id);
            append_string(out, config.degree_bound_id);
            append_string(out, config.srs_type);
            append_string(out, config.srs_id);
            append_string(out, config.privacy_policy);
            append_string(out, config.input_privacy);
            append_string(out, config.output_mode);
            append_string(out, config.opening_mode);
            append_string(out, config.binding_mode);
            append_string(out, config.proof_task);
            append_string(out, config.security_profile_id);
            append_size(out, config.num_classes);
            append_size(out, config.range_bits);
            append_size(out, config.layer_count);
            append_size(out, config.K_out);
            append_size(out, config.batch_graphs);
            append_vector(out, config.hidden_profile, [&](const auto &profile)
                          {
                              append_size(out, profile.head_count);
                              append_size(out, profile.head_dim);
                          });
            append_size_vector(out, config.d_in_profile);
        }

        std::vector<std::uint64_t> read_u64_vector(Reader &reader)
        {
            return read_vector<std::uint64_t>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return inner.u64("u64-vector.value"); });
        }

        std::vector<std::vector<std::uint64_t>> read_u64_matrix(Reader &reader)
        {
            return read_vector<std::vector<std::uint64_t>>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_u64_vector(inner); });
        }

        VerifierConfig read_config(Reader &reader)
        {
            VerifierConfig config;
            config.dataset = reader.string("config.dataset");
            config.task_type = reader.string("config.task_type");
            config.report_unit = reader.string("config.report_unit");
            config.batching_rule = reader.string("config.batching_rule");
            config.subgraph_rule = reader.string("config.subgraph_rule");
            config.self_loop_rule = reader.string("config.self_loop_rule");
            config.edge_sort_rule = reader.string("config.edge_sort_rule");
            config.chunking_rule = reader.string("config.chunking_rule");
            config.quant_cfg_id = reader.string("config.quant_cfg_id");
            auto &q = config.quantization;
            q.s_input = reader.u64("quant.s_input");
            q.s_delta = reader.u64("quant.s_delta");
            q.s_exp = reader.u64("quant.s_exp");
            q.s_h = read_u64_vector(reader);
            q.s_w_hidden = read_u64_matrix(reader);
            q.s_h_prime = read_u64_matrix(reader);
            q.s_a_src_hidden = read_u64_matrix(reader);
            q.s_a_dst_hidden = read_u64_matrix(reader);
            q.s_agg_pre = read_u64_vector(reader);
            q.s_agg = read_u64_vector(reader);
            q.s_cat = read_u64_vector(reader);
            q.s_out_pre = reader.u64("quant.s_out_pre");
            q.s_out_head = reader.u64("quant.s_out_head");
            q.s_out = reader.u64("quant.s_out");
            q.s_w_output = read_u64_vector(reader);
            q.s_y_prime = read_u64_vector(reader);
            q.s_a_src_output = read_u64_vector(reader);
            q.s_a_dst_output = read_u64_vector(reader);
            q.s_b_output = read_u64_vector(reader);
            config.model_arch_id = reader.string("config.model_arch_id");
            config.model_param_id = reader.string("config.model_param_id");
            config.input_preprocess_id = reader.string("config.input_preprocess_id");
            config.static_table_id = reader.string("config.static_table_id");
            config.degree_bound_id = reader.string("config.degree_bound_id");
            config.srs_type = reader.string("config.srs_type");
            config.srs_id = reader.string("config.srs_id");
            config.privacy_policy = reader.string("config.privacy_policy");
            config.input_privacy = reader.string("config.input_privacy");
            config.output_mode = reader.string("config.output_mode");
            config.opening_mode = reader.string("config.opening_mode");
            config.binding_mode = reader.string("config.binding_mode");
            config.proof_task = reader.string("config.proof_task");
            config.security_profile_id = reader.string("config.security_profile_id");
            config.num_classes = reader.size("config.num_classes");
            config.range_bits = reader.size("config.range_bits");
            config.layer_count = reader.size("config.layer_count");
            config.K_out = reader.size("config.K_out");
            config.batch_graphs = reader.size("config.batch_graphs");
            config.hidden_profile = read_vector<util::HiddenLayerProfile>(
                reader, kMaxVectorEntries,
                [](Reader &inner)
                {
                    return util::HiddenLayerProfile{
                        .head_count = inner.size("hidden-profile.head-count"),
                        .head_dim = inner.size("hidden-profile.head-dim")};
                });
            config.d_in_profile = read_size_vector(reader);
            return config;
        }

        void append_dataset(
            std::vector<std::uint8_t> *out,
            const data::GraphDataset &dataset)
        {
            // The candidate writer intentionally excludes labels and floating
            // point/raw feature storage.  Quantized public features live in the
            // LocalGraph section and are copied into dataset.features on load.
            if (!dataset.features_fp.empty() || !dataset.labels.empty() || !dataset.labels_multi.empty())
                throw std::runtime_error("verifier registration cannot contain private dataset material");
            append_string(out, dataset.name);
            append_string(out, dataset.input_preprocess_id);
            append_size(out, dataset.num_nodes);
            append_size(out, dataset.num_features);
            append_size(out, dataset.num_classes);
            append_size(out, dataset.graph_count);
            append_vector(out, dataset.edges, [&](const auto &edge) { append_edge(out, edge); });
            append_size_vector(out, dataset.node_ptr);
            append_size_vector(out, dataset.edge_ptr);
            append_string(out, dataset.task_type);
            append_string(out, dataset.report_unit);
        }

        data::GraphDataset read_dataset(Reader &reader)
        {
            data::GraphDataset dataset;
            dataset.name = reader.string("dataset.name");
            dataset.input_preprocess_id = reader.string("dataset.input_preprocess_id");
            dataset.num_nodes = reader.size("dataset.num_nodes");
            dataset.num_features = reader.size("dataset.num_features");
            dataset.num_classes = reader.size("dataset.num_classes");
            dataset.graph_count = reader.size("dataset.graph_count");
            dataset.edges = read_vector<data::Edge>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_edge(inner); });
            dataset.node_ptr = read_size_vector(reader);
            dataset.edge_ptr = read_size_vector(reader);
            dataset.task_type = reader.string("dataset.task_type");
            dataset.report_unit = reader.string("dataset.report_unit");
            return dataset;
        }

        void append_local(
            std::vector<std::uint8_t> *out,
            const data::LocalGraph &local)
        {
            if (!local.features_fp.empty() || !local.labels.empty() || !local.labels_multi.empty())
                throw std::runtime_error("verifier registration cannot contain raw local model/data material");
            append_string(out, local.name);
            append_string(out, local.input_preprocess_id);
            append_size(out, local.num_nodes);
            append_size(out, local.num_features);
            append_size(out, local.num_classes);
            append_size_vector(out, local.absolute_ids);
            append_field_matrix(out, local.features);
            append_vector(out, local.edges, [&](const auto &edge) { append_edge(out, edge); });
            append_normalized_input(out, local.public_input);
            append_size(out, local.graph_count);
            append_size_vector(out, local.node_ptr);
            append_size_vector(out, local.edge_ptr);
            append_string(out, local.task_type);
            append_string(out, local.report_unit);
            append_string(out, local.batching_rule);
            append_string(out, local.subgraph_rule);
            append_string(out, local.self_loop_rule);
            append_string(out, local.edge_sort_rule);
            append_string(out, local.chunking_rule);
        }

        data::LocalGraph read_local(Reader &reader)
        {
            data::LocalGraph local;
            local.name = reader.string("local.name");
            local.input_preprocess_id = reader.string("local.input_preprocess_id");
            local.num_nodes = reader.size("local.num_nodes");
            local.num_features = reader.size("local.num_features");
            local.num_classes = reader.size("local.num_classes");
            local.absolute_ids = read_size_vector(reader);
            local.features = read_field_matrix(reader);
            local.edges = read_vector<data::Edge>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_edge(inner); });
            local.public_input = read_normalized_input(reader);
            local.graph_count = reader.size("local.graph_count");
            local.node_ptr = read_size_vector(reader);
            local.edge_ptr = read_size_vector(reader);
            local.task_type = reader.string("local.task_type");
            local.report_unit = reader.string("local.report_unit");
            local.batching_rule = reader.string("local.batching_rule");
            local.subgraph_rule = reader.string("local.subgraph_rule");
            local.self_loop_rule = reader.string("local.self_loop_rule");
            local.edge_sort_rule = reader.string("local.edge_sort_rule");
            local.chunking_rule = reader.string("local.chunking_rule");
            return local;
        }

        void append_model_shape(
            std::vector<std::uint8_t> *out,
            const VerifierModelShape &model)
        {
            append_string(out, model.model_arch_id);
            append_string(out, model.model_param_id);
            append_string(out, model.input_preprocess_id);
            append_string(out, model.quant_cfg_id);
            append_string(out, model.static_table_id);
            append_string(out, model.degree_bound_id);
            append_size(out, model.L);
            append_size_vector(out, model.d_in_profile);
            append_vector(out, model.hidden_profile, [&](const auto &shape)
                          {
                              append_size(out, shape.head_count);
                              append_size(out, shape.head_dim);
                          });
            append_size(out, model.K_out);
            append_size(out, model.C);
            append_vector(out, model.hidden_layers, [&](const auto &layer)
                          {
                              append_size(out, layer.layer_index);
                              append_size(out, layer.input_dim);
                              append_size(out, layer.shape.head_count);
                              append_size(out, layer.shape.head_dim);
                              append_size(out, layer.heads.size());
                          });
            append_size(out, model.output_layer.input_dim);
            append_size(out, model.output_layer.output_dim);
            append_size(out, model.output_layer.head_count);
            append_size(out, model.output_layer.heads.size());
        }

        VerifierModelShape read_model_shape(Reader &reader)
        {
            VerifierModelShape model;
            model.model_arch_id = reader.string("model.model_arch_id");
            model.model_param_id = reader.string("model.model_param_id");
            model.input_preprocess_id = reader.string("model.input_preprocess_id");
            model.quant_cfg_id = reader.string("model.quant_cfg_id");
            model.static_table_id = reader.string("model.static_table_id");
            model.degree_bound_id = reader.string("model.degree_bound_id");
            model.L = reader.size("model.L");
            model.d_in_profile = read_size_vector(reader);
            model.hidden_profile = read_vector<model::HiddenLayerShape>(
                reader, kMaxVectorEntries,
                [](Reader &inner)
                {
                    return model::HiddenLayerShape{
                        .head_count = inner.size("model.hidden-profile.head-count"),
                        .head_dim = inner.size("model.hidden-profile.head-dim")};
                });
            model.K_out = reader.size("model.K_out");
            model.C = reader.size("model.C");
            model.hidden_layers = read_vector<VerifierHiddenLayerShape>(
                reader, kMaxVectorEntries,
                [](Reader &inner)
                {
                    VerifierHiddenLayerShape layer;
                    layer.layer_index = inner.size("model.layer.index");
                    layer.input_dim = inner.size("model.layer.input-dim");
                    layer.shape.head_count = inner.size("model.layer.head-count");
                    layer.shape.head_dim = inner.size("model.layer.head-dim");
                    const auto head_count = inner.size("model.layer.head-vector");
                    if (head_count > kMaxVectorEntries)
                        throw std::runtime_error("model head vector exceeds cap");
                    layer.heads.resize(head_count);
                    return layer;
                });
            model.output_layer.input_dim = reader.size("model.output.input-dim");
            model.output_layer.output_dim = reader.size("model.output.output-dim");
            model.output_layer.head_count = reader.size("model.output.head-count");
            const auto output_heads = reader.size("model.output.head-vector");
            if (output_heads > kMaxVectorEntries)
                throw std::runtime_error("model output head vector exceeds cap");
            model.output_layer.heads.resize(output_heads);
            return model;
        }

        void append_tables(
            std::vector<std::uint8_t> *out,
            const StaticTables &tables)
        {
            auto append_pairs = [&](const auto &pairs)
            {
                append_vector(out, pairs, [&](const auto &pair)
                              {
                                  append_field(out, pair.first);
                                  append_field(out, pair.second);
                              });
            };
            append_pairs(tables.lrelu);
            append_pairs(tables.elu);
            append_pairs(tables.exp);
            append_field_vector(out, tables.range);
        }

        StaticTables read_tables(Reader &reader)
        {
            auto read_pairs = [&](std::string_view what)
            {
                return read_vector<std::pair<algebra::FieldElement, algebra::FieldElement>>(
                    reader, kMaxVectorEntries,
                    [&](Reader &inner)
                    {
                        return std::make_pair(
                            read_field(inner, std::string(what) + ".x"),
                            read_field(inner, std::string(what) + ".y"));
                    });
            };
            StaticTables tables;
            tables.lrelu = read_pairs("tables.lrelu");
            tables.elu = read_pairs("tables.elu");
            tables.exp = read_pairs("tables.exp");
            tables.range = read_field_vector(reader);
            return tables;
        }

        void append_domain(
            std::vector<std::uint8_t> *out,
            const algebra::RootOfUnityDomain &domain)
        {
            append_string(out, domain.name);
            append_size(out, domain.size);
            append_field(out, domain.omega);
            append_field(out, domain.inv_size);
        }

        std::shared_ptr<algebra::RootOfUnityDomain> read_domain(Reader &reader)
        {
            const auto name = reader.string("domain.name");
            const auto size = reader.size("domain.size");
            if (size == 0 || (size & (size - 1U)) != 0U)
                throw std::runtime_error("verifier registration domain size is not a power of two");
            const auto omega = read_field(reader, "domain.omega");
            const auto inv_size = read_field(reader, "domain.inv_size");
            auto expected = algebra::RootOfUnityDomain::create(name, size);
            if (expected->omega != omega || expected->inv_size != inv_size
                || expected->name != name || expected->size != size)
                throw std::runtime_error("domain data is not canonical");
            return expected;
        }

        std::shared_ptr<algebra::RootOfUnityDomain> domain_ref(
            const std::unordered_map<std::string, std::shared_ptr<algebra::RootOfUnityDomain>> &domains,
            const std::string &name)
        {
            const auto it = domains.find(name);
            if (it == domains.end())
                throw std::runtime_error("unknown verifier registration domain reference: " + name);
            return it->second;
        }

        void append_domains(
            std::vector<std::uint8_t> *out,
            const WorkDomains &domains)
        {
            const std::array<const std::shared_ptr<algebra::RootOfUnityDomain> *, 7> top = {
                &domains.fh, &domains.edge, &domains.in, &domains.d,
                &domains.cat, &domains.c, &domains.n};
            const std::array<std::string_view, 7> names = {
                "FH", "edge", "in", "d", "cat", "C", "N"};
            for (std::size_t index = 0; index < top.size(); ++index)
            {
                const auto *domain = top[index];
                if (*domain == nullptr)
                    throw std::runtime_error("verifier registration has a missing work domain");
                if ((*domain)->name != names[index])
                    throw std::runtime_error(
                        "verifier registration has a non-canonical work domain name");
                append_domain(out, **domain);
            }
            append_vector(out, domains.hidden_layers, [&](const auto &layer)
                          {
                              append_vector(out, layer, [&](const auto &head)
                                            {
                                                if (head.in == nullptr || head.d == nullptr)
                                                    throw std::runtime_error("verifier registration has a missing head domain");
                                                append_string(out, head.in->name);
                                                append_string(out, head.d->name);
                                            });
                          });
            append_bool(out, domains.output_head.has_value());
            if (domains.output_head.has_value())
            {
                if (domains.output_head->in == nullptr || domains.output_head->d == nullptr)
                    throw std::runtime_error("verifier registration has a missing output domain");
                append_string(out, domains.output_head->in->name);
                append_string(out, domains.output_head->d->name);
            }
        }

        WorkDomains read_domains(Reader &reader)
        {
            std::array<std::shared_ptr<algebra::RootOfUnityDomain>, 7> top;
            const std::array<std::string_view, 7> names = {
                "FH", "edge", "in", "d", "cat", "C", "N"};
            std::unordered_map<std::string, std::shared_ptr<algebra::RootOfUnityDomain>> by_name;
            for (std::size_t i = 0; i < top.size(); ++i)
            {
                top[i] = read_domain(reader);
                if (top[i]->name != names[i])
                    throw std::runtime_error(
                        "verifier registration work-domain order/name mismatch");
                if (!by_name.emplace(top[i]->name, top[i]).second)
                    throw std::runtime_error("duplicate top-level verifier registration domain");
            }
            WorkDomains domains;
            domains.fh = top[0];
            domains.edge = top[1];
            domains.in = top[2];
            domains.d = top[3];
            domains.cat = top[4];
            domains.c = top[5];
            domains.n = top[6];
            domains.hidden_layers = read_vector<std::vector<AttentionHeadDomains>>(
                reader, kMaxVectorEntries,
                [&](Reader &inner)
                {
                    return read_vector<AttentionHeadDomains>(
                        inner, kMaxVectorEntries,
                        [&](Reader &head_reader)
                        {
                            return AttentionHeadDomains{
                                .in = domain_ref(by_name, head_reader.string("head-domain.in")),
                                .d = domain_ref(by_name, head_reader.string("head-domain.d"))};
                        });
                });
            const auto has_output = reader.u8("output-domain.present");
            if (has_output > 1)
                throw std::runtime_error("non-canonical output domain presence");
            if (has_output != 0)
            {
                domains.output_head = AttentionHeadDomains{
                    .in = domain_ref(by_name, reader.string("output-domain.in")),
                    .d = domain_ref(by_name, reader.string("output-domain.d"))};
            }
            return domains;
        }

        void append_srs(
            std::vector<std::uint8_t> *out,
            const VerifierSRSMetadata &srs,
            const crypto::KZGVerifierKey &vk)
        {
            append_string(out, srs.manifest_format);
            append_string(out, srs.srs_id);
            append_string(out, srs.curve_id);
            append_string(out, srs.field_modulus_id);
            append_string(out, srs.srs_type);
            append_string(out, srs.ceremony_transcript_hash);
            append_size(out, srs.degree_bound);
            append_size_map(out, srs.domain_sizes);
            append_string_map(out, srs.domain_generators);
            append_string(out, srs.root_derivation_id);
            append_string(out, srs.row_order);
            append_string(out, srs.protocol_compiler_artifact_hash);
            append_string(out, srs.max_instance_envelope_id);
            append_string(out, srs.max_instance_envelope_hash);
            append_size_map(out, srs.max_instance_envelope);
            append_string(out, srs.proof_schema_template_hash);
            append_string(out, srs.observation_ledger_hash);
            append_string(out, srs.random_identity_ledger_hash);
            append_string(out, srs.security_budget_ledger_hash);
            append_string(out, srs.vk_hash);
            append_string(out, srs.vanishing_g2_hash);
            append_string(out, srs.srs_audit_report_hash);
            append_string(out, srs.srs_consistency_hash);
            append_string(out, srs.manifest_hash);
            append_string(out, srs.generator_id);
            append_string(out, srs.encoding);
            append_string(out, srs.protocol_schema_id);
            append_string(out, srs.kzg_assumption_profile_id);
            append_size(out, srs.kzg_assumption_degree_bound);
            append_size(out, srs.kzg_assumption_security_bits);
            append_string(out, srs.kzg_assumption_domain_catalog_hash);
            append_string(out, srs.domain_generator_catalog_hash);
            append_string(out, srs.security_level);
            append_string(out, vk.srs_id);
            append_string(out, vk.srs_type);
            append_size(out, vk.degree_bound);
            append_point(out, vk.g1_generator);
            append_point(out, vk.g2_one);
            append_point(out, vk.g2_tau);
            std::vector<std::string> keys;
            keys.reserve(vk.domain_zero_g2.size());
            for (const auto &[key, _] : vk.domain_zero_g2)
            {
                (void)_;
                keys.push_back(key);
            }
            std::sort(keys.begin(), keys.end());
            append_size(out, keys.size());
            for (const auto &key : keys)
            {
                append_string(out, key);
                append_point(out, vk.domain_zero_g2.at(key));
            }
            append_string(out, vk.manifest_hash);
        }

        void read_srs(
            Reader &reader,
            VerifierSRSMetadata *srs,
            crypto::KZGVerifierKey *vk)
        {
            srs->manifest_format = reader.string("srs.manifest_format");
            srs->srs_id = reader.string("srs.srs_id");
            srs->curve_id = reader.string("srs.curve_id");
            srs->field_modulus_id = reader.string("srs.field_modulus_id");
            srs->srs_type = reader.string("srs.srs_type");
            srs->ceremony_transcript_hash = reader.string("srs.ceremony_transcript_hash");
            srs->degree_bound = reader.size("srs.degree_bound");
            srs->domain_sizes = read_size_map(reader, "srs.domain_sizes");
            srs->domain_generators = read_string_map(reader, "srs.domain_generators");
            srs->root_derivation_id = reader.string("srs.root_derivation_id");
            srs->row_order = reader.string("srs.row_order");
            srs->protocol_compiler_artifact_hash = reader.string("srs.protocol_compiler_artifact_hash");
            srs->max_instance_envelope_id = reader.string("srs.max_instance_envelope_id");
            srs->max_instance_envelope_hash = reader.string("srs.max_instance_envelope_hash");
            srs->max_instance_envelope = read_size_map(reader, "srs.max_instance_envelope");
            srs->proof_schema_template_hash = reader.string("srs.proof_schema_template_hash");
            srs->observation_ledger_hash = reader.string("srs.observation_ledger_hash");
            srs->random_identity_ledger_hash = reader.string("srs.random_identity_ledger_hash");
            srs->security_budget_ledger_hash = reader.string("srs.security_budget_ledger_hash");
            srs->vk_hash = reader.string("srs.vk_hash");
            srs->vanishing_g2_hash = reader.string("srs.vanishing_g2_hash");
            srs->srs_audit_report_hash = reader.string("srs.srs_audit_report_hash");
            srs->srs_consistency_hash = reader.string("srs.srs_consistency_hash");
            srs->manifest_hash = reader.string("srs.manifest_hash");
            srs->generator_id = reader.string("srs.generator_id");
            srs->encoding = reader.string("srs.encoding");
            srs->protocol_schema_id = reader.string("srs.protocol_schema_id");
            srs->kzg_assumption_profile_id = reader.string("srs.assumption_profile");
            srs->kzg_assumption_degree_bound = reader.size("srs.assumption_degree_bound");
            srs->kzg_assumption_security_bits = reader.size("srs.assumption_security_bits");
            srs->kzg_assumption_domain_catalog_hash = reader.string("srs.assumption_domain_catalog_hash");
            srs->domain_generator_catalog_hash = reader.string("srs.domain_generator_catalog_hash");
            srs->security_level = reader.string("srs.security_level");
            vk->srs_id = reader.string("vk.srs_id");
            vk->srs_type = reader.string("vk.srs_type");
            vk->degree_bound = reader.size("vk.degree_bound");
            vk->g1_generator = read_g1(reader, "vk.g1_generator");
            vk->g2_one = read_g2(reader, "vk.g2_one");
            vk->g2_tau = read_g2(reader, "vk.g2_tau");
            const auto count = reader.size("vk.domain_zero_g2.count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("verifier registration VK domain count exceeds cap");
            std::string previous;
            for (std::size_t i = 0; i < count; ++i)
            {
                auto key = reader.string("vk.domain_zero_g2.name");
                if (key.empty() || (!previous.empty() && key <= previous))
                    throw std::runtime_error("verifier registration VK domain order is not canonical");
                previous = key;
                if (!vk->domain_zero_g2.emplace(key, read_g2(reader, "vk.domain_zero_g2.point")).second)
                    throw std::runtime_error("duplicate verifier registration VK domain");
            }
            vk->manifest_hash = reader.string("vk.manifest_hash");
            vk->g2_one_prepared = std::make_shared<crypto::PreparedG2>(crypto::prepare_g2(vk->g2_one));
            vk->g2_tau_prepared = std::make_shared<crypto::PreparedG2>(crypto::prepare_g2(vk->g2_tau));
        }

        std::vector<std::pair<std::string, VerifierPublicPolynomialDescriptor>> sorted_polynomial_descriptors(
            const std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> &descriptors)
        {
            std::vector<std::pair<std::string, VerifierPublicPolynomialDescriptor>> sorted;
            sorted.reserve(descriptors.size());
            for (const auto &[key, descriptor] : descriptors)
                sorted.emplace_back(key, descriptor);
            std::sort(sorted.begin(), sorted.end(), [](const auto &lhs, const auto &rhs)
                      { return lhs.first < rhs.first; });
            return sorted;
        }

        void append_polynomial_descriptors(
            std::vector<std::uint8_t> *out,
            const std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> &descriptors,
            const WorkDomains &domains)
        {
            auto sorted = sorted_polynomial_descriptors(descriptors);
            append_size(out, sorted.size());
            std::string previous;
            const auto domain_size = [&](const std::string &name)
                -> std::optional<std::size_t>
            {
                for (const auto &domain : {
                         domains.fh, domains.edge, domains.in, domains.d,
                         domains.cat, domains.c, domains.n})
                {
                    if (domain != nullptr && domain->name == name)
                        return domain->size;
                }
                return std::nullopt;
            };
            for (const auto &[key, descriptor] : sorted)
            {
                if (key.empty() || (!previous.empty() && key <= previous)
                    || descriptor.name != key || descriptor.size == 0)
                    throw std::runtime_error("public polynomial descriptor order/name mismatch");
                if (descriptor.basis == algebra::PolynomialBasis::Evaluation)
                {
                    const auto size = domain_size(descriptor.domain);
                    if (!size.has_value() || descriptor.size != *size)
                        throw std::runtime_error(
                            "public polynomial descriptor has unknown domain or size mismatch");
                }
                else if (!descriptor.domain.empty())
                {
                    throw std::runtime_error("coefficient descriptor has a domain reference");
                }
                previous = key;
                append_string(out, key);
                append_u8(out, static_cast<std::uint8_t>(descriptor.basis));
                append_string(out, descriptor.domain);
                append_size(out, descriptor.size);
            }
        }

        std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> read_polynomial_descriptors(
            Reader &reader,
            const WorkDomains &domains)
        {
            std::unordered_map<std::string, std::shared_ptr<algebra::RootOfUnityDomain>> by_name = {
                {domains.fh->name, domains.fh}, {domains.edge->name, domains.edge},
                {domains.in->name, domains.in}, {domains.d->name, domains.d},
                {domains.cat->name, domains.cat}, {domains.c->name, domains.c},
                {domains.n->name, domains.n}};
            const auto count = reader.size("public-polynomials.count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("public polynomial count exceeds cap");
            std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> out;
            out.reserve(count);
            std::string previous;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto key = reader.string("public-polynomial.name");
                if (key.empty() || (!previous.empty() && key <= previous))
                    throw std::runtime_error("public polynomial order is not canonical");
                previous = key;
                const auto basis = reader.u8("public-polynomial.basis");
                if (basis > static_cast<std::uint8_t>(algebra::PolynomialBasis::Evaluation))
                    throw std::runtime_error("invalid public polynomial basis");
                const auto domain_name = reader.string("public-polynomial.domain");
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                if (basis == static_cast<std::uint8_t>(algebra::PolynomialBasis::Evaluation))
                {
                    if (domain_name.empty() || !by_name.contains(domain_name))
                        throw std::runtime_error("public evaluation polynomial has unknown domain");
                    domain = by_name.at(domain_name);
                }
                else if (!domain_name.empty())
                {
                    throw std::runtime_error("coefficient polynomial has a domain reference");
                }
                const auto size = reader.size("public-polynomial.size");
                if (size == 0)
                    throw std::runtime_error("public polynomial descriptor size is zero");
                if (domain != nullptr && size != domain->size)
                    throw std::runtime_error("public polynomial descriptor size/domain mismatch");
                VerifierPublicPolynomialDescriptor descriptor{
                    .name = key,
                    .basis = static_cast<algebra::PolynomialBasis>(basis),
                    .domain = domain_name,
                    .size = size};
                if (!out.emplace(key, std::move(descriptor)).second)
                    throw std::runtime_error("duplicate public polynomial");
            }
            return out;
        }

        void append_point_map(
            std::vector<std::uint8_t> *out,
            const std::unordered_map<std::string, VerifierPointCommitment> &commitments)
        {
            std::vector<std::string> keys;
            keys.reserve(commitments.size());
            for (const auto &[key, commitment] : commitments)
            {
                if (key.empty() || commitment.name != key)
                    throw std::runtime_error("point-only commitment map name mismatch");
                keys.push_back(key);
            }
            std::sort(keys.begin(), keys.end());
            append_size(out, keys.size());
            for (const auto &key : keys)
            {
                append_string(out, key);
                append_point(out, commitments.at(key).point);
            }
        }

        std::unordered_map<std::string, VerifierPointCommitment> read_point_map(Reader &reader)
        {
            const auto count = reader.size("point-map.count");
            if (count > kMaxVectorEntries)
                throw std::runtime_error("point-only commitment map exceeds cap");
            std::unordered_map<std::string, VerifierPointCommitment> out;
            out.reserve(count);
            std::string previous;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto key = reader.string("point-map.name");
                if (key.empty() || (!previous.empty() && key <= previous))
                    throw std::runtime_error("point-only commitment map order is not canonical");
                previous = key;
                const auto point = read_g1(reader, "point-map.point");
                if (!out.emplace(key, VerifierPointCommitment{key, point}).second)
                    throw std::runtime_error("duplicate point-only commitment");
            }
            return out;
        }

        void append_proof_schema_entry(
            std::vector<std::uint8_t> *out,
            const ProofSchemaEntry &entry)
        {
            append_string(out, entry.tag);
            append_string(out, entry.domain);
            append_string(out, entry.owner_block);
            append_string(out, entry.field_type);
            append_string(out, entry.padding_mode);
            append_string(out, entry.owner_policy);
            append_u8(out, static_cast<std::uint8_t>(entry.visibility));
            append_size(out, entry.layer_id);
            append_size(out, entry.head_id);
            append_string(out, entry.opening_policy);
            append_string(out, entry.binding_mode);
            append_string(out, entry.blinding_basis);
            append_bool(out, entry.enters_commitment);
            append_bool(out, entry.enters_opening);
            append_bool(out, entry.enters_quotient);
        }

        ProofSchemaEntry read_proof_schema_entry(Reader &reader)
        {
            ProofSchemaEntry entry;
            entry.tag = reader.string("proof-schema.tag");
            entry.domain = reader.string("proof-schema.domain");
            entry.owner_block = reader.string("proof-schema.owner-block");
            entry.field_type = reader.string("proof-schema.field-type");
            entry.padding_mode = reader.string("proof-schema.padding-mode");
            entry.owner_policy = reader.string("proof-schema.owner-policy");
            const auto visibility = reader.u8("proof-schema.visibility");
            if (visibility > static_cast<std::uint8_t>(PolynomialVisibility::PrivateWitness))
                throw std::runtime_error("invalid proof schema visibility");
            entry.visibility = static_cast<PolynomialVisibility>(visibility);
            entry.layer_id = reader.size("proof-schema.layer-id");
            entry.head_id = reader.size("proof-schema.head-id");
            entry.opening_policy = reader.string("proof-schema.opening-policy");
            entry.binding_mode = reader.string("proof-schema.binding-mode");
            entry.blinding_basis = reader.string("proof-schema.blinding-basis");
            const auto c = reader.u8("proof-schema.enters-commitment");
            const auto o = reader.u8("proof-schema.enters-opening");
            const auto q = reader.u8("proof-schema.enters-quotient");
            if (c > 1 || o > 1 || q > 1)
                throw std::runtime_error("non-canonical proof schema boolean");
            entry.enters_commitment = c != 0;
            entry.enters_opening = o != 0;
            entry.enters_quotient = q != 0;
            return entry;
        }

        void append_ledger_entry(
            std::vector<std::uint8_t> *out,
            const CompilerLedgerEntry &entry)
        {
            append_string(out, entry.id);
            append_string(out, entry.object_or_challenge);
            append_string(out, entry.domain);
            append_string(out, entry.transcript_stage);
            append_string(out, entry.expression_or_certificate);
            append_size(out, entry.degree_bound);
            append_size(out, entry.query_bound);
            append_size(out, entry.exclusion_degree);
            append_size(out, entry.multiplicity);
        }

        CompilerLedgerEntry read_ledger_entry(Reader &reader)
        {
            return CompilerLedgerEntry{
                .id = reader.string("ledger.id"),
                .object_or_challenge = reader.string("ledger.object"),
                .domain = reader.string("ledger.domain"),
                .transcript_stage = reader.string("ledger.stage"),
                .expression_or_certificate = reader.string("ledger.expression"),
                .degree_bound = reader.size("ledger.degree-bound"),
                .query_bound = reader.size("ledger.query-bound"),
                .exclusion_degree = reader.size("ledger.exclusion-degree"),
                .multiplicity = reader.size("ledger.multiplicity")};
        }

        void append_integer_bound(
            std::vector<std::uint8_t> *out,
            const IntegerBoundLedgerEntry &entry)
        {
            append_string(out, entry.id);
            append_string(out, entry.stage);
            append_string(out, entry.source);
            append_u64(out, entry.absolute_bound);
            append_size(out, entry.limb_count);
        }

        IntegerBoundLedgerEntry read_integer_bound(Reader &reader)
        {
            return IntegerBoundLedgerEntry{
                .id = reader.string("integer-bound.id"),
                .stage = reader.string("integer-bound.stage"),
                .source = reader.string("integer-bound.source"),
                .absolute_bound = reader.u64("integer-bound.absolute-bound"),
                .limb_count = reader.size("integer-bound.limb-count")};
        }

        void append_observation_edge(
            std::vector<std::uint8_t> *out,
            const ObservationDependencyEdge &edge)
        {
            append_string(out, edge.id);
            append_string(out, edge.derived_object);
            append_string_vector(out, edge.source_objects);
            append_string_vector(out, edge.serialized_observations);
            append_string(out, edge.transcript_stage);
            append_string(out, edge.propagation_rule);
            append_bool(out, edge.introduces_independent_randomness);
        }

        ObservationDependencyEdge read_observation_edge(Reader &reader)
        {
            const auto id = reader.string("observation-edge.id");
            ObservationDependencyEdge edge;
            edge.id = id;
            edge.derived_object = reader.string("observation-edge.derived-object");
            edge.source_objects = read_vector<std::string>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return inner.string("observation-edge.source"); });
            edge.serialized_observations = read_vector<std::string>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return inner.string("observation-edge.observation"); });
            edge.transcript_stage = reader.string("observation-edge.stage");
            edge.propagation_rule = reader.string("observation-edge.rule");
            const auto random = reader.u8("observation-edge.randomness");
            if (random > 1)
                throw std::runtime_error("non-canonical observation edge boolean");
            edge.introduces_independent_randomness = random != 0;
            return edge;
        }

        void append_compiler(
            std::vector<std::uint8_t> *out,
            const ProtocolCompilerArtifact &artifact)
        {
            append_vector(out, artifact.proof_schema.entries, [&](const auto &entry)
                          { append_proof_schema_entry(out, entry); });
            append_size(out, artifact.proof_schema.private_plain_value_count);
            auto append_string_vec = [&](const auto &values)
            {
                append_string_vector(out, values);
            };
            append_string_vec(artifact.binding_relation_registry);
            append_string_vec(artifact.proof_blocks);
            append_string_vec(artifact.serializer_tag_table);
            append_string_vec(artifact.object_canonical_order);
            append_string_vec(artifact.transcript_stages);
            append_string_vec(artifact.verifier_equation_registry);
            append_string_vec(artifact.ledger_templates);
            append_string_vec(artifact.constraint_registry);
            append_string_vec(artifact.point_group_registry_template);
            append_string_vec(artifact.degree_registry);
            append_string_vec(artifact.padding_registry);
            append_vector(out, artifact.observation_dependency_dag, [&](const auto &edge)
                          { append_observation_edge(out, edge); });
            append_vector(out, artifact.observation_ledger, [&](const auto &entry)
                          { append_ledger_entry(out, entry); });
            append_vector(out, artifact.random_identity_ledger, [&](const auto &entry)
                          { append_ledger_entry(out, entry); });
            append_vector(out, artifact.security_budget_ledger, [&](const auto &entry)
                          { append_ledger_entry(out, entry); });
            append_vector(out, artifact.integer_bound_ledger, [&](const auto &entry)
                          { append_integer_bound(out, entry); });
            append_string(out, artifact.kzg_assumption_profile);
            append_string(out, artifact.shift_algorithm_id);
            append_string(out, artifact.shift_algorithm_spec);
            append_blob(out, artifact.protocol_canonical_bytes);
            append_blob(out, artifact.instance_canonical_bytes);
        }

        ProtocolCompilerArtifact read_compiler(Reader &reader)
        {
            ProtocolCompilerArtifact artifact;
            artifact.proof_schema.entries = read_vector<ProofSchemaEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_proof_schema_entry(inner); });
            artifact.proof_schema.private_plain_value_count = reader.size("compiler.private-plain-count");
            auto read_string_vec = [&](std::string_view what)
            {
                return read_vector<std::string>(
                    reader, kMaxVectorEntries,
                    [&](Reader &inner) { return inner.string(what); });
            };
            artifact.binding_relation_registry = read_string_vec("compiler.binding-relation");
            artifact.proof_blocks = read_string_vec("compiler.proof-block");
            artifact.serializer_tag_table = read_string_vec("compiler.serializer-tag");
            artifact.object_canonical_order = read_string_vec("compiler.object-order");
            artifact.transcript_stages = read_string_vec("compiler.transcript-stage");
            artifact.verifier_equation_registry = read_string_vec("compiler.verifier-equation");
            artifact.ledger_templates = read_string_vec("compiler.ledger-template");
            artifact.constraint_registry = read_string_vec("compiler.constraint");
            artifact.point_group_registry_template = read_string_vec("compiler.point-group");
            artifact.degree_registry = read_string_vec("compiler.degree");
            artifact.padding_registry = read_string_vec("compiler.padding");
            artifact.observation_dependency_dag = read_vector<ObservationDependencyEdge>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_observation_edge(inner); });
            artifact.observation_ledger = read_vector<CompilerLedgerEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_ledger_entry(inner); });
            artifact.random_identity_ledger = read_vector<CompilerLedgerEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_ledger_entry(inner); });
            artifact.security_budget_ledger = read_vector<CompilerLedgerEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_ledger_entry(inner); });
            artifact.integer_bound_ledger = read_vector<IntegerBoundLedgerEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_integer_bound(inner); });
            artifact.kzg_assumption_profile = reader.string("compiler.kzg-profile");
            artifact.shift_algorithm_id = reader.string("compiler.shift-id");
            artifact.shift_algorithm_spec = reader.string("compiler.shift-spec");
            artifact.protocol_canonical_bytes = reader.blob("compiler.protocol-canonical");
            artifact.instance_canonical_bytes = reader.blob("compiler.instance-canonical");
            return artifact;
        }

        void append_registry_entry(
            std::vector<std::uint8_t> *out,
            const VerifierRegistryEntry &entry)
        {
            append_string(out, entry.label);
            append_string(out, entry.domain);
            append_size(out, entry.degree_bound);
            append_u64(out, entry.scale);
            append_u64(out, static_cast<std::uint64_t>(entry.range_min));
            append_u64(out, static_cast<std::uint64_t>(entry.range_max));
            append_point(out, entry.point);
            append_bool(out, entry.bound_certificate.present);
            if (entry.bound_certificate.present)
            {
                const auto &certificate = entry.bound_certificate;
                append_string(out, certificate.label);
                append_size(out, certificate.layer_index);
                append_size(out, certificate.local_head_index);
                append_string(out, certificate.domain);
                append_size(out, certificate.degree_bound);
                append_u64(out, certificate.scale);
                append_u64(out, certificate.absolute_bound);
            }
        }

        VerifierRegistryEntry read_registry_entry(Reader &reader)
        {
            VerifierRegistryEntry entry{
                .label = reader.string("registry-entry.label"),
                .domain = reader.string("registry-entry.domain"),
                .degree_bound = reader.size("registry-entry.degree-bound"),
                .scale = reader.u64("registry-entry.scale"),
                .range_min = static_cast<std::int64_t>(reader.u64("registry-entry.range-min")),
                .range_max = static_cast<std::int64_t>(reader.u64("registry-entry.range-max")),
                .point = read_g1(reader, "registry-entry.point")};
            const auto bound_present =
                reader.u8("registry-entry.bound-present");
            if (bound_present > 1U)
                throw std::runtime_error(
                    "non-canonical model-bound certificate presence flag");
            entry.bound_certificate.present = bound_present != 0U;
            if (entry.bound_certificate.present)
            {
                entry.bound_certificate.label =
                    reader.string("registry-entry.bound-label");
                entry.bound_certificate.layer_index =
                    reader.size("registry-entry.bound-layer");
                entry.bound_certificate.local_head_index =
                    reader.size("registry-entry.bound-head");
                entry.bound_certificate.domain =
                    reader.string("registry-entry.bound-domain");
                entry.bound_certificate.degree_bound =
                    reader.size("registry-entry.bound-degree");
                entry.bound_certificate.scale =
                    reader.u64("registry-entry.bound-scale");
                entry.bound_certificate.absolute_bound =
                    reader.u64("registry-entry.bound-absolute");
            }
            return entry;
        }

        void append_registry(
            std::vector<std::uint8_t> *out,
            const AuthenticatedVerifierRegistry &registry)
        {
            append_string(out, registry.format);
            append_string(out, registry.registry_id);
            append_string(out, registry.model_arch_id);
            append_string(out, registry.model_param_id);
            append_string(out, registry.input_preprocess_id);
            append_string(out, registry.quant_cfg_id);
            append_string(out, registry.static_table_id);
            append_string(out, registry.srs_id);
            append_string(out, registry.protocol_schema_id);
            append_string(out, registry.instance_schema_hash);
            append_string(out, registry.srs_manifest_hash);
            append_size(out, registry.srs_degree_bound);
            append_point(out, registry.srs_g1_generator);
            append_string(out, registry.srs_generator_id);
            append_string(out, registry.srs_encoding);
            append_string(out, registry.srs_spec_id);
            // Keep the SRS ceremony's compiler-artifact hash as a distinct
            // authenticated field.  It binds this registry to the exact
            // protocol compiler file copied into the SRS directory; it must
            // not be conflated with the per-instance typed artifact digest.
            append_string(out, registry.srs_protocol_compiler_artifact_hash);
            append_string(out, registry.compiler_artifact_digest);
            append_string(out, registry.model_layout_digest);
            append_string(out, registry.quant_chain_digest);
            append_string(out, registry.static_table_digest);
            append_string(out, registry.static_table_generator_id);
            append_string(out, registry.static_table_spec_hash);
            append_string(out, registry.range_and_scale_certificate);
            append_string(out, registry.integer_bound_ledger_digest);
            append_string(out, registry.public_input_digest);
            append_string(out, registry.topology_digest);
            append_string(out, registry.security_budget_ledger_digest);
            append_string(out, registry.security_budget_exact);
            append_vector(out, registry.integer_bound_ledger, [&](const auto &entry)
                          { append_integer_bound(out, entry); });
            append_vector(out, registry.model_entries, [&](const auto &entry)
                          { append_registry_entry(out, entry); });
            append_vector(out, registry.static_entries, [&](const auto &entry)
                          { append_registry_entry(out, entry); });
        }

        AuthenticatedVerifierRegistry read_registry(Reader &reader)
        {
            AuthenticatedVerifierRegistry registry;
            registry.format = reader.string("registry.format");
            registry.registry_id = reader.string("registry.id");
            registry.model_arch_id = reader.string("registry.model-arch");
            registry.model_param_id = reader.string("registry.model-param");
            registry.input_preprocess_id = reader.string("registry.input-preprocess");
            registry.quant_cfg_id = reader.string("registry.quant");
            registry.static_table_id = reader.string("registry.static-table");
            registry.srs_id = reader.string("registry.srs-id");
            registry.protocol_schema_id = reader.string("registry.protocol-schema");
            registry.instance_schema_hash = reader.string("registry.instance-schema");
            registry.srs_manifest_hash = reader.string("registry.manifest-hash");
            registry.srs_degree_bound = reader.size("registry.degree-bound");
            registry.srs_g1_generator = read_g1(reader, "registry.g1-generator");
            registry.srs_generator_id = reader.string("registry.generator-id");
            registry.srs_encoding = reader.string("registry.encoding");
            registry.srs_spec_id = reader.string("registry.spec-id");
            registry.srs_protocol_compiler_artifact_hash =
                reader.string("registry.srs-protocol-compiler-artifact-hash");
            registry.compiler_artifact_digest = reader.string("registry.compiler-digest");
            registry.model_layout_digest = reader.string("registry.model-layout-digest");
            registry.quant_chain_digest = reader.string("registry.quant-digest");
            registry.static_table_digest = reader.string("registry.static-digest");
            registry.static_table_generator_id = reader.string("registry.table-generator-id");
            registry.static_table_spec_hash = reader.string("registry.table-spec-hash");
            registry.range_and_scale_certificate = reader.string("registry.range-scale-certificate");
            registry.integer_bound_ledger_digest = reader.string("registry.integer-bound-digest");
            registry.public_input_digest = reader.string("registry.public-input-digest");
            registry.topology_digest = reader.string("registry.topology-digest");
            registry.security_budget_ledger_digest = reader.string("registry.security-digest");
            registry.security_budget_exact = reader.string("registry.security-exact");
            registry.integer_bound_ledger = read_vector<IntegerBoundLedgerEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_integer_bound(inner); });
            registry.model_entries = read_vector<VerifierRegistryEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_registry_entry(inner); });
            registry.static_entries = read_vector<VerifierRegistryEntry>(
                reader, kMaxVectorEntries,
                [](Reader &inner) { return read_registry_entry(inner); });
            return registry;
        }

        std::vector<std::uint8_t> make_section(
            Section section,
            const std::function<void(std::vector<std::uint8_t> *)> &writer)
        {
            std::vector<std::uint8_t> payload;
            writer(&payload);
            if (payload.size() > kMaxSectionBytes)
                throw std::runtime_error("verifier registration section exceeds cap");
            std::vector<std::uint8_t> out;
            append_u32(&out, static_cast<std::uint32_t>(section));
            append_u64(&out, payload.size());
            out.insert(out.end(), payload.begin(), payload.end());
            return out;
        }

        void parse_section(
            Reader &section,
            Section expected,
            VerifierRegistrationArtifact *artifact)
        {
            switch (expected)
            {
            case Section::Config: artifact->config = read_config(section); break;
            case Section::Dataset: artifact->dataset = read_dataset(section); break;
            case Section::LocalGraph: artifact->local = read_local(section); break;
            case Section::ModelShape: artifact->model = read_model_shape(section); break;
            case Section::StaticTables: artifact->tables = read_tables(section); break;
            case Section::Domains: artifact->domains = read_domains(section); break;
            case Section::SrsAndVk: read_srs(section, &artifact->srs, &artifact->kzg_vk); break;
            case Section::PublicPolynomials: artifact->public_polynomial_descriptors = read_polynomial_descriptors(section, artifact->domains); break;
            case Section::PublicCommitments: artifact->public_commitments = read_point_map(section); break;
            case Section::StaticCommitments: artifact->static_commitments = read_point_map(section); break;
            case Section::CompilerArtifact: artifact->compiler_artifact = read_compiler(section); break;
            case Section::Registry: artifact->registry = read_registry(section); break;
            }
            section.require_empty("section");
        }
    }

    std::vector<std::uint8_t> VerifierRegistrationArtifact::canonical_bytes() const
    {
        std::vector<std::uint8_t> out;
        out.insert(out.end(), kMagic.begin(), kMagic.end());
        append_u32(&out, kSectionCount);
        const std::array<std::vector<std::uint8_t>, kSectionCount> sections = {
            make_section(Section::Config, [&](auto *payload) { append_config(payload, config); }),
            make_section(Section::Dataset, [&](auto *payload) { append_dataset(payload, dataset); }),
            make_section(Section::LocalGraph, [&](auto *payload) { append_local(payload, local); }),
            make_section(Section::ModelShape, [&](auto *payload) { append_model_shape(payload, model); }),
            make_section(Section::StaticTables, [&](auto *payload) { append_tables(payload, tables); }),
            make_section(Section::Domains, [&](auto *payload) { append_domains(payload, domains); }),
            make_section(Section::SrsAndVk, [&](auto *payload) { append_srs(payload, srs, kzg_vk); }),
            make_section(Section::PublicPolynomials, [&](auto *payload) { append_polynomial_descriptors(payload, public_polynomial_descriptors, domains); }),
            make_section(Section::PublicCommitments, [&](auto *payload) { append_point_map(payload, public_commitments); }),
            make_section(Section::StaticCommitments, [&](auto *payload) { append_point_map(payload, static_commitments); }),
            make_section(Section::CompilerArtifact, [&](auto *payload) { append_compiler(payload, compiler_artifact); }),
            make_section(Section::Registry, [&](auto *payload) { append_registry(payload, registry); }),
        };
        for (const auto &section : sections)
            out.insert(out.end(), section.begin(), section.end());
        const auto digest = sha256_raw(out);
        out.insert(out.end(), digest.begin(), digest.end());
        if (out.size() > kMaxArtifactBytes)
            throw std::runtime_error("verifier registration artifact exceeds cap");
        return out;
    }

    std::string VerifierRegistrationArtifact::canonical_digest() const
    {
        const auto bytes = canonical_bytes();
        return sha256_hex(bytes);
    }

    std::vector<std::uint8_t> serialize_verifier_registration_artifact(
        const VerifierRegistrationArtifact &artifact)
    {
        return artifact.canonical_bytes();
    }

    VerifierRegistrationArtifact parse_verifier_registration_artifact(
        const std::vector<std::uint8_t> &bytes)
    {
        if (bytes.size() < kMagic.size() + sizeof(std::uint32_t) + SHA256_DIGEST_LENGTH
            || bytes.size() > kMaxArtifactBytes)
            throw std::runtime_error("verifier registration artifact size is invalid");
        const auto body_size = bytes.size() - SHA256_DIGEST_LENGTH;
        if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
            throw std::runtime_error("verifier registration artifact magic mismatch");
        const auto expected = sha256_raw(std::span<const std::uint8_t>(bytes.data(), body_size));
        if (!std::equal(expected.begin(), expected.end(), bytes.begin() + body_size))
            throw std::runtime_error("verifier registration artifact digest mismatch");
        Reader reader(std::span<const std::uint8_t>(bytes.data(), body_size));
        const auto magic = reader.take(kMagic.size(), "header.magic");
        if (!std::equal(magic.begin(), magic.end(), kMagic.begin()))
            throw std::runtime_error("verifier registration artifact magic mismatch");
        if (reader.u32("header.section-count") != kSectionCount)
            throw std::runtime_error("verifier registration artifact section count mismatch");
        VerifierRegistrationArtifact artifact;
        for (std::uint32_t section_id = 1; section_id <= kSectionCount; ++section_id)
        {
            const auto actual = reader.u32("section.id");
            if (actual != section_id)
                throw std::runtime_error("verifier registration section order/duplicate mismatch");
            const auto length = reader.size("section.length");
            if (length > kMaxSectionBytes)
                throw std::runtime_error("verifier registration section exceeds cap");
            Reader section(reader.take(length, "section.payload"));
            parse_section(section, static_cast<Section>(section_id), &artifact);
        }
        reader.require_empty("artifact");
        const auto canonical = artifact.canonical_bytes();
        if (canonical != bytes)
            throw std::runtime_error("verifier registration artifact is not canonical");
        return artifact;
    }
}
