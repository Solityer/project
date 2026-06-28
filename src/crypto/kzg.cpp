#include "gatzk/crypto/kzg.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include <openssl/sha.h>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::crypto
{
    namespace
    {
        using algebra::FieldElement;
        using Clock = std::chrono::steady_clock;

        std::string bytes_to_hex(const unsigned char* bytes, std::size_t size)
        {
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (std::size_t i = 0; i < size; ++i)
            {
                out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
            }
            return out.str();
        }

        std::string sha256_hex(const std::string& data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
        }

        std::string sha256_bytes_hex(const std::vector<std::uint8_t>& data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(data.data(), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
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

        std::string manifest_object(const std::string& text, const std::string& key)
        {
            const auto quoted_key = "\"" + key + "\"";
            const auto pos = text.find(quoted_key);
            if (pos == std::string::npos)
            {
                return {};
            }
            const auto open = text.find('{', pos + quoted_key.size());
            if (open == std::string::npos)
            {
                return {};
            }
            std::size_t depth = 0;
            for (std::size_t i = open; i < text.size(); ++i)
            {
                if (text[i] == '{')
                {
                    ++depth;
                }
                else if (text[i] == '}')
                {
                    --depth;
                    if (depth == 0)
                    {
                        return text.substr(open + 1, i - open - 1);
                    }
                }
            }
            throw std::runtime_error("unterminated SRS manifest object: " + key);
        }

        std::unordered_map<std::string, std::string> parse_string_map(const std::string& object)
        {
            std::unordered_map<std::string, std::string> out;
            std::size_t pos = 0;
            while (true)
            {
                const auto key_begin = object.find('"', pos);
                if (key_begin == std::string::npos)
                {
                    break;
                }
                const auto key_end = object.find('"', key_begin + 1);
                const auto value_begin = object.find('"', object.find(':', key_end) + 1);
                const auto value_end = object.find('"', value_begin + 1);
                if (key_end == std::string::npos || value_begin == std::string::npos || value_end == std::string::npos)
                {
                    throw std::runtime_error("invalid string map in SRS manifest");
                }
                out.emplace(object.substr(key_begin + 1, key_end - key_begin - 1),
                            object.substr(value_begin + 1, value_end - value_begin - 1));
                pos = value_end + 1;
            }
            return out;
        }

        std::unordered_map<std::string, std::size_t> parse_size_map(const std::string& object)
        {
            std::unordered_map<std::string, std::size_t> out;
            std::size_t pos = 0;
            while (true)
            {
                const auto key_begin = object.find('"', pos);
                if (key_begin == std::string::npos)
                {
                    break;
                }
                const auto key_end = object.find('"', key_begin + 1);
                const auto colon = object.find(':', key_end);
                if (key_end == std::string::npos || colon == std::string::npos)
                {
                    throw std::runtime_error("invalid size map in SRS manifest");
                }
                const auto number_begin = object.find_first_of("0123456789", colon + 1);
                const auto number_end = object.find_first_not_of("0123456789", number_begin);
                if (number_begin == std::string::npos)
                {
                    throw std::runtime_error("invalid numeric value in SRS manifest");
                }
                out.emplace(object.substr(key_begin + 1, key_end - key_begin - 1),
                            static_cast<std::size_t>(std::stoull(object.substr(number_begin, number_end - number_begin))));
                pos = number_end;
            }
            return out;
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
            manifest.pk_coef_path = manifest_scalar(text, "pk_coef_path");
            manifest.pk_coef_hash = manifest_scalar(text, "pk_coef_hash");
            manifest.pk_lagrange_paths = parse_string_map(manifest_object(text, "pk_lagrange_paths"));
            manifest.pk_lagrange_hashes = parse_string_map(manifest_object(text, "pk_lagrange_hashes"));
            manifest.domain_sizes = parse_size_map(manifest_object(text, "domain_sizes"));
            manifest.vk_path = manifest_scalar(text, "vk_path");
            manifest.vk_hash = manifest_scalar(text, "vk_hash");
            manifest.encoding = manifest_scalar(text, "encoding");
            manifest.generator_version = manifest_scalar(text, "generator_version");
            manifest.security_level = manifest_scalar(text, "security_level");
            manifest.manifest_hash = sha256_hex(text);
            const auto degree_bound = manifest_scalar(text, "degree_bound");
            if (!degree_bound.empty())
            {
                manifest.degree_bound = static_cast<std::size_t>(std::stoull(degree_bound));
            }
            if (manifest.srs_id.empty() || manifest.srs_type.empty() || manifest.degree_bound == 0
                || manifest.pk_coef_path.empty() || manifest.pk_lagrange_paths.empty() || manifest.vk_path.empty())
            {
                throw std::runtime_error("SRS manifest is missing required material fields");
            }
            return manifest;
        }

        void validate_material_hash_value(
            const std::string& actual_hash,
            const std::string& expected_hash,
            const std::string& label)
        {
            if (expected_hash.empty())
            {
                throw std::runtime_error("SRS manifest is missing material hash: " + label);
            }
            if (actual_hash != expected_hash)
            {
                throw std::runtime_error("SRS material hash mismatch for " + label);
            }
        }

        std::uint64_t read_u64_le(std::istream& input, const std::string& label)
        {
            std::array<std::uint8_t, 8> bytes{};
            input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
            {
                throw std::runtime_error("truncated SRS file while reading " + label);
            }
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return out;
        }

        std::uint64_t read_u64_le_bytes(
            const std::vector<std::uint8_t>& bytes,
            std::size_t& offset,
            const std::string& label)
        {
            if (offset + 8 > bytes.size())
            {
                throw std::runtime_error("truncated SRS file while reading " + label);
            }
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < 8; ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
            }
            offset += 8;
            return out;
        }

        std::uint64_t parse_u64_le_bytes(const std::array<std::uint8_t, 8>& bytes)
        {
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path& path,
            std::string* actual_hash,
            std::size_t max_points);

        std::vector<G1Point> load_raw_g1_vector_file_prefix(
            const std::filesystem::path& path,
            std::streamoff file_size,
            std::size_t max_points,
            std::string* actual_hash)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS G1 material: " + path.string());
            }
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            auto read_hashed = [&](char* out, std::size_t size, const std::string& label) {
                input.read(out, static_cast<std::streamsize>(size));
                if (input.gcount() != static_cast<std::streamsize>(size))
                {
                    throw std::runtime_error("truncated SRS G1 material while reading " + label + ": " + path.string());
                }
                SHA256_Update(&ctx, out, size);
            };

            std::array<char, 8> magic{};
            read_hashed(magic.data(), magic.size(), "G1 magic");
            const std::array<char, 8> expected_raw_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
            if (magic != expected_raw_magic)
            {
                return load_g1_vector_file(path, actual_hash, 0);
            }
            std::array<std::uint8_t, 8> count_bytes{};
            read_hashed(reinterpret_cast<char*>(count_bytes.data()), count_bytes.size(), "G1 point count");
            const auto count = parse_u64_le_bytes(count_bytes);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("SRS G1 material is too large for this platform");
            }
            std::array<std::uint8_t, 8> raw_size_bytes{};
            read_hashed(reinterpret_cast<char*>(raw_size_bytes.data()), raw_size_bytes.size(), "raw G1 point size");
            const auto raw_size = parse_u64_le_bytes(raw_size_bytes);
            if (raw_size != sizeof(mcl::G1))
            {
                throw std::runtime_error("raw SRS G1 material was generated for an incompatible mcl layout");
            }

            const auto point_count = static_cast<std::size_t>(count);
            const auto header_size = static_cast<std::size_t>(24);
            const auto expected_size = header_size + point_count * sizeof(mcl::G1);
            if (expected_size != static_cast<std::size_t>(file_size))
            {
                throw std::runtime_error("raw SRS G1 material size mismatch: " + path.string());
            }
            const auto load_count = max_points == 0 ? point_count : std::min(point_count, max_points);
            std::vector<G1Point> out(load_count);
            if (load_count != 0)
            {
                const auto load_bytes = load_count * sizeof(mcl::G1);
                read_hashed(reinterpret_cast<char*>(&out.front().value), load_bytes, "raw G1 prefix");
            }
            std::array<char, 1 << 20> buffer{};
            std::size_t remaining_bytes = (point_count - load_count) * sizeof(mcl::G1);
            while (remaining_bytes != 0)
            {
                const auto chunk = std::min<std::size_t>(remaining_bytes, buffer.size());
                read_hashed(buffer.data(), chunk, "raw G1 suffix");
                remaining_bytes -= chunk;
            }
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256_Final(digest, &ctx);
            if (actual_hash != nullptr)
            {
                *actual_hash = bytes_to_hex(digest, sizeof(digest));
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path& path,
            std::string* actual_hash = nullptr,
            std::size_t max_points = 0)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS G1 material: " + path.string());
            }
            const auto file_size = input.tellg();
            if (file_size < static_cast<std::streamoff>(16))
            {
                throw std::runtime_error("truncated SRS G1 material: " + path.string());
            }
            if (max_points != 0)
            {
                return load_raw_g1_vector_file_prefix(path, file_size, max_points, actual_hash);
            }
            input.seekg(0);
            std::vector<std::uint8_t> file_bytes(static_cast<std::size_t>(file_size));
            input.read(reinterpret_cast<char*>(file_bytes.data()), file_size);
            if (input.gcount() != file_size)
            {
                throw std::runtime_error("failed to read SRS G1 material: " + path.string());
            }
            if (actual_hash != nullptr)
            {
                *actual_hash = sha256_bytes_hex(file_bytes);
            }
            const std::array<char, 8> expected_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', '\0'}};
            const std::array<char, 8> expected_raw_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
            std::array<char, 8> magic{};
            std::copy_n(reinterpret_cast<const char*>(file_bytes.data()), magic.size(), magic.data());
            if (magic != expected_magic && magic != expected_raw_magic)
            {
                throw std::runtime_error("invalid SRS G1 material header: " + path.string());
            }
            std::size_t offset = 8;
            const auto count = read_u64_le_bytes(file_bytes, offset, "G1 point count");
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("SRS G1 material is too large for this platform");
            }
            if (magic == expected_raw_magic)
            {
                const auto raw_size = read_u64_le_bytes(file_bytes, offset, "raw G1 point size");
                if (raw_size != sizeof(mcl::G1))
                {
                    throw std::runtime_error("raw SRS G1 material was generated for an incompatible mcl layout");
                }
                const auto point_count = static_cast<std::size_t>(count);
                const auto expected_size = offset + point_count * sizeof(mcl::G1);
                if (expected_size != file_bytes.size())
                {
                    throw std::runtime_error("raw SRS G1 material size mismatch: " + path.string());
                }
                std::vector<G1Point> out(point_count);
                if (point_count != 0)
                {
                    std::memcpy(
                        &out.front().value,
                        file_bytes.data() + offset,
                        point_count * sizeof(mcl::G1));
                }
                return out;
            }
            std::vector<std::pair<std::size_t, std::size_t>> payloads;
            payloads.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i)
            {
                const auto size = read_u64_le_bytes(file_bytes, offset, "G1 point size");
                if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
                    || offset + static_cast<std::size_t>(size) > file_bytes.size())
                {
                    throw std::runtime_error("truncated SRS G1 point payload");
                }
                payloads.emplace_back(offset, static_cast<std::size_t>(size));
                offset += static_cast<std::size_t>(size);
            }

            std::vector<G1Point> out;
            out.resize(static_cast<std::size_t>(count));
            const auto worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto chunk_size = (out.size() + worker_count - 1) / worker_count;
            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker)
            {
                const auto begin = worker * chunk_size;
                const auto end = std::min(out.size(), begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&file_bytes, &payloads, &out, begin, end]() {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        const auto [payload_offset, payload_size] = payloads[i];
                        std::vector<std::uint8_t> point_bytes(
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset),
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset + payload_size));
                        out[i] = deserialize_g1(point_bytes);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
            return out;
        }

        std::pair<G2Point, G2Point> load_vk_file(const std::filesystem::path& path, std::string* actual_hash = nullptr)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
            {
                throw std::runtime_error("failed to open SRS verifier key: " + path.string());
            }
            const auto file_size = file.tellg();
            file.seekg(0);
            std::string file_bytes(static_cast<std::size_t>(file_size), '\0');
            file.read(file_bytes.data(), file_size);
            if (file.gcount() != file_size)
            {
                throw std::runtime_error("failed to read SRS verifier key: " + path.string());
            }
            if (actual_hash != nullptr)
            {
                *actual_hash = sha256_hex(file_bytes);
            }
            std::istringstream input(file_bytes, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS verifier key: " + path.string());
            }
            std::string header;
            std::getline(input, header);
            if (header != "GATZK-VK-v1")
            {
                throw std::runtime_error("invalid verifier key header: " + path.string());
            }
            auto read_g2 = [&]() {
                const auto size = read_u64_le(input, "G2 point size");
                std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
                input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
                {
                    throw std::runtime_error("truncated SRS G2 point payload");
                }
                return deserialize_g2(bytes);
            };
            auto g2_one = read_g2();
            auto g2_tau = read_g2();
            return {g2_one, g2_tau};
        }

        void validate_manifest(
            const SRSManifest& manifest,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id)
        {
            if (manifest.curve_id != "bls12_381" || manifest.field_modulus_id != "bls12_381_fr")
            {
                throw std::runtime_error("SRS curve or field mismatch");
            }
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

        double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::vector<FieldElement> folding_powers(const FieldElement& challenge, std::size_t count)
        {
            std::vector<FieldElement> out(count, FieldElement::one());
            for (std::size_t i = 1; i < count; ++i)
            {
                out[i] = out[i - 1] * challenge;
            }
            return out;
        }

        template <class Fn>
        void parallel_for_chunks_capped(
            std::size_t count,
            std::size_t min_parallel_count,
            std::size_t max_worker_count,
            Fn&& fn)
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto available_workers =
                max_worker_count == 0 ? cpu_count : std::min<std::size_t>(cpu_count, max_worker_count);
            if (count < min_parallel_count || available_workers <= 1)
            {
                fn(0, count);
                return;
            }
            const auto task_count = std::min<std::size_t>(
                available_workers,
                std::max<std::size_t>(2, count / min_parallel_count));
            const auto chunk_size = (count + task_count - 1) / task_count;
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                const auto begin = task * chunk_size;
                const auto end = std::min(count, begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&, begin, end]() {
                    fn(begin, end);
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
        }

        template <class Fn>
        void parallel_for_chunks(std::size_t count, std::size_t min_parallel_count, Fn&& fn)
        {
            parallel_for_chunks_capped(count, min_parallel_count, 0, std::forward<Fn>(fn));
        }

        std::string srs_domain_name(const std::string& domain_name)
        {
            return domain_name == "d" ? "d_h" : domain_name;
        }

        std::string srs_exact_domain_name(const std::string& domain_name, std::size_t domain_size)
        {
            return srs_domain_name(domain_name) + "_" + std::to_string(domain_size);
        }

        std::string polynomial_basis_name(const algebra::Polynomial& polynomial)
        {
            return polynomial.basis == algebra::PolynomialBasis::Coefficient ? "coefficient" : "evaluation";
        }

        std::string polynomial_domain_name(const algebra::Polynomial& polynomial)
        {
            if (polynomial.domain == nullptr)
            {
                return "coef";
            }
            return polynomial.domain->name;
        }

        bool progress_enabled()
        {
            return std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        }

        void progress_note(const std::string& label)
        {
            if (progress_enabled())
            {
                std::cerr << "[progress] " << label << '\n';
            }
        }

        std::size_t commit_batch_outer_worker_count(
            std::size_t object_count,
            std::size_t total_scalar_count,
            std::size_t max_scalar_count,
            std::size_t cpu_count)
        {
            if (!util::route2_options().fast_msm
                || object_count <= 1
                || cpu_count < 4
                || total_scalar_count < (1ULL << 15))
            {
                return 1;
            }

            const auto average_scalar_count = total_scalar_count / object_count;
            if (max_scalar_count >= (1ULL << 22) && max_scalar_count * 2 >= total_scalar_count)
            {
                return 1;
            }
            if (object_count <= 8 && average_scalar_count >= (1ULL << 17))
            {
                return std::min<std::size_t>(
                    object_count,
                    std::max<std::size_t>(1, std::min<std::size_t>(3, cpu_count / 8)));
            }
            if (object_count <= 20 && max_scalar_count <= (1ULL << 15))
            {
                const auto by_cpu = std::max<std::size_t>(1, cpu_count / 3);
                const auto by_work = std::max<std::size_t>(2, total_scalar_count / (1ULL << 16));
                return std::min<std::size_t>(
                    object_count,
                    std::max<std::size_t>(1, std::min<std::size_t>(6, std::min(by_cpu, by_work))));
            }

            const auto by_cpu = std::max<std::size_t>(1, cpu_count / 2);
            const auto by_work = std::max<std::size_t>(1, total_scalar_count / (1ULL << 16));
            return std::min<std::size_t>(
                object_count,
                std::min<std::size_t>(10, std::min(by_cpu, by_work)));
        }

        std::vector<std::size_t> commitment_work_order(
            const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials)
        {
            std::vector<std::size_t> work_order(named_polynomials.size());
            std::iota(work_order.begin(), work_order.end(), std::size_t{0});
            auto group_key = [&](std::size_t index) {
                const auto* polynomial = named_polynomials[index].second;
                return std::string(polynomial_basis_name(*polynomial))
                    + "|"
                    + polynomial_domain_name(*polynomial)
                    + "|"
                    + std::to_string(polynomial->domain != nullptr ? polynomial->domain->size : 0);
            };
            std::stable_sort(
                work_order.begin(),
                work_order.end(),
                [&](std::size_t lhs, std::size_t rhs) {
                    const auto lhs_key = group_key(lhs);
                    const auto rhs_key = group_key(rhs);
                    if (lhs_key != rhs_key)
                    {
                        return lhs_key < rhs_key;
                    }
                    return named_polynomials[lhs].second->size() > named_polynomials[rhs].second->size();
                });
            return work_order;
        }

        bool should_fold_constant_evaluation_tail(std::size_t size, std::size_t tail_start)
        {
            if (size < 8192 || tail_start >= size)
            {
                return false;
            }
            const auto tail_length = size - tail_start;
            const auto min_tail_length = std::max<std::size_t>(4096, size / 8);
            return tail_length >= min_tail_length;
        }

        G1Point commit_polynomial_point(
            const algebra::Polynomial& polynomial,
            const KZGProverKey& pk,
            CommitObjectProfile* profile = nullptr,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            G1MSMProfile msm_profile;
            const auto msm_start = Clock::now();
            G1Point point;
            bool constant_tail_folding = false;
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                if (pk.powers_g1 == nullptr)
                {
                    throw std::runtime_error("missing coefficient proving SRS");
                }
                if (polynomial.size() > pk.powers_g1->size() || polynomial.size() > pk.degree_bound + 1)
                {
                    throw std::runtime_error("coefficient polynomial exceeds SRS degree bound: " + polynomial.name);
                }
                point = g1_multi_scalar_mul_profiled(
                    pk.powers_g1->data(),
                    polynomial.values().data(),
                    polynomial.size(),
                    profile != nullptr ? &msm_profile : nullptr,
                    msm_options);
            }
            else
            {
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain: " + polynomial.name);
                }
                const auto base_key = srs_domain_name(polynomial.domain->name);
                const auto exact_key = srs_exact_domain_name(polynomial.domain->name, polynomial.domain->size);
                auto it = pk.lagrange_srs_by_domain.find(exact_key);
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    it = pk.lagrange_srs_by_domain.find(base_key);
                    if (it != pk.lagrange_srs_by_domain.end()
                        && it->second != nullptr
                        && it->second->size() != polynomial.domain->size)
                    {
                        it = pk.lagrange_srs_by_domain.end();
                    }
                }
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    for (auto candidate = pk.lagrange_srs_by_domain.begin();
                         candidate != pk.lagrange_srs_by_domain.end();
                         ++candidate)
                    {
                        if (candidate->second != nullptr && candidate->second->size() == polynomial.domain->size)
                        {
                            it = candidate;
                            break;
                        }
                    }
                }
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    throw std::runtime_error(
                        "missing exact Lagrange SRS for domain: "
                        + base_key
                        + ":"
                        + std::to_string(polynomial.domain->size));
                }
                if (it->second == nullptr)
                {
                    throw std::runtime_error("missing Lagrange proving SRS material for domain: " + exact_key);
                }
                if (polynomial.size() > it->second->size())
                {
                    throw std::runtime_error("evaluation polynomial exceeds Lagrange SRS domain size: " + polynomial.name);
                }
                const auto& values = polynomial.values();
                const auto can_fold_tail = values.size() == polynomial.domain->size;
                std::size_t tail_start = values.size();
                if (can_fold_tail && !values.empty())
                {
                    const auto& tail_value = values.back();
                    while (tail_start > 0 && values[tail_start - 1] == tail_value)
                    {
                        --tail_start;
                    }
                }
                if (can_fold_tail && should_fold_constant_evaluation_tail(values.size(), tail_start))
                {
                    constant_tail_folding = true;
                    const auto& tail_value = values.back();
                    const auto tail_commitment = g1_mul(pk.g1_generator, tail_value);
                    if (tail_start == 0)
                    {
                        point = tail_commitment;
                        if (profile != nullptr)
                        {
                            msm_profile.input_count = polynomial.size();
                            msm_profile.nonzero_scalar_count = 0;
                            msm_profile.effective_scalar_count = 0;
                            msm_profile.zero_scalar_skipping = true;
                            msm_profile.sparse_compaction = true;
                            msm_profile.multithreaded = false;
                        }
                    }
                    else
                    {
                        std::vector<FieldElement> prefix_deltas;
                        prefix_deltas.reserve(tail_start);
                        for (std::size_t i = 0; i < tail_start; ++i)
                        {
                            prefix_deltas.push_back(values[i] - tail_value);
                        }
                        const auto prefix_commitment = g1_multi_scalar_mul_profiled(
                            it->second->data(),
                            prefix_deltas.data(),
                            prefix_deltas.size(),
                            profile != nullptr ? &msm_profile : nullptr,
                            msm_options);
                        if (profile != nullptr)
                        {
                            msm_profile.input_count = polynomial.size();
                        }
                        point = g1_add(tail_commitment, prefix_commitment);
                    }
                }
                else
                {
                    point = g1_multi_scalar_mul_profiled(
                        it->second->data(),
                        values.data(),
                        polynomial.size(),
                        profile != nullptr ? &msm_profile : nullptr,
                        msm_options);
                }
            }
            if (profile != nullptr)
            {
                profile->name = polynomial.name;
                profile->basis = polynomial_basis_name(polynomial);
                profile->domain = polynomial_domain_name(polynomial);
                profile->vector_length = polynomial.size();
                profile->nonzero_scalar_count = msm_profile.nonzero_scalar_count;
                profile->effective_scalar_count = msm_profile.effective_scalar_count;
                profile->zero_scalar_skipping = msm_profile.zero_scalar_skipping;
                profile->sparse_compaction = msm_profile.sparse_compaction;
                profile->constant_tail_folding = constant_tail_folding;
                profile->multithreaded_msm = msm_profile.multithreaded;
                profile->msm_ms = elapsed_ms(msm_start, Clock::now());
            }
            return point;
        }

        Commitment make_commitment(
            const std::string& name,
            algebra::Polynomial polynomial,
            const KZGProverKey& pk,
            CommitObjectProfile* profile = nullptr,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            return Commitment{
                .name = name,
                .point = commit_polynomial_point(*material, pk, profile, msm_options),
                .opening_polynomial = material,
            };
        }

        algebra::Polynomial constant_polynomial(const std::string& name, const FieldElement& value)
        {
            return algebra::Polynomial::from_coefficients(name, {value});
        }

        algebra::Polynomial single_point_quotient_polynomial(
            const algebra::Polynomial& polynomial,
            const FieldElement& point,
            const FieldElement& value)
        {
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                const auto& coefficients = polynomial.values();
                if (coefficients.empty())
                {
                    if (!(value == FieldElement::zero()))
                    {
                        throw std::runtime_error("claimed value does not match committed polynomial");
                    }
                    return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", {FieldElement::zero()});
                }
                std::size_t coefficient_count = coefficients.size();
                while (coefficient_count > 1 && coefficients[coefficient_count - 1] == FieldElement::zero())
                {
                    --coefficient_count;
                }
                if (coefficient_count == 1)
                {
                    if (!(coefficients.front() == value))
                    {
                        throw std::runtime_error("claimed value does not match committed polynomial");
                    }
                    return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", {FieldElement::zero()});
                }
                std::vector<FieldElement> quotient(coefficient_count - 1, FieldElement::zero());
                quotient.back() = coefficients[coefficient_count - 1];
                for (std::size_t i = coefficient_count - 1; i-- > 1;)
                {
                    quotient[i - 1] = coefficients[i] + point * quotient[i];
                }
                const auto remainder = coefficients.front() - value + point * quotient.front();
                if (!(remainder == FieldElement::zero()))
                {
                    throw std::runtime_error("claimed value does not match committed polynomial");
                }
                return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", std::move(quotient));
            }

            if (polynomial.domain == nullptr)
            {
                throw std::runtime_error("evaluation polynomial is missing its domain");
            }
            const auto& values = polynomial.values();
            std::vector<FieldElement> quotient_values(polynomial.domain->size, FieldElement::zero());
            for (std::size_t i = 0; i < polynomial.domain->size; ++i)
            {
                const auto domain_point = polynomial.domain->point_at(i);
                if (domain_point == point)
                {
                    throw std::runtime_error("opening point overlaps evaluation domain; derivative witness path is not implemented");
                }
                const auto row_value = i < values.size() ? values[i] : FieldElement::zero();
                quotient_values[i] = (row_value - value) / (domain_point - point);
            }
            return algebra::Polynomial::from_evaluations(
                polynomial.name + "_witness",
                std::move(quotient_values),
                polynomial.domain);
        }

        algebra::Polynomial scaled_coefficient_single_point_quotient_polynomial(
            const algebra::Polynomial& polynomial,
            const FieldElement& point,
            const FieldElement& scale,
            const FieldElement& scaled_value)
        {
            if (polynomial.basis != algebra::PolynomialBasis::Coefficient)
            {
                throw std::runtime_error("scaled single-point quotient requires coefficient basis");
            }
            const auto& coefficients = polynomial.values();
            if (coefficients.empty() || scale.is_zero())
            {
                if (!(scaled_value == FieldElement::zero()))
                {
                    throw std::runtime_error("scaled claimed value does not match committed polynomial");
                }
                return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", {FieldElement::zero()});
            }
            std::size_t coefficient_count = coefficients.size();
            while (coefficient_count > 1 && coefficients[coefficient_count - 1] == FieldElement::zero())
            {
                --coefficient_count;
            }
            if (coefficient_count == 1)
            {
                if (!(coefficients.front() * scale == scaled_value))
                {
                    throw std::runtime_error("scaled claimed value does not match committed polynomial");
                }
                return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", {FieldElement::zero()});
            }
            std::vector<FieldElement> quotient(coefficient_count - 1, FieldElement::zero());
            quotient.back() = coefficients[coefficient_count - 1] * scale;
            for (std::size_t i = coefficient_count - 1; i-- > 1;)
            {
                quotient[i - 1] = coefficients[i] * scale + point * quotient[i];
            }
            const auto remainder = coefficients.front() * scale - scaled_value + point * quotient.front();
            if (!(remainder == FieldElement::zero()))
            {
                throw std::runtime_error("scaled claimed value does not match committed polynomial");
            }
            return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", std::move(quotient));
        }

        struct EvaluationInverseDenominatorCacheEntry
        {
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            FieldElement base_point;
            std::vector<FieldElement> base_inverse_denominators;
        };

        std::optional<std::size_t> small_rotation_shift(
            const algebra::RootOfUnityDomain& domain,
            const FieldElement& from,
            const FieldElement& to)
        {
            if (from == to)
            {
                return 0;
            }
            FieldElement value = from;
            const auto limit = std::min<std::size_t>(domain.size, 8);
            for (std::size_t shift = 1; shift < limit; ++shift)
            {
                value *= domain.omega;
                if (value == to)
                {
                    return shift;
                }
            }
            return std::nullopt;
        }

        G1Point open_single_point_fold(
            const std::vector<Commitment>& commitments,
            const std::vector<FieldElement>& values,
            const FieldElement& point,
            const std::vector<FieldElement>& powers,
            const KZGProverKey& pk,
            std::vector<EvaluationInverseDenominatorCacheEntry>* inverse_denominator_cache = nullptr,
            BatchOpeningProfile* profile = nullptr,
            std::size_t worker_cap = 0,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            std::vector<EvaluationInverseDenominatorCacheEntry> local_inverse_denominator_cache;
            auto* inverse_cache = inverse_denominator_cache != nullptr
                ? inverse_denominator_cache
                : &local_inverse_denominator_cache;
            auto accumulate_scaled = [worker_cap](
                std::vector<FieldElement>* target,
                const std::vector<FieldElement>& source,
                const FieldElement& scale)
            {
                if (target->size() < source.size())
                {
                    target->resize(source.size(), FieldElement::zero());
                }
                parallel_for_chunks_capped(source.size(), 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        (*target)[i] += source[i] * scale;
                    }
                });
            };
            struct EvaluationFoldGroup
            {
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                struct Term
                {
                    const std::vector<FieldElement>* values = nullptr;
                    std::size_t value_count = 0;
                    FieldElement scale = FieldElement::zero();
                };
                std::vector<Term> terms;
                FieldElement scaled_claim_sum = FieldElement::zero();
            };
            struct FoldedOpeningTerm
            {
                const Commitment* commitment = nullptr;
                FieldElement scale = FieldElement::zero();
                FieldElement scaled_claim = FieldElement::zero();
            };

            std::vector<FieldElement> coefficient_fold;
            std::vector<EvaluationFoldGroup> evaluation_folds;
            std::vector<FoldedOpeningTerm> folded_terms;
            folded_terms.reserve(commitments.size());
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                if (commitments[i].opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial: " + commitments[i].name);
                }
                auto existing = std::find_if(
                    folded_terms.begin(),
                    folded_terms.end(),
                    [&](const FoldedOpeningTerm& term)
                    {
                        return term.commitment != nullptr
                            && term.commitment->opening_polynomial.get() == commitments[i].opening_polynomial.get();
                    });
                if (existing == folded_terms.end())
                {
                    folded_terms.push_back(FoldedOpeningTerm{
                        .commitment = &commitments[i],
                        .scale = powers[i],
                        .scaled_claim = values[i] * powers[i],
                    });
                }
                else
                {
                    existing->scale += powers[i];
                    existing->scaled_claim += values[i] * powers[i];
                }
            }

            auto folded_evaluation_group = [&](const std::shared_ptr<algebra::RootOfUnityDomain>& domain)
                -> EvaluationFoldGroup&
            {
                auto it = std::find_if(
                    evaluation_folds.begin(),
                    evaluation_folds.end(),
                    [&](const EvaluationFoldGroup& group)
                    {
                        return group.domain.get() == domain.get();
                    });
                if (it == evaluation_folds.end())
                {
                    EvaluationFoldGroup group;
                    group.domain = domain;
                    evaluation_folds.push_back(std::move(group));
                    it = std::prev(evaluation_folds.end());
                }
                return *it;
            };

            auto out = g1_zero();
            for (const auto& term : folded_terms)
            {
                if (term.commitment == nullptr || term.commitment->opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial");
                }
                if (term.scale.is_zero() && term.scaled_claim == FieldElement::zero())
                {
                    continue;
                }
                const auto& polynomial = *term.commitment->opening_polynomial;
                if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                {
                    algebra::Polynomial quotient;
                    try
                    {
                        quotient = scaled_coefficient_single_point_quotient_polynomial(
                            polynomial,
                            point,
                            term.scale,
                            term.scaled_claim);
                    }
                    catch (const std::exception& ex)
                    {
                        throw std::runtime_error(
                            "opening mismatch for " + term.commitment->name + ": " + ex.what());
                    }
                    if (coefficient_fold.empty())
                    {
                        coefficient_fold = std::move(quotient.data);
                    }
                    else
                    {
                        accumulate_scaled(&coefficient_fold, quotient.values(), FieldElement::one());
                    }
                    continue;
                }
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain: " + term.commitment->name);
                }
                auto& group = folded_evaluation_group(polynomial.domain);
                const auto& source_values = polynomial.values();
                if (!term.scale.is_zero())
                {
                    group.terms.push_back(EvaluationFoldGroup::Term{
                        .values = &source_values,
                        .value_count = source_values.size(),
                        .scale = term.scale,
                    });
                }
                group.scaled_claim_sum += term.scaled_claim;
            }
            if (!coefficient_fold.empty())
            {
                out = g1_add(
                    out,
                    commit_polynomial_point(
                        algebra::Polynomial::from_coefficients("folded_coefficient_witness", std::move(coefficient_fold)),
                        pk,
                        nullptr,
                        msm_options));
            }
            for (auto& group : evaluation_folds)
            {
                const std::vector<FieldElement>* inverse_denominators = nullptr;
                std::size_t inverse_rotation_shift = 0;
                FieldElement inverse_rotation_scale = FieldElement::one();
                bool reused_inverse_denominators = false;

                const auto inverse_lookup_start = Clock::now();
                for (auto& entry : *inverse_cache)
                {
                    if (entry.domain.get() != group.domain.get())
                    {
                        continue;
                    }
                    const auto shift = small_rotation_shift(*group.domain, entry.base_point, point);
                    if (!shift.has_value())
                    {
                        continue;
                    }
                    inverse_denominators = &entry.base_inverse_denominators;
                    inverse_rotation_shift = *shift;
                    if (inverse_rotation_shift != 0)
                    {
                        inverse_rotation_scale = group.domain->omega.pow(
                            static_cast<std::uint64_t>(group.domain->size - inverse_rotation_shift));
                        reused_inverse_denominators = true;
                    }
                    break;
                }
                if (reused_inverse_denominators && profile != nullptr)
                {
                    profile->denominator_inverse_reuse_hits += 1;
                    profile->denominator_inverse_reuse_ms += elapsed_ms(inverse_lookup_start, Clock::now());
                }

                if (inverse_denominators == nullptr)
                {
                    const auto inverse_start = Clock::now();
                    std::vector<FieldElement> denominators(group.domain->size, FieldElement::zero());
                    parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                        for (std::size_t row = begin; row < end; ++row)
                        {
                            const auto domain_point = group.domain->point_at(row);
                            if (domain_point == point)
                            {
                                throw std::runtime_error("opening point overlaps evaluation domain; derivative witness path is not implemented");
                            }
                            denominators[row] = domain_point - point;
                        }
                    });
                    auto computed_inverse_denominators = algebra::batch_invert(denominators);
                    if (profile != nullptr)
                    {
                        profile->denominator_inverse_ms += elapsed_ms(inverse_start, Clock::now());
                        profile->denominator_inverse_cache_misses += 1;
                    }
                    inverse_cache->push_back(EvaluationInverseDenominatorCacheEntry{
                        .domain = group.domain,
                        .base_point = point,
                        .base_inverse_denominators = std::move(computed_inverse_denominators),
                    });
                    inverse_denominators = &inverse_cache->back().base_inverse_denominators;
                }

                std::vector<FieldElement> quotient_values(group.domain->size, FieldElement::zero());
                struct NativeTerm
                {
                    const std::vector<FieldElement>* values = nullptr;
                    std::size_t value_count = 0;
                    mcl::Fr scale;
                };
                std::vector<NativeTerm> native_terms;
                native_terms.reserve(group.terms.size());
                for (const auto& term : group.terms)
                {
                    native_terms.push_back(NativeTerm{
                        .values = term.values,
                        .value_count = term.value_count,
                        .scale = term.scale.native(),
                    });
                }
                mcl::Fr zero;
                zero.clear();
                mcl::Fr base_numerator;
                mcl::Fr::sub(base_numerator, zero, group.scaled_claim_sum.native());
                const bool has_rotation = inverse_rotation_shift != 0;
                const mcl::Fr rotation_scale_native = inverse_rotation_scale.native();
                parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        mcl::Fr native_numerator = base_numerator;
                        for (const auto& term : native_terms)
                        {
                            if (row < term.value_count)
                            {
                                mcl::Fr product;
                                mcl::Fr::mul(product, (*term.values)[row].native(), term.scale);
                                mcl::Fr::add(native_numerator, native_numerator, product);
                            }
                        }
                        const auto inverse_index =
                            !has_rotation
                            ? row
                            : (row + group.domain->size - inverse_rotation_shift) & (group.domain->size - 1U);
                        mcl::Fr quotient_native;
                        mcl::Fr::mul(
                            quotient_native,
                            native_numerator,
                            (*inverse_denominators)[inverse_index].native());
                        if (has_rotation)
                        {
                            mcl::Fr::mul(quotient_native, quotient_native, rotation_scale_native);
                        }
                        quotient_values[row] = FieldElement::from_native(quotient_native);
                    }
                });
                out = g1_add(
                    out,
                    commit_polynomial_point(
                        algebra::Polynomial::from_evaluations(
                            "folded_" + group.domain->name + "_witness",
                            std::move(quotient_values),
                            group.domain),
                        pk,
                        nullptr,
                        msm_options));
            }
            return out;
        }

        G1Point folded_single_point_commitment(
            const std::vector<Commitment>& commitments,
            const std::vector<FieldElement>& values,
            const FieldElement& challenge,
            const G1Point& g1_generator)
        {
            const auto powers = folding_powers(challenge, commitments.size());
            auto out = g1_zero();
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                const auto value_commitment = g1_mul(g1_generator, values[i]);
                const auto numerator = g1_sub(commitments[i].point, value_commitment);
                out = g1_add(out, g1_mul(numerator, powers[i]));
            }
            return out;
        }

        bool verify_pairing_equation(
            const G1Point& numerator,
            const G1Point& witness,
            const FieldElement& point,
            const KZGVerifierKey& vk)
        {
            const auto rhs = g2_sub(vk.g2_tau, g2_mul(vk.g2_one, point));
            if (util::route2_options().fast_verify_pairing && vk.g2_one_prepared != nullptr)
            {
                return pairing_equal_with_prepared_rhs_a(numerator, *vk.g2_one_prepared, witness, rhs);
            }
            return pairing_equal(numerator, vk.g2_one, witness, rhs);
        }
    }

    std::size_t serialized_size(const Commitment& commitment)
    {
        return serialized_size(commitment.point);
    }

    std::size_t serialized_size(const OpeningProof& proof)
    {
        std::size_t out = 0;
        for (const auto& witness : proof.witnesses)
        {
            out += serialized_size(witness);
        }
        return out;
    }

        std::string required_lagrange_cache_fragment(
            const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains)
        {
            if (required_lagrange_domains.empty())
            {
                return "all";
            }
            std::vector<std::string> parts;
            parts.reserve(required_lagrange_domains.size());
            for (const auto& [domain, size] : required_lagrange_domains)
            {
                parts.push_back(srs_domain_name(domain) + ":" + std::to_string(size));
            }
            std::sort(parts.begin(), parts.end());
            std::ostringstream out;
            for (const auto& part : parts)
            {
                out << '|' << part;
            }
            return out.str();
        }

        void add_unique_domain(std::vector<std::string>* domains, const std::string& domain)
        {
            if (domains == nullptr || domain.empty())
            {
                return;
            }
            if (std::find(domains->begin(), domains->end(), domain) == domains->end())
            {
                domains->push_back(domain);
            }
        }

        std::vector<std::string> select_lagrange_srs_domains(
            const SRSManifest& manifest,
            const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains)
        {
            if (required_lagrange_domains.empty())
            {
                std::vector<std::string> all_domains;
                all_domains.reserve(manifest.pk_lagrange_paths.size());
                for (const auto& [domain, _] : manifest.pk_lagrange_paths)
                {
                    all_domains.push_back(domain);
                }
                std::sort(all_domains.begin(), all_domains.end());
                return all_domains;
            }

            std::vector<std::string> selected;
            for (const auto& [domain_name, domain_size] : required_lagrange_domains)
            {
                const auto base_key = srs_domain_name(domain_name);
                const auto exact_key = srs_exact_domain_name(domain_name, domain_size);
                if (manifest.pk_lagrange_paths.contains(exact_key))
                {
                    add_unique_domain(&selected, exact_key);
                    continue;
                }
                if (manifest.pk_lagrange_paths.contains(base_key))
                {
                    const auto size_it = manifest.domain_sizes.find(base_key);
                    if (size_it != manifest.domain_sizes.end() && size_it->second == domain_size)
                    {
                        add_unique_domain(&selected, base_key);
                        continue;
                    }
                }
                std::vector<std::string> same_size_candidates;
                for (const auto& [candidate, size] : manifest.domain_sizes)
                {
                    if (size == domain_size && manifest.pk_lagrange_paths.contains(candidate))
                    {
                        same_size_candidates.push_back(candidate);
                    }
                }
                std::sort(same_size_candidates.begin(), same_size_candidates.end());
                if (!same_size_candidates.empty())
                {
                    add_unique_domain(&selected, same_size_candidates.front());
                    continue;
                }
                throw std::runtime_error(
                    "SRS manifest lacks Lagrange material for domain "
                    + domain_name
                    + ":"
                    + std::to_string(domain_size));
            }
            std::sort(selected.begin(), selected.end());
            return selected;
        }

    KZGKeyPair KZG::load_keys(
        const std::string& srs_path,
        const std::string& expected_srs_type,
        const std::string& expected_srs_id)
    {
        return load_keys_for_domains(srs_path, expected_srs_type, expected_srs_id, {});
    }

    KZGKeyPair KZG::load_keys_for_domains(
        const std::string& srs_path,
        const std::string& expected_srs_type,
        const std::string& expected_srs_id,
        const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains,
        std::size_t required_coefficient_count)
    {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, KZGKeyPair> cache;
        const auto cache_key =
            srs_path
            + "|"
            + expected_srs_type
            + "|"
            + expected_srs_id
            + "|coef="
            + std::to_string(required_coefficient_count)
            + "|lagrange="
            + required_lagrange_cache_fragment(required_lagrange_domains);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                return it->second;
            }
        }

        const auto manifest = load_manifest_file(srs_path);
        validate_manifest(manifest, expected_srs_type, expected_srs_id);
        const auto srs_root = std::filesystem::path(srs_path);
        const auto selected_lagrange_domains =
            select_lagrange_srs_domains(manifest, required_lagrange_domains);

        std::string pk_coef_hash;
        auto powers_g1 = load_g1_vector_file(
            srs_root / manifest.pk_coef_path,
            &pk_coef_hash,
            required_coefficient_count);
        validate_material_hash_value(pk_coef_hash, manifest.pk_coef_hash, "pk_coef");
        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs;
        for (const auto& domain : selected_lagrange_domains)
        {
            const auto path_it = manifest.pk_lagrange_paths.find(domain);
            if (path_it == manifest.pk_lagrange_paths.end())
            {
                throw std::runtime_error("missing Lagrange SRS path for domain: " + domain);
            }
            const auto hash_it = manifest.pk_lagrange_hashes.find(domain);
            if (hash_it == manifest.pk_lagrange_hashes.end())
            {
                throw std::runtime_error("missing Lagrange SRS hash for domain: " + domain);
            }
            std::string material_hash;
            auto material = load_g1_vector_file(srs_root / path_it->second, &material_hash);
            validate_material_hash_value(material_hash, hash_it->second, "pk_lagrange_" + domain);
            const auto size_it = manifest.domain_sizes.find(domain);
            if (size_it == manifest.domain_sizes.end() || material.size() < size_it->second)
            {
                throw std::runtime_error("Lagrange SRS domain size mismatch: " + domain);
            }
            lagrange_srs.emplace(domain, std::make_shared<const std::vector<G1Point>>(std::move(material)));
        }
        std::string vk_hash;
        const auto [g2_one, g2_tau] = load_vk_file(srs_root / manifest.vk_path, &vk_hash);
        validate_material_hash_value(vk_hash, manifest.vk_hash, "vk");

        G1Point g1_generator = g1_zero();
        mcl::bn::hashAndMapToG1(g1_generator.value, "gatzk:kzg:g1");
        if (powers_g1.empty() || !(powers_g1.front() == g1_generator))
        {
            throw std::runtime_error("SRS coefficient powers do not start with the configured G1 generator");
        }

        const KZGProverKey pk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g1_generator = g1_generator,
            .powers_g1 = std::make_shared<const std::vector<G1Point>>(std::move(powers_g1)),
            .lagrange_srs_by_domain = std::move(lagrange_srs),
            .manifest_hash = manifest.manifest_hash,
        };
        const KZGVerifierKey vk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g1_generator = g1_generator,
            .g2_one = g2_one,
            .g2_tau = g2_tau,
            .g2_one_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_one)),
            .g2_tau_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_tau)),
            .manifest_hash = manifest.manifest_hash,
        };
        const KZGKeyPair key{
            .manifest = manifest,
            .pk = pk,
            .vk = vk,
            .g1_generator = g1_generator,
            .g2_one = g2_one,
            .g2_tau = g2_tau,
            .g2_one_prepared = vk.g2_one_prepared,
            .g2_tau_prepared = vk.g2_tau_prepared,
        };
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, key);
        }
        return key;
    }

    Commitment KZG::commit_constant(
        const std::string& name,
        const FieldElement& value,
        const KZGKeyPair& key)
    {
        return make_commitment(name, constant_polynomial(name, value), key.pk);
    }

    Commitment KZG::commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key)
    {
        return make_commitment(name, polynomial, key.pk);
    }

    std::vector<Commitment> KZG::commit_constant_batch(
        const std::vector<std::pair<std::string, FieldElement>>& named_values,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out;
        out.reserve(named_values.size());
        const auto msm_start = Clock::now();
        for (const auto& [name, value] : named_values)
        {
            out.push_back(commit_constant(name, value, key));
        }
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
        }
        return out;
    }

    std::vector<Commitment> KZG::commit_batch(
        const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        for (const auto& [name, polynomial] : named_polynomials)
        {
            if (polynomial == nullptr)
            {
                throw std::runtime_error("null polynomial in KZG::commit_batch: " + name);
            }
            total_scalar_count += polynomial->size();
            max_scalar_count = std::max(max_scalar_count, polynomial->size());
        }
        const auto outer_worker_count = commit_batch_outer_worker_count(
            named_polynomials.size(),
            total_scalar_count,
            max_scalar_count,
            cpu_count);
        const G1MSMExecutionOptions msm_options{
            .max_worker_count = outer_worker_count > 1
                ? std::max<std::size_t>(1, cpu_count / outer_worker_count)
                : 0,
        };
        auto commit_one = [&](std::size_t index) {
            const auto& [name, polynomial] = named_polynomials[index];
            CommitObjectProfile object_profile;
            out[index] = make_commitment(
                name,
                *polynomial,
                key.pk,
                profile != nullptr ? &object_profile : nullptr,
                outer_worker_count > 1 ? &msm_options : nullptr);
            if (profile != nullptr)
            {
                object_profiles[index] = std::move(object_profile);
            }
        };

        if (outer_worker_count > 1)
        {
            const auto work_order = commitment_work_order(named_polynomials);
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(outer_worker_count);
            for (std::size_t worker = 0; worker < outer_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto index = next.fetch_add(1);
                        if (index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[index]);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
        }
        else
        {
            for (std::size_t index = 0; index < named_polynomials.size(); ++index)
            {
                commit_one(index);
            }
        }
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
            for (auto& object_profile : object_profiles)
            {
                if (!object_profile.name.empty())
                {
                    profile->objects.push_back(std::move(object_profile));
                }
            }
        }
        return out;
    }

    std::vector<Commitment> KZG::commit_owned_batch(
        std::vector<std::pair<std::string, algebra::Polynomial>>&& named_polynomials,
        const KZGKeyPair& key,
        CommitBatchProfile* profile,
        const std::string& progress_prefix)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        std::vector<std::pair<std::string, const algebra::Polynomial*>> view;
        view.reserve(named_polynomials.size());
        for (const auto& [name, polynomial] : named_polynomials)
        {
            total_scalar_count += polynomial.size();
            max_scalar_count = std::max(max_scalar_count, polynomial.size());
            view.push_back({name, &polynomial});
        }

        const auto outer_worker_count = commit_batch_outer_worker_count(
            named_polynomials.size(),
            total_scalar_count,
            max_scalar_count,
            cpu_count);
        const G1MSMExecutionOptions msm_options{
            .max_worker_count = outer_worker_count > 1
                ? std::max<std::size_t>(1, cpu_count / outer_worker_count)
                : 0,
        };
        const auto work_order = outer_worker_count > 1 ? commitment_work_order(view) : std::vector<std::size_t>{};
        auto commit_one = [&](std::size_t index) {
            auto& [name, polynomial] = named_polynomials[index];
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":" + name + ":start");
            }
            CommitObjectProfile object_profile;
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            out[index] = Commitment{
                .name = name,
                .point = commit_polynomial_point(
                    *material,
                    key.pk,
                    profile != nullptr ? &object_profile : nullptr,
                    outer_worker_count > 1 ? &msm_options : nullptr),
                .opening_polynomial = material,
            };
            if (profile != nullptr)
            {
                object_profiles[index] = std::move(object_profile);
            }
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":" + name + ":done");
            }
        };

        if (outer_worker_count > 1)
        {
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(outer_worker_count);
            for (std::size_t worker = 0; worker < outer_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto order_index = next.fetch_add(1);
                        if (order_index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[order_index]);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
        }
        else
        {
            for (std::size_t index = 0; index < named_polynomials.size(); ++index)
            {
                commit_one(index);
            }
        }

        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
            for (auto& object_profile : object_profiles)
            {
                if (!object_profile.name.empty())
                {
                    profile->objects.push_back(std::move(object_profile));
                }
            }
        }
        return out;
    }

    OpeningProof KZG::open_batch(
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
        for (const auto& values : claimed_values)
        {
            if (values.size() != points.size())
            {
                throw std::runtime_error("batch opening value count does not match points");
            }
        }
        const auto precompute_start = Clock::now();
        const auto powers = folding_powers(folding_challenge, commitments.size());
        if (profile != nullptr)
        {
            profile->precompute_ms += elapsed_ms(precompute_start, Clock::now());
        }

        OpeningProof proof;
        proof.witnesses.reserve(points.size());
        std::vector<EvaluationInverseDenominatorCacheEntry> inverse_denominator_cache;
        inverse_denominator_cache.reserve(points.size());
        const auto witness_start = Clock::now();
        auto values_for_point = [&](std::size_t point_index) {
            std::vector<FieldElement> values_for_point;
            values_for_point.reserve(commitments.size());
            for (const auto& values : claimed_values)
            {
                values_for_point.push_back(values[point_index]);
            }
            return values_for_point;
        };
        std::size_t max_evaluation_domain_size = 0;
        std::size_t max_coefficient_size = 0;
        for (const auto& commitment : commitments)
        {
            if (commitment.opening_polynomial == nullptr)
            {
                continue;
            }
            const auto& polynomial = *commitment.opening_polynomial;
            if (polynomial.basis == algebra::PolynomialBasis::Evaluation && polynomial.domain != nullptr)
            {
                max_evaluation_domain_size = std::max(max_evaluation_domain_size, polynomial.domain->size);
            }
            else
            {
                max_coefficient_size = std::max(max_coefficient_size, polynomial.size());
            }
        }
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto point_worker_count = std::min<std::size_t>(points.size(), std::max<std::size_t>(1, cpu_count / 4));
        const bool run_point_parallel =
            util::route2_options().fast_msm
            && points.size() > 1
            && point_worker_count > 1
            && (max_evaluation_domain_size >= (1ULL << 16) || max_coefficient_size >= (1ULL << 16));
        if (run_point_parallel && profile != nullptr)
        {
            profile->point_parallel_batches += 1;
            profile->point_parallel_workers += point_worker_count;
            profile->point_msm_worker_cap += std::max<std::size_t>(1, cpu_count / point_worker_count);
        }
        if (run_point_parallel)
        {
            const auto worker_cap = std::max<std::size_t>(1, cpu_count / point_worker_count);
            const G1MSMExecutionOptions msm_options{.max_worker_count = worker_cap};
            proof.witnesses.resize(points.size());
            std::vector<BatchOpeningProfile> local_profiles(points.size());
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(point_worker_count);
            for (std::size_t worker = 0; worker < point_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto point_index = next.fetch_add(1);
                        if (point_index >= points.size())
                        {
                            break;
                        }
                        std::vector<EvaluationInverseDenominatorCacheEntry> local_inverse_cache;
                        local_inverse_cache.reserve(1);
                        const auto point_values = values_for_point(point_index);
                        proof.witnesses[point_index] = open_single_point_fold(
                            commitments,
                            point_values,
                            points[point_index],
                            powers,
                            key.pk,
                            &local_inverse_cache,
                            profile != nullptr ? &local_profiles[point_index] : nullptr,
                            worker_cap,
                            &msm_options);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
            if (profile != nullptr)
            {
                for (const auto& local : local_profiles)
                {
                    profile->denominator_inverse_ms += local.denominator_inverse_ms;
                    profile->denominator_inverse_reuse_ms += local.denominator_inverse_reuse_ms;
                    profile->denominator_inverse_reuse_hits += local.denominator_inverse_reuse_hits;
                    profile->denominator_inverse_cache_misses += local.denominator_inverse_cache_misses;
                }
            }
        }
        else
        {
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
            {
                const auto point_values = values_for_point(point_index);
                proof.witnesses.push_back(open_single_point_fold(
                commitments,
                point_values,
                points[point_index],
                powers,
                key.pk,
                &inverse_denominator_cache,
                profile));
            }
        }
        if (profile != nullptr)
        {
            profile->finalize_ms += elapsed_ms(witness_start, Clock::now());
        }
        return proof;
    }

    bool KZG::verify_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<FieldElement>& points,
            const std::vector<std::vector<FieldElement>>& claimed_values,
            const FieldElement& folding_challenge,
            const OpeningProof& proof,
            const KZGVerifierKey& vk)
    {
        if (commitments.size() != claimed_values.size() || proof.witnesses.size() != points.size())
        {
            throw std::runtime_error("batch verification shape mismatch");
        }
        for (const auto& values : claimed_values)
        {
            if (values.size() != points.size())
            {
                throw std::runtime_error("batch verification value count does not match points");
            }
        }

        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
        {
            std::vector<FieldElement> values_for_point;
            values_for_point.reserve(commitments.size());
            for (const auto& values : claimed_values)
            {
                values_for_point.push_back(values[point_index]);
            }
            const auto folded_commitment = folded_single_point_commitment(
                commitments,
                values_for_point,
                folding_challenge,
                vk.g1_generator);
            if (!verify_pairing_equation(folded_commitment, proof.witnesses[point_index], points[point_index], vk))
            {
                return false;
            }
        }
        return true;
    }

    OpeningProof KZG::open_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
        const std::vector<FieldElement>& points,
        const FieldElement& folding_challenge,
        const KZGKeyPair& key)
    {
        (void)folding_challenge;
        if (commitments_and_values.size() != points.size())
        {
            throw std::runtime_error("external opening shape mismatch");
        }
        OpeningProof proof;
        proof.witnesses.reserve(commitments_and_values.size());
        for (std::size_t i = 0; i < commitments_and_values.size(); ++i)
        {
            const auto& [commitment, value] = commitments_and_values[i];
            if (commitment.opening_polynomial == nullptr)
            {
                throw std::runtime_error("external commitment is missing prover-side opening polynomial: " + commitment.name);
            }
            const auto quotient = single_point_quotient_polynomial(*commitment.opening_polynomial, points[i], value);
            proof.witnesses.push_back(commit_polynomial_point(quotient, key.pk));
        }
        return proof;
    }

    bool KZG::verify_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
            const std::vector<FieldElement>& points,
            const FieldElement& folding_challenge,
            const OpeningProof& proof,
            const KZGVerifierKey& vk)
    {
        (void)folding_challenge;
        if (commitments_and_values.size() != points.size() || proof.witnesses.size() != points.size())
        {
            throw std::runtime_error("external verification shape mismatch");
        }
        for (std::size_t i = 0; i < commitments_and_values.size(); ++i)
        {
            const auto& [commitment, value] = commitments_and_values[i];
            const auto numerator = g1_sub(commitment.point, g1_mul(vk.g1_generator, value));
            if (!verify_pairing_equation(numerator, proof.witnesses[i], points[i], vk))
            {
                if (std::getenv("GATZK_DEBUG_VERIFY") != nullptr)
                {
                    util::info("external_fold_pairing_fail_index=" + std::to_string(i));
                }
                return false;
            }
        }
        return true;
    }
}
