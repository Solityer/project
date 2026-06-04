#include "gatzk/crypto/kzg.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <openssl/sha.h>

#include "gatzk/algebra/vector_ops.hpp"
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

        std::string sha256_file_hex(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("missing SRS material file: " + path.string());
            }
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            std::array<char, 1 << 16> buffer{};
            while (input)
            {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto read = input.gcount();
                if (read > 0)
                {
                    SHA256_Update(&ctx, buffer.data(), static_cast<std::size_t>(read));
                }
            }
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256_Final(digest, &ctx);
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

        void validate_material_hash(
            const std::filesystem::path& srs_root,
            const std::string& relative_path,
            const std::string& expected_hash,
            const std::string& label)
        {
            if (relative_path.empty())
            {
                throw std::runtime_error("SRS manifest is missing material path: " + label);
            }
            if (expected_hash.empty())
            {
                throw std::runtime_error("SRS manifest is missing material hash: " + label);
            }
            const auto actual = sha256_file_hex(srs_root / relative_path);
            if (actual != expected_hash)
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

        std::vector<G1Point> load_g1_vector_file(const std::filesystem::path& path)
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
            input.seekg(0);
            std::vector<std::uint8_t> file_bytes(static_cast<std::size_t>(file_size));
            input.read(reinterpret_cast<char*>(file_bytes.data()), file_size);
            if (input.gcount() != file_size)
            {
                throw std::runtime_error("failed to read SRS G1 material: " + path.string());
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

        std::pair<G2Point, G2Point> load_vk_file(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary);
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

        std::string srs_domain_name(const std::string& domain_name)
        {
            return domain_name == "d" ? "d_h" : domain_name;
        }

        G1Point commit_polynomial_point(const algebra::Polynomial& polynomial, const KZGProverKey& pk)
        {
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
                return g1_multi_scalar_mul(pk.powers_g1->data(), polynomial.values().data(), polynomial.size());
            }

            if (polynomial.domain == nullptr)
            {
                throw std::runtime_error("evaluation polynomial is missing its domain: " + polynomial.name);
            }
            const auto key = srs_domain_name(polynomial.domain->name);
            const auto it = pk.lagrange_srs_by_domain.find(key);
            if (it == pk.lagrange_srs_by_domain.end())
            {
                throw std::runtime_error("missing Lagrange SRS for domain: " + key);
            }
            if (it->second == nullptr)
            {
                throw std::runtime_error("missing Lagrange proving SRS material for domain: " + key);
            }
            if (polynomial.size() > it->second->size())
            {
                throw std::runtime_error("evaluation polynomial exceeds Lagrange SRS domain size: " + polynomial.name);
            }
            return g1_multi_scalar_mul(it->second->data(), polynomial.values().data(), polynomial.size());
        }

        Commitment make_commitment(const std::string& name, algebra::Polynomial polynomial, const KZGProverKey& pk)
        {
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            return Commitment{
                .name = name,
                .point = commit_polynomial_point(*material, pk),
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
                auto coefficients = polynomial.values();
                if (coefficients.empty())
                {
                    coefficients.push_back(FieldElement::zero());
                }
                coefficients.front() -= value;
                while (coefficients.size() > 1 && coefficients.back() == FieldElement::zero())
                {
                    coefficients.pop_back();
                }
                if (coefficients.size() == 1)
                {
                    if (!(coefficients.front() == FieldElement::zero()))
                    {
                        throw std::runtime_error("claimed value does not match committed polynomial");
                    }
                    return algebra::Polynomial::from_coefficients(polynomial.name + "_witness", {FieldElement::zero()});
                }
                std::vector<FieldElement> quotient(coefficients.size() - 1, FieldElement::zero());
                quotient.back() = coefficients.back();
                for (std::size_t i = coefficients.size() - 1; i-- > 1;)
                {
                    quotient[i - 1] = coefficients[i] + point * quotient[i];
                }
                const auto remainder = coefficients.front() + point * quotient.front();
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

        G1Point open_single_point_fold(
            const std::vector<Commitment>& commitments,
            const std::vector<FieldElement>& values,
            const FieldElement& point,
            const std::vector<FieldElement>& powers,
            const KZGProverKey& pk)
        {
            auto out = g1_zero();
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
            if (commitments[i].opening_polynomial == nullptr)
            {
                throw std::runtime_error("commitment is missing prover-side opening polynomial: " + commitments[i].name);
            }
            algebra::Polynomial quotient;
            try
            {
                quotient = single_point_quotient_polynomial(
                    *commitments[i].opening_polynomial,
                    point,
                    values[i]);
            }
            catch (const std::exception& ex)
            {
                throw std::runtime_error(
                    "opening mismatch for " + commitments[i].name + ": " + ex.what());
            }
            out = g1_add(out, g1_mul(commit_polynomial_point(quotient, pk), powers[i]));
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
        const auto srs_root = std::filesystem::path(srs_path);
        validate_material_hash(srs_root, manifest.pk_coef_path, manifest.pk_coef_hash, "pk_coef");
        validate_material_hash(srs_root, manifest.vk_path, manifest.vk_hash, "vk");
        for (const auto& [domain, path] : manifest.pk_lagrange_paths)
        {
            const auto hash_it = manifest.pk_lagrange_hashes.find(domain);
            if (hash_it == manifest.pk_lagrange_hashes.end())
            {
                throw std::runtime_error("missing Lagrange SRS hash for domain: " + domain);
            }
            validate_material_hash(srs_root, path, hash_it->second, "pk_lagrange_" + domain);
        }

        auto powers_g1 = load_g1_vector_file(srs_root / manifest.pk_coef_path);
        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs;
        for (const auto& [domain, path] : manifest.pk_lagrange_paths)
        {
            auto material = load_g1_vector_file(srs_root / path);
            const auto size_it = manifest.domain_sizes.find(domain);
            if (size_it == manifest.domain_sizes.end() || material.size() < size_it->second)
            {
                throw std::runtime_error("Lagrange SRS domain size mismatch: " + domain);
            }
            lagrange_srs.emplace(domain, std::make_shared<const std::vector<G1Point>>(std::move(material)));
        }
        const auto [g2_one, g2_tau] = load_vk_file(srs_root / manifest.vk_path);

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
        std::vector<Commitment> out;
        out.reserve(named_polynomials.size());
        const auto msm_start = Clock::now();
        for (const auto& [name, polynomial] : named_polynomials)
        {
            if (polynomial == nullptr)
            {
                throw std::runtime_error("null polynomial in KZG::commit_batch: " + name);
            }
            out.push_back(commit(name, *polynomial, key));
        }
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
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
        const auto witness_start = Clock::now();
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
        {
            std::vector<FieldElement> values_for_point;
            values_for_point.reserve(commitments.size());
            for (const auto& values : claimed_values)
            {
                values_for_point.push_back(values[point_index]);
            }
            proof.witnesses.push_back(open_single_point_fold(
                commitments,
                values_for_point,
                points[point_index],
                powers,
                key.pk));
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
                return false;
            }
        }
        return true;
    }
}
