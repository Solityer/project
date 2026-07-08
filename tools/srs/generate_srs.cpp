#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace
{

    void write_u64_le(std::ostream &out, std::uint64_t value)
    {
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
        }
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    void append_u64_le(std::string &out, std::uint64_t value)
    {
        std::array<char, 8> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
        }
        out.append(bytes.data(), bytes.size());
    }

    std::vector<std::uint8_t> serialize_g1(const mcl::G1 &point)
    {
        std::array<std::uint8_t, 256> buffer{};
        const auto size = point.serialize(buffer.data(), buffer.size());
        if (size == 0)
        {
            throw std::runtime_error("failed to serialize G1 point");
        }
        return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
    }

    void append_serialized_g1(std::string &out, const mcl::G1 &point)
    {
        const auto bytes = serialize_g1(point);
        append_u64_le(out, bytes.size());
        out.append(
            reinterpret_cast<const char *>(bytes.data()),
            reinterpret_cast<const char *>(bytes.data()) + static_cast<std::ptrdiff_t>(bytes.size()));
    }

    void append_raw_g1(std::string &out, const mcl::G1 &point)
    {
        const auto *begin = reinterpret_cast<const char *>(&point);
        out.append(begin, begin + static_cast<std::ptrdiff_t>(sizeof(mcl::G1)));
    }

    std::vector<std::uint8_t> serialize_g2(const mcl::G2 &point)
    {
        std::array<std::uint8_t, 512> buffer{};
        const auto size = point.serialize(buffer.data(), buffer.size());
        if (size == 0)
        {
            throw std::runtime_error("failed to serialize G2 point");
        }
        return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
    }

    std::string bytes_to_hex(const unsigned char *bytes, std::size_t size)
    {
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::size_t i = 0; i < size; ++i)
        {
            out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
        }
        return out.str();
    }

    std::string sha256_hex(const std::string &data)
    {
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char *>(data.data()), data.size(), digest);
        return bytes_to_hex(digest, sizeof(digest));
    }

    std::string sha256_file_hex(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("failed to hash " + path.string());
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

    template <typename ChunkFn>
    void write_g1_chunks(const std::filesystem::path &path, std::uint64_t count, ChunkFn &&chunk_fn)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open " + path.string());
        }
        const std::array<char, 8> magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
        out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        write_u64_le(out, count);
        write_u64_le(out, sizeof(mcl::G1));

        const std::uint64_t chunk_size = 8192;
        const std::size_t max_inflight = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::vector<std::future<std::string>> inflight;
        inflight.reserve(max_inflight);
        std::uint64_t next = 0;
        auto launch_next = [&]()
        {
            const auto begin = next;
            const auto end = std::min<std::uint64_t>(count, begin + chunk_size);
            next = end;
            inflight.push_back(std::async(std::launch::async, [&, begin, end]()
                                          { return chunk_fn(begin, end); }));
        };

        while (next < count && inflight.size() < max_inflight)
        {
            launch_next();
        }
        while (!inflight.empty())
        {
            auto bytes = inflight.front().get();
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            inflight.erase(inflight.begin());
            if (next < count)
            {
                launch_next();
            }
        }
    }

    void write_vk(const std::filesystem::path &path, const mcl::G2 &g2_one, const mcl::G2 &g2_tau)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open " + path.string());
        }
        out << "GATZK-VK-v1\n";
        for (const auto &point : {g2_one, g2_tau})
        {
            const auto bytes = serialize_g2(point);
            write_u64_le(out, bytes.size());
            out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
    }

    std::unordered_map<std::string, std::size_t> parse_domains(int argc, char **argv)
    {
        std::unordered_map<std::string, std::size_t> domains;
        for (int i = 4; i < argc; ++i)
        {
            const std::string arg = argv[i];
            const auto eq = arg.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == arg.size())
            {
                throw std::runtime_error("domain arguments must use name=size");
            }
            domains.emplace(arg.substr(0, eq), static_cast<std::size_t>(std::stoull(arg.substr(eq + 1))));
        }
        if (domains.empty())
        {
            domains = {
                {"FH", 1ULL << 22},
                {"edge", 1ULL << 15},
                {"in", 1ULL << 11},
                {"d_h", 1ULL << 4},
                {"cat", 1ULL << 7},
                {"C", 1ULL << 4},
                {"N", 1ULL << 12},
            };
        }
        return domains;
    }

    void write_manifest(
        const std::filesystem::path &out_dir,
        const std::string &srs_id,
        std::size_t degree_bound,
        const std::unordered_map<std::string, std::size_t> &domains,
        const std::unordered_map<std::string, std::string> &lagrange_hashes)
    {
        const auto transcript_hash = sha256_hex("gatzk_srs_setup_v2|" + srs_id + "|" + std::to_string(degree_bound));
        std::ostringstream manifest;
        manifest << "{\n";
        manifest << "  \"srs_id\": \"" << srs_id << "\",\n";
        manifest << "  \"curve_id\": \"bls12_381\",\n";
        manifest << "  \"field_modulus_id\": \"bls12_381_fr\",\n";
        manifest << "  \"srs_type\": \"trusted_srs\",\n";
        manifest << "  \"ceremony_transcript_hash\": \"" << transcript_hash << "\",\n";
        manifest << "  \"setup_transcript_hash\": \"" << transcript_hash << "\",\n";
        manifest << "  \"degree_bound\": " << degree_bound << ",\n";
        manifest << "  \"domain_sizes\": {\n";
        std::size_t index = 0;
        for (const auto &[domain, size] : domains)
        {
            manifest << "    \"" << domain << "\": " << size << (++index == domains.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"pk_coef_path\": \"pk_coef.bin\",\n";
        manifest << "  \"pk_coef_hash\": \"" << sha256_file_hex(out_dir / "pk_coef.bin") << "\",\n";
        manifest << "  \"pk_lagrange_paths\": {\n";
        index = 0;
        for (const auto &[domain, _] : domains)
        {
            (void)_;
            manifest << "    \"" << domain << "\": \"pk_lagrange_" << domain << ".bin\""
                     << (++index == domains.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"pk_lagrange_hashes\": {\n";
        index = 0;
        for (const auto &[domain, hash] : lagrange_hashes)
        {
            manifest << "    \"" << domain << "\": \"" << hash << "\""
                     << (++index == lagrange_hashes.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"vk_path\": \"vk.bin\",\n";
        manifest << "  \"vk_hash\": \"" << sha256_file_hex(out_dir / "vk.bin") << "\",\n";
        manifest << "  \"encoding\": \"raw_mcl_bls12_381_v1\",\n";
        manifest << "  \"generator_version\": \"zkgat_srs_setup_tool_v2\",\n";
        manifest << "  \"security_level\": \"128-bit\",\n";
        manifest << "  \"manifest_hash\": \"" << sha256_hex(manifest.str()) << "\"\n";
        manifest << "}\n";

        std::ofstream out(out_dir / "srs_manifest.json", std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open SRS manifest for writing");
        }
        out << manifest.str();
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 4)
        {
            std::cerr << "usage: generate_srs <output-dir> <degree-bound> <srs-id> [domain=size ...]\n";
            return 2;
        }
        const auto out_dir = std::filesystem::path(argv[1]);
        const auto degree_bound = static_cast<std::size_t>(std::stoull(argv[2]));
        const std::string srs_id = argv[3];
        const auto domains = parse_domains(argc, argv);
        std::filesystem::create_directories(out_dir);

        mcl::bn::initPairing(mcl::BLS12_381);
        mcl::G1 g1;
        mcl::G2 g2;
        mcl::bn::hashAndMapToG1(g1, "gatzk:kzg:g1");
        mcl::bn::hashAndMapToG2(g2, "gatzk:kzg:g2");

        mcl::Fr tau;
        tau.setByCSPRNG();  //随机采样
        if (tau.isZero())
        {
            tau = 1;
        }

        write_g1_chunks(out_dir / "pk_coef.bin", degree_bound + 1, [&](std::uint64_t begin, std::uint64_t end)
                        {
            std::string bytes;
            bytes.reserve(static_cast<std::size_t>(end - begin) * sizeof(mcl::G1));
            mcl::Fr power;
            mcl::Fr::pow(power, tau, begin);
            for (std::uint64_t i = begin; i < end; ++i) {
                (void)i;
                mcl::G1 point;
                mcl::G1::mul(point, g1, power);
                append_raw_g1(bytes, point);
                mcl::Fr::mul(power, power, tau);
            }
            return bytes; });

        mcl::G2 g2_tau;
        mcl::G2::mul(g2_tau, g2, tau);
        write_vk(out_dir / "vk.bin", g2, g2_tau);

        const auto tau_field = gatzk::algebra::FieldElement::from_native(tau);
        std::unordered_map<std::string, std::string> lagrange_hashes;
        for (const auto &[domain_name, domain_size] : domains)
        {
            const auto omega = gatzk::algebra::FieldElement::root_of_unity(domain_size);
            const auto inv_size = gatzk::algebra::FieldElement(static_cast<std::uint64_t>(domain_size)).inv();
            const auto zero_at_tau = tau_field.pow(domain_size) - gatzk::algebra::FieldElement::one();
            const auto path = out_dir / ("pk_lagrange_" + domain_name + ".bin");
            write_g1_chunks(path, domain_size, [&](std::uint64_t begin, std::uint64_t end)
                            {
                std::string bytes;
                bytes.reserve(static_cast<std::size_t>(end - begin) * sizeof(mcl::G1));
                auto domain_point = omega.pow(begin);
                for (std::uint64_t i = begin; i < end; ++i) {
                    (void)i;
                    if (tau_field == domain_point) {
                        throw std::runtime_error("setup sampled tau inside domain " + domain_name);
                    }
                    const auto scalar = zero_at_tau * domain_point * inv_size / (tau_field - domain_point);
                    mcl::G1 point;
                    mcl::G1::mul(point, g1, scalar.native());
                    append_raw_g1(bytes, point);
                    domain_point *= omega;
                }
                return bytes; });
            lagrange_hashes.emplace(domain_name, sha256_file_hex(path));
        }
        write_manifest(out_dir, srs_id, degree_bound, domains, lagrange_hashes);

        tau.clear();
        std::cout << "wrote SRS material for " << srs_id << " degree_bound=" << degree_bound << "\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "generate_srs: " << ex.what() << "\n";
        return 1;
    }
}
