#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

namespace {

void write_u64_le(std::ostream& out, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> serialize_g1(const mcl::G1& point) {
    std::array<std::uint8_t, 256> buffer{};
    const auto size = point.serialize(buffer.data(), buffer.size());
    if (size == 0) {
        throw std::runtime_error("failed to serialize G1 point");
    }
    return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
}

std::vector<std::uint8_t> serialize_g2(const mcl::G2& point) {
    std::array<std::uint8_t, 512> buffer{};
    const auto size = point.serialize(buffer.data(), buffer.size());
    if (size == 0) {
        throw std::runtime_error("failed to serialize G2 point");
    }
    return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
}

void write_g1_vector(const std::filesystem::path& path, const std::vector<mcl::G1>& points) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open " + path.string());
    }
    const std::array<char, 8> magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', '\0'}};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u64_le(out, points.size());
    for (const auto& point : points) {
        const auto bytes = serialize_g1(point);
        write_u64_le(out, bytes.size());
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

void write_vk(const std::filesystem::path& path, const mcl::G2& g2_one, const mcl::G2& g2_tau) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open " + path.string());
    }
    out << "GATZK-VK-v1\n";
    for (const auto& point : {g2_one, g2_tau}) {
        const auto bytes = serialize_g2(point);
        write_u64_le(out, bytes.size());
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

void write_fr32_le(const std::filesystem::path& path, const mcl::Fr& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::array<std::uint8_t, 32> bytes{};
    value.getLittleEndian(bytes.data(), bytes.size());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: generate_srs <output-dir> <degree-bound> <srs-id>\n";
            return 2;
        }
        const auto out_dir = std::filesystem::path(argv[1]);
        const auto degree_bound = static_cast<std::size_t>(std::stoull(argv[2]));
        std::filesystem::create_directories(out_dir);

        mcl::bn::initPairing(mcl::BLS12_381);
        mcl::G1 g1;
        mcl::G2 g2;
        mcl::bn::hashAndMapToG1(g1, "gatzk:kzg:g1");
        mcl::bn::hashAndMapToG2(g2, "gatzk:kzg:g2");

        mcl::Fr tau;
        tau.setByCSPRNG();
        if (tau.isZero()) {
            tau = 1;
        }

        std::vector<mcl::G1> powers;
        powers.reserve(degree_bound + 1);
        mcl::Fr power = 1;
        for (std::size_t i = 0; i <= degree_bound; ++i) {
            mcl::G1 point;
            mcl::G1::mul(point, g1, power);
            powers.push_back(point);
            mcl::Fr::mul(power, power, tau);
        }

        mcl::G2 g2_tau;
        mcl::G2::mul(g2_tau, g2, tau);
        write_g1_vector(out_dir / "pk_coef.bin", powers);
        write_vk(out_dir / "vk.bin", g2, g2_tau);
        write_fr32_le(out_dir / "prover_aux.bin", tau);

        tau.clear();
        power.clear();
        std::cout << "wrote SRS material for " << argv[3] << " degree_bound=" << degree_bound << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "generate_srs: " << ex.what() << "\n";
        return 1;
    }
}
