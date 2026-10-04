#include "gat_adapter/output_stream.hpp"

#include <filesystem>
#include <iostream>

int main() {
    const std::filesystem::path full_device("/dev/full");
    if (!std::filesystem::exists(full_device)) {
        std::cerr << "output_stream_failure_check requires /dev/full\n";
        return 2;
    }
    std::ofstream output(full_device, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "cannot open /dev/full\n";
        return 2;
    }
    output << "force a short output write\n";
    try {
        zkdl_gat::finish_output_stream(output, full_device, "test artifact");
    } catch (const std::exception &) {
        std::cout << "output_stream_failure_check PASS short-write-rejected\n";
        return 0;
    }
    std::cerr << "short write was accepted\n";
    return 1;
}
