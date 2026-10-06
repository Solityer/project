#include "gat_adapter/checkpoint_manifest_json.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {

std::string read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open checkpoint manifest: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void require_rejected(const std::string &text, const std::string &label) {
    try {
        (void)zkdl_gat::checkpoint_manifest_json::tensor_shapes(text);
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("parser accepted invalid manifest case: " + label);
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::runtime_error("expected the five canonical checkpoint manifest paths");
        for (int index = 1; index < argc; ++index) {
            const auto shapes = zkdl_gat::checkpoint_manifest_json::tensor_shapes(read_file(argv[index]));
            const auto first = shapes.find("hidden/layer0/head0/attn_dst/kernel");
            if (shapes.size() != 56U || first == shapes.end() || first->second.size() != 3U ||
                first->second.front() != 1U || first->second.back() != 1U) {
                throw std::runtime_error(std::string("canonical manifest lost its direct first tensor descriptor: ") + argv[index]);
            }
        }

        const auto nested = zkdl_gat::checkpoint_manifest_json::tensor_shapes(
            "{\"metadata\":{\"shape\":[99]},\"tensor_index\":{\"first\":{\"metadata\":{\"shape\":[99]},\"shape\":[1,8,1]}}}\n");
        if (nested.size() != 1U || nested.at("first") != std::vector<std::uint64_t>({1U, 8U, 1U}))
            throw std::runtime_error("parser did not select only the direct tensor descriptor shape");

        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[1]}},\"tensor_index\":{\"b\":{\"shape\":[2]}}}", "duplicate root key");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[1],\"shape\":[2]}}}", "duplicate descriptor key");
        require_rejected("{\"tensor_index\":{\"a\":{\"nested\":{\"shape\":[1]}}}}", "nested-only shape");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[]}}}", "empty shape");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[0]}}}", "zero dimension");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[-1]}}}", "negative dimension");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[1.0]}}}", "fractional dimension");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[true]}}}", "boolean dimension");
        require_rejected("{\"tensor_index\":", "truncated JSON");
        require_rejected("{\"tensor_index\":{\"a\":{\"shape\":[1]}}}garbage", "trailing bytes");
        require_rejected("{\"tensor_index\":[{\"shape\":[1]}]}", "wrong tensor index type");
        std::cout << "checkpoint_manifest_parser=PASS real_manifests=5 mutations=11\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "checkpoint_manifest_parser=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
