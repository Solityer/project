#include "gat_adapter/integer_bound_ledger.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

std::string read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open integer-bound ledger: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void expect_rejected(const std::string &text, const std::string &label) {
    try {
        (void)zkdl_gat::integer_bound_ledger::parse(text);
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("native ledger parser accepted invalid case: " + label);
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::runtime_error("expected five canonical integer-bound ledger paths");
        std::string first_ledger;
        for (int index = 1; index < argc; ++index) {
            const std::string text = read_file(argv[index]);
            if (index == 1) first_ledger = text;
            const auto ledger = zkdl_gat::integer_bound_ledger::parse(text);
            const auto actual = zkdl_gat::integer_bound_ledger::digest(ledger);
            if (actual != ledger.digest) throw std::runtime_error(std::string("embedded digest differs from production native digest: ") + argv[index]);
            std::cout << actual << '\n';
        }

        const std::string family_marker = "\"family_id\": \"hidden/layer0/head0\"";
        auto missing_family_id = first_ledger;
        const auto family_position = missing_family_id.find(family_marker);
        if (family_position == std::string::npos) throw std::runtime_error("canonical first family marker is missing");
        missing_family_id.erase(family_position, family_marker.size() + 1U);
        expect_rejected(missing_family_id, "missing family_id");

        auto wrong_output_type = first_ledger;
        const auto output_position = wrong_output_type.find("\"output\": 0");
        if (output_position == std::string::npos) throw std::runtime_error("canonical output role marker is missing");
        wrong_output_type.replace(output_position, std::string("\"output\": 0").size(), "\"output\": true");
        expect_rejected(wrong_output_type, "boolean output role");

        auto duplicate_root = first_ledger;
        const auto root_marker = duplicate_root.find("\"schema\": \"zkDL_gat_integer_bound_ledger_v1\"");
        if (root_marker == std::string::npos) throw std::runtime_error("canonical schema marker is missing");
        duplicate_root.insert(root_marker, "\"schema\": \"zkDL_gat_integer_bound_ledger_v1\", ");
        expect_rejected(duplicate_root, "duplicate root key");

        auto fractional_coordinate = first_ledger;
        const auto coordinate_position = fractional_coordinate.find("\"layer_index\": 0");
        if (coordinate_position == std::string::npos) throw std::runtime_error("canonical layer coordinate marker is missing");
        fractional_coordinate.replace(coordinate_position, std::string("\"layer_index\": 0").size(), "\"layer_index\": 0.0");
        expect_rejected(fractional_coordinate, "fractional layer coordinate");

        std::cerr << "native_ledger_parser=PASS real_ledgers=5 mutations=4\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native_ledger_parser=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
