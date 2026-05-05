#include "zkdl_gat/experiment.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return zkdl_gat::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "zkdl_gat_full_forward: " << error.what() << '\n';
        return 2;
    }
}
