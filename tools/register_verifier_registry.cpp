#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/config.hpp"

namespace
{
    std::string require_value(int &index, int argc, char **argv)
    {
        if (index + 1 >= argc)
            throw std::runtime_error("missing value for --config/--output");
        return argv[++index];
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::string config_path;
        std::string output_path;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--config")
                config_path = require_value(index, argc, argv);
            else if (option == "--output")
                output_path = require_value(index, argc, argv);
            else
                throw std::runtime_error("unknown option: " + option);
        }
        if (config_path.empty())
            throw std::runtime_error(
                "usage: gatzk_registry_register --config <canonical.cfg> [--output <registry.txt>]");

        const auto config = gatzk::util::load_config(config_path);
        const auto context = gatzk::protocol::build_context(config, nullptr);
        if (output_path.empty())
            output_path = gatzk::protocol::canonical_verifier_registry_path(context);
        auto registry_path = std::filesystem::path(output_path);
        if (registry_path.is_relative())
            registry_path = std::filesystem::path(config.project_root) / registry_path;
        const auto candidate_artifact =
            gatzk::protocol::make_verifier_registration_artifact(context);
        gatzk::protocol::write_verifier_registry(context, registry_path.string());

        std::cout << "candidate_registry_path=" << registry_path.string()
                  << " registry_id=" << candidate_artifact.registry.registry_id
                  << " candidate_registry_digest="
                  << candidate_artifact.canonical_digest() << '\n';
        std::cout << "The candidate digest must be reviewed and supplied as "
                     "verifier_registry_digest in the formal config before a run.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
