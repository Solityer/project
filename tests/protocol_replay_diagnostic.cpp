#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    struct ScopedFileRemove
    {
        std::filesystem::path path;

        ~ScopedFileRemove()
        {
            if (path.empty())
            {
                return;
            }

            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    std::string require_value(
        int &index,
        int argc,
        char **argv,
        const std::string &option)
    {
        if (index + 1 >= argc)
        {
            throw std::runtime_error(
                "missing value for " + option);
        }

        return argv[++index];
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::string config_path;
        std::string compute_backend = "cpu";

        for (int index = 1;
             index < argc;
             ++index)
        {
            const std::string option =
                argv[index];

            if (option == "--config")
            {
                config_path =
                    require_value(
                        index,
                        argc,
                        argv,
                        option);
            }
            else if (
                option ==
                "--compute-backend")
            {
                compute_backend =
                    require_value(
                        index,
                        argc,
                        argv,
                        option);
            }
            else
            {
                throw std::runtime_error(
                    "unknown option: " +
                    option);
            }
        }

        if (config_path.empty())
        {
            throw std::runtime_error(
                "usage: "
                "gatzk_protocol_replay_diag "
                "--config <path> "
                "[--compute-backend "
                "cpu|cuda_hotspots]");
        }

        if (
            compute_backend != "cpu" &&
            compute_backend !=
                "cuda_hotspots")
        {
            throw std::runtime_error(
                "unsupported compute backend: " +
                compute_backend);
        }

        if (
            ::setenv(
                "GATZK_ALGEBRA_BACKEND",
                "cpu",
                1) != 0)
        {
            throw std::runtime_error(
                "failed to set "
                "GATZK_ALGEBRA_BACKEND");
        }

        gatzk::util::Route2Options route;

        const bool use_cuda =
            compute_backend ==
            "cuda_hotspots";

        route.cuda_trace_hotspots =
            use_cuda;
        route.cuda_field_hotspots =
            use_cuda;

        gatzk::util::set_route2_options(
            route);

        if (
            use_cuda &&
            !gatzk::algebra::
                cuda_backend_available())
        {
            throw std::runtime_error(
                "cuda_hotspots requested "
                "but CUDA backend is unavailable");
        }

        const auto config =
            gatzk::util::load_config(
                config_path);

        std::optional<
            gatzk::protocol::Proof>
            proof;

        std::map<
            std::string,
            gatzk::algebra::FieldElement>
            trace_challenges;

        std::string registry_digest;

        std::size_t graph_count = 0;
        std::size_t dataset_graph_count = 0;
        std::size_t node_count = 0;
        std::size_t edge_count = 0;

        const auto registry_path =
            std::filesystem::
                temp_directory_path() /
            ("gatzk_protocol_replay_" +
             gatzk::protocol::
                 fresh_proof_nonce() +
             ".registry");

        ScopedFileRemove registry_cleanup{
            registry_path};

        {
            auto context =
                gatzk::protocol::
                    build_context(
                        config,
                        nullptr);

            graph_count =
                context.local.graph_count;

            dataset_graph_count =
                context.dataset.graph_count;

            node_count =
                context.local.num_nodes;

            edge_count =
                context.local.edges.size();

            // This executable is intentionally
            // forbidden from becoming a full-dataset
            // runner.  Formal full runs remain owned
            // exclusively by gatzk_run.
            if (
                dataset_graph_count <= 1 ||
                graph_count >=
                    dataset_graph_count)
            {
                throw std::runtime_error(
                    "protocol replay diagnostic "
                    "requires a strict partial "
                    "multi-graph instance");
            }

            if (
                context.local.subgraph_rule !=
                "whole_graph")
            {
                throw std::runtime_error(
                    "protocol replay diagnostic "
                    "must consume complete graphs, "
                    "not node/edge subgraphs");
            }

            auto trace =
                gatzk::protocol::
                    build_trace(
                        context,
                        nullptr);

            trace_challenges =
                trace.challenges;

            proof.emplace(
                gatzk::protocol::prove(
                    context,
                    trace,
                    nullptr));

            for (
                const auto &[label, value] :
                trace_challenges)
            {
                const auto replayed =
                    proof->challenges.find(
                        label);

                if (
                    replayed ==
                        proof->challenges.end() ||
                    replayed->second != value)
                {
                    throw std::runtime_error(
                        "prove challenge diverged "
                        "from finalized trace: " +
                        label);
                }
            }

            const auto
                registration_artifact =
                    gatzk::protocol::
                        make_verifier_registration_artifact(
                            context);

            registry_digest =
                registration_artifact
                    .canonical_digest();

            gatzk::protocol::
                write_verifier_registry(
                    context,
                    registry_path.string());
        }

        // The proving context and trace are already
        // destroyed here.  Verification therefore
        // proceeds only from the authenticated
        // point-only registry and serialized proof.
        auto verifier_context =
            gatzk::protocol::
                load_verifier_context_from_registry(
                    registry_path.string(),
                    registry_digest);

        if (
            !gatzk::protocol::
                validate_verifier_context(
                    verifier_context))
        {
            throw std::runtime_error(
                "independent verifier context "
                "validation failed");
        }

        const auto proof_bytes =
            gatzk::protocol::
                serialize_proof(*proof);

        auto parsed_proof =
            gatzk::protocol::
                parse_proof(
                    verifier_context,
                    proof_bytes);

        // parse_proof reconstructs the canonical
        // verifier-side Fiat-Shamir challenges.
        if (
            parsed_proof.challenges !=
            proof->challenges)
        {
            throw std::runtime_error(
                "serialized proof challenge "
                "replay mismatch");
        }

        const auto reencoded =
            gatzk::protocol::
                serialize_proof(
                    parsed_proof);

        if (reencoded != proof_bytes)
        {
            throw std::runtime_error(
                "canonical proof codec "
                "round-trip mismatch");
        }

        if (
            gatzk::protocol::
                private_plain_value_count(
                    verifier_context,
                    parsed_proof) != 0)
        {
            throw std::runtime_error(
                "private plaintext leaked "
                "into proof payload");
        }

        if (
            !gatzk::protocol::verify(
                verifier_context,
                parsed_proof,
                nullptr))
        {
            throw std::runtime_error(
                "independent verifier rejected "
                "diagnostic proof");
        }

        std::cout
            << "PROTOCOL_REPLAY_DIAG_OK"
            << " backend="
            << compute_backend
            << " graphs="
            << graph_count
            << "/"
            << dataset_graph_count
            << " nodes="
            << node_count
            << " edges="
            << edge_count
            << " proof_bytes="
            << proof_bytes.size()
            << '\n';

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "PROTOCOL_REPLAY_DIAG_FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}