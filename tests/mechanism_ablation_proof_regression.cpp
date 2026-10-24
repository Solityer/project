#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/mechanism/runtime.hpp"
#include "../src/mechanism/codec.hpp"

namespace
{
    using namespace gatzk;
    using namespace gatzk::mechanism;

    constexpr std::string_view kTestOnlySrsPrefix = "mechanism_ablation_test_only";

    struct ProofCase
    {
        std::string id;
        Mechanism mechanism;
        Variant variant;
        std::size_t width;
    };

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot read test-only proof artifact: " + path.string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot write test-only proof artifact: " + path.string());
        output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output)
            throw std::runtime_error("failed writing test-only proof artifact: " + path.string());
    }

    std::vector<ProofCase> proof_cases()
    {
        std::vector<ProofCase> cases;
        for (const auto width : {3U, 5U, 6U, 7U})
        {
            cases.push_back({"ff-original-d" + std::to_string(width), Mechanism::FeatureFolding, Variant::Original, width});
            cases.push_back({"ff-control-d" + std::to_string(width), Mechanism::FeatureFolding, Variant::Control, width});
        }
        cases.push_back({"sn-original-d8", Mechanism::SharedNeighborhood, Variant::Original, 8});
        cases.push_back({"sn-control-d8", Mechanism::SharedNeighborhood, Variant::Control, 8});
        return cases;
    }

    void prove_and_verify_case(const ProofCase &test_case, std::size_t ordinal, const std::filesystem::path &srs_path,
                               const std::filesystem::path &root, bool test_tamper)
    {
        const auto case_dir = root / test_case.id;
        std::filesystem::create_directories(case_dir);
        const auto seed_path = case_dir / "witness_seed.hex";
        {
            std::ofstream seed(seed_path, std::ios::binary | std::ios::trunc);
            if (!seed)
                throw std::runtime_error("cannot write test-only witness seed");
            seed << std::string(64, '7') << '\n';
        }

        RunOptions run_options;
        run_options.mechanism = test_case.mechanism;
        run_options.variant = test_case.variant;
        run_options.backend = Backend::Cpu;
        run_options.nodes = 2;
        run_options.edges = 2;
        run_options.width = test_case.width;
        run_options.graphs = 1;
        run_options.seed = 20261024U + ordinal;
        run_options.witness_seed_file = seed_path;
        run_options.srs_path = srs_path;
        run_options.proof_out = case_dir / "proof.bin";
        run_options.statement_out = case_dir / "statement.json";
        run_options.result_json = case_dir / "prove_receipt.json";
        const auto run_receipt = run(run_options);
        require(run_receipt.srs_id.rfind(kTestOnlySrsPrefix, 0) == 0,
                "proof regression loaded an SRS without the explicit test-only identity");
        require(!run_receipt.real_input && run_receipt.proof_size_bytes > 0 &&
                    std::filesystem::file_size(run_options.proof_out) == run_receipt.proof_size_bytes &&
                    std::filesystem::file_size(run_options.statement_out) > 0,
                "test-only proof did not produce serialized statement/proof bytes");

        VerifyOptions verify_options;
        verify_options.proof_path = run_options.proof_out;
        verify_options.statement_path = run_options.statement_out;
        verify_options.srs_path = srs_path;
        verify_options.result_json = case_dir / "verify_receipt.json";
        const auto verify_receipt = verify(verify_options);
        require(verify_receipt.proof_sha256 == run_receipt.proof_sha256 &&
                    verify_receipt.statement_sha256 == run_receipt.statement_sha256 &&
                    verify_receipt.srs_id == run_receipt.srs_id,
                "independent verifier receipt is not bound to the serialized proof, statement, and test-only SRS");

        if (test_tamper)
        {
            auto decoded = internal::decode_proof(read_bytes(run_options.proof_out));
            require(!decoded.opening_groups.empty() && !decoded.opening_groups.front().values.empty(),
                    "test-only proof has no opening claim to tamper");
            decoded.opening_groups.front().values.front() += algebra::FieldElement::one();
            const auto tampered_path = case_dir / "proof-tampered.bin";
            write_bytes(tampered_path, internal::encode_proof(decoded));
            verify_options.proof_path = tampered_path;
            verify_options.result_json = case_dir / "tampered_verify_receipt.json";
            bool rejected = false;
            try
            {
                (void)verify(verify_options);
            }
            catch (const std::exception &)
            {
                rejected = true;
            }
            require(rejected, "independent verifier accepted a proof with a tampered opening claim");
        }

        std::cout << "MECHANISM_ABLATION_TEST_ONLY_PROOF_OK case=" << test_case.id
                  << " bytes=" << run_receipt.proof_size_bytes << " independent_verify=PASS"
                  << (test_tamper ? " tampered_claim=REJECTED" : "") << '\n';
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("usage: gatzk_mechanism_ablation_proof_regression TEST_ONLY_SRS_DIR [all|CASE_ID]");
        const std::filesystem::path srs_path(argv[1]);
        const std::string filter = argc == 3 ? argv[2] : "all";
        const auto all_cases = proof_cases();
        std::vector<ProofCase> selected_cases;
        for (const auto &test_case : all_cases)
            if (filter == "all" || filter == test_case.id)
                selected_cases.push_back(test_case);
        require(!selected_cases.empty(), "unknown mechanism ablation proof case filter: " + filter);

        const auto root = std::filesystem::temp_directory_path() /
                          ("gatzk_mechanism_ablation_proof_test_only_" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        for (std::size_t i = 0; i < selected_cases.size(); ++i)
            prove_and_verify_case(selected_cases[i], i, srs_path, root, i == 0);
        std::filesystem::remove_all(root);
        std::cout << "MECHANISM_ABLATION_TEST_ONLY_PROOF_REGRESSION_OK cases=" << selected_cases.size()
                  << " independent_verify=PASS real_dataset_admission=NOT_TESTED formal_SRS=NOT_USED\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "mechanism ablation test-only proof regression: " << error.what() << '\n';
        return 1;
    }
}
