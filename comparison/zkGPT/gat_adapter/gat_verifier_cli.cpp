#include "gat_proof.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

// Independent verifier entry point.  Its only statement inputs are the
// complete point-only verifier context, the separately supplied public Y,
// and the verifier-facing proof artifact.  No registration JSON, checkpoint,
// model tensor, witness, prover secret, or private path is accepted.
namespace {

using zkgpt::gat::GatProofArtifact;
using zkgpt::gat::Int;
using zkgpt::gat::Tensor2D;
using zkgpt::gat::VerifierContext;

struct Options {
    std::string baseline_root = ".";
    std::string verifier_context;
    std::string public_logits;
    std::string artifact;
    std::string report;
    bool help = false;
    bool capabilities = false;
};

std::string trim_root(std::string root) {
    while (root.size() > 1 && (root.back() == '/' || root.back() == '\\')) root.pop_back();
    return root.empty() ? "." : root;
}

bool absolute_path(const std::string &path) {
    return !path.empty() && (path[0] == '/' || path[0] == '\\');
}

std::string local_path(const std::string &root, const std::string &relative,
                       const char *description) {
    if (relative.empty() || absolute_path(relative)) {
        throw std::runtime_error(std::string(description) + " must be baseline-relative");
    }
    std::size_t begin = 0;
    while (begin <= relative.size()) {
        const std::size_t slash = relative.find('/', begin);
        const std::string part = relative.substr(
            begin, slash == std::string::npos ? std::string::npos : slash - begin);
        if (part.empty() || part == "." || part == "..") {
            throw std::runtime_error(std::string(description) + " contains a non-canonical path component");
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return trim_root(root) + "/" + relative;
}

void require_regular(const std::string &path, const char *description) {
    struct stat info {};
    if (::lstat(path.c_str(), &info) != 0) {
        throw std::runtime_error(std::string("missing ") + description + ": " + path);
    }
    if (S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_nlink != 1) {
        throw std::runtime_error(std::string(description) + " must be a regular non-aliased file: " + path);
    }
}

std::string read_text(const std::string &path, const char *description,
                      std::size_t limit = 256U * 1024U * 1024U) {
    require_regular(path, description);
    struct stat info {};
    if (::stat(path.c_str(), &info) != 0 || info.st_size < 0 ||
        static_cast<std::uint64_t>(info.st_size) > limit) {
        throw std::runtime_error(std::string(description) + " exceeds the verifier input limit: " + path);
    }
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + description + ": " + path);
    std::ostringstream out;
    out << in.rdbuf();
    if (!in.good() && !in.eof()) throw std::runtime_error(std::string("cannot read ") + description + ": " + path);
    return out.str();
}

Tensor2D parse_public_logits(const std::string &text,
                             const Tensor2D &expected) {
    std::istringstream input(text);
    unsigned long long rows = 0;
    unsigned long long cols = 0;
    if (!(input >> rows >> cols) || rows != expected.rows || cols != expected.cols) {
        throw std::runtime_error("public logits dimensions do not match verifier context");
    }
    Tensor2D parsed(expected.rows, expected.cols);
    for (std::size_t i = 0; i < parsed.values.size(); ++i) {
        long long value = 0;
        if (!(input >> value)) throw std::runtime_error("public logits are truncated");
        parsed.values[i] = static_cast<Int>(value);
    }
    std::string trailing;
    if (input >> trailing) throw std::runtime_error("public logits contain trailing tokens");
    return parsed;
}

std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) out << "\\u00" << std::hex << static_cast<unsigned>(ch) << std::dec;
                else out << static_cast<char>(ch);
        }
    }
    return out.str();
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") options.help = true;
        else if (arg == "--capabilities") options.capabilities = true;
        else if (arg == "--baseline-root" && i + 1 < argc) options.baseline_root = argv[++i];
        else if (arg == "--verifier-context" && i + 1 < argc) options.verifier_context = argv[++i];
        else if (arg == "--public-logits" && i + 1 < argc) options.public_logits = argv[++i];
        else if (arg == "--artifact" && i + 1 < argc) options.artifact = argv[++i];
        else if (arg == "--report" && i + 1 < argc) options.report = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (options.help || options.capabilities) return options;
    if (options.verifier_context.empty() || options.public_logits.empty() || options.artifact.empty()) {
        throw std::runtime_error("--verifier-context, --public-logits and --artifact are required");
    }
    return options;
}

void write_report(const std::string &path, const std::string &json) {
    if (path.empty()) return;
    const std::string temporary = path + ".partial";
    std::ofstream out(temporary.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open verifier report: " + path);
    out << json;
    if (!out.good()) throw std::runtime_error("cannot write verifier report: " + path);
    out.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot finalize verifier report: " + path);
    }
}

std::string report_json(const std::string &status, const std::string &reason,
                        const std::string &statement_digest,
                        const std::string &context, const std::string &logits,
                        const std::string &artifact,
                        const std::string &artifact_fs,
                        const VerifierContext *public_context = nullptr,
                        const GatProofArtifact *proof_artifact = nullptr) {
    const std::string dataset = public_context == nullptr
        ? std::string() : public_context->registration.dataset;
    const std::string backend = public_context == nullptr
        ? std::string() : public_context->backend;
    const std::string plan_digest = public_context == nullptr
        ? std::string() : public_context->plan.plan_digest;
    const std::string circuit_id = public_context == nullptr
        ? std::string() : public_context->plan.circuit_id;
    const std::string registration_digest = public_context == nullptr
        ? std::string() : public_context->registration.registration_digest;
    const std::string registration_binding_digest = public_context == nullptr
        ? std::string() : public_context->registration.binding_digest;
    const std::string registration_layout_digest = public_context == nullptr
        ? std::string() : public_context->registration.layout_digest;
    const std::string com_w_point = public_context == nullptr
        ? std::string() : public_context->registration.com_W.point_hex;
    const std::string full_dataset_identity = public_context == nullptr
        ? std::string() : public_context->full_dataset_identity;
    const std::string checkpoint_identity = public_context == nullptr
        ? std::string() : public_context->checkpoint_identity;
    const std::string cache_identity = public_context == nullptr
        ? std::string() : public_context->cache_identity;
    const std::string proof_digest = proof_artifact == nullptr
        ? std::string() : zkgpt::gat::canonical_digest64(proof_artifact->transcript_digest);
    std::string artifact_sha256;
    std::uint64_t artifact_size_bytes = 0;
    if (proof_artifact != nullptr) {
        try {
            struct stat info {};
            if (::stat(artifact_fs.c_str(), &info) == 0 && info.st_size >= 0) {
                artifact_size_bytes = static_cast<std::uint64_t>(info.st_size);
                artifact_sha256 = zkgpt::gat::sha256_file_digest(artifact_fs);
            }
        } catch (...) {
            artifact_sha256.clear();
            artifact_size_bytes = 0;
        }
    }
    const std::size_t slash = context.find_last_of("/\\");
    const std::string statement = slash == std::string::npos
        ? "public_statement.json"
        : context.substr(0, slash + 1) + "public_statement.json";
    std::ostringstream out;
    out << "{\n"
        << "  \"schema\": \"zkgpt-gat-independent-verifier-v3\",\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"independent\": true,\n"
        << "  \"masked_full\": true,\n"
        << "  \"verifier_reads_witness\": false,\n"
        << "  \"verifier_reads_checkpoint\": false,\n"
        << "  \"dataset\": \"" << json_escape(dataset) << "\",\n"
        << "  \"backend\": \"" << json_escape(backend) << "\",\n"
        << "  \"statement_digest\": \"" << json_escape(statement_digest) << "\",\n"
        << "  \"plan_digest\": \"" << json_escape(plan_digest) << "\",\n"
        << "  \"circuit_id\": \"" << json_escape(circuit_id) << "\",\n"
        << "  \"proof_digest\": \"" << json_escape(proof_digest) << "\",\n"
        << "  \"registration_digest\": \"" << json_escape(registration_digest) << "\",\n"
        << "  \"registration_binding_digest\": \"" << json_escape(registration_binding_digest) << "\",\n"
        << "  \"registration_layout_digest\": \"" << json_escape(registration_layout_digest) << "\",\n"
        << "  \"com_W_point\": \"" << json_escape(com_w_point) << "\",\n"
        << "  \"full_dataset_identity\": \"" << json_escape(full_dataset_identity) << "\",\n"
        << "  \"checkpoint_identity\": \"" << json_escape(checkpoint_identity) << "\",\n"
        << "  \"cache_identity\": \"" << json_escape(cache_identity) << "\",\n"
        << "  \"statement_path\": \"" << json_escape(statement) << "\",\n"
        << "  \"verifier_context_path\": \"" << json_escape(context) << "\",\n"
        << "  \"public_logits_path\": \"" << json_escape(logits) << "\",\n"
        << "  \"artifact_path\": \"" << json_escape(artifact) << "\",\n"
        << "  \"artifact_sha256\": \"" << json_escape(artifact_sha256) << "\",\n"
        << "  \"artifact_size_bytes\": " << artifact_size_bytes << ",\n"
        << "  \"reason\": \"" << json_escape(reason) << "\"\n"
        << "}\n";
    return out.str();
}

void print_help() {
    std::cout << "Usage: zkgpt_gat_verifier --verifier-context <relative-binary> "
                 "--public-logits <relative-text> --artifact <relative-artifact> "
                 "[--baseline-root <zkGPT-root>] [--report <relative-json>]\n"
                 "       zkgpt_gat_verifier --capabilities\n"
                 "The verifier reads only the complete v3 public context, public logits and proof artifact.\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        if (options.capabilities) {
            std::cout << "{\"schema\":\"zkgpt-gat-verifier-capabilities-v2\","
                         "\"independent\":true,\"masked_full\":true,"
                         "\"reads_witness\":false,\"reads_checkpoint\":false}\n";
            return 0;
        }

        const std::string root = trim_root(options.baseline_root);
        const std::string context_rel = options.verifier_context;
        const std::string logits_rel = options.public_logits;
        const std::string artifact_rel = options.artifact;
        const std::string context_path = local_path(root, context_rel, "verifier context");
        const std::string logits_path = local_path(root, logits_rel, "public logits");
        const std::string artifact_path = local_path(root, artifact_rel, "proof artifact");

        zkgpt::gat::initialize_gat_field();
        // The v3 context reader reconstructs the layered plan from its compact
        // recipe.  Keep it off the small process stack: malformed or missing
        // input must still fail closed without a stack overflow.
        std::unique_ptr<VerifierContext> context_storage(new VerifierContext());
        VerifierContext &context = *context_storage;
        std::string context_reason;
        if (!zkgpt::gat::read_verifier_context(context_path, context, &context_reason)) {
            const std::string report = report_json("REJECTED", context_reason,
                                                    "", context_rel, logits_rel,
                                                    artifact_rel, artifact_path);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }
        const std::string statement_digest = zkgpt::gat::verifier_context_digest(context);

        const Tensor2D external_logits = parse_public_logits(
            read_text(logits_path, "public logits"), context.public_logits);
        if (external_logits.rows != context.public_logits.rows ||
            external_logits.cols != context.public_logits.cols ||
            external_logits.values != context.public_logits.values) {
            const std::string report = report_json(
                "REJECTED", "separately supplied public logits do not match v3 context",
                statement_digest, context_rel, logits_rel, artifact_rel,
                artifact_path, &context, nullptr);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }

        std::unique_ptr<GatProofArtifact> artifact_storage(new GatProofArtifact());
        GatProofArtifact &artifact = *artifact_storage;
        std::string artifact_reason;
        if (!GatProofArtifact::read(artifact_path, artifact, &artifact_reason)) {
            const std::string report = report_json("REJECTED", artifact_reason,
                                                    statement_digest, context_rel,
                                                    logits_rel, artifact_rel,
                                                    artifact_path, &context, nullptr);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }
        std::string verify_reason;
        if (!zkgpt::gat::verify_gat(context, artifact, &verify_reason)) {
            const std::string report = report_json(
                "REJECTED", verify_reason.empty() ? "verify_gat rejected proof" : verify_reason,
                statement_digest, context_rel, logits_rel, artifact_rel,
                artifact_path, &context, &artifact);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }
        const std::string report = report_json("VERIFY_OK", "", statement_digest,
                                               context_rel, logits_rel, artifact_rel,
                                               artifact_path, &context, &artifact);
        write_report(options.report, report);
        std::cout << "VERIFY_OK\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "VERIFIER_REJECTED: " << error.what() << '\n';
        return 1;
    }
}
