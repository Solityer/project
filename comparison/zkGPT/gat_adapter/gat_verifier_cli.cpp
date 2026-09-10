#include "gat_proof.hpp"

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

// This executable is deliberately a separate process from gat_cli.  Its
// input surface is limited to the serialized verifier context, public logits,
// and proof artifact.  In particular it never receives a checkpoint path,
// witness path, model object, or prover state.
namespace {

using zkgpt::gat::GatProofArtifact;

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
        if (part == "..") {
            throw std::runtime_error(std::string(description) + " escapes baseline root");
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return trim_root(root) + "/" + relative;
}

void require_regular(const std::string &path, const char *description) {
    struct stat info;
    if (::lstat(path.c_str(), &info) != 0) {
        throw std::runtime_error(std::string("missing ") + description + ": " + path);
    }
    if (S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_nlink != 1) {
        throw std::runtime_error(std::string(description) + " must be a private regular file: " + path);
    }
}

std::string read_text(const std::string &path, const char *description,
                      std::size_t limit = 256U * 1024U * 1024U) {
    require_regular(path, description);
    struct stat info;
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

// The producer emits a deliberately small, canonical JSON object.  This
// parser only accepts JSON string fields and is used for identity checks; it
// is not a second proof or statement encoder.
bool json_string_field(const std::string &json, const std::string &key,
                       std::string &value) {
    const std::string needle = "\"" + key + "\"";
    std::size_t cursor = 0;
    while ((cursor = json.find(needle, cursor)) != std::string::npos) {
        std::size_t colon = cursor + needle.size();
        while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' ||
                                       json[colon] == '\r' || json[colon] == '\t')) ++colon;
        if (colon >= json.size() || json[colon] != ':') {
            cursor += needle.size();
            continue;
        }
        ++colon;
        while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' ||
                                       json[colon] == '\r' || json[colon] == '\t')) ++colon;
        if (colon >= json.size() || json[colon] != '"') return false;
        ++colon;
        std::ostringstream decoded;
        bool escaped = false;
        for (; colon < json.size(); ++colon) {
            const char ch = json[colon];
            if (escaped) {
                switch (ch) {
                    case '"': decoded << '"'; break;
                    case '\\': decoded << '\\'; break;
                    case '/': decoded << '/'; break;
                    case 'n': decoded << '\n'; break;
                    case 'r': decoded << '\r'; break;
                    case 't': decoded << '\t'; break;
                    default: return false;
                }
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                value = decoded.str();
                return true;
            } else {
                decoded << ch;
            }
        }
        return false;
    }
    return false;
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
                        const std::string &dataset, const std::string &backend,
                        const std::string &statement_digest,
                        const std::string &context, const std::string &logits,
                        const std::string &artifact) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema\": \"zkgpt-gat-independent-verifier-v1\",\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"independent\": true,\n"
        << "  \"verifier_reads_witness\": false,\n"
        << "  \"verifier_reads_checkpoint\": false,\n"
        << "  \"dataset\": \"" << json_escape(dataset) << "\",\n"
        << "  \"backend\": \"" << json_escape(backend) << "\",\n"
        << "  \"statement_digest\": \"" << json_escape(statement_digest) << "\",\n"
        << "  \"verifier_context_path\": \"" << json_escape(context) << "\",\n"
        << "  \"public_logits_path\": \"" << json_escape(logits) << "\",\n"
        << "  \"artifact_path\": \"" << json_escape(artifact) << "\",\n"
        << "  \"reason\": \"" << json_escape(reason) << "\"\n"
        << "}\n";
    return out.str();
}

void print_help() {
    std::cout << "Usage: zkgpt_gat_verifier --verifier-context <relative-json> "
                 "--public-logits <relative-text> --artifact <relative-artifact> "
                 "[--baseline-root <zkGPT-root>] [--report <relative-json>]\n"
                 "       zkgpt_gat_verifier --capabilities\n"
                 "The verifier process reads only the serialized public context, public logits and proof artifact.\n";
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
            std::cout << "{\"schema\":\"zkgpt-gat-verifier-capabilities-v1\","
                         "\"independent\":true,\"masked_full\":false,"
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
        const std::string context_json = read_text(context_path, "verifier context");
        const std::string logits_text = read_text(logits_path, "public logits");
        if (logits_text.empty()) throw std::runtime_error("public logits are empty");

        std::string schema;
        std::string relation;
        std::string profile;
        std::string dataset;
        std::string backend;
        std::string statement_digest;
        std::string context_logits;
        std::string registration_rel;
        std::string context_registration_digest;
        std::string context_registration_layout_digest;
        std::string context_registration_binding_digest;
        std::string context_com_w_point;
        if (!json_string_field(context_json, "schema", schema) || schema != "zkgpt-gat-verifier-context-v1") {
            throw std::runtime_error("unsupported verifier context schema");
        }
        if (!json_string_field(context_json, "relation_id", relation) || relation != "R_GAT-zkGPT") {
            throw std::runtime_error("verifier context relation mismatch");
        }
        if (!json_string_field(context_json, "privacy_extension_profile_id", profile) || profile != "masked_full") {
            throw std::runtime_error("unsupported privacy profile in verifier context");
        }
        if (!json_string_field(context_json, "statement_digest", statement_digest) || statement_digest.empty()) {
            throw std::runtime_error("verifier context statement digest is missing");
        }
        (void)json_string_field(context_json, "dataset", dataset);
        (void)json_string_field(context_json, "backend", backend);
        if (!json_string_field(context_json, "public_logits_path", context_logits) || context_logits != logits_rel) {
            throw std::runtime_error("public logits path does not match verifier context");
        }
        if (!json_string_field(context_json, "registration_path", registration_rel) || registration_rel.empty()) {
            throw std::runtime_error("verifier context registration path is missing");
        }
        if (!json_string_field(context_json, "registration_digest", context_registration_digest) || context_registration_digest.empty() ||
            !json_string_field(context_json, "registration_layout_digest", context_registration_layout_digest) || context_registration_layout_digest.empty() ||
            !json_string_field(context_json, "registration_binding_digest", context_registration_binding_digest) || context_registration_binding_digest.empty() ||
            !json_string_field(context_json, "com_W_point", context_com_w_point) || context_com_w_point.empty()) {
            throw std::runtime_error("verifier context registration identity is incomplete");
        }
        // The registration artifact is public verifier material.  It is read
        // from the same baseline-local run directory and contains only
        // commitment/identity metadata; this process never opens a
        // checkpoint, tensor bundle, or witness path.
        const std::string registration_path = local_path(root, registration_rel, "model registration");
        const std::string registration_json = read_text(registration_path, "model registration");
        std::string registration_schema;
        std::string registration_status;
        std::string registration_digest;
        std::string registration_layout_digest;
        std::string registration_binding_digest;
        std::string registration_selector;
        std::string registration_opening_policy;
        std::string registration_commitment_scheme;
        std::string registration_point;
        std::string registration_field_id;
        if (!json_string_field(registration_json, "schema", registration_schema) || registration_schema != "zkgpt-gat-model-registration-v1" ||
            !json_string_field(registration_json, "status", registration_status) || registration_status != "registered_long_lived_com_W" ||
            !json_string_field(registration_json, "registration_digest", registration_digest) ||
            !json_string_field(registration_json, "layout_digest", registration_layout_digest) ||
            !json_string_field(registration_json, "binding_digest", registration_binding_digest) ||
            !json_string_field(registration_json, "selector", registration_selector) ||
            !json_string_field(registration_json, "opening_policy", registration_opening_policy) ||
            !json_string_field(registration_json, "commitment_scheme", registration_commitment_scheme) ||
            !json_string_field(registration_json, "point_hex", registration_point) ||
            !json_string_field(registration_json, "field_id", registration_field_id)) {
            throw std::runtime_error("model registration artifact is incomplete");
        }
        if (registration_digest != context_registration_digest ||
            registration_layout_digest != context_registration_layout_digest ||
            registration_binding_digest != context_registration_binding_digest ||
            registration_point != context_com_w_point ||
            registration_selector != "00:X_pub,01:W,10:Aux,11:L_qry" ||
            registration_opening_policy != "same_r_three_openings_v1" ||
            registration_commitment_scheme != "hyrax-pedersen-bn254" ||
            registration_field_id != "BN254") {
            throw std::runtime_error("model registration artifact does not match public verifier context");
        }

        GatProofArtifact artifact;
        std::string artifact_reason;
        if (!GatProofArtifact::read(artifact_path, artifact, &artifact_reason)) {
            const std::string report = report_json("REJECTED", artifact_reason,
                                                    dataset, backend, statement_digest,
                                                    context_rel, logits_rel, artifact_rel);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }
        if (artifact.statement_digest != statement_digest) {
            const std::string report = report_json("REJECTED", "proof statement digest does not match public context",
                                                    dataset, backend, statement_digest,
                                                    context_rel, logits_rel, artifact_rel);
            write_report(options.report, report);
            std::cout << report;
            return 1;
        }

        // Current official zkGPT does not expose a serializable masked_full
        // verifier transcript.  A valid-looking wire is therefore still an
        // unsupported proof, never a successful verification.
        const std::string report = report_json(
            "UNSUPPORTED",
            "unsupported_privacy_profile: native masked_full verifier transcript is unavailable",
            dataset, backend, statement_digest, context_rel, logits_rel, artifact_rel);
        write_report(options.report, report);
        std::cout << report;
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "VERIFIER_REJECTED: " << error.what() << '\n';
        return 1;
    }
}
