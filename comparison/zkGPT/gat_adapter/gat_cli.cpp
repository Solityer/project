#include "gat_cuda.hpp"
#include "gat_proof.hpp"
#include "gat_types.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

using namespace zkgpt::gat;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset;
    std::string backend;
    std::string baseline_root = ".";
    std::string config;
    bool help = false;
    bool register_only = false;
};

struct Record {
    std::string schema = "zkgpt-gat-result-v1";
    std::string dataset;
    std::string backend;
    std::string status = "FAILED";
    std::string verification_status = "FAILED";
    std::string proof_scope = "full_gat";
    std::string privacy_policy = "model_and_trace_hidden";
    std::string privacy_profile = "masked_full";
    std::string input_privacy = "public_graph_public_features";
    std::string output_mode = "public_logits";
    std::string native_backend;
    std::string failure_reason;
    std::string statement_digest;
    std::string circuit_id;
    std::string plan_digest;
    std::string run_dir;
    std::string statement_path;
    std::string public_logits_path;
    std::string verifier_context_path;
    std::string registration_path;
    std::string registration_digest;
    std::string registration_binding_digest;
    std::string registration_layout_digest;
    std::string artifact_path;
    std::string proof_digest;
    std::string full_dataset_identity;
    std::string checkpoint_identity;
    std::string cache_identity;
    std::string com_w_point;
    bool proof_generated = false;
    bool verifier_called = false;
    bool verifier_ok = false;
    bool verifier_reads_witness = false;
    bool cpu_fallback = false;
    bool gpu_compiled = false;
    bool gpu_device_available = false;
    bool gpu_kernel_executed = false;
    std::uint64_t gpu_kernel_invocations = 0;
    std::uint64_t proof_message_bytes = 0;
    std::uint64_t proof_size_bytes = 0;
    double load_ms = 0.0;
    double registration_ms = 0.0;
    double witness_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    double serialize_ms = 0.0;
};

double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

bool absolute_path(const std::string &path) {
    return !path.empty() && (path[0] == '/' || path[0] == '\\');
}

std::string without_trailing_slash(std::string path) {
    while (path.size() > 1 && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    return path;
}

std::string join_path(const std::string &left, const std::string &right) {
    if (right.empty()) return left;
    if (left.empty() || left == ".") return right;
    return without_trailing_slash(left) + "/" + right;
}

std::string local_path(const std::string &root, const std::string &relative,
                       const char *what) {
    if (relative.empty() || absolute_path(relative)) {
        throw std::runtime_error(std::string("baseline-local ") + what + " must be a relative path");
    }
    std::size_t begin = 0;
    while (begin <= relative.size()) {
        const std::size_t slash = relative.find('/', begin);
        const std::string part = relative.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);
        if (part == "..") throw std::runtime_error(std::string("baseline-local ") + what + " escapes the baseline root");
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return join_path(root, relative);
}

void make_directory(const std::string &path) {
    if (path.empty() || path == ".") return;
    std::string current;
    if (path[0] == '/') current = "/";
    std::size_t begin = path[0] == '/' ? 1 : 0;
    while (begin <= path.size()) {
        const std::size_t slash = path.find('/', begin);
        const std::string part = path.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);
        if (!part.empty() && part != ".") {
            if (!current.empty() && current.back() != '/') current.push_back('/');
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("cannot create baseline output directory: " + current);
            }
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
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
                if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch) << std::dec;
                else out << static_cast<char>(ch);
        }
    }
    return out.str();
}

void write_text(const std::string &path, const std::string &content) {
    const std::string temporary = path + ".partial";
    {
        std::ofstream out(temporary.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open output file: " + path);
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!out.good()) throw std::runtime_error("cannot write output file: " + path);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot finalize output file: " + path);
    }
}

std::string rel_display(const std::string &root, const std::string &path) {
    const std::string clean_root = without_trailing_slash(root);
    if (clean_root == ".") return path;
    const std::string prefix = clean_root + "/";
    return path.compare(0, prefix.size(), prefix) == 0 ? path.substr(prefix.size()) : path;
}

std::string bool_json(bool value) { return value ? "true" : "false"; }

void json_string(std::ostringstream &out, const char *key, const std::string &value,
                 bool comma = true) {
    out << "  \"" << key << "\": \"" << json_escape(value) << "\"" << (comma ? ",\n" : "\n");
}

void json_bool(std::ostringstream &out, const char *key, bool value, bool comma = true) {
    out << "  \"" << key << "\": " << bool_json(value) << (comma ? ",\n" : "\n");
}

void json_number(std::ostringstream &out, const char *key, double value, bool comma = true) {
    out << "  \"" << key << "\": " << std::fixed << std::setprecision(3) << value << (comma ? ",\n" : "\n");
}

void write_result(const std::string &path, const Record &record) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", record.schema);
    json_string(out, "dataset", record.dataset);
    json_string(out, "backend", record.backend);
    json_string(out, "status", record.status);
    json_string(out, "verification_status", record.verification_status);
    json_string(out, "proof_scope", record.proof_scope);
    json_string(out, "privacy_policy", record.privacy_policy);
    json_string(out, "privacy_profile", record.privacy_profile);
    json_string(out, "input_privacy", record.input_privacy);
    json_string(out, "output_mode", record.output_mode);
    json_string(out, "native_backend", record.native_backend);
    json_string(out, "failure_reason", record.failure_reason);
    json_string(out, "statement_digest", record.statement_digest);
    json_string(out, "proof_digest", record.proof_digest);
    json_string(out, "circuit_id", record.circuit_id);
    json_string(out, "plan_digest", record.plan_digest);
    json_string(out, "full_dataset_identity", record.full_dataset_identity);
    json_string(out, "checkpoint_identity", record.checkpoint_identity);
    json_string(out, "cache_identity", record.cache_identity);
    json_string(out, "run_dir", record.run_dir);
    json_string(out, "statement_path", record.statement_path);
    json_string(out, "public_logits_path", record.public_logits_path);
    json_string(out, "verifier_context_path", record.verifier_context_path);
    json_string(out, "registration_path", record.registration_path);
    json_string(out, "registration_digest", record.registration_digest);
    json_string(out, "registration_binding_digest", record.registration_binding_digest);
    json_string(out, "registration_layout_digest", record.registration_layout_digest);
    json_string(out, "artifact_path", record.artifact_path);
    json_string(out, "com_W_point", record.com_w_point);
    json_bool(out, "proof_generated", record.proof_generated);
    json_bool(out, "verifier_called", record.verifier_called);
    json_bool(out, "verifier_ok", record.verifier_ok);
    json_bool(out, "verifier_reads_witness", record.verifier_reads_witness);
    json_bool(out, "cpu_fallback", record.cpu_fallback);
    json_bool(out, "gpu_compiled", record.gpu_compiled);
    json_bool(out, "gpu_device_available", record.gpu_device_available);
    json_bool(out, "gpu_kernel_executed", record.gpu_kernel_executed);
    out << "  \"gpu_kernel_invocations\": " << record.gpu_kernel_invocations << ",\n";
    out << "  \"proof_message_bytes\": " << record.proof_message_bytes << ",\n";
    out << "  \"proof_size_bytes\": " << record.proof_size_bytes << ",\n";
    out << "  \"timing_ms\": {\n";
    out << "    \"load\": " << std::fixed << std::setprecision(3) << record.load_ms << ",\n";
    out << "    \"registration\": " << record.registration_ms << ",\n";
    out << "    \"witness\": " << record.witness_ms << ",\n";
    out << "    \"prove\": " << record.prove_ms << ",\n";
    out << "    \"verify\": " << record.verify_ms << ",\n";
    out << "    \"serialize\": " << record.serialize_ms << "\n";
    out << "  }\n";
    out << "}\n";
    write_text(path, out.str());
}

void write_logits(const std::string &path, const Tensor2D &logits) {
    std::ostringstream out;
    out << logits.rows << ' ' << logits.cols << '\n';
    for (std::size_t row = 0; row < logits.rows; ++row) {
        for (std::size_t col = 0; col < logits.cols; ++col) {
            if (col != 0) out << ' ';
            out << logits.at(row, col);
        }
        out << '\n';
    }
    write_text(path, out.str());
}

void write_registration(const std::string &path, const ModelRegistration &registration,
                        const std::string &root, const std::string &dataset) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", "zkgpt-gat-model-registration-v1");
    json_string(out, "dataset", dataset);
    json_string(out, "status", registration.status);
    json_string(out, "model_arch_id", registration.model_arch_id);
    json_string(out, "quant_cfg_id", registration.quant_cfg_id);
    json_string(out, "static_tables_id", registration.static_tables_id);
    json_string(out, "field_id", registration.field_id);
    json_string(out, "field_modulus", registration.field_modulus);
    json_string(out, "manifest_digest", registration.manifest_digest);
    json_string(out, "model_digest", registration.model_digest);
    json_string(out, "bounds_digest", registration.bounds_digest);
    json_string(out, "quant_digest", registration.quant_digest);
    json_string(out, "layout_digest", registration.layout_digest);
    json_string(out, "selector", registration.selector);
    json_string(out, "opening_policy", registration.opening_policy);
    json_string(out, "binding_digest", registration.binding_digest);
    json_string(out, "registration_digest", registration.registration_digest);
    out << "  \"n_raw\": " << registration.n_raw << ",\n";
    out << "  \"n_in\": " << registration.n_in << ",\n";
    out << "  \"input_segments\": [\n";
    for (std::size_t index = 0; index < registration.input_segments.size(); ++index) {
        const InputSegment &segment = registration.input_segments[index];
        out << "    {\n";
        json_string(out, "name", segment.name);
        out << "      \"offset\": " << segment.offset << ",\n";
        out << "      \"logical_length\": " << segment.logical_length << ",\n";
        out << "      \"padded_length\": " << segment.padded_length << "\n";
        out << "    }" << (index + 1 == registration.input_segments.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    json_string(out, "commitment_scheme", registration.com_W.scheme);
    json_string(out, "basis_id", registration.com_W.basis_id);
    out << "  \"logical_size\": " << registration.com_W.logical_size << ",\n";
    out << "  \"padded_size\": " << registration.com_W.padded_size << ",\n";
    json_string(out, "point_hex", registration.com_W.point_hex, false);
    out << "}\n";
    (void)root;
    write_text(path, out.str());
}

void write_public_statement(const std::string &path, const Record &record,
                            const Graph &graph, const Tensor2D &features,
                            const QuantConfig &quant, const StaticTables &tables,
                            const GatCircuitPlan &plan) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", "zkgpt-gat-public-statement-v1");
    json_string(out, "relation_id", "R_GAT-zkGPT");
    json_string(out, "dataset", record.dataset);
    json_string(out, "backend", record.backend);
    json_string(out, "proof_task", "node_classification_full_gat");
    json_string(out, "proof_scope", "full_gat");
    json_string(out, "privacy_policy", "model_and_trace_hidden");
    json_string(out, "privacy_extension_profile_id", "masked_full");
    json_string(out, "input_privacy", "public_graph_public_features");
    json_string(out, "output_mode", "public_logits");
    json_string(out, "quant_cfg_id", quant.quant_cfg_id);
    json_string(out, "static_tables_id", tables.table_id);
    json_string(out, "static_tables_digest", tables.digest);
    json_string(out, "registration_digest", record.registration_digest);
    json_string(out, "registration_layout_digest", record.registration_layout_digest);
    json_string(out, "registration_binding_digest", record.registration_binding_digest);
    json_string(out, "circuit_id", plan.circuit_id);
    json_string(out, "plan_digest", plan.plan_digest);
    json_string(out, "statement_digest", record.statement_digest);
    out << "  \"num_nodes\": " << graph.num_nodes << ",\n";
    out << "  \"num_edges\": " << graph.edges() << ",\n";
    out << "  \"graph_count\": " << graph.graph_count << ",\n";
    out << "  \"num_features\": " << features.cols << ",\n";
    out << "  \"num_classes\": " << graph.num_classes << ",\n";
    json_string(out, "com_W_point", record.com_w_point, false);
    out << "}\n";
    write_text(path, out.str());
}

void write_verifier_context(const std::string &path, const Record &record,
                            const Graph &graph, const Tensor2D &features,
                            const QuantConfig &quant, const StaticTables &tables,
                            const GatCircuitPlan &plan) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", "zkgpt-gat-verifier-context-v1");
    json_string(out, "relation_id", "R_GAT-zkGPT");
    json_string(out, "dataset", record.dataset);
    json_string(out, "backend", record.backend);
    json_string(out, "privacy_extension_profile_id", "masked_full");
    json_string(out, "quant_cfg_id", quant.quant_cfg_id);
    json_string(out, "static_tables_id", tables.table_id);
    json_string(out, "static_tables_digest", tables.digest);
    json_string(out, "statement_digest", record.statement_digest);
    json_string(out, "public_statement_path", record.statement_path);
    json_string(out, "public_logits_path", record.public_logits_path);
    json_string(out, "registration_path", record.registration_path);
    json_string(out, "circuit_id", plan.circuit_id);
    json_string(out, "plan_digest", plan.plan_digest);
    out << "  \"num_nodes\": " << graph.num_nodes << ",\n";
    out << "  \"num_edges\": " << graph.edges() << ",\n";
    out << "  \"graph_count\": " << graph.graph_count << ",\n";
    out << "  \"public_features_rows\": " << features.rows << ",\n";
    out << "  \"public_features_cols\": " << features.cols << ",\n";
    json_string(out, "com_W_point", record.com_w_point, false);
    out << "}\n";
    write_text(path, out.str());
}

std::vector<Int> quantized_first_projection(const Model &model, const QuantConfig &quant) {
    const Tensor &tensor = model.tensor("hidden/layer0/head0/seq/kernel");
    if (tensor.shape.dims.size() != 3 || tensor.shape.dims[0] != 1) {
        throw std::runtime_error("canonical first projection tensor has invalid shape");
    }
    const Int scale = quant.s_w_hidden[0].at(0);
    std::vector<Int> values;
    values.reserve(tensor.values.size());
    for (double value : tensor.values) {
        const long double scaled = static_cast<long double>(value) * scale;
        const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
        if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) ||
            rounded < static_cast<long double>(std::numeric_limits<Int>::min())) {
            throw std::overflow_error("canonical projection weight does not fit int64");
        }
        values.push_back(static_cast<Int>(rounded));
    }
    return values;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            options.dataset = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--baseline-root" && i + 1 < argc) {
            options.baseline_root = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            options.config = argv[++i];
        } else if (arg == "--register-only") {
            options.register_only = true;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (options.help) return options;
    const std::vector<std::string> datasets = {"cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"};
    bool known_dataset = false;
    for (const std::string &dataset : datasets) if (options.dataset == dataset) known_dataset = true;
    if (!known_dataset) throw std::runtime_error("--dataset must be one of cora,citeseer,pubmed,ppi,ogbn_arxiv");
    if (options.backend != "cpu" && options.backend != "gpu") throw std::runtime_error("--backend must be cpu or gpu");
    if (options.baseline_root.empty()) throw std::runtime_error("--baseline-root is empty");
    if (options.config.empty()) options.config = "baseline-local/configs/" + options.dataset + "_secure.cfg";
    return options;
}

void print_help() {
    std::cout << "Usage: gat_cli --dataset <cora|citeseer|pubmed|ppi|ogbn_arxiv> "
                 "--backend <cpu|gpu> [--baseline-root <zkGPT-root>] [--config <baseline-relative.cfg>]\n"
                 "       gat_cli ... --register-only\n"
                 "The normal command emits one baseline-local run record and exits nonzero on FAILED or UNSUPPORTED.\n"
                 "--register-only authenticates and writes only the long-lived model registration; it does not load graph data.\n";
}

} // namespace

int main(int argc, char **argv) {
    Record record;
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        record.dataset = options.dataset;
        record.backend = options.backend;
        record.native_backend = options.backend == "gpu" ? "cuda_relation_bridge" : "cpu_native_zkgpt";
        const std::string root = without_trailing_slash(options.baseline_root);
        const std::string fallback_run_rel = "baseline-local/runs/" + options.dataset + "_" + options.backend;
        std::string run_rel = fallback_run_rel;
        std::string run_dir = local_path(root, run_rel, "run directory");
        make_directory(run_dir);
        const auto load_begin = Clock::now();

        const std::string config_path = local_path(root, options.config, "config");
        const CanonicalConfig config = CanonicalConfig::load(config_path);
        if (config.get("dataset") != options.dataset) throw std::runtime_error("dataset/config identity mismatch");
        if (config.get("privacy_policy") != record.privacy_policy ||
            config.get("input_privacy") != record.input_privacy ||
            config.get("output_mode") != record.output_mode ||
            config.get("proof_task") != "node_classification_full_gat") {
            throw std::runtime_error("config does not declare the strict full-GAT privacy/proof boundary");
        }
        const std::string configured_run = config.get("export_dir");
        if (!configured_run.empty()) {
            run_rel = configured_run + "/" + options.backend;
            run_dir = local_path(root, run_rel, "run directory");
            make_directory(run_dir);
        }
        record.run_dir = run_rel;
        const std::string cache_rel = config.get("cache_root");
        const std::string bundle_rel = config.get("checkpoint_bundle");
        if (cache_rel.empty() || bundle_rel.empty()) throw std::runtime_error("config is missing baseline-local cache/checkpoint paths");
        const std::string cache_dir = local_path(root, cache_rel + "/" + options.dataset, "cache");
        const std::string bundle_dir = local_path(root, bundle_rel, "checkpoint bundle");
        const std::string quant_path = local_path(root, "baseline-local/configs/quantization_scale_chain.cfg", "quantization config");
        const QuantConfig quant = QuantConfig::load(quant_path);
        if (quant.quant_cfg_id != config.get("quant_cfg_id")) throw std::runtime_error("quantization config identity mismatch");

        if (options.register_only) {
            const auto registration_begin = Clock::now();
            const Graph graph = load_graph(cache_dir);
            const Tensor2D features = load_features(cache_dir, config, graph);
            const Model model = Model::load(bundle_dir);
            model.validate_identity(config);
            const ForwardResult forward = run_forward(graph, features, model, quant);
            const GatCircuitPlan plan = build_gat_circuit_plan(graph, model, quant, forward);
            const ModelRegistration registration = register_model(model, quant, plan.input_layout, model.manifest.model_arch_id);
            record.registration_ms = elapsed_ms(registration_begin, Clock::now());
            record.com_w_point = registration.com_W.point_hex;
            record.registration_digest = registration.registration_digest;
            record.registration_binding_digest = registration.binding_digest;
            record.registration_layout_digest = registration.layout_digest;
            record.registration_path = run_rel + "/model_registration.json";
            record.checkpoint_identity = "baseline-local/checkpoints/" + options.dataset + "_gat/integrity.sha256";
            write_registration(run_dir + "/model_registration.json", registration, root, options.dataset);
            std::cout << "REGISTERED dataset=" << options.dataset
                      << " backend=" << options.backend
                      << " registration=" << record.registration_path << '\n';
            return 0;
        }

        const Graph graph = load_graph(cache_dir);
        const Tensor2D features = load_features(cache_dir, config, graph);
        (void)load_labels(cache_dir, graph);
        const Model model = Model::load(bundle_dir);
        model.validate_identity(config);
        const StaticTables tables = StaticTables::build(quant);
        if (!tables.authenticate()) throw std::runtime_error("canonical static table authentication failed");
        record.full_dataset_identity = "canonical:" + options.dataset;
        record.checkpoint_identity = "baseline-local/checkpoints/" + options.dataset + "_gat/integrity.sha256";
        record.cache_identity = "baseline-local/data/cache/" + options.dataset + "/integrity.sha256";
        record.load_ms = elapsed_ms(load_begin, Clock::now());

        if (options.backend == "gpu") {
            const std::vector<Int> weights = quantized_first_projection(model, quant);
            const GpuExecutionEvidence gpu = execute_gpu_matmul_aggregation(
                features, weights, features.cols,
                model.tensor("hidden/layer0/head0/seq/kernel").shape.dims.at(2), graph);
            record.gpu_compiled = gpu.compiled;
            record.gpu_device_available = gpu.device_available;
            record.gpu_kernel_executed = gpu.kernel_executed;
            record.gpu_kernel_invocations = gpu.kernel_invocations;
            if (!gpu.compiled || !gpu.device_available || !gpu.kernel_executed || gpu.kernel_invocations == 0) {
                record.status = gpu.compiled ? "FAILED" : "UNSUPPORTED";
                record.verification_status = record.status;
                record.failure_reason = gpu.diagnostic.empty() ? "GPU relation bridge did not execute" : gpu.diagnostic;
                write_result(run_dir + "/result.json", record);
                std::cerr << record.status << ": " << record.failure_reason << '\n';
                return record.status == "UNSUPPORTED" ? 2 : 1;
            }
        }

        const auto witness_begin = Clock::now();
        const ForwardResult forward = run_forward(graph, features, model, quant);
        if (!validate_relation(graph, features, model, quant, forward)) throw std::runtime_error("canonical GAT relation validation failed");
        const GatCircuitPlan plan = build_gat_circuit_plan(graph, model, quant, forward);
        if (!plan.official_layered || plan.has_placeholder) throw std::runtime_error("official layered GAT circuit plan is unavailable");
        record.witness_ms = elapsed_ms(witness_begin, Clock::now());

        const auto registration_begin = Clock::now();
        const ModelRegistration registration = register_model(model, quant, plan.input_layout, model.manifest.model_arch_id);
        record.registration_ms = elapsed_ms(registration_begin, Clock::now());
        record.com_w_point = registration.com_W.point_hex;
        record.registration_digest = registration.registration_digest;
        record.registration_binding_digest = registration.binding_digest;
        record.registration_layout_digest = registration.layout_digest;
        record.registration_path = run_rel + "/model_registration.json";
        write_registration(run_dir + "/model_registration.json", registration, root, options.dataset);

        const VerifierContext context = make_verifier_context(
            graph, features, quant, tables, forward.logits, registration,
            plan, record.privacy_profile);
        record.statement_digest = verifier_context_digest(context);
        record.circuit_id = plan.circuit_id;
        record.plan_digest = plan.plan_digest;
        record.statement_path = run_rel + "/public_statement.json";
        record.public_logits_path = run_rel + "/public_logits.txt";
        record.verifier_context_path = run_rel + "/verifier_context.json";
        write_logits(run_dir + "/public_logits.txt", forward.logits);
        write_public_statement(run_dir + "/public_statement.json", record, graph, features, quant, tables, plan);
        write_verifier_context(run_dir + "/verifier_context.json", record, graph, features, quant, tables, plan);

        const auto prove_begin = Clock::now();
        const ProveResult proved = prove_gat(plan, graph, features, model, quant, forward,
                                             registration, record.privacy_profile);
        record.prove_ms = elapsed_ms(prove_begin, Clock::now());
        if (proved.status != ProveStatus::GENERATED || !proved.has_artifact) {
            record.status = proved.status == ProveStatus::UNSUPPORTED_PRIVACY_PROFILE ? "UNSUPPORTED" : "FAILED";
            record.verification_status = record.status;
            record.failure_reason = proved.diagnostic.empty() ? "native prover did not generate a proof artifact" : proved.diagnostic;
            write_result(run_dir + "/result.json", record);
            std::cerr << record.status << ": " << record.failure_reason << '\n';
            return record.status == "UNSUPPORTED" ? 2 : 1;
        }

        const std::string artifact_file = run_dir + "/proof.artifact";
        const auto serialize_begin = Clock::now();
        std::string artifact_reason;
        if (!proved.artifact.write(artifact_file, &artifact_reason)) throw std::runtime_error("proof artifact serialization failed: " + artifact_reason);
        GatProofArtifact artifact;
        if (!GatProofArtifact::read(artifact_file, artifact, &artifact_reason)) throw std::runtime_error("serialized proof artifact cannot be read: " + artifact_reason);
        record.serialize_ms = elapsed_ms(serialize_begin, Clock::now());
        record.artifact_path = run_rel + "/proof.artifact";
        record.proof_generated = true;
        record.proof_digest = artifact.transcript_digest;
        for (const ProofSection *section : {&artifact.Pi_Lasso, &artifact.Pi_MatMul, &artifact.Pi_GKR, &artifact.Pi_combine, &artifact.Pi_open}) {
            for (const TranscriptMessage &message : section->messages) {
                record.proof_message_bytes += static_cast<std::uint64_t>(message.label.size() + message.payload.size());
            }
        }
        struct stat artifact_stat;
        if (::stat(artifact_file.c_str(), &artifact_stat) != 0 || artifact_stat.st_size < 0) throw std::runtime_error("cannot stat proof artifact");
        record.proof_size_bytes = static_cast<std::uint64_t>(artifact_stat.st_size);

        const auto verify_begin = Clock::now();
        std::string verify_reason;
        record.verifier_called = true;
        record.verifier_ok = verify_gat(context, artifact, &verify_reason);
        record.verify_ms = elapsed_ms(verify_begin, Clock::now());
        if (!record.verifier_ok) {
            record.status = "FAILED";
            record.verification_status = "FAILED";
            record.failure_reason = verify_reason.empty() ? "independent verifier rejected proof artifact" : verify_reason;
        } else {
            record.status = "VERIFY_OK";
            record.verification_status = "VERIFY_OK";
        }
        write_result(run_dir + "/result.json", record);
        std::cout << record.status << " dataset=" << record.dataset << " backend=" << record.backend
                  << " result=" << record.run_dir + "/result.json" << '\n';
        return record.status == "VERIFY_OK" ? 0 : 1;
    } catch (const std::exception &error) {
        record.status = record.status == "UNSUPPORTED" ? "UNSUPPORTED" : "FAILED";
        record.verification_status = record.status;
        record.failure_reason = error.what();
        try {
            const std::string root = without_trailing_slash(record.run_dir.empty() ? "." : ".");
            (void)root;
        } catch (...) {
        }
        std::cerr << record.status << ": " << record.failure_reason << '\n';
        return record.status == "UNSUPPORTED" ? 2 : 1;
    }
}
