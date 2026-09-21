#include "gat_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <set>

namespace zkgpt {
namespace gat {
namespace {

std::string trim(const std::string &in) {
    const std::string ws = " \t\r\n";
    const std::size_t first = in.find_first_not_of(ws);
    if (first == std::string::npos) return std::string();
    const std::size_t last = in.find_last_not_of(ws);
    return in.substr(first, last - first + 1);
}

std::string strip_comment(const std::string &in) {
    const std::size_t p = in.find('#');
    return trim(p == std::string::npos ? in : in.substr(0, p));
}

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> out;
    std::string token;
    std::istringstream stream(s);
    while (std::getline(stream, token, delimiter)) out.push_back(trim(token));
    return out;
}

std::string require_file(const std::string &path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error("canonical file is not readable: " + path);
    return path;
}

std::string sha256_hex(const std::string &bytes) {
    unsigned char digest[32] = {};
    if (mcl::fp::sha256(digest, sizeof(digest), bytes.data(),
                        static_cast<std::uint32_t>(bytes.size())) != sizeof(digest)) {
        throw std::runtime_error("SHA-256 implementation failed");
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string sha256_file(const std::string &path) {
    require_file(path);
    std::ifstream in(path.c_str(), std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    return sha256_hex(contents.str());
}

std::string basename_of(const std::string &path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

void verify_sha256_manifest(const std::string &root, const std::string &manifest,
                            const std::vector<std::string> &required) {
    require_file(manifest);
    std::ifstream in(manifest.c_str());
    std::map<std::string, std::string> expected;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string digest;
        std::string relative;
        if (!(row >> digest >> relative)) throw std::runtime_error("malformed SHA-256 manifest: " + manifest);
        std::string trailing;
        if (row >> trailing) throw std::runtime_error("extra fields in SHA-256 manifest: " + manifest);
        if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos ||
            relative.empty() || relative[0] == '/' || relative.find("..") != std::string::npos ||
            relative.find('/') != std::string::npos || relative.find('\\') != std::string::npos) {
            throw std::runtime_error("invalid SHA-256 manifest entry: " + manifest);
        }
        if (!expected.emplace(relative, digest).second) throw std::runtime_error("duplicate SHA-256 manifest entry: " + relative);
    }
    for (const std::string &name : required) {
        const auto it = expected.find(name);
        if (it == expected.end()) throw std::runtime_error("SHA-256 manifest missing required file: " + name);
        const std::string actual = sha256_file(root + "/" + name);
        if (actual != it->second) throw std::runtime_error("SHA-256 mismatch for " + root + "/" + name);
    }
    for (const auto &entry : expected) {
        const std::string actual = sha256_file(root + "/" + entry.first);
        if (actual != entry.second) throw std::runtime_error("SHA-256 mismatch for " + root + "/" + entry.first);
    }
}

void verify_config_auth(const std::string &path) {
    const std::string sidecar = path + ".sha256";
    const std::string root = path.substr(0, path.find_last_of("/\\"));
    verify_sha256_manifest(root.empty() ? "." : root, sidecar, {basename_of(path)});
}

void verify_cache_auth(const std::string &cache_dir) {
    const std::string manifest = cache_dir + "/integrity.sha256";
    require_file(manifest);
    std::vector<std::string> required{"meta.cfg", "edges.txt"};
    if (std::ifstream((cache_dir + "/features.txt").c_str())) required.push_back("features.txt");
    else required.push_back("features.npy");
    if (std::ifstream((cache_dir + "/labels.txt").c_str())) required.push_back("labels.txt");
    else if (std::ifstream((cache_dir + "/labels_multi.txt").c_str())) {
        required.push_back("labels_multi.txt");
        if (std::ifstream((cache_dir + "/labels_multi.npy").c_str())) required.push_back("labels_multi.npy");
    } else required.push_back("labels_multi.npy");
    // These files are part of the canonical PPI payload whenever present;
    // include them in the authenticated set instead of leaving optional
    // masks/pointers outside the integrity boundary.
    const char *optional[] = {"node_ptr.txt", "test.index", "train_mask.npy", "val_mask.npy", "test_mask.npy"};
    for (const char *name : optional) if (std::ifstream((cache_dir + "/" + name).c_str())) required.push_back(name);
    verify_sha256_manifest(cache_dir, manifest, required);
}

void verify_checkpoint_auth(const std::string &bundle_dir) {
    verify_sha256_manifest(bundle_dir, bundle_dir + "/integrity.sha256",
                           {"manifest.json", "tensors.txt", "tensors.npz",
                            "best_model.pt", "qat_best_logits.npy", "training_summary.json"});
}

Int quantize_double(double value, Int scale) {
    if (scale <= 0) throw std::runtime_error("quantization scale must be positive");
    const long double scaled = static_cast<long double>(value) * scale;
    const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L)
                                            : std::ceil(scaled - 0.5L);
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) ||
        rounded < static_cast<long double>(std::numeric_limits<Int>::min())) {
        throw std::overflow_error("canonical input does not fit int64");
    }
    return static_cast<Int>(rounded);
}

std::map<std::string, std::string> read_key_values(const std::string &path) {
    require_file(path);
    std::ifstream in(path.c_str());
    std::map<std::string, std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        line = strip_comment(line);
        if (line.empty()) continue;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) throw std::runtime_error("malformed config line in " + path);
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty()) throw std::runtime_error("empty config key in " + path);
        out[key] = value;
    }
    return out;
}

bool config_bool(const CanonicalConfig &config, const std::string &key, bool fallback) {
    const std::string value = config.get(key, fallback ? "true" : "false");
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::runtime_error("invalid boolean for " + key + ": " + value);
}

Int parse_int(const std::string &s, const std::string &what) {
    try {
        std::size_t used = 0;
        const long long value = std::stoll(trim(s), &used, 0);
        if (used != trim(s).size()) throw std::runtime_error("trailing characters");
        return static_cast<Int>(value);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid integer for " + what + ": " + s);
    }
}

Int checked_mul_for_config(Int lhs, Int rhs, const std::string &what) {
    const Wide product = static_cast<Wide>(lhs) * static_cast<Wide>(rhs);
    if (product > static_cast<Wide>(std::numeric_limits<Int>::max()) ||
        product < static_cast<Wide>(std::numeric_limits<Int>::min())) {
        throw std::runtime_error("integer overflow while parsing " + what);
    }
    return static_cast<Int>(product);
}

std::string manifest_string(const std::string &text, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) return std::string();
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return std::string();
    const std::size_t first = text.find('"', colon + 1);
    if (first == std::string::npos) return std::string();
    const std::size_t last = text.find('"', first + 1);
    if (last == std::string::npos) return std::string();
    return text.substr(first + 1, last - first - 1);
}

std::size_t manifest_number(const std::string &text, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) return 0;
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return 0;
    std::size_t first = colon + 1;
    while (first < text.size() && (text[first] == ' ' || text[first] == '\n' || text[first] == '\r' || text[first] == '\t')) ++first;
    std::size_t last = first;
    while (last < text.size() && text[last] >= '0' && text[last] <= '9') ++last;
    return first == last ? 0 : static_cast<std::size_t>(std::stoull(text.substr(first, last - first)));
}

std::string manifest_required_string(const std::string &text, const std::string &key) {
    const std::string value = manifest_string(text, key);
    if (value.empty()) throw std::runtime_error("checkpoint manifest is missing required field: " + key);
    return value;
}

struct NpyArray {
    std::string descr;
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

std::size_t little_u16(const std::string &bytes, std::size_t p) {
    return static_cast<unsigned char>(bytes[p]) |
           (static_cast<std::size_t>(static_cast<unsigned char>(bytes[p + 1])) << 8);
}

std::size_t little_u32(const std::string &bytes, std::size_t p) {
    return static_cast<std::size_t>(static_cast<unsigned char>(bytes[p])) |
           (static_cast<std::size_t>(static_cast<unsigned char>(bytes[p + 1])) << 8) |
           (static_cast<std::size_t>(static_cast<unsigned char>(bytes[p + 2])) << 16) |
           (static_cast<std::size_t>(static_cast<unsigned char>(bytes[p + 3])) << 24);
}

NpyArray load_npy(const std::string &path) {
    require_file(path);
    std::ifstream in(path.c_str(), std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 10 || static_cast<unsigned char>(bytes[0]) != 0x93 ||
        bytes.substr(1, 5) != "NUMPY") {
        throw std::runtime_error("invalid NumPy header: " + path);
    }
    const unsigned char major = static_cast<unsigned char>(bytes[6]);
    std::size_t header_len = 0;
    std::size_t header_pos = 0;
    if (major == 1) {
        header_len = little_u16(bytes, 8);
        header_pos = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12) throw std::runtime_error("truncated NumPy header: " + path);
        header_len = little_u32(bytes, 8);
        header_pos = 12;
    } else {
        throw std::runtime_error("unsupported NumPy version in " + path);
    }
    if (header_pos + header_len > bytes.size()) throw std::runtime_error("truncated NumPy payload: " + path);
    const std::string header = bytes.substr(header_pos, header_len);
    const std::size_t descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) throw std::runtime_error("NumPy descr missing: " + path);
    const std::size_t descr_first = header.find('\'', header.find(':', descr_pos) + 1);
    const std::size_t descr_last = descr_first == std::string::npos ? std::string::npos : header.find('\'', descr_first + 1);
    if (descr_first == std::string::npos || descr_last == std::string::npos) throw std::runtime_error("NumPy descr malformed: " + path);
    NpyArray array;
    array.descr = header.substr(descr_first + 1, descr_last - descr_first - 1);
    const std::size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) throw std::runtime_error("NumPy shape missing: " + path);
    const std::size_t open = header.find('(', shape_pos);
    const std::size_t close = open == std::string::npos ? std::string::npos : header.find(')', open + 1);
    if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("NumPy shape malformed: " + path);
    for (const std::string &part : split(header.substr(open + 1, close - open - 1), ',')) {
        if (!part.empty()) array.shape.push_back(static_cast<std::size_t>(parse_int(part, "NumPy shape")));
    }
    if (array.shape.empty()) array.shape.push_back(1);
    std::size_t count = 1;
    for (std::size_t d : array.shape) {
        if (d == 0 || count > std::numeric_limits<std::size_t>::max() / d) throw std::runtime_error("invalid NumPy shape");
        count *= d;
    }
    const std::size_t item_size = array.descr == "<f4" || array.descr == "|f4" || array.descr == "<i4" ? 4 :
                                  array.descr == "<f8" || array.descr == "|f8" || array.descr == "<i8" ? 8 :
                                  array.descr == "<u1" || array.descr == "|u1" ? 1 : 0;
    if (item_size == 0) throw std::runtime_error("unsupported NumPy dtype " + array.descr);
    const std::size_t data_pos = header_pos + header_len;
    if (count > (bytes.size() - data_pos) / item_size ||
        data_pos + count * item_size != bytes.size()) {
        throw std::runtime_error("NumPy payload has an invalid element count or trailing bytes: " + path);
    }
    array.values.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const char *p = bytes.data() + data_pos + i * item_size;
        if (array.descr == "<f4" || array.descr == "|f4") {
            float v;
            std::memcpy(&v, p, sizeof(v));
            array.values[i] = v;
        } else if (array.descr == "<f8" || array.descr == "|f8") {
            double v;
            std::memcpy(&v, p, sizeof(v));
            array.values[i] = v;
        } else if (array.descr == "<i4") {
            std::int32_t v;
            std::memcpy(&v, p, sizeof(v));
            array.values[i] = static_cast<double>(v);
        } else if (array.descr == "<i8") {
            std::int64_t v;
            std::memcpy(&v, p, sizeof(v));
            array.values[i] = static_cast<double>(v);
        } else {
            array.values[i] = static_cast<unsigned char>(*p);
        }
        if (!std::isfinite(array.values[i])) throw std::runtime_error("non-finite NumPy value: " + path);
    }
    return array;
}

std::vector<Int> parse_integer_line(const std::string &line) {
    std::vector<Int> values;
    std::istringstream in(line);
    std::string token;
    while (in >> token) values.push_back(parse_int(token, "integer row"));
    return values;
}

void validate_tensor(const Tensor &tensor) {
    if (tensor.shape.dims.empty() || tensor.values.size() != tensor.shape.elements()) {
        throw std::runtime_error("tensor payload/shape mismatch: " + tensor.name);
    }
}

std::string hidden_name(std::size_t layer, std::size_t head, const char *kind) {
    std::ostringstream out;
    out << "hidden/layer" << layer << "/head" << head << "/" << kind;
    return out.str();
}

std::string output_name(std::size_t head, const char *kind) {
    std::ostringstream out;
    out << "output/head" << head << "/" << kind;
    return out.str();
}

} // namespace

std::string sha256_file_digest(const std::string &path) {
    // Keep one implementation for authenticated resources and native
    // artifact telemetry.  The helper above reads the exact bytes and uses
    // mcl's SHA-256 primitive; no shell command or digest normalization is
    // involved.
    return sha256_file(path);
}

std::size_t Shape::elements() const {
    std::size_t result = 1;
    for (std::size_t d : dims) {
        if (d == 0 || result > std::numeric_limits<std::size_t>::max() / d) throw std::overflow_error("tensor shape overflow");
        result *= d;
    }
    return result;
}

CanonicalConfig CanonicalConfig::load(const std::string &path) {
    CanonicalConfig config;
    config.source_path = path;
    verify_config_auth(path);
    config.values = read_key_values(path);
    return config;
}

std::string CanonicalConfig::get(const std::string &key, const std::string &fallback) const {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
}

Int CanonicalConfig::get_int(const std::string &key, Int fallback) const {
    const auto it = values.find(key);
    return it == values.end() ? fallback : parse_int(it->second, key);
}

std::vector<Int> CanonicalConfig::get_int_list(const std::string &key) const {
    const auto it = values.find(key);
    if (it == values.end()) return std::vector<Int>();
    std::string value = it->second;
    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());
    std::vector<Int> out;
    for (const std::string &part : split(value, ',')) if (!part.empty()) out.push_back(parse_int(part, key));
    return out;
}

QuantConfig QuantConfig::load(const std::string &path) {
    const CanonicalConfig config = CanonicalConfig::load(path);
    QuantConfig quant;
    quant.schema = config.get("schema");
    quant.quant_cfg_id = config.get("quant_cfg_id");
    quant.s_input = config.get_int("s_input");
    quant.s_delta = config.get_int("s_delta");
    quant.s_exp = config.get_int("s_exp");
    const std::vector<Int> h = config.get_int_list("s_h");
    if (h.size() != 3) throw std::runtime_error("s_h must contain three scales");
    std::copy(h.begin(), h.end(), quant.s_h.begin());
    for (std::size_t l = 0; l < 2; ++l) {
        std::ostringstream key;
        key << "s_w_hidden_" << l;
        quant.s_w_hidden[l] = config.get_int_list(key.str());
        key.str(""); key.clear(); key << "s_h_prime_hidden_" << l;
        quant.s_hprime_hidden[l] = config.get_int_list(key.str());
        key.str(""); key.clear(); key << "s_a_src_hidden_" << l;
        quant.s_a_src_hidden[l] = config.get_int_list(key.str());
        key.str(""); key.clear(); key << "s_a_dst_hidden_" << l;
        quant.s_a_dst_hidden[l] = config.get_int_list(key.str());
        if (quant.s_w_hidden[l].size() != 8 || quant.s_hprime_hidden[l].size() != 8 ||
            quant.s_a_src_hidden[l].size() != 8 || quant.s_a_dst_hidden[l].size() != 8) {
            throw std::runtime_error("each hidden scale family must contain eight entries");
        }
    }
    const std::vector<Int> agg_pre = config.get_int_list("s_agg_pre");
    const std::vector<Int> agg = config.get_int_list("s_agg");
    const std::vector<Int> cat = config.get_int_list("s_cat");
    if (agg_pre.size() != 2 || agg.size() != 2 || cat.size() != 2) throw std::runtime_error("aggregation scales must contain two entries");
    std::copy(agg_pre.begin(), agg_pre.end(), quant.s_agg_pre.begin());
    std::copy(agg.begin(), agg.end(), quant.s_agg.begin());
    std::copy(cat.begin(), cat.end(), quant.s_cat.begin());
    quant.s_out_pre = config.get_int("s_out_pre");
    quant.s_out_head = config.get_int("s_out_head");
    quant.s_out = config.get_int("s_out");
    const std::vector<Int> wo = config.get_int_list("s_w_output");
    const std::vector<Int> yp = config.get_int_list("s_y_prime");
    const std::vector<Int> as = config.get_int_list("s_a_src_output");
    const std::vector<Int> ad = config.get_int_list("s_a_dst_output");
    const std::vector<Int> bo = config.get_int_list("s_b_output");
    if (wo.size() != 2 || yp.size() != 2 || as.size() != 2 || ad.size() != 2 || bo.size() != 2) throw std::runtime_error("output scales must contain two entries");
    std::copy(wo.begin(), wo.end(), quant.s_w_output.begin());
    std::copy(yp.begin(), yp.end(), quant.s_yprime_output.begin());
    std::copy(as.begin(), as.end(), quant.s_a_src_output.begin());
    std::copy(ad.begin(), ad.end(), quant.s_a_dst_output.begin());
    std::copy(bo.begin(), bo.end(), quant.s_b_output.begin());
    quant.rounding = config.get("round");
    quant.rescale = config.get("rescale");
    quant.output_average = config.get("output_average");
    // The canonical per-dataset configs use the singular key
    // `static_table_id`; older adapter drafts used `static_tables_id`.
    // Accept both spellings at this boundary, but expose one identity to the
    // rest of the proof path.
    quant.static_tables_id = config.get(
        "static_table_id", config.get("static_tables_id", quant.static_tables_id));
    quant.static_table_generator_id = config.get(
        "static_table_generator_id", quant.static_table_generator_id);
    quant.static_row_order = config.get("static_row_order", quant.static_row_order);
    quant.exp_approx_id = config.get("exp_approx_id", quant.exp_approx_id);
    quant.elu_approx_id = config.get("elu_approx_id", quant.elu_approx_id);
    quant.lrelu_slope_num = config.get_int("lrelu_slope_num", quant.lrelu_slope_num);
    quant.lrelu_slope_den = config.get_int("lrelu_slope_den", quant.lrelu_slope_den);
    quant.range_bits = config.get_int("range_bits", quant.range_bits);

    // A scale-chain file may omit the table interval fields.  In that case
    // derive the fixed, public interval from its explicit range_bits policy;
    // never derive it from an observed forward trace.  Explicit interval
    // fields, when present in a future canonical chain, take precedence.
    const bool has_lrelu_min = config.values.count("static_lrelu_min") != 0;
    const bool has_lrelu_max = config.values.count("static_lrelu_max") != 0;
    const bool has_exp_min = config.values.count("static_exp_min") != 0;
    const bool has_exp_max = config.values.count("static_exp_max") != 0;
    const bool has_elu_min = config.values.count("static_elu_min") != 0;
    const bool has_elu_max = config.values.count("static_elu_max") != 0;
    if (quant.range_bits <= 0 || quant.range_bits > 22) {
        throw std::runtime_error("canonical range_bits must be in [1,22] for zkGPT static tables");
    }
    const Int range = static_cast<Int>(Int(1) << quant.range_bits);
    if (!has_lrelu_min) quant.static_lrelu_min = -range;
    else quant.static_lrelu_min = config.get_int("static_lrelu_min");
    if (!has_lrelu_max) quant.static_lrelu_max = range;
    else quant.static_lrelu_max = config.get_int("static_lrelu_max");
    if (!has_exp_min) quant.static_exp_min = 0;
    else quant.static_exp_min = config.get_int("static_exp_min");
    if (!has_exp_max) quant.static_exp_max = checked_mul_for_config(range, 2, "static_exp_max");
    else quant.static_exp_max = config.get_int("static_exp_max");
    if (!has_elu_min) quant.static_elu_min = -range;
    else quant.static_elu_min = config.get_int("static_elu_min");
    if (!has_elu_max) quant.static_elu_max = range;
    else quant.static_elu_max = config.get_int("static_elu_max");
    quant.validate();
    return quant;
}

void QuantConfig::validate() const {
    if (schema != "gat_l3_scale_chain" || quant_cfg_id != "gat_l3_scale_chain") throw std::runtime_error("unsupported quantization schema");
    if (s_input <= 0 || s_delta <= 0 || s_exp <= 0 || s_h[0] != s_input ||
        s_out <= 0 || s_out_head <= 0 || s_out_pre <= 0 || range_bits <= 0 ||
        range_bits > 22) throw std::runtime_error("invalid canonical scale chain");
    if (rounding != "nearest_ties_away" || rescale != "none" || output_average != "signed_round_div") throw std::runtime_error("non-canonical rounding/rescale policy");
    if (static_tables_id != "tables:lrelu+elu+exp+range" ||
        static_table_generator_id != "zkgpt-gat-static-tables-v2" ||
        static_row_order != "ascending_input" ||
        exp_approx_id != "exp_neg_quantized" || elu_approx_id != "elu_expm1_quantized" ||
        lrelu_slope_num <= 0 || lrelu_slope_den <= 0 ||
        lrelu_slope_num >= lrelu_slope_den || static_lrelu_min >= static_lrelu_max ||
        static_exp_min != 0 || static_exp_max < static_exp_min ||
        static_elu_min >= static_elu_max) {
        throw std::runtime_error("non-canonical static-table policy");
    }
    const std::uint64_t table_cap = 1ULL << 24;
    const auto interval_size = [](Int minimum, Int maximum) -> std::uint64_t {
        if (minimum > maximum) throw std::runtime_error("invalid static-table interval");
        const Wide span = static_cast<Wide>(maximum) - static_cast<Wide>(minimum) + 1;
        if (span <= 0 || span > static_cast<Wide>(std::numeric_limits<std::uint64_t>::max())) {
            throw std::runtime_error("static-table interval overflow");
        }
        return static_cast<std::uint64_t>(span);
    };
    if (static_lrelu_min > 0 || static_lrelu_max < 0 ||
        static_exp_min != 0 || static_exp_max < 0 ||
        static_elu_min > 0 || static_elu_max < 0 ||
        interval_size(static_lrelu_min, static_lrelu_max) > table_cap ||
        interval_size(static_exp_min, static_exp_max) > table_cap ||
        interval_size(static_elu_min, static_elu_max) > table_cap) {
        throw std::runtime_error("static-table domain exceeds authenticated cap");
    }
    const auto require_positive = [](const std::vector<Int> &values, const char *name) {
        if (values.empty()) throw std::runtime_error(std::string(name) + " is empty");
        for (Int value : values) if (value <= 0) throw std::runtime_error(std::string(name) + " contains a non-positive scale");
    };
    for (Int value : s_h) if (value <= 0) throw std::runtime_error("hidden scales must be positive");
    for (std::size_t l = 0; l < 2; ++l) {
        for (std::size_t h = 0; h < 8; ++h) {
            require_positive(s_w_hidden[l], "hidden projection scales");
            require_positive(s_hprime_hidden[l], "hidden projection-output scales");
            require_positive(s_a_src_hidden[l], "hidden source-attention scales");
            require_positive(s_a_dst_hidden[l], "hidden destination-attention scales");
            if (static_cast<Wide>(s_h[l]) * s_w_hidden[l][h] != s_hprime_hidden[l][h]) throw std::runtime_error("hidden projection scale mismatch");
            if (static_cast<Wide>(s_hprime_hidden[l][h]) * s_a_src_hidden[l][h] != s_delta || static_cast<Wide>(s_hprime_hidden[l][h]) * s_a_dst_hidden[l][h] != s_delta) throw std::runtime_error("hidden attention scale mismatch");
            if (s_agg_pre[l] <= 0 || s_agg[l] <= 0 || s_cat[l] <= 0) throw std::runtime_error("invalid hidden aggregation scale");
        }
        if (l == 0 && s_cat[l] != s_h[1]) throw std::runtime_error("hidden concat scale mismatch");
        if (l == 1 && s_cat[l] != s_h[2]) throw std::runtime_error("final hidden scale mismatch");
    }
    for (std::size_t h = 0; h < 2; ++h) {
        if (s_w_output[h] <= 0 || s_yprime_output[h] <= 0 || s_a_src_output[h] <= 0 || s_a_dst_output[h] <= 0 || s_b_output[h] <= 0) throw std::runtime_error("invalid output scales");
        if (static_cast<Wide>(s_h[2]) * s_w_output[h] != s_yprime_output[h]) throw std::runtime_error("output projection scale mismatch");
        if (static_cast<Wide>(s_yprime_output[h]) * s_a_src_output[h] != s_delta || static_cast<Wide>(s_yprime_output[h]) * s_a_dst_output[h] != s_delta) throw std::runtime_error("output attention scale mismatch");
        if (s_b_output[h] != s_out_head) throw std::runtime_error("output bias scale mismatch");
    }
    if (s_out != s_out_head || s_exp <= 0) throw std::runtime_error("output scale mismatch");
}

std::pair<std::size_t, std::size_t> Graph::node_range(std::size_t gid) const {
    if (gid >= graph_count || node_ptr.size() != graph_count + 1) throw std::out_of_range("graph node range");
    return std::make_pair(node_ptr[gid], node_ptr[gid + 1]);
}

std::pair<std::size_t, std::size_t> Graph::edge_range(std::size_t gid) const {
    if (gid >= graph_count || edge_ptr.size() != graph_count + 1) throw std::out_of_range("graph edge range");
    return std::make_pair(edge_ptr[gid], edge_ptr[gid + 1]);
}

void Graph::validate() const {
    if (num_nodes == 0 || num_features == 0 || num_classes == 0 || graph_count == 0 || src.empty()) throw std::runtime_error("empty canonical graph");
    if (src.size() != dst.size() || src.size() != edge_gid.size()) throw std::runtime_error("edge vectors have different lengths");
    if (node_ptr.size() != graph_count + 1 || edge_ptr.size() != graph_count + 1 || node_ptr.front() != 0 || edge_ptr.front() != 0 || node_ptr.back() != num_nodes || edge_ptr.back() != edges()) throw std::runtime_error("invalid graph boundary arrays");
    for (std::size_t g = 0; g < graph_count; ++g) {
        if (node_ptr[g] >= node_ptr[g + 1] || edge_ptr[g] >= edge_ptr[g + 1]) throw std::runtime_error("graph boundary must be strictly increasing");
    }
    // The endpoint checks below establish that every destination belongs to
    // exactly one graph/node range.  Record incoming and self-loop
    // existence while scanning the edges so the final requirement checks are
    // O(num_nodes + num_edges), rather than rescanning every graph's edges
    // once for every node (which is O(V*E) for large single-graph datasets).
    std::vector<unsigned char> node_edge_flags(num_nodes, 0);
    for (std::size_t e = 0; e < edges(); ++e) {
        if (edge_gid[e] < 0 || static_cast<std::size_t>(edge_gid[e]) >= graph_count) throw std::runtime_error("edge graph id out of range");
        const std::size_t g = static_cast<std::size_t>(edge_gid[e]);
        if (e < edge_ptr[g] || e >= edge_ptr[g + 1]) throw std::runtime_error("edge graph ranges are not contiguous");
        if (src[e] < static_cast<Int>(node_ptr[g]) || src[e] >= static_cast<Int>(node_ptr[g + 1]) || dst[e] < static_cast<Int>(node_ptr[g]) || dst[e] >= static_cast<Int>(node_ptr[g + 1])) throw std::runtime_error("edge endpoint crosses graph boundary");
        if (e > 0 && (edge_gid[e - 1] > edge_gid[e] || (edge_gid[e - 1] == edge_gid[e] && dst[e - 1] > dst[e]))) throw std::runtime_error("edges are not canonical stable-sorted");
        const std::size_t destination = static_cast<std::size_t>(dst[e]);
        node_edge_flags[destination] |= 1u;
        if (src[e] == dst[e]) node_edge_flags[destination] |= 2u;
    }
    // Self-loops are part of the canonical graph statement.  The loader must
    // reject a cache missing one instead of silently changing the topology.
    for (std::size_t node = 0; node < num_nodes; ++node) {
        if ((node_edge_flags[node] & 1u) == 0)
            throw std::runtime_error("canonical graph node has no incoming edge");
        if ((node_edge_flags[node] & 2u) == 0)
            throw std::runtime_error("canonical graph is missing a required self-loop");
    }
}

Graph load_graph(const std::string &cache_dir, const CanonicalConfig &config) {
    verify_cache_auth(cache_dir);
    CanonicalConfig meta;
    meta.source_path = cache_dir + "/meta.cfg";
    meta.values = read_key_values(meta.source_path);
    if (config.get("self_loop_rule") != "per_node") {
        throw std::runtime_error("canonical graph requires self_loop_rule=per_node");
    }
    if (config.get("edge_sort_rule") != "edge_gid_then_dst_stable") {
        throw std::runtime_error("canonical graph requires edge_sort_rule=edge_gid_then_dst_stable");
    }
    const bool deduplicate_edges = config_bool(config, "deduplicate_edges", true);
    const bool symmetrize_edges = config_bool(config, "symmetrize_edges", false);
    Graph graph;
    graph.num_nodes = static_cast<std::size_t>(meta.get_int("num_nodes"));
    graph.num_features = static_cast<std::size_t>(meta.get_int("num_features"));
    graph.num_classes = static_cast<std::size_t>(meta.get_int("num_classes"));
    const Int declared_graph_count = meta.get_int("graph_count", 1);
    if (declared_graph_count <= 0) throw std::runtime_error("canonical graph_count must be positive");
    graph.graph_count = static_cast<std::size_t>(declared_graph_count);
    const std::string node_ptr_path = cache_dir + "/node_ptr.txt";
    std::ifstream np(node_ptr_path.c_str());
    if (np) {
        std::string node_line;
        while (std::getline(np, node_line)) {
            node_line = strip_comment(node_line);
            if (node_line.empty()) continue;
            const std::vector<Int> values = parse_integer_line(node_line);
            for (Int value : values) {
                if (value < 0 || static_cast<std::size_t>(value) > graph.num_nodes) throw std::runtime_error("invalid node_ptr");
                graph.node_ptr.push_back(static_cast<std::size_t>(value));
            }
        }
        if (graph.node_ptr.size() != graph.graph_count + 1) throw std::runtime_error("PPI node_ptr length does not match graph_count");
    } else {
        if (graph.graph_count != 1) throw std::runtime_error("multi-graph cache is missing node_ptr.txt");
        graph.node_ptr = {0, graph.num_nodes};
    }
    if (graph.node_ptr.empty() || graph.node_ptr.front() != 0 || graph.node_ptr.back() != graph.num_nodes) {
        throw std::runtime_error("canonical graph node_ptr does not cover the graph");
    }
    for (std::size_t g = 0; g < graph.graph_count; ++g) {
        if (graph.node_ptr[g] >= graph.node_ptr[g + 1]) throw std::runtime_error("canonical graph node_ptr must be strictly increasing");
    }
    if (config.get("dataset") == "ppi") {
        if (graph.graph_count != 295 || config.get_int("batch_graphs", 0) != 295) {
            throw std::runtime_error("PPI canonical graph must contain exactly 295 graph batches");
        }
    }
    struct Edge { Int gid, src, dst; std::size_t order; };
    std::vector<Edge> raw_edges;
    std::ifstream ein(require_file(cache_dir + "/edges.txt"));
    std::string line;
    std::size_t order = 0;
    while (std::getline(ein, line)) {
        line = strip_comment(line);
        if (line.empty()) continue;
        const std::vector<Int> values = parse_integer_line(line);
        if (values.size() != 2 && values.size() != 3) throw std::runtime_error("edge line must contain two or three integers");
        if (values.size() == 2 && graph.graph_count != 1) {
            throw std::runtime_error("multi-graph cache edge rows must contain graph_id,src,dst");
        }
        const Edge edge = values.size() == 2
            ? Edge{0, values[0], values[1], order++}
            : Edge{values[0], values[1], values[2], order++};
        if (edge.gid < 0 || static_cast<std::size_t>(edge.gid) >= graph.graph_count) {
            throw std::runtime_error("edge graph id out of range");
        }
        const std::size_t gid = static_cast<std::size_t>(edge.gid);
        if (edge.src < static_cast<Int>(graph.node_ptr[gid]) ||
            edge.src >= static_cast<Int>(graph.node_ptr[gid + 1]) ||
            edge.dst < static_cast<Int>(graph.node_ptr[gid]) ||
            edge.dst >= static_cast<Int>(graph.node_ptr[gid + 1])) {
            throw std::runtime_error("edge endpoint crosses graph boundary");
        }
        raw_edges.push_back(edge);
    }
    graph.src.reserve(raw_edges.size() + graph.num_nodes);
    graph.dst.reserve(raw_edges.size() + graph.num_nodes);
    graph.edge_gid.reserve(raw_edges.size() + graph.num_nodes);
    graph.edge_ptr.assign(graph.graph_count + 1, 0);
    for (std::size_t gid = 0; gid < graph.graph_count; ++gid) {
        std::vector<Edge> graph_edges;
        std::set<std::pair<Int, Int>> seen;
        std::size_t stable_index = 0;
        const auto append_edge = [&](Int src, Int dst) {
            const bool first = seen.emplace(src, dst).second;
            if (deduplicate_edges && !first) return;
            graph_edges.push_back(Edge{static_cast<Int>(gid), src, dst, stable_index++});
        };
        for (const Edge &edge : raw_edges) {
            if (static_cast<std::size_t>(edge.gid) != gid) continue;
            append_edge(edge.src, edge.dst);
            if (symmetrize_edges && edge.src != edge.dst) append_edge(edge.dst, edge.src);
        }
        for (std::size_t node = graph.node_ptr[gid]; node < graph.node_ptr[gid + 1]; ++node) {
            const auto key = std::make_pair(static_cast<Int>(node), static_cast<Int>(node));
            if (seen.find(key) == seen.end()) append_edge(key.first, key.second);
        }
        std::stable_sort(graph_edges.begin(), graph_edges.end(), [](const Edge &a, const Edge &b) {
            if (a.gid != b.gid) return a.gid < b.gid;
            if (a.dst != b.dst) return a.dst < b.dst;
            return a.order < b.order;
        });
        for (const Edge &edge : graph_edges) {
            graph.src.push_back(edge.src);
            graph.dst.push_back(edge.dst);
            graph.edge_gid.push_back(edge.gid);
        }
        graph.edge_ptr[gid + 1] = graph.src.size();
    }
    graph.validate();
    return graph;
}

Tensor2D load_features(const std::string &cache_dir, const CanonicalConfig &config,
                       const QuantConfig &quant, const Graph &graph) {
    verify_cache_auth(cache_dir);
    const bool normalize = config.get("input_preprocess_id") == "row_normalize";
    const Int input_scale = quant.s_input;
    if (input_scale <= 0) throw std::runtime_error("canonical config must provide positive s_input");
    const std::string text_path = cache_dir + "/features.txt";
    std::ifstream text(text_path.c_str());
    Tensor2D features(graph.num_nodes, graph.num_features);
    if (text) {
        std::size_t rows = 0, cols = 0;
        if (!(text >> rows >> cols) || rows != graph.num_nodes || cols != graph.num_features) throw std::runtime_error("canonical feature header mismatch");
        std::string line;
        std::getline(text, line);
        for (std::size_t r = 0; r < rows; ++r) {
            if (!std::getline(text, line)) throw std::runtime_error("truncated canonical features");
            std::map<std::size_t, double> sparse;
            double sum = 0;
            for (const std::string &item : split(trim(line), ' ')) {
                if (item.empty()) continue;
                const std::size_t colon = item.find(':');
                if (colon == std::string::npos) throw std::runtime_error("malformed sparse feature");
                const std::size_t index = static_cast<std::size_t>(parse_int(item.substr(0, colon), "feature index"));
                if (index >= cols) throw std::runtime_error("feature index out of range");
                const std::string value_text = item.substr(colon + 1);
                std::size_t used = 0;
                const double value = std::stod(value_text, &used);
                if (used != value_text.size() || !std::isfinite(value)) throw std::runtime_error("non-finite or malformed sparse feature");
                if (!sparse.emplace(index, value).second) throw std::runtime_error("duplicate sparse feature index");
                sum += value;
                if (!std::isfinite(sum)) throw std::runtime_error("non-finite sparse feature row");
            }
            const double denom = normalize && sum != 0 ? sum : 1.0;
            for (const auto &entry : sparse) features.at(r, entry.first) = quantize_double(entry.second / denom, input_scale);
        }
        return features;
    }
    const NpyArray array = load_npy(require_file(cache_dir + "/features.npy"));
    if (array.shape.size() != 2 || array.shape[0] != graph.num_nodes || array.shape[1] != graph.num_features) throw std::runtime_error("NumPy feature shape mismatch");
    for (std::size_t r = 0; r < graph.num_nodes; ++r) {
        double sum = 0;
        for (std::size_t c = 0; c < graph.num_features; ++c) {
            sum += array.values[r * graph.num_features + c];
            if (!std::isfinite(sum)) throw std::runtime_error("non-finite feature row");
        }
        const double denom = normalize && sum != 0 ? sum : 1.0;
        for (std::size_t c = 0; c < graph.num_features; ++c) features.at(r, c) = quantize_double(array.values[r * graph.num_features + c] / denom, input_scale);
    }
    return features;
}

std::vector<Int> load_labels(const std::string &cache_dir, const Graph &graph) {
    verify_cache_auth(cache_dir);
    const std::string txt = cache_dir + "/labels.txt";
    std::ifstream in(txt.c_str());
    const std::string meta_path = cache_dir + "/meta.cfg";
    CanonicalConfig meta;
    meta.source_path = meta_path;
    meta.values = read_key_values(meta.source_path);
    const bool multi_label = meta.get("label_format") == "multi_hot" ||
                             meta.get("task_type") == "inductive_multi_graph_node_classification";
    if (multi_label && in) {
        throw std::runtime_error("PPI multi-label cache must not provide labels.txt");
    }
    if (!in) {
        const std::string multi_txt = cache_dir + "/labels_multi.txt";
        std::ifstream multi(multi_txt.c_str());
        if (!multi) {
            const NpyArray array = load_npy(cache_dir + "/labels_multi.npy");
            if (array.shape.size() != 2 || array.shape[0] != graph.num_nodes || array.shape[1] != graph.num_classes) throw std::runtime_error("multi-label shape mismatch");
            std::vector<Int> labels;
            labels.reserve(array.values.size());
            for (double value : array.values) {
                if (!std::isfinite(value) || value != std::floor(value) || (value != 0.0 && value != 1.0)) throw std::runtime_error("multi-label value is not binary integer");
                labels.push_back(static_cast<Int>(value));
            }
            return labels;
        }
        std::vector<Int> labels;
        std::string line;
        bool first_payload = true;
        while (std::getline(multi, line)) {
            line = strip_comment(line);
            if (line.empty()) continue;
            const std::vector<Int> row = parse_integer_line(line);
            // The staged PPI text encoding is self-describing: its first
            // non-empty line is `num_nodes num_classes`, followed by one
            // 121-column multi-hot row per node.  Treat that header as
            // metadata only after checking it against the authenticated
            // cache shape.  A malformed or inconsistent header must fail
            // closed; it must never be accepted as a short data row.
            if (first_payload && row.size() == 2 &&
                row[0] == static_cast<Int>(graph.num_nodes) &&
                row[1] == static_cast<Int>(graph.num_classes)) {
                first_payload = false;
                continue;
            }
            first_payload = false;
            if (row.size() != graph.num_classes) throw std::runtime_error("multi-label row width mismatch");
            for (Int value : row) if (value != 0 && value != 1) throw std::runtime_error("multi-label value is not binary integer");
            labels.insert(labels.end(), row.begin(), row.end());
        }
        if (labels.size() != graph.num_nodes * graph.num_classes) throw std::runtime_error("multi-label row count mismatch");
        // When both canonical encodings are staged (as for PPI), compare
        // them.  A verifier must not silently choose one unauthenticated
        // payload over the other.
        const std::string multi_npy = cache_dir + "/labels_multi.npy";
        if (std::ifstream(multi_npy.c_str())) {
            const NpyArray array = load_npy(multi_npy);
            if (array.shape.size() != 2 || array.shape[0] != graph.num_nodes || array.shape[1] != graph.num_classes || array.values.size() != labels.size()) {
                throw std::runtime_error("multi-label NumPy shape mismatch");
            }
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (!std::isfinite(array.values[i]) || array.values[i] != std::floor(array.values[i]) ||
                    (array.values[i] != 0.0 && array.values[i] != 1.0) ||
                    labels[i] != static_cast<Int>(array.values[i])) {
                    throw std::runtime_error("multi-label text/NumPy payload mismatch");
                }
            }
        }
        return labels;
    }
    if (multi_label) throw std::runtime_error("multi-label cache is missing labels_multi payload");
    std::vector<Int> labels;
    std::string line;
    while (std::getline(in, line)) {
        line = strip_comment(line);
        if (line.empty()) continue;
        const std::vector<Int> values = parse_integer_line(line);
        if (values.size() != 1) throw std::runtime_error("single-label row must contain exactly one integer");
        labels.push_back(values.front());
    }
    if (labels.size() != graph.num_nodes) throw std::runtime_error("single-label count mismatch");
    for (Int value : labels) if (value < 0 || value >= static_cast<Int>(graph.num_classes)) throw std::runtime_error("single-label class index out of range");
    return labels;
}

Model Model::load(const std::string &bundle_dir) {
    Model model;
    verify_checkpoint_auth(bundle_dir);
    const std::string manifest_path = bundle_dir + "/manifest.json";
    model.manifest.source_path = manifest_path;
    const std::string manifest_text = [&]() {
        require_file(manifest_path);
        std::ifstream in(manifest_path.c_str());
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }();
    model.manifest.dataset = manifest_string(manifest_text, "dataset");
    model.manifest.model_arch_id = manifest_string(manifest_text, "model_arch_id");
    model.manifest.quant_cfg_id = manifest_string(manifest_text, "quant_cfg_id");
    model.manifest.model_param_id = manifest_required_string(manifest_text, "model_param_id");
    model.manifest.static_table_id = manifest_required_string(manifest_text, "static_table_id");
    model.manifest.input_preprocess_id = manifest_required_string(manifest_text, "input_preprocess_id");
    model.manifest.batching_rule = manifest_required_string(manifest_text, "batching_rule");
    model.manifest.edge_sort_rule = manifest_required_string(manifest_text, "edge_sort_rule");
    model.manifest.self_loop_rule = manifest_required_string(manifest_text, "self_loop_rule");
    model.manifest.label_format = manifest_required_string(manifest_text, "label_format");
    model.manifest.output_average_rule = manifest_required_string(manifest_text, "output_average_rule");
    model.manifest.quantization_scale_chain = manifest_required_string(manifest_text, "quantization_scale_chain");
    model.manifest.tensor_count = manifest_number(manifest_text, "tensor_count");
    model.manifest.layers = manifest_number(manifest_text, "L");
    model.manifest.output_heads = manifest_number(manifest_text, "K_out");
    model.manifest.classes = manifest_number(manifest_text, "C");
    const std::string tensor_path = bundle_dir + "/tensors.txt";
    require_file(tensor_path);
    std::ifstream in(tensor_path.c_str());
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::istringstream header(line);
        std::string tag, name;
        std::size_t rank = 0;
        if (!(header >> tag >> name >> rank) || tag != "TENSOR") throw std::runtime_error("malformed tensor header in " + tensor_path);
        Tensor tensor;
        tensor.name = name;
        tensor.shape.dims.resize(rank);
        std::size_t count = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            if (!(header >> tensor.shape.dims[i]) || tensor.shape.dims[i] == 0 ||
                count > std::numeric_limits<std::size_t>::max() / tensor.shape.dims[i]) {
                throw std::runtime_error("tensor rank/shape mismatch: " + name);
            }
            count *= tensor.shape.dims[i];
        }
        std::size_t declared_count = 0;
        if (!(header >> declared_count) || declared_count != count) throw std::runtime_error("tensor count mismatch: " + name);
        tensor.values.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (!(in >> tensor.values[i]) || !std::isfinite(tensor.values[i])) throw std::runtime_error("invalid tensor payload: " + name);
        }
        validate_tensor(tensor);
        if (!model.tensors.emplace(name, std::move(tensor)).second) throw std::runtime_error("duplicate tensor: " + name);
    }
    std::string trailing;
    if (in >> trailing) throw std::runtime_error("trailing tensor payload in " + tensor_path);
    model.validate_schema();
    return model;
}

Model Model::load_shape_only(const std::string &bundle_dir) {
    // Authenticate the complete checkpoint bundle before consuming any
    // metadata.  This routine intentionally never stores tensor payloads;
    // it only parses their canonical headers and skips the authenticated
    // scalar stream.  Capacity/registration callers therefore cannot
    // accidentally construct a private witness or run a forward pass.
    Model model;
    verify_checkpoint_auth(bundle_dir);
    const std::string manifest_path = bundle_dir + "/manifest.json";
    model.manifest.source_path = manifest_path;
    require_file(manifest_path);
    std::ifstream manifest_in(manifest_path.c_str());
    const std::string manifest_text((std::istreambuf_iterator<char>(manifest_in)),
                                    std::istreambuf_iterator<char>());
    model.manifest.dataset = manifest_string(manifest_text, "dataset");
    model.manifest.model_arch_id = manifest_string(manifest_text, "model_arch_id");
    model.manifest.quant_cfg_id = manifest_string(manifest_text, "quant_cfg_id");
    model.manifest.model_param_id = manifest_required_string(manifest_text, "model_param_id");
    model.manifest.static_table_id = manifest_required_string(manifest_text, "static_table_id");
    model.manifest.input_preprocess_id = manifest_required_string(manifest_text, "input_preprocess_id");
    model.manifest.batching_rule = manifest_required_string(manifest_text, "batching_rule");
    model.manifest.edge_sort_rule = manifest_required_string(manifest_text, "edge_sort_rule");
    model.manifest.self_loop_rule = manifest_required_string(manifest_text, "self_loop_rule");
    model.manifest.label_format = manifest_required_string(manifest_text, "label_format");
    model.manifest.output_average_rule = manifest_required_string(manifest_text, "output_average_rule");
    model.manifest.quantization_scale_chain = manifest_required_string(manifest_text, "quantization_scale_chain");
    model.manifest.tensor_count = manifest_number(manifest_text, "tensor_count");
    model.manifest.layers = manifest_number(manifest_text, "L");
    model.manifest.output_heads = manifest_number(manifest_text, "K_out");
    model.manifest.classes = manifest_number(manifest_text, "C");

    const std::string tensor_path = bundle_dir + "/tensors.txt";
    require_file(tensor_path);
    std::ifstream in(tensor_path.c_str());
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::istringstream header(line);
        std::string tag, name;
        std::size_t rank = 0;
        if (!(header >> tag >> name >> rank) || tag != "TENSOR" || name.empty())
            throw std::runtime_error("malformed tensor header in " + tensor_path);
        Tensor tensor;
        tensor.name = name;
        tensor.shape.dims.resize(rank);
        std::size_t count = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            if (!(header >> tensor.shape.dims[i]) || tensor.shape.dims[i] == 0 ||
                count > std::numeric_limits<std::size_t>::max() / tensor.shape.dims[i])
                throw std::runtime_error("tensor rank/shape mismatch: " + name);
            count *= tensor.shape.dims[i];
        }
        std::size_t declared_count = 0;
        if (!(header >> declared_count) || declared_count != count)
            throw std::runtime_error("tensor count mismatch: " + name);
        // Do not retain private values.  The whole file was authenticated
        // above, and schema validation below only needs a shape-sized zero
        // vector to exercise the canonical rank/dimension checks.
        tensor.values.assign(count, 0.0);
        if (!model.tensors.emplace(name, std::move(tensor)).second)
            throw std::runtime_error("duplicate tensor: " + name);
        double ignored = 0.0;
        for (std::size_t i = 0; i < count; ++i)
            if (!(in >> ignored)) throw std::runtime_error("truncated tensor payload: " + name);
    }
    std::string trailing;
    if (in >> trailing) throw std::runtime_error("trailing tensor payload in " + tensor_path);
    model.validate_schema();
    return model;
}

const Tensor &Model::tensor(const std::string &name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("canonical tensor missing: " + name);
    return it->second;
}

std::size_t Model::input_dim() const { return tensor(hidden_name(0, 0, "seq/kernel")).shape.dims.at(1); }
std::size_t Model::hidden_dim() const { return tensor(hidden_name(0, 0, "seq/kernel")).shape.dims.at(2); }
std::size_t Model::hidden_heads() const {
    if (manifest.layers < 2 || manifest.output_heads == 0 ||
        manifest.tensor_count < manifest.output_heads * 4) {
        return 0;
    }
    const std::size_t hidden_tensors = manifest.tensor_count - manifest.output_heads * 4;
    const std::size_t hidden_layers = manifest.layers - 1;
    const std::size_t tensors_per_head = hidden_layers * 3;
    if (tensors_per_head == 0 || hidden_tensors % tensors_per_head != 0) return 0;
    return hidden_tensors / tensors_per_head;
}
std::size_t Model::classes() const { return tensor(output_name(0, "output_bias")).shape.dims.at(0); }

void Model::validate_schema() const {
    const std::size_t hidden_layers = manifest.layers > 0 ? manifest.layers - 1 : 0;
    const std::size_t hidden_heads_count = hidden_heads();
    const std::size_t expected_tensors = hidden_layers * hidden_heads_count * 3 + manifest.output_heads * 4;
    if (hidden_layers != 2 || hidden_heads_count == 0 || manifest.output_heads == 0 ||
        manifest.output_heads > 2 ||
        tensors.size() != expected_tensors || manifest.tensor_count != expected_tensors) {
        throw std::runtime_error("canonical model tensor/head profile is invalid");
    }
    if (manifest.layers != 3 || manifest.classes == 0 ||
        manifest.dataset.empty() || manifest.model_arch_id.empty() || manifest.quant_cfg_id.empty() ||
        manifest.model_param_id.empty() || manifest.static_table_id.empty() ||
        manifest.input_preprocess_id.empty() || manifest.batching_rule.empty() ||
        manifest.edge_sort_rule.empty() || manifest.self_loop_rule.empty() ||
        manifest.label_format.empty() || manifest.output_average_rule.empty() ||
        manifest.quantization_scale_chain.empty()) {
        throw std::runtime_error("canonical model manifest identity is incomplete");
    }
    if (manifest.static_table_id != "tables:lrelu+elu+exp+range" ||
        manifest.edge_sort_rule != "edge_gid_then_dst_stable" ||
        manifest.self_loop_rule != "per_node" ||
        manifest.output_average_rule != "per_head_bias_then_arithmetic_mean") {
        throw std::runtime_error("canonical model manifest policy is not supported");
    }
    for (std::size_t l = 0; l < hidden_layers; ++l) for (std::size_t h = 0; h < hidden_heads_count; ++h) {
        const Tensor &seq = tensor(hidden_name(l, h, "seq/kernel"));
        const Tensor &as = tensor(hidden_name(l, h, "attn_src/kernel"));
        const Tensor &ad = tensor(hidden_name(l, h, "attn_dst/kernel"));
        if (seq.shape.dims.size() != 3 || seq.shape.dims[0] != 1 || as.shape.dims.size() != 3 || ad.shape.dims.size() != 3 || as.shape.dims[0] != 1 || ad.shape.dims[0] != 1 || as.shape.dims[2] != 1 || ad.shape.dims[2] != 1 || seq.shape.dims[2] != as.shape.dims[1] || seq.shape.dims[2] != ad.shape.dims[1]) throw std::runtime_error("hidden tensor shape mismatch");
        if (l == 0 && seq.shape.dims[1] != input_dim()) throw std::runtime_error("hidden input shape mismatch");
        if (l == 1 && seq.shape.dims[1] != hidden_heads_count * hidden_dim()) throw std::runtime_error("hidden concat shape mismatch");
    }
    for (std::size_t h = 0; h < manifest.output_heads; ++h) {
        const Tensor &seq = tensor(output_name(h, "seq/kernel"));
        const Tensor &as = tensor(output_name(h, "attn_src/kernel"));
        const Tensor &ad = tensor(output_name(h, "attn_dst/kernel"));
        const Tensor &bias = tensor(output_name(h, "output_bias"));
        if (seq.shape.dims.size() != 3 || seq.shape.dims[0] != 1 || seq.shape.dims[1] != hidden_heads_count * hidden_dim() || as.shape.dims.size() != 3 || ad.shape.dims.size() != 3 || as.shape.dims[1] != classes() || ad.shape.dims[1] != classes() || bias.shape.dims.size() != 1 || bias.shape.dims[0] != classes()) throw std::runtime_error("output tensor shape mismatch");
    }
    if (manifest.classes != 0 && manifest.classes != classes()) throw std::runtime_error("manifest class count mismatch");
}

void Model::validate_identity(const CanonicalConfig &config) const {
    validate_schema();
    const auto require_equal = [&](const std::string &key, const std::string &actual) {
        const std::string expected = config.get(key);
        if (expected.empty() || expected != actual) throw std::runtime_error("model/config identity mismatch for " + key);
    };
    require_equal("dataset", manifest.dataset);
    require_equal("model_arch_id", manifest.model_arch_id);
    require_equal("quant_cfg_id", manifest.quant_cfg_id);
    if (!config.get("static_table_id").empty() && config.get("static_table_id") != manifest.static_table_id) {
        throw std::runtime_error("model/config identity mismatch for static_table_id");
    }
    if (!config.get("model_param_id").empty() && config.get("model_param_id") != manifest.model_param_id) {
        throw std::runtime_error("model/config identity mismatch for model_param_id");
    }
    if (!config.get("input_preprocess_id").empty() && config.get("input_preprocess_id") != manifest.input_preprocess_id) {
        throw std::runtime_error("model/config identity mismatch for input_preprocess_id");
    }
    if (!config.get("batching_rule").empty() && config.get("batching_rule") != manifest.batching_rule) {
        throw std::runtime_error("model/config identity mismatch for batching_rule");
    }
    if (config.get("checkpoint_bundle").empty()) throw std::runtime_error("canonical config missing checkpoint_bundle");
}

} // namespace gat
} // namespace zkgpt
