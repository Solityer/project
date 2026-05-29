#include <mcl/bn.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using mcl::bn::Fr;
using mcl::bn::G1;
using mcl::bn::hashAndMapToG1;
using mcl::bn::initPairing;

namespace {

std::string json_escape(const std::string& value)
{
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << ch;
        }
    }
    return out.str();
}

std::vector<Fr> read_field_vector(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path);
    }
    std::vector<Fr> values;
    std::string token;
    while (in >> token) {
        Fr x;
        x.setStr(token, 10);
        values.push_back(x);
    }
    if (values.empty()) {
        values.emplace_back(0);
    }
    return values;
}

std::string read_text(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::size_t next_power_of_two(std::size_t value)
{
    std::size_t out = 1;
    while (out < std::max<std::size_t>(value, 1)) {
        out <<= 1;
    }
    return out;
}

std::size_t log2_exact(std::size_t value)
{
    std::size_t out = 0;
    while ((std::size_t(1) << out) < value) {
        ++out;
    }
    if ((std::size_t(1) << out) != value) {
        throw std::runtime_error("value is not a power of two");
    }
    return out;
}

Fr hash_fr(const std::string& label)
{
    Fr out;
    out.setHashOf(label.c_str(), label.size());
    return out;
}

G1 hash_g1(const std::string& label)
{
    G1 out;
    hashAndMapToG1(out, label.c_str(), label.size());
    return out;
}

std::vector<G1> generators(const std::string& label, std::size_t count)
{
    std::vector<G1> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(hash_g1("gat-zkgpt-hyrax-generator|" + label + "|" + std::to_string(i)));
    }
    return out;
}

G1 pedersen_commit(const std::vector<G1>& gens, const G1& h, const std::vector<Fr>& row, const Fr& blind)
{
    G1 out;
    out.clear();
    for (std::size_t i = 0; i < row.size(); ++i) {
        G1 term;
        G1::mul(term, gens[i], row[i]);
        G1::add(out, out, term);
    }
    G1 blind_term;
    G1::mul(blind_term, h, blind);
    G1::add(out, out, blind_term);
    return out;
}

std::vector<Fr> eq_table(const std::vector<Fr>& r)
{
    std::vector<Fr> values(std::size_t(1) << r.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        Fr acc = 1;
        for (std::size_t bit = 0; bit < r.size(); ++bit) {
            if ((i >> bit) & 1U) {
                acc *= r[bit];
            } else {
                acc *= (Fr(1) - r[bit]);
            }
        }
        values[i] = acc;
    }
    return values;
}

void write_json(
    const std::string& path,
    const std::string& label,
    std::size_t original_length,
    std::size_t padded_length,
    std::size_t row_bits,
    std::size_t col_bits,
    const std::vector<G1>& row_commitments,
    const G1& h,
    const G1& t_prime,
    const std::vector<Fr>& lt,
    const Fr& opening_blind,
    const Fr& eval)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << "{\n";
    out << "  \"backend\": \"mcl-bn254-hyrax-mle-v1\",\n";
    out << "  \"label\": \"" << json_escape(label) << "\",\n";
    out << "  \"original_length\": " << original_length << ",\n";
    out << "  \"padded_length\": " << padded_length << ",\n";
    out << "  \"row_bits\": " << row_bits << ",\n";
    out << "  \"col_bits\": " << col_bits << ",\n";
    out << "  \"hiding_generator\": \"" << h.serializeToHexStr() << "\",\n";
    out << "  \"row_commitments\": [";
    for (std::size_t i = 0; i < row_commitments.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "\"" << row_commitments[i].serializeToHexStr() << "\"";
    }
    out << "],\n";
    out << "  \"opening\": {\n";
    out << "    \"mode\": \"OpenMasked\",\n";
    out << "    \"t_prime\": \"" << t_prime.serializeToHexStr() << "\",\n";
    out << "    \"lt\": [";
    for (std::size_t i = 0; i < lt.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "\"" << lt[i].serializeToHexStr() << "\"";
    }
    out << "],\n";
    out << "    \"opening_blind\": \"" << opening_blind.serializeToHexStr() << "\",\n";
    out << "    \"masked_evaluation\": \"" << eval.serializeToHexStr() << "\"\n";
    out << "  }\n";
    out << "}\n";
}

std::string extract_string(const std::string& text, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON string key: " + key);
    }
    const auto colon = text.find(':', key_pos + needle.size());
    const auto first_quote = text.find('"', colon + 1);
    const auto second_quote = text.find('"', first_quote + 1);
    if (colon == std::string::npos || first_quote == std::string::npos || second_quote == std::string::npos) {
        throw std::runtime_error("malformed JSON string key: " + key);
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::size_t extract_size(const std::string& text, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON numeric key: " + key);
    }
    const auto colon = text.find(':', key_pos + needle.size());
    const auto begin = text.find_first_of("0123456789", colon + 1);
    const auto end = text.find_first_not_of("0123456789", begin);
    if (colon == std::string::npos || begin == std::string::npos) {
        throw std::runtime_error("malformed JSON numeric key: " + key);
    }
    return static_cast<std::size_t>(std::stoull(text.substr(begin, end - begin)));
}

std::vector<std::string> extract_string_array(const std::string& text, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON array key: " + key);
    }
    const auto begin = text.find('[', key_pos + needle.size());
    const auto end = text.find(']', begin + 1);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("malformed JSON array key: " + key);
    }
    std::vector<std::string> out;
    std::size_t pos = begin + 1;
    while (pos < end) {
        const auto first_quote = text.find('"', pos);
        if (first_quote == std::string::npos || first_quote >= end) {
            break;
        }
        const auto second_quote = text.find('"', first_quote + 1);
        if (second_quote == std::string::npos || second_quote > end) {
            throw std::runtime_error("malformed string in JSON array: " + key);
        }
        out.push_back(text.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }
    return out;
}

G1 parse_g1_hex(const std::string& hex)
{
    G1 out;
    out.deserializeHexStr(hex);
    return out;
}

Fr parse_fr_hex(const std::string& hex)
{
    Fr out;
    out.deserializeHexStr(hex);
    return out;
}

int run_commit_open(const std::vector<std::string>& args)
{
    if (args.size() != 5) {
        throw std::runtime_error("usage: hyrax_mle_cli commit-open <label> <vector.txt> <point.txt> <out.json>");
    }
    const std::string& label = args[1];
    std::vector<Fr> values = read_field_vector(args[2]);
    const std::size_t original_length = values.size();
    const std::size_t padded_length = next_power_of_two(values.size());
    values.resize(padded_length, Fr(0));

    const std::size_t variable_count = log2_exact(padded_length);
    std::vector<Fr> point = read_field_vector(args[3]);
    if (point.size() != variable_count) {
        throw std::runtime_error("opening point length does not match MLE variable count");
    }

    const std::size_t row_bits = variable_count / 2;
    const std::size_t col_bits = variable_count - row_bits;
    const std::size_t row_count = std::size_t(1) << row_bits;
    const std::size_t col_count = std::size_t(1) << col_bits;

    const auto gens = generators(label, col_count);
    const G1 h = hash_g1("gat-zkgpt-hyrax-hiding|" + label);
    std::vector<Fr> row_blinds(row_count);
    std::vector<G1> row_commitments(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        row_blinds[row] = hash_fr("gat-zkgpt-row-blind|" + label + "|" + std::to_string(row));
        std::vector<Fr> slice(values.begin() + static_cast<std::ptrdiff_t>(row * col_count),
                              values.begin() + static_cast<std::ptrdiff_t>((row + 1) * col_count));
        row_commitments[row] = pedersen_commit(gens, h, slice, row_blinds[row]);
    }

    const std::vector<Fr> col_point(point.begin(), point.begin() + static_cast<std::ptrdiff_t>(col_bits));
    const std::vector<Fr> row_point(point.begin() + static_cast<std::ptrdiff_t>(col_bits), point.end());
    const auto col_eq = eq_table(col_point);
    const auto row_eq = eq_table(row_point);

    std::vector<Fr> lt(col_count, Fr(0));
    Fr opening_blind = 0;
    G1 t_prime;
    t_prime.clear();
    for (std::size_t row = 0; row < row_count; ++row) {
        opening_blind += row_eq[row] * row_blinds[row];
        G1 term;
        G1::mul(term, row_commitments[row], row_eq[row]);
        G1::add(t_prime, t_prime, term);
        for (std::size_t col = 0; col < col_count; ++col) {
            lt[col] += row_eq[row] * values[row * col_count + col];
        }
    }

    Fr eval = 0;
    for (std::size_t col = 0; col < col_count; ++col) {
        eval += col_eq[col] * lt[col];
    }

    const G1 recomputed = pedersen_commit(gens, h, lt, opening_blind);
    if (recomputed != t_prime) {
        throw std::runtime_error("internal Hyrax opening consistency check failed");
    }

    write_json(args[4], label, original_length, padded_length, row_bits, col_bits, row_commitments, h, t_prime, lt, opening_blind, eval);
    return 0;
}

int run_verify_open(const std::vector<std::string>& args)
{
    if (args.size() != 5) {
        throw std::runtime_error("usage: hyrax_mle_cli verify-open <label> <proof.json> <point.txt> <out.json>");
    }
    const std::string& label = args[1];
    const auto text = read_text(args[2]);
    const auto row_bits = extract_size(text, "row_bits");
    const auto col_bits = extract_size(text, "col_bits");
    const auto variable_count = row_bits + col_bits;
    std::vector<Fr> point = read_field_vector(args[3]);
    if (point.size() != variable_count) {
        throw std::runtime_error("opening point length does not match proof metadata");
    }

    const auto row_hex = extract_string_array(text, "row_commitments");
    const auto lt_hex = extract_string_array(text, "lt");
    const std::size_t row_count = std::size_t(1) << row_bits;
    const std::size_t col_count = std::size_t(1) << col_bits;
    if (row_hex.size() != row_count || lt_hex.size() != col_count) {
        throw std::runtime_error("Hyrax proof dimensions do not match row/column bit lengths");
    }

    std::vector<G1> row_commitments;
    row_commitments.reserve(row_hex.size());
    for (const auto& item : row_hex) {
        row_commitments.push_back(parse_g1_hex(item));
    }
    std::vector<Fr> lt;
    lt.reserve(lt_hex.size());
    for (const auto& item : lt_hex) {
        lt.push_back(parse_fr_hex(item));
    }

    const G1 h = parse_g1_hex(extract_string(text, "hiding_generator"));
    const G1 t_prime = parse_g1_hex(extract_string(text, "t_prime"));
    const Fr opening_blind = parse_fr_hex(extract_string(text, "opening_blind"));
    const Fr claimed_eval = parse_fr_hex(extract_string(text, "masked_evaluation"));

    const std::vector<Fr> col_point(point.begin(), point.begin() + static_cast<std::ptrdiff_t>(col_bits));
    const std::vector<Fr> row_point(point.begin() + static_cast<std::ptrdiff_t>(col_bits), point.end());
    const auto col_eq = eq_table(col_point);
    const auto row_eq = eq_table(row_point);

    G1 combined_rows;
    combined_rows.clear();
    for (std::size_t row = 0; row < row_count; ++row) {
        G1 term;
        G1::mul(term, row_commitments[row], row_eq[row]);
        G1::add(combined_rows, combined_rows, term);
    }

    const auto gens = generators(label, col_count);
    const G1 recomputed = pedersen_commit(gens, h, lt, opening_blind);
    Fr recomputed_eval = 0;
    for (std::size_t col = 0; col < col_count; ++col) {
        recomputed_eval += col_eq[col] * lt[col];
    }
    const bool ok = (combined_rows == t_prime) && (recomputed == t_prime) && (recomputed_eval == claimed_eval);

    std::ofstream out(args[4]);
    if (!out) {
        throw std::runtime_error("failed to open verifier output: " + args[4]);
    }
    out << "{\n";
    out << "  \"status\": \"" << (ok ? "VERIFY_OK" : "VERIFY_FAILED") << "\",\n";
    out << "  \"backend\": \"mcl-bn254-hyrax-mle-v1\",\n";
    out << "  \"label\": \"" << json_escape(label) << "\",\n";
    out << "  \"row_commitment_fold_ok\": " << (combined_rows == t_prime ? "true" : "false") << ",\n";
    out << "  \"opening_commitment_ok\": " << (recomputed == t_prime ? "true" : "false") << ",\n";
    out << "  \"evaluation_ok\": " << (recomputed_eval == claimed_eval ? "true" : "false") << "\n";
    out << "}\n";
    return ok ? 0 : 3;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        initPairing(mcl::BN254);
        std::vector<std::string> args(argv + 1, argv + argc);
        if (args.empty()) {
            throw std::runtime_error("usage: hyrax_mle_cli <commit-open> ...");
        }
        if (args[0] == "commit-open") {
            return run_commit_open(args);
        }
        if (args[0] == "verify-open") {
            return run_verify_open(args);
        }
        throw std::runtime_error("unknown command: " + args[0]);
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << "\n";
        return 1;
    }
}
