#include "gatzk/mechanism/runtime.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include <sys/stat.h>
#include <unistd.h>

namespace
{
    using gatzk::mechanism::RunOptions;
    using gatzk::mechanism::VerifyOptions;

    class CliError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    void print_usage(std::ostream &out)
    {
        out << "Usage:\n"
            << "  gatzk_mechanism run --mechanism feature_folding|shared_neighborhood --variant original|control \\\n"
            << "    --nodes N --edges E --width D --graphs G --seed S --backend cpu|cuda [--device INDEX] \\\n"
            << "    --witness-seed-file PATH --srs PATH --proof-out PATH --statement-out PATH --result-json PATH\n"
            << "  gatzk_mechanism verify --proof PATH --statement PATH --srs PATH --result-json PATH\n"
            << "  gatzk_mechanism --help\n";
    }

    bool starts_with(std::string_view value, std::string_view prefix)
    {
        return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
    }

    std::map<std::string, std::string> parse_flags(int argc, char **argv, int first, const std::set<std::string> &allowed)
    {
        std::map<std::string, std::string> values;
        for (int i = first; i < argc;)
        {
            const std::string name = argv[i++];
            if (!starts_with(name, "--"))
                throw CliError("unexpected positional argument: " + name);
            if (!allowed.contains(name))
                throw CliError("unknown option: " + name);
            if (i >= argc || starts_with(argv[i], "--"))
                throw CliError("missing value for option: " + name);
            if (!values.emplace(name, argv[i++]).second)
                throw CliError("option was supplied more than once: " + name);
        }
        return values;
    }

    const std::string &required(const std::map<std::string, std::string> &values, const std::string &name)
    {
        const auto it = values.find(name);
        if (it == values.end() || it->second.empty())
            throw CliError("required option is missing: " + name);
        return it->second;
    }

    std::uint64_t parse_u64(const std::string &value, const std::string &name)
    {
        if (value.empty() || value.front() == '-')
            throw CliError(name + " must be a non-negative decimal integer");
        std::uint64_t parsed = 0;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
            throw CliError(name + " must be a non-negative decimal integer");
        return parsed;
    }

    std::size_t parse_size(const std::string &value, const std::string &name)
    {
        const auto parsed = parse_u64(value, name);
        if (parsed > std::numeric_limits<std::size_t>::max())
            throw CliError(name + " exceeds this platform's size_t range");
        return static_cast<std::size_t>(parsed);
    }

    int parse_device(const std::string &value)
    {
        const auto parsed = parse_u64(value, "--device");
        if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw CliError("--device exceeds the supported device index range");
        return static_cast<int>(parsed);
    }

    std::filesystem::path required_path(const std::map<std::string, std::string> &values, const std::string &name)
    {
        return std::filesystem::path(required(values, name));
    }

    void require_private_seed_file(const std::filesystem::path &path)
    {
        std::error_code error;
        const auto link_status = std::filesystem::symlink_status(path, error);
        if (error || std::filesystem::is_symlink(link_status) || !std::filesystem::is_regular_file(link_status))
            throw CliError("witness seed must be a regular non-symlink file");
        struct stat info
        {
        };
        if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode))
            throw CliError("cannot inspect witness seed permissions");
        if (info.st_uid != ::geteuid() || (info.st_mode & 0777) != 0600)
            throw CliError("witness seed file must be owned by the current user with mode 0600");
    }

    RunOptions parse_run_options(int argc, char **argv)
    {
        const std::set<std::string> allowed{
            "--mechanism", "--variant", "--nodes", "--edges", "--width", "--graphs", "--seed", "--backend",
            "--device", "--witness-seed-file", "--srs", "--proof-out", "--statement-out", "--result-json",
        };
        const auto values = parse_flags(argc, argv, 2, allowed);
        RunOptions options;
        options.mechanism = gatzk::mechanism::parse_mechanism(required(values, "--mechanism"));
        options.variant = gatzk::mechanism::parse_variant(required(values, "--variant"));
        options.backend = gatzk::mechanism::parse_backend(required(values, "--backend"));
        options.nodes = parse_size(required(values, "--nodes"), "--nodes");
        options.edges = parse_size(required(values, "--edges"), "--edges");
        options.width = parse_size(required(values, "--width"), "--width");
        options.graphs = parse_size(required(values, "--graphs"), "--graphs");
        options.seed = parse_u64(required(values, "--seed"), "--seed");
        options.witness_seed_file = required_path(values, "--witness-seed-file");
        options.srs_path = required_path(values, "--srs");
        options.proof_out = required_path(values, "--proof-out");
        options.statement_out = required_path(values, "--statement-out");
        options.result_json = required_path(values, "--result-json");

        const auto device = values.find("--device");
        if (options.backend == gatzk::mechanism::Backend::Cuda)
        {
            if (device == values.end())
                throw CliError("CUDA proving requires --device INDEX");
            options.device = parse_device(device->second);
        }
        else
        {
            if (device != values.end())
                throw CliError("--device is only valid with --backend cuda");
            options.device = -1;
        }

        if (options.nodes == 0 || options.edges < options.nodes || options.graphs == 0 || options.graphs > options.nodes)
            throw CliError("workload requires nodes >= graphs > 0 and edges >= nodes");
        if (options.width == 0 || options.width > 32)
            throw CliError("--width must be in the supported range 1..32");
        require_private_seed_file(options.witness_seed_file);
        return options;
    }

    VerifyOptions parse_verify_options(int argc, char **argv)
    {
        const std::set<std::string> allowed{"--proof", "--statement", "--srs", "--result-json"};
        const auto values = parse_flags(argc, argv, 2, allowed);
        VerifyOptions options;
        options.proof_path = required_path(values, "--proof");
        options.statement_path = required_path(values, "--statement");
        options.srs_path = required_path(values, "--srs");
        options.result_json = required_path(values, "--result-json");
        return options;
    }

    int dispatch(int argc, char **argv)
    {
        if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "help")
        {
            print_usage(std::cout);
            return argc < 2 ? 2 : 0;
        }
        const std::string command = argv[1];
        if ((command == "run" || command == "verify") && argc == 3 && std::string_view(argv[2]) == "--help")
        {
            print_usage(std::cout);
            return 0;
        }
        if (command == "run")
        {
            const auto options = parse_run_options(argc, argv);
            const auto receipt = gatzk::mechanism::run(options);
            std::cout << "MECHANISM_PROOF_WRITTEN mechanism=" << gatzk::mechanism::to_string(options.mechanism)
                      << " variant=" << gatzk::mechanism::to_string(options.variant)
                      << " backend=" << gatzk::mechanism::to_string(options.backend)
                      << " actual_cuda=" << (receipt.actual_cuda ? "true" : "false")
                      << " commitment_time_ms=" << receipt.commitment_time_ms
                      << " prove_time_ms=" << receipt.prove_time_ms
                      << " proof_size_bytes=" << receipt.proof_size_bytes
                      << " proof_sha256=" << receipt.proof_sha256 << '\n';
            return 0;
        }
        if (command == "verify")
        {
            const auto options = parse_verify_options(argc, argv);
            const auto receipt = gatzk::mechanism::verify(options);
            std::cout << "VERIFY_OK proof_sha256=" << receipt.proof_sha256
                      << " statement_sha256=" << receipt.statement_sha256
                      << " verify_time_ms=" << receipt.verify_time_ms << '\n';
            return 0;
        }
        throw CliError("unknown command: " + command);
    }
}

int main(int argc, char **argv)
{
    try
    {
        return dispatch(argc, argv);
    }
    catch (const CliError &error)
    {
        std::cerr << "gatzk_mechanism: " << error.what() << '\n';
        print_usage(std::cerr);
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "gatzk_mechanism: " << error.what() << '\n';
        return 1;
    }
}
