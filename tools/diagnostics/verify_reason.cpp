#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

std::filesystem::path repo_root() {
    auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / "configs" / "cora_secure.cfg")) {
        return cwd;
    }
    if (std::filesystem::exists(cwd.parent_path() / "configs" / "cora_secure.cfg")) {
        return cwd.parent_path();
    }
    return cwd;
}

}  // namespace

int main() {
    try {
        auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
        auto context = gatzk::protocol::build_context(config, nullptr);
        auto trace = gatzk::protocol::build_trace(context, nullptr);
        auto proof = gatzk::protocol::prove(context, trace, nullptr);
        (void)::setenv("GATZK_DEBUG_VERIFY", "1", 1);
        gatzk::protocol::RunMetrics metrics;
        const auto accepted = gatzk::protocol::verify(context, proof, &metrics);
        std::cout << "accepted=" << (accepted ? "true" : "false") << "\n";
        return accepted ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
