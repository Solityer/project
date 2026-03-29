#include "gatzk/util/route2.hpp"

#include <mutex>

namespace gatzk::util {
namespace {

Route2Options& mutable_options() {
    static Route2Options options;
    return options;
}

std::mutex& options_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

const Route2Options& route2_options() {
    return mutable_options();
}

void set_route2_options(const Route2Options& options) {
    std::lock_guard<std::mutex> lock(options_mutex());
    mutable_options() = options;
}

std::string route2_feature_label(const Route2Options& options) {
    std::string label;
    if (options.fast_msm) {
        label += "msm";
    }
    if (options.parallel_fft) {
        if (!label.empty()) {
            label += "_";
        }
        label += "fft";
    }
    if (options.fft_backend_upgrade) {
        if (!label.empty()) {
            label += "_";
        }
        label += "packed";
    }
    if (options.fft_backend_upgrade && options.fft_kernel_upgrade) {
        if (!label.empty()) {
            label += "_";
        }
        label += "kernel";
    }
    if (options.trace_layout_upgrade) {
        if (!label.empty()) {
            label += "_";
        }
        label += "layout";
    }
    if (options.fast_verify_pairing) {
        if (!label.empty()) {
            label += "_";
        }
        label += "pairing";
    }
    return label.empty() ? "legacy" : label;
}

std::string route2_feature_notes(const Route2Options& options) {
    return "enabled_fast_msm=" + std::string(options.fast_msm ? "true" : "false")
        + "; enabled_parallel_fft=" + std::string(options.parallel_fft ? "true" : "false")
        + "; enabled_fft_backend_upgrade=" + std::string(options.fft_backend_upgrade ? "true" : "false")
        + "; enabled_fft_kernel_upgrade=" + std::string(options.fft_kernel_upgrade ? "true" : "false")
        + "; enabled_trace_layout_upgrade=" + std::string(options.trace_layout_upgrade ? "true" : "false")
        + "; enabled_fast_verify_pairing=" + std::string(options.fast_verify_pairing ? "true" : "false");
}

}  // namespace gatzk::util
