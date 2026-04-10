#pragma once

#include <string>

namespace gatzk::util {

struct Route2Options {
    // These switches only select low-level implementation routes. They must not
    // change transcript order, proof shape or verifier equations.
    bool fast_msm = GATZK_ENABLE_FAST_MSM_DEFAULT != 0;
    bool parallel_fft = GATZK_ENABLE_PARALLEL_FFT_DEFAULT != 0;
    bool fft_backend_upgrade = GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT != 0;
    bool fft_kernel_upgrade = GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT != 0;
    bool trace_layout_upgrade = GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT != 0;
    bool fast_verify_pairing = GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT != 0;
    bool cuda_trace_hotspots = false;
};

const Route2Options& route2_options();
void set_route2_options(const Route2Options& options);
std::string route2_feature_label(const Route2Options& options);
std::string route2_feature_notes(const Route2Options& options);

}  // namespace gatzk::util
