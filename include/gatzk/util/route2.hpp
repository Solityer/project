#pragma once

#include <string>

namespace gatzk::util
{

    struct Route2Options
    {
        bool fast_msm = GATZK_ENABLE_FAST_MSM_DEFAULT != 0;
        bool parallel_fft = GATZK_ENABLE_PARALLEL_FFT_DEFAULT != 0;
        bool packed_fft_backend = GATZK_ENABLE_PACKED_FFT_DEFAULT != 0;
        bool rotated_fft_kernel = GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT != 0;
        bool compact_trace_layout = GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT != 0;
        bool fast_verify_pairing = GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT != 0;
        bool cuda_trace_hotspots = false;
        bool cuda_field_hotspots = false;
        bool cuda_force_ntt_for_mechanism = false;
    };

    const Route2Options &route2_options();
    void set_route2_options(const Route2Options &options);
    std::string route2_feature_label(const Route2Options &options);
    std::string route2_feature_notes(const Route2Options &options);

}
