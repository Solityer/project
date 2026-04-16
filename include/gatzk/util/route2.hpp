#pragma once
// 本文件定义正式主线允许使用的执行层路由开关。
// 这些开关只能改变 CPU / GPU 热点放置，不能改变 transcript、proof 形状或验证方程。

#include <string>

namespace gatzk::util {

struct Route2Options {
    // 这些开关只选择底层实现路线，不允许改动 formal 语义。
    bool fast_msm = GATZK_ENABLE_FAST_MSM_DEFAULT != 0;
    bool parallel_fft = GATZK_ENABLE_PARALLEL_FFT_DEFAULT != 0;
    bool fft_backend_upgrade = GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT != 0;
    bool fft_kernel_upgrade = GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT != 0;
    bool trace_layout_upgrade = GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT != 0;
    bool fast_verify_pairing = GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT != 0;
    bool cuda_trace_hotspots = false;
    bool cuda_field_hotspots = false;
};

// 读取当前全局生效的路由选项。
const Route2Options& route2_options();
// 设置一次运行使用的统一路由选项。
void set_route2_options(const Route2Options& options);
// 将路由选项压缩成 manifest / benchmark 使用的短标签。
std::string route2_feature_label(const Route2Options& options);
// 生成人类可读的路由说明，写入 notes 供 artifact 与 benchmark 汇总追踪。
std::string route2_feature_notes(const Route2Options& options);

}  // namespace gatzk::util
