#include "gatzk/util/route2.hpp"

#include <mutex>

// 保存一次运行生效的 route2 选项，并把开关压缩成 manifest / benchmark 可读标签。

namespace gatzk::util
{
    namespace
    {

        Route2Options &mutable_options()
        {
            static Route2Options options;
            return options;
        }

        std::mutex &options_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

    }

    const Route2Options &route2_options()
    {
        return mutable_options();
    }

    // 全局路由选项只在 run 启动阶段设置一次，后续 proving / verifying 只读
    void set_route2_options(const Route2Options &options)
    {
        std::lock_guard<std::mutex> lock(options_mutex());
        mutable_options() = options;
    }

    // 生成精简标签，帮助 README / latest / summary 快速区分执行策略
    std::string route2_feature_label(const Route2Options &options)
    {
        std::string label;
        if (options.fast_msm)
        {
            label += "msm";
        }
        if (options.parallel_fft)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "fft";
        }
        if (options.packed_fft_backend)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "packed";
        }
        if (options.packed_fft_backend && options.rotated_fft_kernel)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "kernel";
        }
        if (options.compact_trace_layout)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "layout";
        }
        if (options.fast_verify_pairing)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "pairing";
        }
        if (options.cuda_trace_hotspots)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "cuda";
        }
        if (options.cuda_field_hotspots)
        {
            if (!label.empty())
            {
                label += "_";
            }
            label += "cuda_field";
        }
        return label.empty() ? "baseline" : label;
    }

    // 生成更完整的布尔开关串，写入 notes 以便回溯正式 benchmark 配置。
    std::string route2_feature_notes(const Route2Options &options)
    {
        return "enabled_fast_msm=" + std::string(options.fast_msm ? "true" : "false") +
               "; enabled_parallel_fft=" + std::string(options.parallel_fft ? "true" : "false") +
               "; enabled_packed_fft_backend=" + std::string(options.packed_fft_backend ? "true" : "false") +
               "; enabled_rotated_fft_kernel=" + std::string(options.rotated_fft_kernel ? "true" : "false") +
               "; enabled_compact_trace_layout=" + std::string(options.compact_trace_layout ? "true" : "false") +
               "; enabled_fast_verify_pairing=" + std::string(options.fast_verify_pairing ? "true" : "false") +
               "; enabled_cuda_trace_hotspots=" + std::string(options.cuda_trace_hotspots ? "true" : "false") +
               "; enabled_cuda_field_hotspots=" + std::string(options.cuda_field_hotspots ? "true" : "false");
    }

}
