#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

// 简单 wall-clock 计时器；runner 用它记录 commitment/prove/verify 时间。
class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::duration<double> totalTime;
    bool isRunning;

public:
    Timer() : totalTime(0.0), isRunning(false) {}

    // 开始或恢复计时；无输出副作用。
    void start() {
        if (!isRunning) {
            startTime = std::chrono::high_resolution_clock::now();
            isRunning = true;
        }
    }

    // 停止计时并累加秒数；用于 official timing 字段。
    void stop() {
        if (isRunning) {
            std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
            totalTime += std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
            isRunning = false;
        }
    }

    // 清空累计时间；用于单个阶段重新计时。
    void reset() {
        isRunning = false;
        totalTime = std::chrono::duration<double>(0.0);
    }

    // 返回累计秒数；exporter 后续换算为毫秒字段。
    double getTotalTime() const {
        return totalTime.count();
    }
};

#endif  // TIMER_HPP_INCLUDED
