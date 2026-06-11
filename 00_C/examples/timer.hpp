// timer.hpp
#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>
#include <unordered_map>

class MultiTimer {
public:
    // 라벨을 지정해서 시작 시간 저장
    void start(const std::string& label) {
        timers[label] = std::chrono::steady_clock::now();
    }

    long long elapsed_ns(const std::string& label) const {
        auto it = timers.find(label);
        if (it == timers.end()) return -1.0;

        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - it->second).count();
    }

    // 해당 라벨의 경과 시간(ms)을 반환
    long long elapsed_ms(const std::string& label) const {
        auto it = timers.find(label);
        if (it == timers.end()) return -1.0; // 존재하지 않으면 -1 반환

        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second).count();
    }

    // 해당 라벨의 경과 시간(s)을 반환
    double elapsed_sec(const std::string& label) const {
        auto it = timers.find(label);
        if (it == timers.end()) return -1.0;

        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - it->second).count();
    }

private:
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> timers;
};

#endif // TIMER_HPP
