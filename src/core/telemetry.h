// GPU telemetry and the live readout.
//
// Telemetry comes from NVML, which ships with the NVIDIA driver on Linux and
// Windows and is loaded at runtime rather than linked. On AMD it is simply
// absent, and the readout degrades to "n/a" rather than the miner refusing to
// start.

#pragma once

#include <stdint.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <filesystem>
#endif

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace om {

// ------------------------------------------------------------------ ansi --
#define C_RESET "\033[0m"
#define C_DIM "\033[2m"
#define C_BOLD "\033[1m"
#define C_CYAN "\033[36m"
#define C_GREEN "\033[1;32m"
#define C_YELLOW "\033[1;33m"
#define C_RED "\033[1;31m"
#define C_BLUE "\033[38;5;39m"
#define C_ORANGE "\033[38;5;208m"

/**
 * Windows consoles do not interpret ANSI escapes unless virtual-terminal
 * processing is switched on, so without this the readout renders as literal
 * escape-code garbage on Windows 10/11. Enabling it is a no-op elsewhere.
 */
inline void enableAnsiOnWindows() {
#if defined(_WIN32)
    static bool done = false;
    if (done) return;
    done = true;
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) return;
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

inline bool stdoutIsTty() {
#if defined(_WIN32)
    enableAnsiOnWindows();
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

struct GpuTelemetry {
    bool valid = false;
    unsigned powerMilliwatts = 0;
    unsigned temperatureC = 0;
    unsigned fanPercent = 0;
    unsigned smClockMhz = 0;
    unsigned memClockMhz = 0;
};

/** Runtime-loaded NVML. Optional by design. */
class Nvml {
   public:
    bool open(unsigned deviceIndex = 0) {
#if defined(_WIN32)
        lib_ = (void *)LoadLibraryA("nvml.dll");
        if (!lib_)
            lib_ = (void *)LoadLibraryA(
                "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib_) lib_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif
        if (!lib_) return false;
        init_ = (IntFn)sym("nvmlInit_v2");
        shutdown_ = (IntFn)sym("nvmlShutdown");
        getHandle_ = (HandleFn)sym("nvmlDeviceGetHandleByIndex_v2");
        getPower_ = (UintFn)sym("nvmlDeviceGetPowerUsage");
        getTemp_ = (TempFn)sym("nvmlDeviceGetTemperature");
        getFan_ = (UintFn)sym("nvmlDeviceGetFanSpeed");
        getClock_ = (ClockFn)sym("nvmlDeviceGetClockInfo");
        if (!init_ || !getHandle_ || init_() != 0) return false;
        if (getHandle_(deviceIndex, &dev_) != 0) return false;
        ok_ = true;
        return true;
    }

    GpuTelemetry sample() const {
        GpuTelemetry t;
        if (!ok_) return t;
        unsigned v = 0;
        if (getPower_ && getPower_(dev_, &v) == 0) t.powerMilliwatts = v;
        if (getTemp_ && getTemp_(dev_, 0, &v) == 0) t.temperatureC = v;
        if (getFan_ && getFan_(dev_, &v) == 0) t.fanPercent = v;
        if (getClock_ && getClock_(dev_, 1, &v) == 0) t.smClockMhz = v;
        if (getClock_ && getClock_(dev_, 2, &v) == 0) t.memClockMhz = v;
        t.valid = true;
        return t;
    }

    void close() {
        if (ok_ && shutdown_) shutdown_();
        ok_ = false;
    }

   private:
    void *sym(const char *n) {
#if defined(_WIN32)
        return (void *)GetProcAddress((HMODULE)lib_, n);
#else
        return dlsym(lib_, n);
#endif
    }
    using IntFn = int (*)();
    using HandleFn = int (*)(unsigned, void **);
    using UintFn = int (*)(void *, unsigned *);
    using TempFn = int (*)(void *, int, unsigned *);
    using ClockFn = int (*)(void *, int, unsigned *);
    void *lib_ = nullptr, *dev_ = nullptr;
    bool ok_ = false;
    IntFn init_ = nullptr, shutdown_ = nullptr;
    HandleFn getHandle_ = nullptr;
    UintFn getPower_ = nullptr, getFan_ = nullptr;
    TempFn getTemp_ = nullptr;
    ClockFn getClock_ = nullptr;
};


/**
 * AMD telemetry via sysfs hwmon.
 *
 * NVML is NVIDIA-only, so on AMD the readout previously showed nothing for
 * power, temperature and fan. amdgpu exposes all of it under
 * /sys/class/drm/card*\/device/hwmon/hwmon*\/ instead.
 *
 * Picking the right card matters on a machine with an iGPU: this selects the
 * amdgpu node with the largest VRAM, which is the same rule the Vulkan
 * backend uses to choose its device, so the two agree.
 */
class AmdSysfs {
   public:
    bool open() {
#if defined(_WIN32)
        return false;  // sysfs is Linux-only
#else
        namespace fs = std::filesystem;
        unsigned long long bestVram = 0;
        std::error_code ec;
        for (const auto &card : fs::directory_iterator("/sys/class/drm", ec)) {
            const std::string name = card.path().filename().string();
            if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos)
                continue;
            const fs::path dev = card.path() / "device";

            const fs::path hw = dev / "hwmon";
            if (!fs::exists(hw, ec)) continue;

            for (const auto &h : fs::directory_iterator(hw, ec)) {
                if (readStr(h.path() / "name") != "amdgpu") continue;
                const unsigned long long vram =
                    readU64(dev / "mem_info_vram_total");
                if (vram >= bestVram) {
                    bestVram = vram;
                    hwmon_ = h.path().string();
                    device_ = dev.string();
                }
            }
        }
        ok_ = !hwmon_.empty();
        return ok_;
#endif
    }

    GpuTelemetry sample() const {
        GpuTelemetry t;
        if (!ok_) return t;
        // power1_average is the meaningful one on discrete cards; some parts
        // only expose power1_input.
        unsigned long long uw = readU64(hwmon_ + "/power1_average");
        if (uw == 0) uw = readU64(hwmon_ + "/power1_input");
        t.powerMilliwatts = (unsigned)(uw / 1000ULL);
        t.temperatureC = (unsigned)(readU64(hwmon_ + "/temp1_input") / 1000ULL);

        const unsigned long long pwm = readU64(hwmon_ + "/pwm1");
        if (pwm > 0) t.fanPercent = (unsigned)(pwm * 100ULL / 255ULL);

        // freq1_input is the shader clock, freq2_input the memory clock.
        t.smClockMhz = (unsigned)(readU64(hwmon_ + "/freq1_input") / 1000000ULL);
        t.memClockMhz = (unsigned)(readU64(hwmon_ + "/freq2_input") / 1000000ULL);
        t.valid = true;
        return t;
    }

    void close() { ok_ = false; }

   private:
    static std::string readStr(const std::string &p) {
        std::ifstream f(p);
        std::string s;
        if (f) std::getline(f, s);
        return s;
    }
#if !defined(_WIN32)
    static std::string readStr(const std::filesystem::path &p) {
        return readStr(p.string());
    }
    static unsigned long long readU64(const std::filesystem::path &p) {
        return readU64(p.string());
    }
#endif
    static unsigned long long readU64(const std::string &p) {
        const std::string s = readStr(p);
        if (s.empty()) return 0;
        errno = 0;
        return strtoull(s.c_str(), nullptr, 10);
    }

    std::string hwmon_, device_;
    bool ok_ = false;
};

/** Tries NVML first, then AMD sysfs. Whichever answers wins. */
class GpuMonitor {
   public:
    void open() {
        if (nvml_.open(0)) { which_ = Nv; return; }
        if (amd_.open()) { which_ = Amd; return; }
        which_ = None;
    }
    GpuTelemetry sample() const {
        switch (which_) {
            case Nv: return nvml_.sample();
            case Amd: return amd_.sample();
            default: return GpuTelemetry{};
        }
    }
    void close() {
        if (which_ == Nv) nvml_.close();
        else if (which_ == Amd) amd_.close();
    }
    bool available() const { return which_ != None; }

   private:
    enum Which { None, Nv, Amd };
    Which which_ = None;
    Nvml nvml_;
    AmdSysfs amd_;
};

/** Everything the readout displays. */
struct MinerStats {
    std::string gpuName, algo, source, arch, backend;
    double gpuMemGB = 0;
    double hashrate = 0, hashrateAvg = 0, datasetGB = 0;
    uint64_t epoch = 0, totalNonces = 0;
    uint64_t accepted = 0, rejected = 0, jobs = 0, epochs = 0;
    double uptimeSeconds = 0;
    std::vector<double> history;

    void pushSample(double v) {
        history.push_back(v);
        if (history.size() > 48) history.erase(history.begin());
    }
};

inline std::string formatDuration(double s) {
    const long t = (long)s;
    char buf[48];
    if (t >= 86400)
        snprintf(buf, sizeof(buf), "%ldd %02ld:%02ld:%02ld", t / 86400,
                 (t % 86400) / 3600, (t / 60) % 60, t % 60);
    else
        snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60,
                 t % 60);
    return buf;
}

inline std::string formatCount(double n) {
    char buf[32];
    if (n >= 1e12) snprintf(buf, sizeof(buf), "%.2fT", n / 1e12);
    else if (n >= 1e9) snprintf(buf, sizeof(buf), "%.2fG", n / 1e9);
    else if (n >= 1e6) snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);
    else snprintf(buf, sizeof(buf), "%.0f", n);
    return buf;
}

/** A unicode sparkline of recent hashrate, scaled to its own min/max. */
inline std::string sparkline(const std::vector<double> &v) {
    static const char *bars[] = {"\u2581", "\u2582", "\u2583", "\u2584",
                                 "\u2585", "\u2586", "\u2587", "\u2588"};
    if (v.size() < 2) return "";
    double lo = v[0], hi = v[0];
    for (double x : v) {
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    const double span = (hi - lo) > 1e-9 ? (hi - lo) : 1.0;
    std::string out;
    for (double x : v) {
        int idx = (int)((x - lo) / span * 7.0 + 0.5);
        if (idx < 0) idx = 0;
        if (idx > 7) idx = 7;
        out += bars[idx];
    }
    return out;
}

/** Colour a temperature by how worried you should be. */
inline const char *tempColor(unsigned c) {
    if (c >= 80) return C_RED;
    if (c >= 70) return C_YELLOW;
    return C_GREEN;
}

inline void printBanner(const MinerStats &s, bool tty) {
    if (!tty) {
        printf("SOAT Miner | %s | %.1f GB | %s | algo=%s | backend=%s | source=%s\n",
               s.gpuName.c_str(), s.gpuMemGB, s.arch.c_str(), s.algo.c_str(),
               s.backend.c_str(), s.source.c_str());
        fflush(stdout);
        return;
    }
    printf("\n");
    printf(C_ORANGE "   ___  ___  _ _____   __  __ _" C_RESET "\n");
    printf(C_ORANGE "  / __|/ _ \\/ \\_   _| |  \\/  (_)_ _  ___ _ _" C_RESET "\n");
    printf(C_ORANGE "  \\__ \\ (_) | |_| |   | |\\/| | | ' \\/ -_) '_|" C_RESET "\n");
    printf(C_ORANGE "  |___/\\___/|_(_)_|   |_|  |_|_|_||_\\___|_|" C_RESET
           "   " C_DIM "open source, no dev fee" C_RESET "\n\n");
    printf("  " C_DIM "GPU     " C_RESET "%s " C_DIM "(%s, %.1f GB)" C_RESET "\n",
           s.gpuName.c_str(), s.arch.c_str(), s.gpuMemGB);
    printf("  " C_DIM "Algo    " C_RESET "%s " C_DIM "via" C_RESET " %s\n",
           s.algo.c_str(), s.backend.c_str());
    printf("  " C_DIM "Source  " C_RESET "%s\n\n", s.source.c_str());
    fflush(stdout);
}

/** A single event line: startup, epoch change, solution, error. */
inline void logLine(bool tty, const char *kind, const std::string &msg) {
    if (!tty) {
        printf("{\"event\":\"%s\",\"msg\":\"%s\"}\n", kind, msg.c_str());
        fflush(stdout);
        return;
    }
    const char *col = C_CYAN;
    const char *tag = "info";
    if (!strcmp(kind, "ok")) { col = C_GREEN; tag = " ok "; }
    else if (!strcmp(kind, "warn")) { col = C_YELLOW; tag = "warn"; }
    else if (!strcmp(kind, "error")) { col = C_RED; tag = "fail"; }
    printf("\n  %s[%s]%s %s\n", col, tag, C_RESET, msg.c_str());
    fflush(stdout);
}

/**
 * The live readout.
 *
 * Boxed panel on a terminal, one JSON object per interval otherwise. The miner
 * is meant to run under systemd, and redrawing a panel into a log file is
 * unreadable noise.
 */
inline void printReadout(const MinerStats &s, const GpuTelemetry &t, bool tty) {
    const double watts = t.valid ? t.powerMilliwatts / 1000.0 : 0.0;
    const double eff = (watts > 1.0) ? s.hashrate / watts : 0.0;

    if (!tty) {
        printf(
            "{\"uptime_s\":%.0f,\"algo\":\"%s\",\"backend\":\"%s\",\"mhs\":%.2f,"
            "\"mhs_avg\":%.2f,\"watts\":%.1f,\"temp_c\":%u,\"fan_pct\":%u,"
            "\"eff_mh_w\":%.3f,\"epoch\":%llu,\"accepted\":%llu,"
            "\"rejected\":%llu,\"nonces\":%llu}\n",
            s.uptimeSeconds, s.algo.c_str(), s.backend.c_str(), s.hashrate,
            s.hashrateAvg, watts, t.temperatureC, t.fanPercent, eff,
            (unsigned long long)s.epoch, (unsigned long long)s.accepted,
            (unsigned long long)s.rejected, (unsigned long long)s.totalNonces);
        fflush(stdout);
        return;
    }

    // Redraw in place. The cursor must move back up by EXACTLY the number of
    // lines printed - hardcoding it drifted by 1-2 lines per refresh and
    // stacked a fresh copy of the frame every interval.
    int lines = 0;
    const char *B = "  " C_DIM;

    printf("\r%s┌──────────────────────────────────────────────────────────┐" C_RESET "\n", B);
    lines++;

    printf("%s│" C_RESET "  " C_BOLD C_GREEN "%9.2f MH/s" C_RESET "  " C_DIM "avg" C_RESET
           " %7.2f   %s%-18s" C_RESET " %s│" C_RESET "\n",
           B, s.hashrate, s.hashrateAvg, C_BLUE, sparkline(s.history).c_str(), B);
    lines++;

    if (t.valid) {
        printf("%s│" C_RESET "  %6.0f W   %s%3u\u00b0C" C_RESET "   fan %3u%%   " C_DIM
               "eff" C_RESET " %5.2f MH/W        %s│" C_RESET "\n",
               B, watts, tempColor(t.temperatureC), t.temperatureC, t.fanPercent,
               eff, B);
        lines++;
        printf("%s│" C_RESET "  " C_DIM "core" C_RESET " %5u MHz   " C_DIM "mem" C_RESET
               " %6u MHz                       %s│" C_RESET "\n",
               B, t.smClockMhz, t.memClockMhz, B);
        lines++;
    } else {
        printf("%s│" C_RESET "  " C_DIM "power/thermals unavailable" C_RESET
               "                            %s│" C_RESET "\n", B, B);
        lines++;
    }

    printf("%s├──────────────────────────────────────────────────────────┤" C_RESET "\n", B);
    lines++;

    printf("%s│" C_RESET "  " C_DIM "epoch" C_RESET " %-9llu " C_DIM "dataset" C_RESET
           " %5.2f GB   " C_DIM "nonces" C_RESET " %-8s   %s│" C_RESET "\n",
           B, (unsigned long long)s.epoch, s.datasetGB,
           formatCount((double)s.totalNonces).c_str(), B);
    lines++;

    printf("%s│" C_RESET "  " C_GREEN "%llu accepted" C_RESET, B,
           (unsigned long long)s.accepted);
    if (s.rejected)
        printf("   " C_RED "%llu rejected" C_RESET, (unsigned long long)s.rejected);
    printf("   " C_DIM "up" C_RESET " %-12s", formatDuration(s.uptimeSeconds).c_str());
    printf("      %s│" C_RESET "\n", B);
    lines++;

    printf("%s└──────────────────────────────────────────────────────────┘" C_RESET "\n", B);
    lines++;

    printf("\033[%dA", lines);  // back to the top of the frame
    fflush(stdout);
}

}  // namespace om
