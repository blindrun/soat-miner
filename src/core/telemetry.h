// GPU telemetry and the live readout.
//
// Telemetry comes from NVML, which ships with the NVIDIA driver on both Linux
// and Windows, and is loaded at runtime rather than linked. If it is missing
// the miner still runs - it just reports "n/a" for power and temperature
// instead of refusing to start.

#pragma once

#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace om {

struct GpuTelemetry {
    bool valid = false;
    unsigned powerMilliwatts = 0;
    unsigned temperatureC = 0;
    unsigned fanPercent = 0;
    unsigned smClockMhz = 0;
    unsigned memClockMhz = 0;
};

/** Runtime-loaded NVML. Optional: absence degrades the readout, not the miner. */
class Nvml {
   public:
    bool open(unsigned deviceIndex = 0) {
#if defined(_WIN32)
        lib_ = (void *)LoadLibraryA("nvml.dll");
        if (!lib_) lib_ = (void *)LoadLibraryA(
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
        if (getTemp_ && getTemp_(dev_, 0 /*GPU*/, &v) == 0) t.temperatureC = v;
        if (getFan_ && getFan_(dev_, &v) == 0) t.fanPercent = v;
        if (getClock_ && getClock_(dev_, 1 /*SM*/, &v) == 0) t.smClockMhz = v;
        if (getClock_ && getClock_(dev_, 2 /*MEM*/, &v) == 0) t.memClockMhz = v;
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

    void *lib_ = nullptr;
    void *dev_ = nullptr;
    bool ok_ = false;
    IntFn init_ = nullptr, shutdown_ = nullptr;
    HandleFn getHandle_ = nullptr;
    UintFn getPower_ = nullptr, getFan_ = nullptr;
    TempFn getTemp_ = nullptr;
    ClockFn getClock_ = nullptr;
};

/** Everything the readout displays. */
struct MinerStats {
    std::string gpuName;
    std::string algo;
    std::string source;
    int smMajor = 0, smMinor = 0;
    double hashrate = 0;      // MH/s, current interval
    double hashrateAvg = 0;   // MH/s, session
    double datasetGB = 0;
    uint64_t epoch = 0;
    uint64_t totalNonces = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t stale = 0;
    double uptimeSeconds = 0;
};

inline std::string formatDuration(double s) {
    const long t = (long)s;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60);
    return buf;
}

inline std::string formatCount(double n) {
    char buf[32];
    if (n >= 1e12) snprintf(buf, sizeof(buf), "%.2f T", n / 1e12);
    else if (n >= 1e9) snprintf(buf, sizeof(buf), "%.2f G", n / 1e9);
    else if (n >= 1e6) snprintf(buf, sizeof(buf), "%.2f M", n / 1e6);
    else snprintf(buf, sizeof(buf), "%.0f", n);
    return buf;
}

/**
 * The readout.
 *
 * Two modes on purpose: a boxed panel when attached to a terminal, and one
 * structured line per interval when it is not. The miner is meant to run under
 * systemd, and redrawing a box into a log file produces unreadable noise.
 */
inline void printReadout(const MinerStats &s, const GpuTelemetry &t, bool tty) {
    const double watts = t.valid ? t.powerMilliwatts / 1000.0 : 0.0;
    const double eff = (watts > 1.0) ? s.hashrate / watts : 0.0;

    if (!tty) {
        printf(
            "{\"ts\":%.0f,\"algo\":\"%s\",\"mhs\":%.2f,\"mhs_avg\":%.2f,"
            "\"watts\":%.1f,\"temp_c\":%u,\"fan_pct\":%u,\"eff_mh_w\":%.3f,"
            "\"epoch\":%llu,\"accepted\":%llu,\"rejected\":%llu,"
            "\"nonces\":%llu,\"uptime_s\":%.0f}\n",
            s.uptimeSeconds, s.algo.c_str(), s.hashrate, s.hashrateAvg, watts,
            t.temperatureC, t.fanPercent, eff, (unsigned long long)s.epoch,
            (unsigned long long)s.accepted, (unsigned long long)s.rejected,
            (unsigned long long)s.totalNonces, s.uptimeSeconds);
        fflush(stdout);
        return;
    }

    printf("\033[H\033[J");  // home + clear
    printf("\033[1m  SOAT Miner\033[0m  \033[2m%s\033[0m\n", s.algo.c_str());
    printf("  ────────────────────────────────────────────────────────────\n");
    printf("  GPU        %s (sm_%d%d)\n", s.gpuName.c_str(), s.smMajor, s.smMinor);
    printf("  Source     %s\n", s.source.c_str());
    printf("  ────────────────────────────────────────────────────────────\n");
    printf("  Hashrate   \033[1;32m%8.2f MH/s\033[0m    avg %.2f MH/s\n",
           s.hashrate, s.hashrateAvg);
    if (t.valid) {
        printf("  Power      %8.0f W       efficiency %.2f MH/W\n", watts, eff);
        printf("  Temp       %8u C       fan %u%%\n", t.temperatureC, t.fanPercent);
        printf("  Clocks     %8u MHz     mem %u MHz\n", t.smClockMhz, t.memClockMhz);
    } else {
        printf("  Power      %8s          (NVML unavailable)\n", "n/a");
    }
    printf("  ────────────────────────────────────────────────────────────\n");
    printf("  Epoch      %8llu         dataset %.2f GB\n",
           (unsigned long long)s.epoch, s.datasetGB);
    printf("  Solutions  \033[1;32m%llu accepted\033[0m", (unsigned long long)s.accepted);
    if (s.rejected) printf("  \033[1;31m%llu rejected\033[0m", (unsigned long long)s.rejected);
    if (s.stale) printf("  %llu stale", (unsigned long long)s.stale);
    printf("\n");
    printf("  Nonces     %8s\n", formatCount((double)s.totalNonces).c_str());
    printf("  Uptime     %8s\n", formatDuration(s.uptimeSeconds).c_str());
    printf("  ────────────────────────────────────────────────────────────\n");
    fflush(stdout);
}

}  // namespace om
