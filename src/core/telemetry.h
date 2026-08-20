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
inline bool g_asciiOnly = false;

inline void enableAnsiOnWindows() {
#if defined(_WIN32)
    static bool done = false;
    if (done) return;
    done = true;

    // Without this the box-drawing and sparkline characters, which are UTF-8,
    // render as mojibake: Windows consoles default to CP437/1252. Setting the
    // mode alone (below) is not enough - that only handles ANSI escapes.
    if (!SetConsoleOutputCP(CP_UTF8)) g_asciiOnly = true;

    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) return;
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
    if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        g_asciiOnly = true;  // no VT support: fall back to a plain frame
#endif
}

// Frame glyphs. The ASCII set is used when the console cannot be put into
// UTF-8 + VT mode, and via --ascii.
struct FrameChars {
    const char *tl, *tr, *bl, *br, *h, *v, *ml, *mr;
};
inline FrameChars frameChars() {
    if (g_asciiOnly)
        return {"+", "+", "+", "+", "-", "|", "+", "+"};
    return {"\u250c", "\u2510", "\u2514", "\u2518",
            "\u2500", "\u2502", "\u251c", "\u2524"};
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
        setMemOffset_ = (OffsetSetFn)sym("nvmlDeviceSetMemClkVfOffset");
        getMemOffset_ = (OffsetGetFn)sym("nvmlDeviceGetMemClkVfOffset");
        if (!init_ || !getHandle_ || init_() != 0) return false;
        if (getHandle_(deviceIndex, &dev_) != 0) return false;
        ok_ = true;
        return true;
    }

    /**
     * Memory clock offset, in MHz of transfer rate (so half of it lands on the
     * clock itself).
     *
     * This exists because CUDA and Vulkan both force performance state P2,
     * which runs the memory below its own rated speed - 10251 against 10501 on
     * a 4090. Autolykos is memory bound, so that is straight hashrate, and
     * `nvidia-smi --lock-memory-clocks` does not override it. Offsetting back
     * up to the rated clock is not an overclock, it is undoing the downclock.
     *
     * Needs root on Linux and a driver new enough to export the call.
     */
    bool setMemOffsetMhz(int mhz) {
        if (!ok_ || !setMemOffset_) return false;
        return setMemOffset_(dev_, mhz) == 0;
    }

    bool memOffsetMhz(int *out) const {
        if (!ok_ || !getMemOffset_) return false;
        return getMemOffset_(dev_, out) == 0;
    }

    bool canSetMemOffset() const { return ok_ && setMemOffset_ != nullptr; }

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
    using OffsetSetFn = int (*)(void *, int);
    using OffsetGetFn = int (*)(void *, int *);
    OffsetSetFn setMemOffset_ = nullptr;
    OffsetGetFn getMemOffset_ = nullptr;
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

    void close() { resetMemClock(); ok_ = false; }

#if !defined(_WIN32)
    // --- memory overclock via the amdgpu overdrive sysfs table ------------
    // Only usable when amdgpu.ppfeaturemask has enabled overdrive (otherwise
    // the OD table is empty) and the process is root. Clocks are MHz.

    // Stock top memory state (OD_MCLK level 1); 0 if overdrive is unavailable.
    int odMemTopMhz() const { return odLevel("OD_MCLK", 1); }
    // Highest memory clock the driver will accept (OD_RANGE MCLK upper bound).
    int odMemMaxMhz() const { return odRangeHi("MCLK"); }

    // Raise the top memory state to (true stock + bump). Resets the OD table to
    // defaults FIRST so a bump never compounds on a clock that was already
    // raised (e.g. by a prior run or a persistent OC service). Clamps to the
    // OD_RANGE max. Returns the clock written, or 0 on failure (no overdrive /
    // not root). `stockOut` (if non-null) receives the true stock clock.
    int applyMemBumpMhz(int bump, int *stockOut = nullptr) {
        writeOd("r");   // back to defaults so we read TRUE stock, not a prior OC
        writeOd("c");
        const int stock = odMemTopMhz();
        if (stockOut) *stockOut = stock;
        if (stock <= 0) return 0;      // overdrive off / not root
        odStock_ = stock;              // mark that reset() should run on close
        int target = stock + bump;
        const int mx = odMemMaxMhz();
        if (mx > 0 && target > mx) target = mx;
        if (target < stock) target = stock;
        if (!writeOd("m 1 " + std::to_string(target))) return 0;
        if (!writeOd("c")) return 0;
        return target;
    }

    // Restore the memory clock to the driver defaults. Safe to call anytime.
    void resetMemClock() {
        if (odStock_ == 0) return;
        writeOd("r");
        writeOd("c");
        odStock_ = 0;
    }
#else
    int odMemTopMhz() const { return 0; }
    int applyMemBumpMhz(int, int *stockOut = nullptr) { if (stockOut) *stockOut = 0; return 0; }
    void resetMemClock() {}
#endif

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

#if !defined(_WIN32)
    // Every integer on a line, in order (so "1: 1250MHz" -> {1,1250} and
    // "MCLK:  97Mhz  1500Mhz" -> {97,1500}), tolerating the Mhz/MHz spelling mix.
    static std::vector<int> ints(const std::string &s) {
        std::vector<int> v;
        for (size_t i = 0; i < s.size();) {
            if (s[i] >= '0' && s[i] <= '9') {
                v.push_back((int)strtol(s.c_str() + i, nullptr, 10));
                while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
            } else {
                i++;
            }
        }
        return v;
    }
    bool writeOd(const std::string &cmd) {
        std::ofstream f(device_ + "/pp_od_clk_voltage");
        if (!f) return false;
        f << cmd << "\n";
        f.flush();
        return f.good();
    }
    // Clock for `level` under an "OD_SCLK:"/"OD_MCLK:" section.
    int odLevel(const std::string &section, int level) const {
        std::ifstream f(device_ + "/pp_od_clk_voltage");
        std::string line;
        bool in = false;
        while (std::getline(f, line)) {
            if (line.rfind("OD_", 0) == 0) { in = line.rfind(section + ":", 0) == 0; continue; }
            if (!in) continue;
            std::vector<int> v = ints(line);
            if (v.size() >= 2 && v[0] == level) return v[1];
        }
        return 0;
    }
    // Upper bound for `field` (e.g. "MCLK") under the "OD_RANGE:" section.
    int odRangeHi(const std::string &field) const {
        std::ifstream f(device_ + "/pp_od_clk_voltage");
        std::string line;
        bool in = false;
        while (std::getline(f, line)) {
            if (line.rfind("OD_RANGE:", 0) == 0) { in = true; continue; }
            if (line.rfind("OD_", 0) == 0) { in = false; continue; }
            if (!in) continue;
            if (line.rfind(field + ":", 0) == 0) {
                std::vector<int> v = ints(line);
                if (v.size() >= 2) return v.back();
            }
        }
        return 0;
    }
    int odStock_ = 0;
#endif

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

    /**
     * Undo the P2 memory downclock. NVIDIA only; AMD does not do this to
     * itself. Returns the resulting offset, or a negative value if it could
     * not be applied.
     */
    int applyMemOffsetMhz(int mhz) {
        if (which_ != Nv || !nvml_.canSetMemOffset()) return -1;
        if (!nvml_.setMemOffsetMhz(mhz)) return -1;
        int got = 0;
        if (!nvml_.memOffsetMhz(&got)) return mhz;
        return got;
    }
    bool canSetMemOffset() const { return which_ == Nv && nvml_.canSetMemOffset(); }

    bool isAmd() const { return which_ == Amd; }
    bool isNvidia() const { return which_ == Nv; }
    // AMD memory-overclock pass-throughs (Linux amdgpu overdrive sysfs).
    int amdMemTopMhz() { return which_ == Amd ? amd_.odMemTopMhz() : 0; }
    int amdApplyMemBump(int bump, int *stockOut = nullptr) {
        return which_ == Amd ? amd_.applyMemBumpMhz(bump, stockOut) : 0;
    }
    void amdResetMemClock() { if (which_ == Amd) amd_.resetMemClock(); }

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

// A four-frame activity pulse. Deliberately blocky rather than a spinner:
// it reads at a glance from across a room, matches the ASCII banner, and does
// not look like a "loading" throbber, which would imply the miner is waiting
// on something.
static const char *const kPulse[] = {"\u25B0\u25B1\u25B1\u25B1",
                                     "\u25B1\u25B0\u25B1\u25B1",
                                     "\u25B1\u25B1\u25B0\u25B1",
                                     "\u25B1\u25B1\u25B1\u25B0"};
static const unsigned kPulseFrames = 4;

/** A unicode sparkline of recent hashrate, scaled to its own min/max. */
inline std::string sparkline(const std::vector<double> &v) {
    static const char *uni[] = {"\u2581", "\u2582", "\u2583", "\u2584",
                                "\u2585", "\u2586", "\u2587", "\u2588"};
    static const char *ascii[] = {"_", ".", ",", "-", "=", "+", "*", "#"};
    const char *const *bars = g_asciiOnly ? ascii : uni;
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
    // Each letter is one glyph wide with a space between, so nothing runs into
    // its neighbour - the old art had the A collapsed against its neighbours
    // and it did not read as an A at all. Widest line is 57 columns; the
    // tagline sits on its own line so an 80-column terminal never wraps it.
    printf("\n");
    printf(C_ORANGE "   ___    ___      _     _____     __  __ _" C_RESET "\n");
    printf(C_ORANGE "  / __|  / _ \\    /_\\   |_   _|   |  \\/  (_)_ _  ___ _ _"
           C_RESET "\n");
    printf(C_ORANGE "  \\__ \\ | (_) |  / _ \\    | |     | |\\/| | | ' \\/ -_) '_|"
           C_RESET "\n");
    printf(C_ORANGE "  |___/  \\___/  /_/ \\_\\   |_|     |_|  |_|_|_||_\\___|_|"
           C_RESET "\n");
    // Pure ASCII on purpose: these lines print identically under --ascii and
    // on a Windows console that never got UTF-8, so there is no second layout
    // to keep in sync.
    printf("  " C_ORANGE "May Your Hashrate Be High and Watts Low" C_RESET "\n");
    printf("  " C_DIM "open source, no dev fee - sonofatech.com" C_RESET "\n\n");
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

    // Redrawn in place each interval.
    //
    // Two things this deliberately avoids, both of which broke it before:
    //  * a hardcoded cursor-up count - it must match the lines actually
    //    printed, or the frame drifts and stacks a new copy every refresh;
    //  * a right-hand border - padding to a fixed column is unreliable once
    //    ANSI colour codes are in the string, since they take width in the
    //    buffer but none on screen.
    const FrameChars fc = frameChars();
    std::string rule;
    for (int i = 0; i < 60; i++) rule += fc.h;

    int lines = 0;
    printf("\r  %s%s%s\n", C_DIM, rule.c_str(), C_RESET);
    lines++;

    // The sparkline that used to live here is gone. It scaled to its OWN
    // min/max, so a rock-steady hashrate drew a wild jagged graph - it
    // amplified sampling noise to full height and made healthy mining look
    // broken. A chart whose y-axis is invisible and self-scaling tells the
    // reader nothing.
    //
    // What replaces it answers the question people actually have while
    // watching this: is it still working? The pulse advances every readout, so
    // motion means alive; the count is what the pool has actually taken.
    // Deliberately NOT the accepted count: that already has its own line lower
    // down, and printing it twice on one screen makes the reader check whether
    // the two disagree. This says one thing - the loop is turning.
    static unsigned pulse = 0;
    pulse++;
    char act[64];
    snprintf(act, sizeof(act), "%s%s MINING%s",
             C_ORANGE, kPulse[pulse % kPulseFrames], C_RESET);
    printf("   " C_BOLD C_GREEN "%9.2f MH/s" C_RESET "   " C_DIM "avg" C_RESET
           " %7.2f   %s\033[K\n",
           s.hashrate, s.hashrateAvg, act);
    lines++;

    if (t.valid) {
        printf("   %6.0f W   %s%3u C" C_RESET "   fan %3u%%   " C_DIM "eff" C_RESET
               " " C_ORANGE "%5.2f" C_RESET " MH/W\033[K\n",
               watts, tempColor(t.temperatureC), t.temperatureC, t.fanPercent, eff);
        lines++;
        if (t.smClockMhz || t.memClockMhz) {
            printf("   " C_DIM "core" C_RESET " %5u MHz   " C_DIM "mem" C_RESET
                   " %6u MHz\033[K\n", t.smClockMhz, t.memClockMhz);
            lines++;
        }
    } else {
        printf("   " C_DIM "power/thermals unavailable" C_RESET "\033[K\n");
        lines++;
    }

    printf("   " C_DIM "epoch" C_RESET " %-9llu " C_DIM "dataset" C_RESET
           " %5.2f GB   " C_DIM "nonces" C_RESET " %s\033[K\n",
           (unsigned long long)s.epoch, s.datasetGB,
           formatCount((double)s.totalNonces).c_str());
    lines++;

    // Colour carries meaning here, one job each: ORANGE is what you earned,
    // GREEN is raw speed, RED is trouble and nothing else. Before this, green
    // did both speed and accepted, so nothing on screen distinguished "fast"
    // from "paid".
    printf("   " C_ORANGE "%llu accepted" C_RESET, (unsigned long long)s.accepted);
    if (s.rejected)
        printf("   " C_RED "%llu rejected" C_RESET, (unsigned long long)s.rejected);
    printf("   " C_DIM "up" C_RESET " %s\033[K\n", formatDuration(s.uptimeSeconds).c_str());
    lines++;

    printf("  %s%s%s\033[K\n", C_DIM, rule.c_str(), C_RESET);
    lines++;

    printf("\033[%dA", lines);  // back to the top of the frame
    fflush(stdout);
}

}  // namespace om
