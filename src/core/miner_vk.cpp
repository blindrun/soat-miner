// SOAT Miner - Vulkan build entry point (AMD, NVIDIA, Intel).
//
// Same run loop as the CUDA build; only the Algorithm differs. This is the
// portable binary: the SPIR-V shader is compiled into it and every vendor's
// Vulkan driver consumes the same module, so one build runs anywhere.

#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "run.h"

namespace om {
Algorithm *makeAutolykos2VK(int deviceIndex);
void vkListDevices();
const char *vkDeviceName();
double vkDeviceMemGB();
const char *vkDriverVersion();
}  // namespace om

static volatile sig_atomic_t g_stop = 0;
static void onSignal(int) { g_stop = 1; }

#if defined(_WIN32)
static BOOL WINAPI consoleHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_BREAK_EVENT) {
        g_stop = 1;
        return TRUE;
    }
    return FALSE;
}
#endif

int main(int argc, char **argv) {
    using namespace om;

    RunOptions opt;
    opt.backendLabel = "Vulkan";
    int deviceIndex = -1;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (a == "--pool") {
            std::string v = next();
            const size_t c = v.rfind(':');
            if (c != std::string::npos) {
                opt.poolHost = v.substr(0, c);
                opt.poolPort = atoi(v.c_str() + c + 1);
            } else {
                opt.poolHost = v;
            }
        }
        else if (a == "--wallet" || a == "--user") opt.wallet = next();
        else if (a == "--worker") opt.worker = next();
        else if (a == "--lithos") opt.lithos = true;
        else if (a == "--pass") opt.password = next();
        else if (a == "--node") opt.target.host = next();
        else if (a == "--port") opt.target.port = atoi(next().c_str());
        else if (a == "--api-key") opt.target.apiKey = next();
        else if (a == "--batch") opt.batch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench") opt.bench = true;
        else if (a == "--bench-height") opt.benchEpoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench-epoch-secs") opt.benchEpochSeconds = atoi(next().c_str());
        else if (a == "--mclk-offset") opt.memOffsetMhz = atoi(next().c_str());
        // No --cache-dag here. Building ahead was measured on this backend and
        // it loses: see the README. Accepting the flag and ignoring it would be
        // worse than not having it.
        else if (a == "--plain") opt.plain = true;
        else if (a == "--ascii") om::g_asciiOnly = true;
        else if (a == "--interval") opt.reportSeconds = atoi(next().c_str());
        else if (a == "--algo") { /* only autolykos2 for now */ (void)next(); }
        else if (a == "--device") deviceIndex = atoi(next().c_str());
        else if (a == "--list-devices") { vkListDevices(); return 0; }
        else if (a == "--list-algos") { printf("autolykos2\n"); return 0; }
        else if (a == "--help" || a == "-h") {
            printf(
                "SOAT Miner (Vulkan) - open-source GPU miner\n\n"
                "  --device N        GPU index (default: largest VRAM)\n"
                "  --list-devices    list Vulkan GPUs and exit\n"
                "  --pool HOST:PORT  stratum pool (omit for solo via node)\n"
                "  --wallet ADDR     payout address (pool mode)\n"
                "  --worker NAME     worker name (default soat)\n"
                "  --lithos          mine to a Lithos client (decentralised\n"
                "                    pool protocol; same Autolykos v2 PoW).\n"
                "                    Defaults --pool to 127.0.0.1:4444 and\n"
                "                    does not require an Ergo address\n"
                "  --pass P          pool password (default x)\n"
                "  --node HOST       Ergo node host (default 127.0.0.1)\n"
                "  --port N          node API port (default 9053)\n"
                "  --api-key KEY     node API key\n"
                "  --batch N         nonces per launch (default 4194304)\n"
                "  --interval N      seconds between readouts (default 5)\n"
                "  --bench-epoch-secs N  benchmark: change height every N s\n"
                "  --plain           one-line JSON output, for logs/systemd\n"
                "  --ascii           plain ASCII frame (no box-drawing glyphs)\n"
                "  --mclk-offset N   memory clock offset in MHz of transfer\n"
                "                    rate. CUDA and Vulkan both force state P2,\n"
                "                    which runs memory under its rated speed;\n"
                "                    500 restores a 4090 to stock for +2.3%%.\n"
                "                    Needs root on Linux. 0 leaves it alone\n"
                "  --bench           benchmark, no node required\n"
                "  --bench-height H  height to benchmark at\n");
            return 0;
        }
    }

    // The Lithos reference client listens on 127.0.0.1:4444 by default.
    // Applied after parsing so an explicit --pool wins regardless of flag order.
    if (opt.lithos && opt.poolHost.empty()) {
        opt.poolHost = "127.0.0.1";
        opt.poolPort = 4444;
    }

    if (!platformInit()) {
        fprintf(stderr, "network init failed\n");
        return 1;
    }
    signal(SIGINT, onSignal);
#if defined(SIGTERM)
    signal(SIGTERM, onSignal);
#endif
#if defined(_WIN32)
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif

    std::unique_ptr<Algorithm> algo(makeAutolykos2VK(deviceIndex));
    if (!algo) {
        fprintf(stderr, "failed to initialise Vulkan backend\n");
        return 1;
    }

    std::string arch = vkDriverVersion();
    const int rc = runMiner(algo.get(), opt, vkDeviceName(), vkDeviceMemGB(),
                            arch.c_str(), &g_stop);
    platformShutdown();
    return rc;
}
