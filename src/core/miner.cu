// SOAT Miner - CUDA build entry point (NVIDIA).
//
// All the logic lives in run.cpp; this only picks the algorithm, reports the
// device, and hands over.

#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "run.h"

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

    std::string algoName = "autolykos2";
    RunOptions opt;
    opt.backendLabel = "CUDA";

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (a == "--algo") algoName = next();
                else if (a == "--pool") {
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
        else if (a == "--pass") opt.password = next();
        else if (a == "--node") opt.target.host = next();
        else if (a == "--port") opt.target.port = atoi(next().c_str());
        else if (a == "--api-key") opt.target.apiKey = next();
        else if (a == "--batch") opt.batch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench") opt.bench = true;
        else if (a == "--bench-height") opt.benchEpoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench-epoch-secs") opt.benchEpochSeconds = atoi(next().c_str());
        else if (a == "--mclk-offset") opt.memOffsetMhz = atoi(next().c_str());
        else if (a == "--cache-dag") {
            const std::string v = next();
            opt.prefetch = (v == "on" || v == "1") ? 1
                         : (v == "off" || v == "0") ? 0
                         : -1;
        }
        else if (a == "--plain") opt.plain = true;
        else if (a == "--ascii") om::g_asciiOnly = true;
        else if (a == "--interval") opt.reportSeconds = atoi(next().c_str());
        else if (a == "--list-algos") {
            for (const auto &n : availableAlgorithms()) printf("%s\n", n.c_str());
            return 0;
        } else if (a == "--list-devices") {
            // The Vulkan build has always had this; the CUDA build did not,
            // and silently ignored it - so `soat-miner.sh --list-devices` on
            // an NVIDIA machine started mining instead of listing anything.
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                printf("no CUDA device found\n");
                return 1;
            }
            for (int d = 0; d < n; d++) {
                cudaDeviceProp p{};
                if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
                printf("  [%d] %-38s %5.1f GB  sm_%d%d  %d SMs\n", d, p.name,
                       p.totalGlobalMem / 1e9, p.major, p.minor,
                       p.multiProcessorCount);
            }
            return 0;
        } else if (a == "--help" || a == "-h") {
            printf(
                "SOAT Miner (CUDA) - open-source GPU miner\n\n"
                "  --algo NAME       algorithm (default autolykos2)\n"
                "  --list-algos      list compiled-in algorithms\n"
                "  --list-devices    list CUDA GPUs and exit\n"
                "  --pool HOST:PORT  stratum pool (omit for solo via node)\n"
                "  --wallet ADDR     payout address (pool mode)\n"
                "  --worker NAME     worker name (default soat)\n"
                "  --pass P          pool password (default x)\n"
                "  --node HOST       Ergo node host (default 127.0.0.1)\n"
                "  --port N          node API port (default 9053)\n"
                "  --api-key KEY     node API key\n"
                "  --batch N         nonces per launch (default 4194304)\n"
                "  --interval N      seconds between readouts (default 5)\n"
                "  --plain           one-line JSON output, for logs/systemd\n"
                "  --ascii           plain ASCII frame (no box-drawing glyphs)\n"
                "  --cache-dag MODE  build the next block's dataset ahead of\n"
                "                    time: auto (default), on, off. Needs a\n"
                "                    second dataset resident, so auto skips it\n"
                "                    on cards without the spare VRAM\n"
                "  --mclk-offset N   memory clock offset in MHz of transfer\n"
                "                    rate. CUDA and Vulkan both force state P2,\n"
                "                    which runs memory under its rated speed;\n"
                "                    500 restores a 4090 to stock for +2.3%%.\n"
                "                    Needs root on Linux. 0 leaves it alone\n"
                "  --bench           benchmark, no node required\n"
                "  --bench-height H  height to benchmark at\n"
                "  --bench-epoch-secs N  benchmark: change height every N s,\n"
                "                    so the cost of a new block is measured\n");
            return 0;
        }
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

    std::unique_ptr<Algorithm> algo(createAlgorithm(algoName));
    if (!algo) {
        fprintf(stderr, "unknown algorithm '%s' (try --list-algos)\n", algoName.c_str());
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    char arch[32];
    snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);

    const int rc = runMiner(algo.get(), opt, prop.name, prop.totalGlobalMem / 1e9,
                            arch, &g_stop);
    platformShutdown();
    return rc;
}
