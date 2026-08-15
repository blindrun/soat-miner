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
        else if (a == "--plain") opt.plain = true;
        else if (a == "--ascii") om::g_asciiOnly = true;
        else if (a == "--interval") opt.reportSeconds = atoi(next().c_str());
        else if (a == "--list-algos") {
            for (const auto &n : availableAlgorithms()) printf("%s\n", n.c_str());
            return 0;
        } else if (a == "--help" || a == "-h") {
            printf(
                "SOAT Miner (CUDA) - open-source GPU miner\n\n"
                "  --algo NAME       algorithm (default autolykos2)\n"
                "  --list-algos      list compiled-in algorithms\n"
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
                "  --bench           benchmark, no node required\n"
                "  --bench-height H  height to benchmark at\n");
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
