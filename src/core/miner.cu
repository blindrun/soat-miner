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
    int deviceIndex = 0;  // which CUDA GPU; one process per GPU for a rig
    bool gatewaySpecified = false;
    bool gatewayPortSpecified = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (a == "--algo") algoName = next();
        else if (a == "--device") deviceIndex = atoi(next().c_str());
        else if (a == "--gateway") {
            gatewaySpecified = true;
            std::string v = next();
            const size_t c = v.rfind(':');
            if (c != std::string::npos) {
                gatewayPortSpecified = true;
                opt.pearlHost = v.substr(0, c);
                opt.pearlPort = atoi(v.c_str() + c + 1);
            } else {
                opt.pearlHost = v;
            }
        }
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
        else if (a == "--pearl-transcript") opt.pearlTranscript = next();
        else if (a == "--lithos") opt.lithos = true;
        else if (a == "--pass") opt.password = next();
        else if (a == "--node") opt.target.host = next();
        else if (a == "--port") opt.target.port = atoi(next().c_str());
        else if (a == "--api-key") opt.target.apiKey = next();
        else if (a == "--batch") {
            // The grid launches 256 threads per block and advances the nonce
            // counter by exactly this many, so it must be a multiple of 256
            // and non-zero, or batches overlap, skip, or retest forever.
            uint64_t b = strtoull(next().c_str(), nullptr, 10);
            if (b < 256) b = 256;
            opt.batch = b & ~UINT64_C(255);
        }
        else if (a == "--bench") opt.bench = true;
        else if (a == "--bench-height") opt.benchEpoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench-epoch-secs") opt.benchEpochSeconds = atoi(next().c_str());
        else if (a == "--mclk-offset") opt.memOffsetMhz = atoi(next().c_str());
        else if (a == "--mem-oc") opt.memOc = true;
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
                "  --device N        GPU index (default 0; one process per GPU)\n"
                "  --pool HOST:PORT  stratum pool; Pearl uses its pool protocol\n"
                "                    and needs --wallet\n"
                "  --gateway H[:PORT] pearl-gateway for pearl-pow solo (default\n"
                "                    127.0.0.1:8337). --pool takes precedence;\n"
                "                    the gateway builds the proof for your node\n"
                "  --wallet ADDR     payout address (pool mode)\n"
                "  --worker NAME     worker name (default soat)\n"
                "  --pearl-transcript FILE  test-only Pearl pool metadata log;\n"
                "                    redacts PRL addresses and never logs wallet/auth\n"
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
                "  --mem-oc          conservative per-generation memory OC\n"
                "                    (AMD: needs overdrive+root; NVIDIA: use\n"
                "                    --mclk-offset instead)\n"
                "  --bench           benchmark, no node required\n"
                "  --bench-height H  height to benchmark at\n"
                "  --bench-epoch-secs N  benchmark: change height every N s,\n"
                "                    so the cost of a new block is measured\n");
            return 0;
        }
    }

    // The Lithos reference client listens on 127.0.0.1:4444 by default.
    // Applied after parsing so an explicit --pool wins regardless of flag order.
    if (opt.lithos && opt.poolHost.empty()) {
        opt.poolHost = "127.0.0.1";
        opt.poolPort = 4444;
    }

    // `--pool` with an empty value swallows whatever follows it, so a launcher
    // that expands a variable to nothing quietly mines to a host called
    // "--worker". Refuse it rather than spend a connection finding out.
    if (!opt.poolHost.empty() &&
        (opt.poolHost[0] == '-' || opt.poolPort <= 0 ||
         opt.poolPort > 65535)) {
        fprintf(stderr,
                "--pool needs HOST:PORT, got '%s:%d'. An empty value in a "
                "launcher script is the usual cause.\n",
                opt.poolHost.c_str(), opt.poolPort);
        return 1;
    }

    // A typed gateway must be an endpoint, not a silently repaired typo. A
    // host without a colon deliberately remains valid: it requests the
    // gateway's own default port below. This runs before CUDA setup, so a bad
    // launcher fails without waking a GPU.
    if (gatewaySpecified &&
        (opt.pearlHost.empty() || opt.pearlHost[0] == '-' ||
         (gatewayPortSpecified &&
          (opt.pearlPort <= 0 || opt.pearlPort > 65535)))) {
        fprintf(stderr, "--gateway needs HOST[:PORT], got '%s:%d'.\n",
                opt.pearlHost.c_str(), opt.pearlPort);
        return 1;
    }

    // Pearl can mine to a pool or through a local gateway. The gateway is the
    // default only for the latter, while runMiner() gives an explicit pool
    // precedence. 8337 is pearl-gateway's own default port.
    if (algoName == "pearl-pow" && opt.pearlHost.empty()) {
        opt.pearlHost = "127.0.0.1";
        opt.pearlPort = 8337;
    }
    if (opt.pearlPort == 0 && !opt.pearlHost.empty()) opt.pearlPort = 8337;
    if (!opt.pearlHost.empty() && algoName != "pearl-pow") {
        fprintf(stderr,
                "--gateway is for --algo pearl-pow; '%s' mines to a pool or a "
                "node instead\n",
                algoName.c_str());
        return 1;
    }
    if (!opt.pearlTranscript.empty() &&
        (algoName != "pearl-pow" || opt.poolHost.empty())) {
        fprintf(stderr, "--pearl-transcript requires --algo pearl-pow with --pool.\n");
        return 1;
    }

    if (!platformInit()) {
        fprintf(stderr, "network init failed\n");
        return 1;
    }
    signal(SIGINT, onSignal);
#if defined(SIGTERM)
    signal(SIGTERM, onSignal);
#endif
#if defined(SIGPIPE)
    signal(SIGPIPE, SIG_IGN);  // a pool disconnecting mid-send must not kill us
#endif
#if defined(_WIN32)
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif

    // Pick the GPU before anything allocates on it. Run one process per GPU
    // (each with its own --device and --mem-oc) to drive a multi-GPU rig.
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    if (deviceIndex < 0 || deviceIndex >= devCount) {
        fprintf(stderr, "--device %d out of range (%d CUDA GPU%s, 0-%d)\n",
                deviceIndex, devCount, devCount == 1 ? "" : "s", devCount - 1);
        return 1;
    }
    cudaSetDevice(deviceIndex);

    std::unique_ptr<Algorithm> algo(createAlgorithm(algoName));
    if (!algo) {
        fprintf(stderr, "unknown algorithm '%s' (try --list-algos)\n", algoName.c_str());
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceIndex) != cudaSuccess) {
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
