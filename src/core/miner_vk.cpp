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
#include "vk_common.h"

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
    // Autolykos was the only algorithm this backend had, so it stays the
    // default: every existing launcher and config.txt omits --algo entirely.
    std::string algoName = "autolykos2";

    bool gatewaySpecified = false, gatewayPortSpecified = false;
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
        // No --cache-dag here. Building ahead was measured on this backend and
        // it loses: see the README. Accepting the flag and ignoring it would be
        // worse than not having it.
        // Pearl runs on this backend too now, and its diagnostic transcript is
        // what the mock-pool test asserts against. Without the flag here the
        // Vulkan binary silently ignored it and wrote nothing.
        // Pearl runs on this backend now, so its two Pearl-specific flags
        // belong here too. Both were CUDA-only and the Vulkan parser simply
        // did not have them: --gateway meant solo mining was impossible on
        // Vulkan, and --pearl-transcript was silently swallowed. The
        // unknown-option refusal added in the same session is how both were
        // found - they went from "ignored" to "refused", loudly.
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
        else if (a == "--pearl-transcript") opt.pearlTranscript = next();
        else if (a == "--plain") opt.plain = true;
        else if (a == "--ascii") om::g_asciiOnly = true;
        else if (a == "--interval") opt.reportSeconds = atoi(next().c_str());
        else if (a == "--algo") {
            // Swallowing an unknown algorithm here meant `--algo pearl-pow`
            // silently mined Ergo instead: the launcher picks Vulkan on
            // Blackwell, this binary ignored the flag, and the user got
            // "'prl1...' is not a valid Ergo address" with no clue why.
            // Refuse instead, and say which algorithms this build actually has
            // rather than naming one - the registry is the source of truth now.
            algoName = next();
            bool known = false;
            for (const auto &n : availableVulkanAlgorithms())
                if (n == algoName) { known = true; break; }
            if (!known) {
                fprintf(stderr,
                        "the Vulkan build does not have '%s'.\n"
                        "  it has:", algoName.c_str());
                for (const auto &n : availableVulkanAlgorithms())
                    fprintf(stderr, " %s", n.c_str());
                fprintf(stderr, "\n");
                // Pearl is still CUDA-only, so name the way out rather than
                // leaving the user to guess which binary to run.
                // Pearl is no longer CUDA-only, so this build has it and
                // the branch is unreachable for that name. Left as a general
                // hint rather than deleted: the next algorithm to ship on one
                // backend first will want exactly this message.
                if (algoName == "pearl-pow")
                    fprintf(stderr,
                            "  This build has no Pearl support. Run "
                            "./soat-miner (or set BACKEND=cuda in config.txt) "
                            "instead of the Vulkan binary.\n");
                return 1;
            }
        }
        else if (a == "--device") deviceIndex = atoi(next().c_str());
        else if (a == "--list-devices") { vkListDevices(); return 0; }
        else if (a == "--list-algos") {
            for (const auto &n : availableVulkanAlgorithms())
                printf("%s\n", n.c_str());
            return 0;
        }
        else if (a == "--help" || a == "-h") {
            printf(
                "SOAT Miner (Vulkan) - open-source GPU miner\n\n"
                "  --device N        GPU index (default: largest VRAM)\n"
                "  --list-devices    list Vulkan GPUs and exit\n"
                "  --algo NAME       algorithm (default autolykos2)\n"
                "  --list-algos      list this build's algorithms and exit\n"
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
                "  --mem-oc          conservative per-generation memory OC\n"
                "                    (AMD: needs overdrive+root; NVIDIA: use\n"
                "                    --mclk-offset instead)\n"
                "  --bench           benchmark, no node required\n"
                "  --bench-height H  height to benchmark at\n");
            return 0;
        }
        else {
            // AN UNKNOWN ARGUMENT IS AN ERROR, NOT A SHRUG.
            //
            // This loop used to fall off the end and ignore anything it did
            // not recognise. A typo, a renamed flag, or a probe like
            // `--version` (which this miner does not have) therefore did not
            // fail - it started mining, on whatever GPU was there, with
            // whatever defaults the rest of the line implied. That is how a
            // one-line check turned into an unclaimed run on a card another
            // lane could have been measuring on.
            //
            // The same mistake is already documented ten lines above for
            // --algo, where swallowing an unknown algorithm silently mined
            // Ergo instead of Pearl. Fixing it there and not here left the
            // hole open for every other flag.
            fprintf(stderr,
                    "unknown option '%s'. Run with --help for the list.\n",
                    a.c_str());
            return 1;
        }
    }

    // The transcript is written by PearlPoolSource and by nothing else, so
    // asking for one on the gateway path produces no file and no complaint.
    // The CUDA parser already refused that combination; I copied the flag here
    // without the guard, and then spent two hours of a real gateway run whose
    // --pearl-transcript wrote nothing. Third time this session that a flag was
    // accepted and silently ineffective - it is the same failure as an unknown
    // option being ignored, one level in.
    if (!opt.pearlTranscript.empty() &&
        (algoName != "pearl-pow" || opt.poolHost.empty())) {
        fprintf(stderr, "--pearl-transcript requires --algo pearl-pow with --pool.\n");
        return 1;
    }

    // Same validation and defaults as the CUDA path, deliberately identical.
    // A typed gateway must be an endpoint, not a silently repaired typo; a
    // host without a colon stays valid and takes the default port below.
    if (gatewaySpecified &&
        (opt.pearlHost.empty() || opt.pearlHost[0] == '-' ||
         (gatewayPortSpecified &&
          (opt.pearlPort <= 0 || opt.pearlPort > 65535)))) {
        fprintf(stderr, "--gateway needs HOST[:PORT], got '%s:%d'.\n",
                opt.pearlHost.c_str(), opt.pearlPort);
        return 1;
    }
    // 8337 is pearl-gateway's own default port. The gateway is Pearl's default
    // source; runMiner() gives an explicit --pool precedence over it.
    if (algoName == "pearl-pow" && opt.pearlHost.empty() && opt.poolHost.empty()) {
        opt.pearlHost = "127.0.0.1";
        opt.pearlPort = 8337;
    }
    if (opt.pearlPort == 0 && !opt.pearlHost.empty()) opt.pearlPort = 8337;
    if (!opt.pearlHost.empty() && algoName != "pearl-pow") {
        fprintf(stderr,
                "--gateway is for --algo pearl-pow; '%s' mines to a pool or a "
                "node instead\n", algoName.c_str());
        return 1;
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

    std::unique_ptr<Algorithm> algo(
        createVulkanAlgorithm(algoName, deviceIndex));
    if (!algo) {
        fprintf(stderr, "failed to initialise the Vulkan backend for %s\n",
                algoName.c_str());
        return 1;
    }

    std::string arch = vkDriverVersion();
    const int rc = runMiner(algo.get(), opt, vkDeviceName(), vkDeviceMemGB(),
                            arch.c_str(), &g_stop);
    platformShutdown();
    return rc;
}
