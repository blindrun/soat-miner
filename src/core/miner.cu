// SOAT Miner - an open-source GPU miner.
//
// The main loop is algorithm-agnostic: it owns the nonce counter, the job
// lifecycle, stop signals and reporting, and delegates everything
// chain-specific to an Algorithm (algo.h) and a JobSource.
//
// Currently shipping: autolykos2 (Ergo), solo against your own node.

#include <signal.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "telemetry.h"

namespace om {

/**
 * Where work comes from and where solutions go.
 * Implement this again for stratum; nothing else has to change.
 */
class JobSource {
   public:
    virtual ~JobSource() = default;
    virtual const char *describe() const = 0;
    virtual bool fetch(Job *job) = 0;
    virtual bool submit(const Job &job, const Solution &sol, std::string *err) = 0;
};

/** Solo mining against an Ergo node's /mining API. */
class ErgoNodeSource : public JobSource {
   public:
    explicit ErgoNodeSource(HttpTarget t) : t_(std::move(t)) {
        desc_ = "ergo node " + t_.host + ":" + std::to_string(t_.port) + " (solo)";
    }

    const char *describe() const override { return desc_.c_str(); }

    bool fetch(Job *job) override {
        bool ok = false;
        const std::string body =
            httpRequest(t_, "GET", "/mining/candidate", "", &ok);
        if (!ok) return false;

        std::string msgHex, bStr, hStr;
        if (!jsonString(body, "msg", &msgHex)) return false;
        if (!jsonNumber(body, "b", &bStr)) return false;
        if (!jsonNumber(body, "h", &hStr)) return false;
        jsonString(body, "pk", &job->extra);

        if (!hexToBytes(msgHex, job->msg, 32)) return false;
        decimalToLimbs(bStr, job->target);
        job->epoch = strtoull(hStr.c_str(), nullptr, 10);
        job->valid = true;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override {
        char nhex[17];
        snprintf(nhex, sizeof(nhex), "%016llx", (unsigned long long)sol.nonce);
        const std::string payload =
            "{\"pk\":\"" + job.extra + "\",\"n\":\"" + std::string(nhex) + "\"}";
        bool ok = false;
        const std::string r =
            httpRequest(t_, "POST", "/mining/solution", payload, &ok);
        if (!ok) *err = r;
        return ok;
    }

   private:
    HttpTarget t_;
    std::string desc_;
};

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

static bool stdoutIsTty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

int main(int argc, char **argv) {
    using namespace om;

    std::string algoName = "autolykos2";
    HttpTarget target;
    uint64_t batch = 1ULL << 22;
    bool bench = false;
    bool forcePlain = false;
    uint64_t benchEpoch = 1851444;
    int reportSeconds = 5;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (a == "--algo") algoName = next();
        else if (a == "--node") target.host = next();
        else if (a == "--port") target.port = atoi(next().c_str());
        else if (a == "--api-key") target.apiKey = next();
        else if (a == "--batch") batch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--bench") bench = true;
        else if (a == "--bench-height") benchEpoch = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--plain") forcePlain = true;
        else if (a == "--interval") reportSeconds = atoi(next().c_str());
        else if (a == "--list-algos") {
            for (const auto &n : availableAlgorithms()) printf("%s\n", n.c_str());
            return 0;
        } else if (a == "--help" || a == "-h") {
            printf(
                "SOAT Miner - open-source GPU miner\n\n"
                "  --algo NAME       algorithm (default autolykos2)\n"
                "  --list-algos      list compiled-in algorithms\n"
                "  --node HOST       node host (default 127.0.0.1)\n"
                "  --port N          node API port (default 9053)\n"
                "  --api-key KEY     node API key\n"
                "  --batch N         nonces per launch (default 4194304)\n"
                "  --interval N      seconds between readouts (default 5)\n"
                "  --plain           force one-line JSON output (for logs)\n"
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
        fprintf(stderr, "unknown algorithm '%s' (try --list-algos)\n",
                algoName.c_str());
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }

    Nvml nvml;
    nvml.open(0);

    const bool tty = stdoutIsTty() && !forcePlain;

    MinerStats stats;
    stats.gpuName = prop.name;
    stats.algo = algo->name();
    stats.smMajor = prop.major;
    stats.smMinor = prop.minor;
    stats.source = bench ? "benchmark (no node)" : "";

    std::unique_ptr<JobSource> source;
    if (!bench) {
        source.reset(new ErgoNodeSource(target));
        stats.source = source->describe();
    }

    if (!tty) {
        printf("SOAT Miner | %s | %.1f GB | sm_%d%d | algo=%s | source=%s\n",
               prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor,
               algo->name(), stats.source.c_str());
        fflush(stdout);
    }

    Job job;
    uint64_t preparedEpoch = ~0ULL;
    uint64_t nonce = ((uint64_t)time(nullptr) << 24);
    uint64_t intervalHashes = 0;
    const auto tStart = std::chrono::steady_clock::now();
    auto tReport = tStart;
    std::vector<Solution> sols;

    if (bench) {
        job.epoch = benchEpoch;
        memset(job.msg, 0xab, 32);
        for (int i = 0; i < 4; i++) job.target[i] = 0;  // never hits
        job.valid = true;
    }

    while (!g_stop) {
        if (!bench) {
            Job fresh;
            if (!source->fetch(&fresh)) {
                if (!tty) printf("{\"error\":\"node unreachable\"}\n");
                else printf("\n  cannot reach %s - retrying\n", source->describe());
                fflush(stdout);
                sleepSeconds(5);
                continue;
            }
            if (memcmp(fresh.msg, job.msg, 32) != 0 || fresh.epoch != job.epoch) {
                job = fresh;
            }
        }
        if (!job.valid) { sleepSeconds(1); continue; }

        if (job.epoch != preparedEpoch) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!algo->prepare(job)) return 1;
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            stats.datasetGB = algo->memoryBytes(job) / 1e9;
            if (!tty)
                printf("{\"event\":\"prepared\",\"epoch\":%llu,\"gb\":%.2f,\"secs\":%.2f}\n",
                       (unsigned long long)job.epoch, stats.datasetGB, secs);
            fflush(stdout);
            preparedEpoch = job.epoch;
        }

        sols.clear();
        if (!algo->search(job, nonce, batch, &sols)) return 1;

        for (const auto &s : sols) {
            if (!algo->verify(job, s)) {
                stats.rejected++;
                printf("\n!! candidate failed host verification (unstable clocks?) "
                       "nonce=%016llx - NOT submitted\n",
                       (unsigned long long)s.nonce);
                fflush(stdout);
                continue;
            }
            if (bench) continue;
            std::string err;
            if (source->submit(job, s, &err)) {
                stats.accepted++;
                printf("\n*** SOLUTION ACCEPTED nonce=%016llx\n",
                       (unsigned long long)s.nonce);
            } else {
                stats.rejected++;
                printf("\n*** solution REJECTED nonce=%016llx %s\n",
                       (unsigned long long)s.nonce, err.c_str());
            }
            fflush(stdout);
        }

        nonce += batch;
        intervalHashes += batch;
        stats.totalNonces += batch;

        const auto now = std::chrono::steady_clock::now();
        const double el = std::chrono::duration<double>(now - tReport).count();
        if (el >= (double)reportSeconds) {
            stats.uptimeSeconds = std::chrono::duration<double>(now - tStart).count();
            stats.hashrate = intervalHashes / el / 1e6;
            stats.hashrateAvg =
                stats.uptimeSeconds > 0
                    ? stats.totalNonces / stats.uptimeSeconds / 1e6
                    : 0.0;
            stats.epoch = job.epoch;
            printReadout(stats, nvml.sample(), tty);
            intervalHashes = 0;
            tReport = now;
        }
    }

    if (tty) printf("\n");
    printf("stopping\n");
    algo->release();
    nvml.close();
    platformShutdown();
    return 0;
}
