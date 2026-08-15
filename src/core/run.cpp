// The shared run loop. Backend-agnostic: it owns the nonce counter, job
// lifecycle, stop signal and reporting, and only ever asks the Algorithm to
// "test these nonces".

#include "run.h"
#include "stratum.h"
#include <thread>

namespace om {

int runMiner(Algorithm *algo, const RunOptions &opt, const char *gpuName,
             double gpuMemGB, const char *archLabel, volatile sig_atomic_t *stop) {
    GpuMonitor gpu;
    gpu.open();

    const bool tty = stdoutIsTty() && !opt.plain;

    MinerStats stats;
    stats.gpuName = gpuName ? gpuName : "unknown GPU";
    stats.algo = algo->name();
    stats.arch = archLabel ? archLabel : "";
    stats.backend = opt.backendLabel;
    stats.gpuMemGB = gpuMemGB;

    // Banner before any network or GPU work. Connecting to a pool and
    // building the dataset take ~15s combined, and printing nothing for that
    // long reads as a hang - it is the first thing a new user sees.
    stats.source = opt.bench ? "benchmark (no pool/node)"
                 : (!opt.poolHost.empty()
                        ? opt.poolHost + ":" + std::to_string(opt.poolPort) + " (pool)"
                        : "ergo node " + opt.target.host + ":" +
                              std::to_string(opt.target.port) + " (solo)");
    printBanner(stats, tty);

    std::unique_ptr<JobSource> source;
    uint64_t noncePrefix = 0;
    int nonceBits = 64;
    if (opt.bench) {
        stats.source = "benchmark (no pool/node)";
    } else if (!opt.poolHost.empty()) {
        logLine(tty, "info", "connecting to " + opt.poolHost + ":" +
                                 std::to_string(opt.poolPort) + " ...");
        auto *st = new StratumSource(opt.poolHost, opt.poolPort, opt.wallet,
                                     opt.worker, opt.password);
        std::string err;
        if (!st->start(&err)) {
            fprintf(stderr, "pool connect failed: %s\n", err.c_str());
            delete st;
            return 1;
        }
        source.reset(st);
        stats.source = source->describe();
        logLine(tty, "ok", "connected, waiting for first job");
        // Give the pool a moment to deliver extranonce + first job.
        for (int i = 0; i < 50; i++) {
            Job probe;
            if (st->fetch(&probe)) break;
            sleepSeconds(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        noncePrefix = st->noncePrefix();
        nonceBits = st->nonceBitsOwned();
    } else {
        source.reset(new ErgoNodeSource(opt.target));
        stats.source = source->describe();
    }

    Job job;
    uint64_t preparedEpoch = ~0ULL;
    const uint64_t nonceMask =
        (nonceBits >= 64) ? ~0ULL : ((1ULL << nonceBits) - 1ULL);
    uint64_t nonceCounter = ((uint64_t)time(nullptr) << 16) & nonceMask;
    uint64_t nonce = noncePrefix | nonceCounter;
    uint64_t intervalHashes = 0;
    const auto tStart = std::chrono::steady_clock::now();
    auto tReport = tStart;
    std::vector<Solution> sols;

    if (opt.bench) {
        job.epoch = opt.benchEpoch;
        memset(job.msg, 0xab, 32);
        for (int i = 0; i < 4; i++) job.target[i] = 0;  // never hits
        job.valid = true;
    }

    while (!*stop) {
        if (!opt.bench) {
            Job fresh;
            if (!source->fetch(&fresh)) {
                logLine(tty, "warn", std::string("cannot reach ") +
                                         source->describe() + " - retrying in 5s");
                sleepSeconds(5);
                continue;
            }
            if (memcmp(fresh.msg, job.msg, 32) != 0 || fresh.epoch != job.epoch) {
                const bool newEpoch = fresh.epoch != job.epoch;
                job = fresh;
                if (!newEpoch) stats.jobs++;
            }
        }
        if (!job.valid) { sleepSeconds(1); continue; }

        if (job.epoch != preparedEpoch) {
            {
                char pre[160];
                snprintf(pre, sizeof(pre),
                         "epoch %llu - building %.2f GB dataset, please wait...",
                         (unsigned long long)job.epoch,
                         algo->memoryBytes(job) / 1e9);
                logLine(tty, "info", pre);
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (!algo->prepare(job)) return 1;
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            stats.datasetGB = algo->memoryBytes(job) / 1e9;
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "dataset ready in %.2fs - mining", secs);
            logLine(tty, "ok", buf);
            preparedEpoch = job.epoch;
            stats.epochs++;
        }

        sols.clear();
        if (!algo->search(job, nonce, opt.batch, &sols)) return 1;

        for (const auto &s : sols) {
            char buf[160];
            if (!algo->verify(job, s)) {
                stats.rejected++;
                snprintf(buf, sizeof(buf),
                         "candidate failed host verification (unstable clocks?) "
                         "nonce=%016llx - NOT submitted",
                         (unsigned long long)s.nonce);
                logLine(tty, "error", buf);
                continue;
            }
            if (opt.bench) continue;
            std::string err;
            if (source->submit(job, s, &err)) {
                stats.accepted++;
                snprintf(buf, sizeof(buf), "SOLUTION ACCEPTED  nonce=%016llx",
                         (unsigned long long)s.nonce);
                logLine(tty, "ok", buf);
            } else {
                stats.rejected++;
                snprintf(buf, sizeof(buf), "solution rejected  nonce=%016llx  %s",
                         (unsigned long long)s.nonce, err.c_str());
                logLine(tty, "error", buf);
            }
        }

        nonceCounter = (nonceCounter + opt.batch) & nonceMask;
        nonce = noncePrefix | nonceCounter;
        intervalHashes += opt.batch;
        stats.totalNonces += opt.batch;

        const auto now = std::chrono::steady_clock::now();
        const double el = std::chrono::duration<double>(now - tReport).count();
        if (el >= (double)opt.reportSeconds) {
            stats.uptimeSeconds = std::chrono::duration<double>(now - tStart).count();
            stats.hashrate = intervalHashes / el / 1e6;
            stats.hashrateAvg =
                stats.uptimeSeconds > 0
                    ? stats.totalNonces / stats.uptimeSeconds / 1e6
                    : 0.0;
            stats.epoch = job.epoch;
            stats.pushSample(stats.hashrate);
            printReadout(stats, gpu.sample(), tty);
            intervalHashes = 0;
            tReport = now;
        }
    }

    if (tty) printf("\n");
    logLine(tty, "info", "stopping");
    algo->release();
    gpu.close();
    return 0;
}

}  // namespace om
