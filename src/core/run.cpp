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
    stats.source = opt.bench ? "BENCHMARK ONLY - not mining, nothing submitted"
                 : (!opt.poolHost.empty()
                        ? opt.poolHost + ":" + std::to_string(opt.poolPort) + " (pool)"
                        : "ergo node " + opt.target.host + ":" +
                              std::to_string(opt.target.port) + " (solo)");
    printBanner(stats, tty);

    if (opt.bench) {
        logLine(tty, "warn",
                "benchmark mode: measuring hashrate only. No pool, no wallet, "
                "no shares, no payouts.");
        logLine(tty, "info",
                "to actually mine, use a mine_ergo_* script (edit WALLET first) "
                "or pass --pool HOST:PORT --wallet <address>");
    }

    algo->setPrefetch(opt.prefetch);

    std::unique_ptr<JobSource> source;
    uint64_t noncePrefix = 0;
    int nonceBits = 64;
    if (opt.bench) {
        stats.source = "benchmark (no pool/node)";
    } else if (!opt.poolHost.empty()) {
        // Catch the obvious cases before spending a connection on them.
        const std::string &w = opt.wallet;
        const bool looksLikeErgo =
            w.size() >= 40 && w.size() <= 60 && w[0] == '9' &&
            w.find_first_of(" \t") == std::string::npos;
        if (w.empty() || w.rfind("9YOUR", 0) == 0 || !looksLikeErgo) {
            logLine(tty, "error",
                    w.empty() ? "no --wallet given; pool mining needs your Ergo "
                                "payout address"
                              : "'" + w + "' is not a valid Ergo address");
            logLine(tty, "error",
                    "edit WALLET in the mine_ergo_* script (or config.txt) to a "
                    "real address - they start with 9 and are about 51 chars");
            return 1;
        }
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
        logLine(tty, "ok", "TCP connected, waiting for first job...");
        // Wait up to 20s: some pools are slow to send the first mining.notify,
        // and 5s was not always enough.
        bool gotJob = false;
        for (int i = 0; i < 200; i++) {
            Job probe;
            if (st->fetch(&probe)) { gotJob = true; break; }
            if (st->loginRejected()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!gotJob && !st->loginRejected()) {
            logLine(tty, "error",
                    "connected to the pool but it sent no job in 20s. The "
                    "socket is open, so this is not a firewall - the pool "
                    "may have refused the wallet silently, or the port is "
                    "for a different algorithm.");
            return 1;
        }
        if (st->loginRejected()) {
            logLine(tty, "error",
                    "pool REJECTED the login: " + st->loginError());
            logLine(tty, "error",
                    "check --wallet is a real Ergo address (starts with 9, "
                    "~51 chars). The placeholder in the example scripts will "
                    "not work.");
            return 1;
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
    uint64_t hostRejected = 0;  // failures this side of the wire, kept separate
                                // from what the pool rejected
    const auto tStart = std::chrono::steady_clock::now();
    auto tReport = tStart;
    std::vector<Solution> sols;

    if (opt.bench) {
        job.epoch = opt.benchEpoch;
        memset(job.msg, 0xab, 32);
        for (int i = 0; i < 4; i++) job.target[i] = 0;  // never hits
        job.valid = true;
    }

    auto tEpoch = tStart;
    std::string prefetchNote;

    while (!*stop) {
        if (opt.bench && opt.benchEpochSeconds > 0) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - tEpoch).count() >=
                (double)opt.benchEpochSeconds) {
                job.epoch++;
                tEpoch = now;
            }
        }
        if (!opt.bench) {
            Job fresh;
            if (!source->fetch(&fresh)) {
                auto *st = dynamic_cast<StratumSource *>(source.get());
                if (st && st->loginRejected()) {
                    logLine(tty, "error",
                            "pool REJECTED the login: " + st->loginError());
                    logLine(tty, "error",
                            "check --wallet is a real Ergo address (starts "
                            "with 9, ~51 chars).");
                    return 1;
                }
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
            // Only warn about a wait when there is going to be one. Building
            // ahead makes this instant, and "please wait" followed immediately
            // by "ready in 0.00s" is just noise.
            if (!algo->prefetchReadyFor(job)) {
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
            snprintf(buf, sizeof(buf), "dataset ready in %.2fs%s - mining", secs,
                     algo->servedFromPrefetch() ? " (built ahead)" : "");
            logLine(tty, "ok", buf);
            preparedEpoch = job.epoch;
            stats.epochs++;

            // Say once what build-ahead decided, and why if it declined.
            const std::string note = algo->prefetchNote();
            if (note != prefetchNote) {
                prefetchNote = note;
                if (!note.empty()) logLine(tty, "info", note);
            }
        }

        sols.clear();
        if (!algo->search(job, nonce, opt.batch, &sols)) return 1;

        for (const auto &s : sols) {
            char buf[160];
            if (!algo->verify(job, s)) {
                hostRejected++;
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
                uint64_t a = 0, r = 0, p = 0;
                std::string le;
                // "submitted" is all that is known at this point for a pool -
                // the verdict comes back on the reader thread. Only solo, which
                // cannot report counters, can call it accepted here.
                if (source->poolCounters(&a, &r, &p, &le)) {
                    snprintf(buf, sizeof(buf), "share submitted   nonce=%016llx",
                             (unsigned long long)s.nonce);
                    logLine(tty, "info", buf);
                } else {
                    stats.accepted++;
                    snprintf(buf, sizeof(buf), "SOLUTION ACCEPTED  nonce=%016llx",
                             (unsigned long long)s.nonce);
                    logLine(tty, "ok", buf);
                }
            } else {
                hostRejected++;
                snprintf(buf, sizeof(buf), "solution rejected  nonce=%016llx  %s",
                         (unsigned long long)s.nonce, err.c_str());
                logLine(tty, "error", buf);
            }
        }

        nonceCounter = (nonceCounter + opt.batch) & nonceMask;
        nonce = noncePrefix | nonceCounter;
        intervalHashes += opt.batch;
        stats.totalNonces += opt.batch;

        // Surface each pool rejection once, with the pool's own words. Without
        // this a miner that is being refused every share looks identical to
        // one that is being paid for every share.
        {
            auto *st = dynamic_cast<StratumSource *>(source.get());
            if (st) {
                const std::string v = st->takeSubmitVerdict();
                if (!v.empty())
                    logLine(tty, "error", "pool REJECTED the share: " + v);
                uint64_t a = 0, r = 0, p = 0;
                std::string le;
                st->poolCounters(&a, &r, &p, &le);
                stats.accepted = a;
                stats.rejected = hostRejected + r;
            }
        }

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
