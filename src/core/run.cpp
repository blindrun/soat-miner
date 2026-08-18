// The shared run loop. Backend-agnostic: it owns the nonce counter, job
// lifecycle, stop signal and reporting, and only ever asks the Algorithm to
// "test these nonces".

#include "run.h"
#include "pearl_gateway.h"
#include "pearl_pool.h"
#include "stratum.h"
#include <thread>

namespace om {

// A validated-conservative memory bump (MHz over stock) for --mem-oc, keyed by
// GPU generation since a generation shares one memory spec. Only generations
// proven on real hardware get a value; anything else returns 0 and the card is
// left alone. RDNA3 measured stable at +100 on a 7900 XT (1350 MHz,
// correctness-verified) and crashes at +150, so ship +75. RDNA2 is the same
// GDDR6 class but untested here, so a smaller +50.
static int conservativeAmdMemBumpMhz(const char *gpuName) {
    std::string n;
    for (const char *p = gpuName ? gpuName : ""; *p; ++p)
        n += (*p >= 'A' && *p <= 'Z') ? (char)(*p + 32) : *p;
    auto has = [&](const char *s) { return n.find(s) != std::string::npos; };
    if (has("navi 3") || has("navi3") || has("rx 79") || has("rx 78") ||
        has("rx 77") || has("rx 76") || has("7900") || has("7800") ||
        has("7700") || has("7600"))
        return 75;  // RDNA3
    if (has("navi 2") || has("navi2") || has("rx 69") || has("rx 68") ||
        has("rx 67") || has("rx 66") || has("6900") || has("6800") ||
        has("6700") || has("6600"))
        return 50;  // RDNA2
    return 0;
}

// A conservative memory-clock offset (MHz of transfer rate, applied via NVML)
// for --mem-oc on NVIDIA, keyed by MEMORY TYPE. GDDR6X cards force performance
// state P2 under load, which runs the memory below its rated speed; +500
// restores rated on a 4090 (validated, ~+2.6%) and the P2 downclock is the same
// across GDDR6X, so this is a restore-to-spec, not a real overclock. GDDR6 and
// GDDR7 have no such downclock, so they get nothing - a +offset there would be a
// genuine overclock. NVML clock control is Linux + root only (fails on Windows).
static int conservativeNvidiaMemOffsetMhz(const char *gpuName) {
    std::string n;
    for (const char *p = gpuName ? gpuName : ""; *p; ++p)
        n += (*p >= 'A' && *p <= 'Z') ? (char)(*p + 32) : *p;
    auto has = [&](const char *s) { return n.find(s) != std::string::npos; };
    // GDDR6X: Ada 4090/4080/4070-family, Ampere 3090/3080 and the 3070 Ti.
    // NOT 4060/4060 Ti, NOT the bare 3070/3060 (GDDR6), NOT 50-series (GDDR7).
    if (has("4090") || has("4080") || has("4070") || has("3090") ||
        has("3080") || has("3070 ti"))
        return 500;
    return 0;
}

// Opt-in conservative memory overclock (--mem-oc), keyed by memory type. AMD
// GDDR6 raises the clock over stock via overdrive sysfs; NVIDIA GDDR6X restores
// its P2 downclock via NVML. Both are Linux-only.
static void applyMemOc(GpuMonitor &gpu, const char *gpuName) {
    if (gpu.isAmd()) {
        const int bump = conservativeAmdMemBumpMhz(gpuName);
        if (bump == 0) {
            fprintf(stderr,
                    "--mem-oc: no conservative profile for this GPU, memory clock "
                    "left alone.\n");
            return;
        }
        int stock = 0;
        const int got = gpu.amdApplyMemBump(bump, &stock);
        if (stock <= 0) {
            fprintf(stderr,
                    "--mem-oc: AMD overdrive unavailable. Add "
                    "amdgpu.ppfeaturemask=0xffffffff to the kernel command line, "
                    "reboot, and run as root.\n");
            return;
        }
        if (got <= 0)
            fprintf(stderr, "--mem-oc: could not set the memory clock (need root).\n");
        else
            fprintf(stderr, "mem-oc: memory %d -> %d MHz (+%d, conservative)\n",
                    stock, got, got - stock);
    } else if (gpu.isNvidia()) {
        const int off = conservativeNvidiaMemOffsetMhz(gpuName);
        if (off == 0) {
            fprintf(stderr,
                    "--mem-oc: no conservative profile for this NVIDIA GPU (only "
                    "GDDR6X cards have a P2 downclock to restore). Use --mclk-offset "
                    "N by hand if you know your card.\n");
            return;
        }
        const int got = gpu.applyMemOffsetMhz(off);
        if (got < 0)
            fprintf(stderr,
                    "--mem-oc: could not set the memory clock offset - NVML clock "
                    "control needs Linux and root, and is unavailable on Windows.\n");
        else
            fprintf(stderr,
                    "mem-oc: memory clock offset +%d (GDDR6X, restore P2 to rated)\n",
                    off);
    }
}

int runMiner(Algorithm *algo, const RunOptions &opt, const char *gpuName,
             double gpuMemGB, const char *archLabel, volatile sig_atomic_t *stop) {
    GpuMonitor gpu;
    gpu.open();

    // Undo the P2 memory downclock before anything is measured. Autolykos is
    // memory bound, so on a 4090 this is worth about 2.3%.
    if (opt.memOffsetMhz != 0) {
        const int got = gpu.applyMemOffsetMhz(opt.memOffsetMhz);
        if (got < 0)
            fprintf(stderr,
                    "could not set the memory clock offset. Needs an NVIDIA "
                    "card, a recent driver, and root on Linux.\n");
    }

    // Opt-in conservative memory OC (reset on exit via gpu.close()).
    if (opt.memOc) applyMemOc(gpu, gpuName);

    const bool tty = stdoutIsTty() && !opt.plain;

    MinerStats stats;
    stats.gpuName = gpuName ? gpuName : "unknown GPU";
    stats.algo = algo->name();
    stats.arch = archLabel ? archLabel : "";
    stats.backend = opt.backendLabel;
    stats.gpuMemGB = gpuMemGB;

    const bool isPearlBanner = algo && algo->name() && !strcmp(algo->name(), "pearl-pow");
    // Banner before any network or GPU work. Connecting to a pool and
    // building the dataset take ~15s combined, and printing nothing for that
    // long reads as a hang - it is the first thing a new user sees.
    // --pool wins over the gateway when both look set, because pearlHost has a
    // DEFAULT and an explicitly typed pool does not. Ordering this the other
    // way made the banner announce the gateway while the miner used the pool.
    stats.source =
        opt.bench ? "BENCHMARK ONLY - not mining, nothing submitted"
        : !opt.poolHost.empty()
            ? opt.poolHost + ":" + std::to_string(opt.poolPort) +
                  (isPearlBanner ? " (pearl pool)" : " (pool)")
        : !opt.pearlHost.empty()
            ? opt.pearlHost + ":" + std::to_string(opt.pearlPort) +
                  " (pearl-gateway)"
            : "ergo node " + opt.target.host + ":" +
                  std::to_string(opt.target.port) + " (solo)";
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

    // Which transport a Pearl job comes over depends on whether the user has
    // a node. Both end in the same PlainProof.
    const bool isPearl = algo && algo->name() && !strcmp(algo->name(), "pearl-pow");

    std::unique_ptr<JobSource> source;
    uint64_t noncePrefix = 0;
    int nonceBits = 64;
    if (opt.bench) {
        stats.source = "benchmark (no pool/node)";
    } else if (isPearl && !opt.poolHost.empty()) {
        // Pearl to a pool. The pool runs the node and the prover, so this
        // needs nothing but a wallet - which is the whole point, since the
        // gateway path below requires running a Pearl node yourself.
        if (opt.wallet.empty()) {
            logLine(tty, "error",
                    "mining Pearl to a pool needs --wallet with your PRL "
                    "address (it starts with prl1).");
            return 1;
        }
        auto *pp = new PearlPoolSource(opt.poolHost, opt.poolPort, opt.wallet,
                                       opt.worker);
        source.reset(pp);
        stats.source = source->describe();
    } else if (!opt.pearlHost.empty()) {
        // Pearl straight to your own node, through pearl-gateway.
        auto *pg = new PearlGatewaySource(opt.pearlHost, opt.pearlPort);
        Job probe;
        if (!pg->fetch(&probe)) {
            logLine(tty, "error",
                    "no pearl-gateway answering on " + opt.pearlHost + ":" +
                        std::to_string(opt.pearlPort) + ".");
            logLine(tty, "info",
                    "Pearl mining goes through pearl-gateway, not straight to a "
                    "node: a block carries a proof only the gateway can "
                    "generate. Start one against your node first.");
            delete pg;
            return 1;
        }
        source.reset(pg);
        stats.source = source->describe();
        logLine(tty, "ok", "gateway answered with work");
    } else if (!opt.poolHost.empty()) {
        // Catch the obvious cases before spending a connection on them. Skipped
        // for Lithos, where the address is a label rather than a payout
        // identifier - see RunOptions::lithos.
        const std::string &w = opt.wallet;
        if (opt.lithos) {
            logLine(tty, "info",
                    "Lithos mode: rewards are settled on-chain by the Lithos "
                    "client from your node, not by this address.");
        } else {
            const bool looksLikeErgo =
                w.size() >= 40 && w.size() <= 60 && w[0] == '9' &&
                w.find_first_of(" \t") == std::string::npos;
            if (w.empty() || w.rfind("9YOUR", 0) == 0 || !looksLikeErgo) {
                logLine(tty, "error",
                        w.empty() ? "no --wallet given; pool mining needs your "
                                    "Ergo payout address"
                                  : "'" + w + "' is not a valid Ergo address");
                logLine(tty, "error",
                        "edit WALLET in the mine_ergo_* script (or config.txt) "
                        "to a real address - they start with 9 and are about 51 "
                        "chars");
                logLine(tty, "error",
                        "mining to a Lithos client instead? pass --lithos");
                return 1;
            }
        }
        // Name the payout address on every startup. On a conventional pool this
        // is who gets paid, and shipping it unseen is how someone ends up mining
        // to a default they never noticed (github issue #1). Lithos is exempt:
        // there the address is a label, not a payout identifier.
        if (!opt.lithos) {
            logLine(tty, "info", "payout address: " + opt.wallet);
        }
        logLine(tty, "info", "connecting to " + opt.poolHost + ":" +
                                 std::to_string(opt.poolPort) + " ...");
        // Lithos only logs the worker name, so an empty address is fine there -
        // but send something rather than a bare ".worker".
        const std::string walletArg =
            (opt.lithos && opt.wallet.empty()) ? std::string("lithos")
                                               : opt.wallet;
        auto *st = new StratumSource(opt.poolHost, opt.poolPort, walletArg,
                                     opt.worker, opt.password);
        std::string err;
        if (!st->start(&err)) {
            if (opt.lithos) {
                // The overwhelmingly common Lithos failure: no local Lithos
                // client is running yet. Lithos has no central pool - you run
                // your own node + client and payouts go to YOUR wallet - so a
                // refused connection here almost always means "not set up yet",
                // not a network problem. Say exactly what to do.
                logLine(tty, "error",
                        "no Lithos client is answering on " + opt.poolHost +
                        ":" + std::to_string(opt.poolPort) + ".");
                logLine(tty, "info",
                        "Lithos has no central pool. You run a local Ergo node "
                        "+ Lithos client, and your mining payouts go to your "
                        "own wallet on that node.");
                logLine(tty, "info",
                        "One command sets all of it up (node, client, wallet) - "
                        "the script ships next to this miner:");
                logLine(tty, "info",
                        "    sudo ./lithos-quickstart.sh --network testnet");
                logLine(tty, "info",
                        "Then wait for it to sync (watch: ./lithos-status), "
                        "start the client, and re-run this miner. Payouts land "
                        "in the wallet the setup created.");
            } else {
                fprintf(stderr, "pool connect failed: %s\n", err.c_str());
            }
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
                    opt.lithos
                        ? "the Lithos client refused the login - check its log; "
                          "it rejects miners before its node has finished "
                          "syncing."
                        : "check --wallet is a real Ergo address (starts with "
                          "9, ~51 chars). The placeholder in the example "
                          "scripts will not work.");
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
    uint64_t nonceMask =
        (nonceBits >= 64) ? ~0ULL : ((1ULL << nonceBits) - 1ULL);
    uint64_t nonceCounter = ((uint64_t)time(nullptr) << 16) & nonceMask;
    uint64_t nonce = noncePrefix | nonceCounter;

    // Non-null only in pool mode; used to pick up a mid-session extranonce
    // reassignment, which Lithos issues when two miners collide on a prefix.
    auto *stratumSrc = dynamic_cast<StratumSource *>(source.get());
    uint64_t extranonceGen =
        stratumSrc ? stratumSrc->extranonceGeneration() : 0;
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
                            opt.lithos
                                ? "the Lithos client refused the login - check "
                                  "its log."
                                : "check --wallet is a real Ergo address "
                                  "(starts with 9, ~51 chars).");
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
            // Not every algorithm has a dataset. Autolykos rebuilds 7.27 GB
            // and the wait is the first thing a user notices; Pearl allocates
            // a couple of hundred megabytes in a few milliseconds and has no
            // epoch at all, so telling someone to please wait for a 0.01 GB
            // dataset is just wrong.
            const double gb = algo->memoryBytes(job) / 1e9;
            if (!algo->prefetchReadyFor(job) && gb >= 1.0) {
                char pre[160];
                snprintf(pre, sizeof(pre),
                         "epoch %llu - building %.2f GB dataset, please wait...",
                         (unsigned long long)job.epoch, gb);
                logLine(tty, "info", pre);
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (!algo->prepare(job)) return 1;
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            stats.datasetGB = algo->memoryBytes(job) / 1e9;
            char buf[160];
            if (stats.datasetGB >= 1.0)
                snprintf(buf, sizeof(buf), "dataset ready in %.2fs%s - mining",
                         secs, algo->servedFromPrefetch() ? " (built ahead)" : "");
            else
                snprintf(buf, sizeof(buf), "ready in %.2fs (%.0f MB) - mining",
                         secs, stats.datasetGB * 1e3);
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
        // Never let a launch cross the owned-subspace boundary. The kernel adds
        // the thread index to `nonce` by plain addition, so a batch straddling
        // the boundary carries into the pool's extranonce prefix and mines
        // nonces we do not own (which the pool then rejects). Shrink the launch
        // that would cross so it lands exactly on the boundary; the next one
        // starts fresh at counter 0. `remaining` wraps correctly when the whole
        // 64-bit nonce is ours (nonceMask all ones), which is the case with no
        // pool prefix - so this is a no-op for solo and for Pearl.
        //
        // NB: `batch` is a nonce count on the pool/node path, but some algorithms
        // (Pearl) treat it as a candidate budget that is not 1:1 with nonces.
        // This clamp only ever SHRINKS a launch, which is safe either way - do
        // not "fix" it into an assumption that exactly `batch` nonces are
        // consumed, nor into expanding a launch to fill a range.
        const uint64_t remaining = nonceMask - nonceCounter + 1;
        const uint64_t batchThisLaunch =
            (opt.batch < remaining) ? opt.batch : remaining;
        if (!algo->search(job, nonce, batchThisLaunch, &sols)) return 1;

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

        // Adopt a new extranonce the moment the pool issues one. Keeping the
        // old prefix would mean mining in a subspace the pool has just handed
        // to somebody else.
        if (stratumSrc) {
            const uint64_t gen = stratumSrc->extranonceGeneration();
            if (gen != extranonceGen) {
                extranonceGen = gen;
                noncePrefix = stratumSrc->noncePrefix();
                nonceBits = stratumSrc->nonceBitsOwned();
                nonceMask =
                    (nonceBits >= 64) ? ~0ULL : ((1ULL << nonceBits) - 1ULL);
                nonceCounter &= nonceMask;
                char xb[96];
                snprintf(xb, sizeof(xb),
                         "pool reassigned the extranonce: prefix=%016llx, %d "
                         "nonce bits ours",
                         (unsigned long long)noncePrefix, nonceBits);
                logLine(tty, "info", xb);
            }
        }

        nonceCounter = (nonceCounter + batchThisLaunch) & nonceMask;
        nonce = noncePrefix | nonceCounter;
        intervalHashes += batchThisLaunch;
        stats.totalNonces += batchThisLaunch;

        // Surface each pool rejection once, with the pool's own words. Without
        // this a miner that is being refused every share looks identical to
        // one that is being paid for every share.
        {
            auto *st = dynamic_cast<StratumSource *>(source.get());
            if (st) {
                const std::string v = st->takeSubmitVerdict();
                if (!v.empty())
                    logLine(tty, "error", "pool REJECTED the share: " + v);
                const std::string jw = st->takeJobWarning();
                if (!jw.empty()) logLine(tty, "warn", jw);
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
    // Put the card back the way it was found.
    if (opt.memOffsetMhz != 0) gpu.applyMemOffsetMhz(0);
    algo->release();
    gpu.close();
    return 0;
}

}  // namespace om
