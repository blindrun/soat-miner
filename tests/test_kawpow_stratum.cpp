// The KawPow stratum client, against the shapes live pools actually send.
//
// Every subscribe reply below was captured from a real pool by
// tools/probe-stratum.py on 2026-08-20 and is pasted here verbatim, because
// the interesting failures are not malformed JSON - they are well-formed
// replies in a shape the client did not expect.
//
// No socket, no wallet, no GPU. The protocol is the part worth testing and it
// needs none of those; the client is driven by handing it lines directly and
// reading back the exact bytes it would have put on the wire.
//
// What it cannot cover, and why: a real mining.notify and an accepted share
// need an authorized session, which needs a Ravencoin payout address this
// project does not have. The notify shapes here are built from kawpowminer's
// parser rather than captured, and that is recorded as an open item rather
// than quietly presented as verified.

#include <cstdio>
#include <cstring>
#include <string>

#include "algos/kawpow/job.h"
#include "algos/kawpow/kawpow.h"
#include "algos/progpow/ethash.h"
#include "core/stratum_kawpow.h"

using om::KawpowStratumSource;

namespace {

int g_fail = 0;

void check(bool ok, const std::string &what) {
    if (ok) {
        printf("  ok    %s\n", what.c_str());
    } else {
        printf("  FAIL  %s\n", what.c_str());
        g_fail++;
    }
}

void checkEq(const std::string &got, const std::string &want,
             const std::string &what) {
    if (got == want) {
        printf("  ok    %s\n", what.c_str());
    } else {
        printf("  FAIL  %s\n         want %s\n         got  %s\n", what.c_str(),
               want.c_str(), got.c_str());
        g_fail++;
    }
}

std::string seedHexFor(int height) {
    const int epoch =
        om::progpow::epochNumber(height, om::kawpow::kEpochLength);
    uint32_t seed[8];
    om::progpow::epochSeed(epoch, seed);
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 8; i++)
        for (int b = 0; b < 4; b++) {
            const uint8_t v = (uint8_t)(seed[i] >> (b * 8));
            s += d[v >> 4];
            s += d[v & 15];
        }
    return s;
}

std::string notifyFor(int height, const std::string &seedHex,
                      const std::string &target) {
    return "{\"id\":null,\"method\":\"mining.notify\",\"params\":["
           "\"job42\","
           "\"8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7\","
           "\"" + seedHex + "\","
           "\"" + target + "\","
           "true," + std::to_string(height) + ",\"1b06292b\"]}";
}

KawpowStratumSource *make(KawpowStratumSource::Mode m = KawpowStratumSource::kAuto) {
    return new KawpowStratumSource("pool.invalid", 1140, "RWallet", "rig", "x", m);
}

}  // namespace

int main() {
    // --- flavour detection, from real captured replies -------------------
    {
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        check(c->mode() == KawpowStratumSource::kStratum,
              "herominers:1140 [null,\"36c3\"] reads as the stratum flavour");
        check(c->noncePrefix() == 0x36c3000000000000ULL,
              "extranonce 36c3 left-justifies into the nonce");
        check(c->nonceBitsOwned() == 48,
              "a 4-character extranonce leaves 48 bits, not 32");
        delete c;
    }
    {
        auto *c = make();
        c->feedLineForTest(
            "{\"id\":1,\"error\":null,\"result\":[[\"mining.notify\","
            "\"85420ca4-510b-4ed5-a0fa-2445f47dc1b4\",\"EthereumStratum/1.0.0\"],"
            "\"adb4\"]}");
        check(c->mode() == KawpowStratumSource::kNiceHash,
              "herominers:1150 announced form reads as the NiceHash flavour");
        check(c->noncePrefix() == 0xadb4000000000000ULL,
              "extranonce is taken from the announced form too");
        delete c;
    }
    {
        // Same pool, different port, different protocol. This pair is the
        // whole reason the flavour is detected rather than configured.
        auto *a = make(), *b = make();
        a->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        b->feedLineForTest(
            "{\"id\":1,\"error\":null,\"result\":[[\"mining.notify\",\"x\","
            "\"EthereumStratum/1.0.0\"],\"adb4\"]}");
        check(a->mode() != b->mode(),
              "one pool's two ports are not assumed to speak the same protocol");
        delete a;
        delete b;
    }
    {
        auto *c = make();
        c->feedLineForTest(
            "{\"id\":1,\"result\":[null,\"27ff92db\",4],\"error\":null}");
        check(c->nonceBitsOwned() == 32,
              "suprnova's 8-character extranonce leaves 32 bits");
        delete c;
    }
    {
        // 2miners. Neither documented shape; the client guesses and must say
        // so, because a wrong guess here rejects every share for a reason
        // nothing else in the miner reports.
        auto *c = make();
        c->feedLineForTest(
            "{\"id\":1,\"result\":[\"06fdf419\",\"50\"],\"error\":null}");
        const std::string w = c->takeJobWarning();
        check(w.find("does not recognise") != std::string::npos,
              "an unrecognised subscribe reply warns instead of failing silently");
        delete c;
    }
    {
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"abc\"]}");
        const std::string w = c->takeJobWarning();
        check(!w.empty() && c->nonceBitsOwned() == 64,
              "an odd-length extranonce is refused, not silently rounded");
        delete c;
    }

    // --- mining.notify ----------------------------------------------------
    const int height = 4501000;
    const std::string seed = seedHexFor(height);
    const std::string target =
        "000000000006292b000000000000000000000000000000000000000000000000";
    {
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        c->feedLineForTest(notifyFor(height, seed, target));
        om::Job j;
        check(c->currentJobForTest(&j), "a well-formed notify produces a job");
        checkEq(std::to_string(j.epoch), "600",
                "Job::epoch carries the Ethash epoch, not the height");
        checkEq(std::to_string(om::kawpow::jobHeight(j.extra, j.epoch)),
                std::to_string(height),
                "the height survives in Job::extra, so the 3-block program is right");
        check(j.target[3] == 0x000000000006292bULL && j.target[0] == 0,
              "the share target lands most-significant-limb first");
        checkEq(j.jobId, "job42", "the job id travels with the job");
        delete c;
    }
    {
        // The height is what selects the program. Without it a miner hashes
        // beautifully and is rejected forever, so refuse instead.
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        c->feedLineForTest(
            "{\"id\":null,\"method\":\"mining.notify\",\"params\":[\"j\","
            "\"8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7\","
            "\"" + seed + "\",\"" + target + "\",true]}");
        om::Job j;
        check(!c->currentJobForTest(&j),
              "a notify with no block height is refused, not mined");
        check(c->takeJobWarning().find("height") != std::string::npos,
              "and the refusal says the height is why");
        delete c;
    }
    {
        // A seed hash that disagrees with ours means a different epoch length,
        // which is exactly the KAWPOW_EPOCH_LENGTH mutant, arriving over the
        // wire instead of at compile time.
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        c->feedLineForTest(notifyFor(
            height,
            "1111111111111111111111111111111111111111111111111111111111111111",
            target));
        om::Job j;
        check(!c->currentJobForTest(&j),
              "a seed hash we cannot reproduce stops mining rather than "
              "building the wrong DAG");
        delete c;
    }

    // --- mining.submit ----------------------------------------------------
    {
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        c->feedLineForTest(notifyFor(height, seed, target));
        om::Job j;
        c->currentJobForTest(&j);

        om::Solution s;
        s.nonce = 0x329dd53ca526c2cdULL;
        s.extra =
            "7b1e003a71d7bf9f81d95c0c68126f72876de28b69b175c84e4c11234ad5d7c7";
        std::string err;
        check(c->submit(j, s, &err), "a solution with a mix hash submits");
        checkEq(
            c->lastSentForTest(),
            "{\"id\":10,\"method\":\"mining.submit\",\"params\":[\"RWallet.rig\","
            "\"job42\",\"0x329dd53ca526c2cd\","
            "\"0x8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7\","
            "\"0x7b1e003a71d7bf9f81d95c0c68126f72876de28b69b175c84e4c11234ad5d7c7\"]}",
            "the stratum-flavour submit carries nonce, header and mix");
        delete c;
    }
    {
        // A share without a mix hash is not a KawPow share. Sending one would
        // be rejected by every pool; refusing locally names the reason.
        auto *c = make();
        c->feedLineForTest("{\"id\":1,\"error\":null,\"result\":[null,\"36c3\"]}");
        c->feedLineForTest(notifyFor(height, seed, target));
        om::Job j;
        c->currentJobForTest(&j);
        om::Solution s;
        s.nonce = 1;
        std::string err;
        check(!c->submit(j, s, &err) && err.find("mix hash") != std::string::npos,
              "a solution with no mix hash is refused before it reaches a pool");
        delete c;
    }
    {
        auto *c = make();
        c->feedLineForTest(
            "{\"id\":1,\"error\":null,\"result\":[[\"mining.notify\",\"x\","
            "\"EthereumStratum/1.0.0\"],\"adb4\"]}");
        c->feedLineForTest(notifyFor(height, seed, target));
        om::Job j;
        c->currentJobForTest(&j);
        om::Solution s;
        s.nonce = 0xadb4d53ca526c2cdULL;
        s.extra =
            "7b1e003a71d7bf9f81d95c0c68126f72876de28b69b175c84e4c11234ad5d7c7";
        std::string err;
        c->submit(j, s, &err);
        // Four characters of extranonce cut, not four bytes. Cutting eight
        // would send a nonce the pool cannot reassemble.
        checkEq(c->lastSentForTest(),
                "{\"id\":10,\"method\":\"mining.submit\",\"params\":[\"RWallet.rig\","
                "\"job42\",\"d53ca526c2cd\"]}",
                "the NiceHash-flavour submit cuts hex characters, not bytes");
        delete c;
    }

    // --- share verdicts ---------------------------------------------------
    {
        auto *c = make();
        c->feedLineForTest("{\"id\":10,\"result\":true,\"error\":null}");
        c->feedLineForTest(
            "{\"id\":11,\"result\":false,\"error\":[21,\"Job not found\",null]}");
        uint64_t a = 0, r = 0, p = 0;
        std::string e;
        c->poolCounters(&a, &r, &p, &e);
        check(a == 1 && r == 1,
              "accepted and rejected are counted from what the pool said");
        delete c;
    }

    if (g_fail == 0) printf("KawPow stratum: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
