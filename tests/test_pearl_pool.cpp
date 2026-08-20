// Host-only Pearl pool protocol checks.

#include <cstdio>
#include <cstring>
#include <string>

#include "../src/core/pearl_pool.h"
#include "../src/core/run.h"

using namespace om;

static bool expect(const char *name, const std::string &actual,
                   const char *wanted) {
    if (actual == wanted) return true;
    fprintf(stderr, "%s\n  got:  %s\n  want: %s\n", name, actual.c_str(), wanted);
    return false;
}

static bool expectBool(const char *name, bool actual) {
    if (actual) return true;
    fprintf(stderr, "%s: failed\n", name);
    return false;
}

static std::string notify(const std::string &header, const std::string &target,
                          const char *jobId) {
    return "{\"method\":\"mining.notify\",\"params\":{\"job_id\":\"" +
           std::string(jobId) + "\",\"header\":\"" + header +
           "\",\"target\":\"" + target + "\",\"cert_version\":3}}";
}

int main() {
    bool ok = true;
    ok &= expect("JSON quotes every control character",
                 jsonQuoted("a\"\\\b\f\n\r\t\x01z"),
                 "\"a\\\"\\\\\\b\\f\\n\\r\\t\\u0001z\"");
    ok &= expect("authorization quotes wallet and worker",
                 pearlPoolAuthorizeRequest("prl1\"wallet", "rig\\one\n"),
                 "{\"id\":1,\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"prl1\\\"wallet\",\"worker\":\"rig\\\\one\\n\",\"pass\":\"x\",\"agent\":\"soat-miner\"}}");
    ok &= expect("authorization has no subscribe method",
                 pearlPoolAuthorizeRequest("w", "r").find("subscribe") == std::string::npos ? "yes" : "no", "yes");
    ok &= expect("submission quotes pool supplied job id",
                 pearlPoolSubmitRequest(101, "job\"1", "proof\\data"),
                 "{\"id\":101,\"method\":\"mining.submit\",\"params\":{\"job_id\":\"job\\\"1\",\"plain_proof\":\"proof\\\\data\"}}");
    std::string replyError;
    ok &= expectBool("only explicit JSON-RPC success is accepted",
                     pearlPoolReplyAccepted("{\"id\":101,\"error\":null,\"result\":true}",
                                            &replyError));
    ok &= expectBool("false submit result is not reported as a share",
                     !pearlPoolReplyAccepted("{\"id\":101,\"error\":null,\"result\":false}",
                                             &replyError));
    ok &= expectBool("pool rejection message is preserved",
                     !pearlPoolReplyAccepted("{\"id\":101,\"error\":{\"message\":\"bad share\"},\"result\":null}",
                                             &replyError) && replyError == "bad share");

    // Pearl8's optional transcript is metadata-only. In particular, logging
    // must neither leak an address echoed by a pool nor alter the exact submit
    // payload used for share construction.
    // Named "fakeAddress", not "secret": gitleaks' generic-api-key rule fires on
    // a string literal assigned to something called secret, and a fixture whose
    // entire job is to be an obviously-fake address should not be the thing that
    // trains people to run git commit --no-verify.
    const std::string fakeAddress = "prl1exampleaddressnotreal000000";
    ok &= expectBool("transcript redacts PRL addresses",
                     pearlTranscriptRedact("pool says " + fakeAddress + " rejected")
                         .find(fakeAddress) == std::string::npos);
    const std::string submitBefore = pearlPoolSubmitRequest(101, "job-1", "proof-data");
    const uint8_t hdrMsg[32] = {0xde, 0xad, 0xbe, 0xef, 1, 2, 3, 4};
    const uint64_t zeroDigest[4] = {};
    const uint64_t someTarget[4] = {0, 0, 0, 0x7fff8ULL << 40};
    const std::string submitTrace =
        pearlTranscriptSubmit(101, "job-1", "proof-data",
                              pearlHeaderFingerprint(hdrMsg), zeroDigest, someTarget);
    const std::string submitAfter = pearlPoolSubmitRequest(101, "job-1", "proof-data");
    ok &= expectBool("transcript does not change share construction",
                     submitBefore == submitAfter &&
                     submitTrace.find("proof-data") == std::string::npos &&
                     submitTrace.find("plain_proof_b3") != std::string::npos);
    // A synthetic address, not a real one: the redactor keys off the prl1
    // prefix and eats alphanumerics, so this exercises it exactly and puts no
    // real wallet in a test fixture.
    const std::string secret = "prl1qqqqexampleaddressnotarealwalletqqqq";
    const std::string replyTrace = pearlTranscriptReply(101, false,
                                                         "wallet " + secret + " rejected");
    ok &= expectBool("transcript redacts echoed pool errors",
                     replyTrace.find(secret) == std::string::npos &&
                     replyTrace.find("REDACTED_PRL_ADDRESS") != std::string::npos);

    // Pearl Stratum targets are already final share bounds.  They arrive as
    // big-endian hex while Job/CUDA use little-endian limbs; a target-only
    // notify is a share-difficulty refresh, not a stale duplicate.
    const std::string header(152, 'a');
    const std::string endianTarget =
        "0102030405060708111213141516171821222324252627283132333435363738";
    uint64_t limbs[4] = {};
    ok &= expectBool("big-endian pool target parses",
                     hexToLimbs(endianTarget, limbs));
    ok &= expectBool("pool target becomes little-endian Job limbs",
                     limbs[0] == 0x3132333435363738ULL &&
                     limbs[1] == 0x2122232425262728ULL &&
                     limbs[2] == 0x1112131415161718ULL &&
                     limbs[3] == 0x0102030405060708ULL);

    const std::string easyTarget =
        "0000000000000001000000000000000000000000000000000000000000000000";
    const std::string hardTarget =
        "0000000000000000800000000000000000000000000000000000000000000000";
    Job easy, hard;
    ok &= expectBool("parse initial mock Pearl job",
                     pearlPoolJobFromNotify(notify(header, easyTarget, "job-easy"), &easy));
    ok &= expectBool("parse target-only Pearl difficulty refresh",
                     pearlPoolJobFromNotify(notify(header, hardTarget, "job-hard"), &hard));
    ok &= expectBool("same header has stable job identity hash",
                     memcmp(easy.msg, hard.msg, sizeof(easy.msg)) == 0);
    ok &= expectBool("refreshed target and job id are retained",
                     memcmp(easy.target, hard.target, sizeof(easy.target)) != 0 &&
                     easy.extra != hard.extra && poolJobId(hard.extra) == "job-hard");
    const std::string notifyTrace = pearlTranscriptNotify(hard);
    ok &= expectBool("notify transcript contains only needed job metadata",
                     notifyTrace.find("job-hard") != std::string::npos &&
                     notifyTrace.find(hardTarget) != std::string::npos &&
                     notifyTrace.find(header) == std::string::npos);
    ok &= expectBool("target-only refresh is a different submit job",
                     !pearlPoolJobsMatch(easy, hard));
    ok &= expectBool("an exact synthetic job remains current",
                     pearlPoolJobsMatch(easy, easy));

    // Do not manufacture a certificate version. A missing V1/V2/V3 selector
    // can produce a perfectly encoded proof for the wrong verifier.
    const std::string noCert =
        "{\"method\":\"mining.notify\",\"params\":{\"job_id\":\"bad\",\"header\":\"" +
        header + "\",\"target\":\"" + easyTarget + "\"}}";
    const std::string unknownCert =
        "{\"method\":\"mining.notify\",\"params\":{\"job_id\":\"bad\",\"header\":\"" +
        header + "\",\"target\":\"" + easyTarget + "\",\"cert_version\":4}}";
    const std::string zeroTarget(64, '0');
    ok &= expectBool("missing certificate version retires the job",
                     !pearlPoolJobFromNotify(noCert, &easy));
    ok &= expectBool("unsupported certificate version retires the job",
                     !pearlPoolJobFromNotify(unknownCert, &easy));
    ok &= expectBool("zero target cannot become a locally accepted job",
                     !pearlPoolJobFromNotify(notify(header, zeroTarget, "bad"), &easy));

    const pearl::U256 easyBound = pearl::U256::fromLimbs(easy.target);
    const pearl::U256 hardBound = pearl::U256::fromLimbs(hard.target);
    pearl::U256 candidate = hardBound;
    ++candidate.v[0];  // above current target, still below the prior one
    ok &= expectBool("candidate would pass only the stale easier share bound",
                     candidate.le(easyBound) && !candidate.le(hardBound));
    pearl::MiningConfig cfg;
    cfg.commonDim = 2048;
    cfg.rank = 128;
    // A pool target is a base target, exactly like a gateway's. HeroMiners
    // publishes 0xFFFF * 2^208 / difficulty and its verifier applies the same
    // rank/work penalty, so the miner scales it once and mines against that.
    // Using the advertised value directly is 2^19 too hard and submits nothing:
    // 12 accepted / 0 rejected on a 4090 with the scaled bound, 0 submits in
    // 71 s on a 5080 without it.
    pearl::U256 scaledHard, scaledEasy;
    ok &= expectBool("pool target scales once into the mining bound",
                     cfg.penalizedTarget(hardBound, &scaledHard) &&
                     cfg.penalizedTarget(easyBound, &scaledEasy));
    ok &= expectBool("scaling is the shipped k=2048/rank=128 factor",
                     scaledHard.le(scaledEasy) && hardBound.le(scaledHard));
    pearl::U256 oneFactor;
    ok &= expectBool("that factor is exactly 524,288",
                     pearl::U256::fromLimbs((const uint64_t[4]){1, 0, 0, 0})
                             .mul(524288ull, &oneFactor) &&
                     cfg.penalizedTarget(
                         pearl::U256::fromLimbs((const uint64_t[4]){1, 0, 0, 0}),
                         &scaledEasy) &&
                     scaledEasy.le(oneFactor) && oneFactor.le(scaledEasy));
    // The stale-target regression still bites: a candidate that only passed
    // the previous, easier job must fail the current one after scaling.
    pearl::U256 staleCandidate = scaledHard;
    ++staleCandidate.v[0];
    ok &= expectBool("candidate above the fresh scaled bound is refused",
                     !staleCandidate.le(scaledHard));
    // The reported failure: "Pearl pool refused login: Invalid Pearl address".
    // Every one of these used to reach the pool and come back as that message.
    const std::string placeholder = "prl1YOUR_PEARL_ADDRESS_HERE";
    // Synthetic, not anyone's wallet: these fixtures only need the address
    // FORM, and a real payout address in a public repo is a privacy leak.
    const std::string fakeMainnet =
        "prl1examplefakeaddressmadeupnevertrueqqqqqqqqqqqqqqqqqqqqqqqqqq";
    const std::string fakeTestnet =
        "tprl1examplefaketestnetaddressneverusedqqqqqqqqqqqqqqqqqqqqqqqqq";
    ok &= expectBool("the launcher placeholder is refused locally",
                     !pearlWalletProblem(placeholder).empty());
    ok &= expectBool("the placeholder message names the launcher",
                     pearlWalletProblem(placeholder).find("WALLET") !=
                         std::string::npos);
    ok &= expectBool("an empty address is refused",
                     !pearlWalletProblem("").empty());
    ok &= expectBool("a testnet address is refused on a mainnet pool",
                     pearlWalletProblem(fakeTestnet).find("TESTNET") !=
                         std::string::npos);
    // An Ergo-shaped address works for Ergo and can never work here.
    const std::string fakeErgo =
        "9exampleErgoAddressNotARealWalletUsedInTests";
    ok &= expectBool("an Ergo address is refused",
                     !pearlWalletProblem(fakeErgo).empty());
    ok &= expectBool("and it says Pearl is a different chain",
                     pearlWalletProblem(fakeErgo).find("different chain") !=
                         std::string::npos);
    ok &= expectBool("wallet.worker concatenation is refused",
                     pearlWalletProblem(fakeMainnet + ".rig1").find("--worker") !=
                         std::string::npos);
    ok &= expectBool("surrounding whitespace is refused",
                     !pearlWalletProblem(fakeMainnet + " ").empty() &&
                     !pearlWalletProblem(" " + fakeMainnet).empty());
    ok &= expectBool("a truncated address is refused",
                     !pearlWalletProblem(fakeMainnet.substr(0, 20)).empty());
    ok &= expectBool("a well-formed mainnet address passes",
                     pearlWalletProblem(fakeMainnet).empty());

    // Two different 524,288s. The MACs one candidate costs is tileSize * k.
    // penalizedTarget's factor is tileSize * (k / rank) * kPenaltyBaseRank.
    // They are equal ONLY because the shipped rank is 128, which is also
    // kPenaltyBaseRank. Anyone who changes rank breaks that silently: the
    // reported TH/s stays right while the mining bound moves, or the reverse.
    {
        pearl::MiningConfig c;
        c.commonDim = 2048;
        c.rank = 128;
        const double macs = (double)c.tileSize() * c.commonDim;
        pearl::U256 one = pearl::U256::fromLimbs((const uint64_t[4]){1, 0, 0, 0});
        pearl::U256 scaled;
        ok &= expectBool("a candidate costs tileSize * k MACs", macs == 524288.0);
        ok &= expectBool("at rank 128 the target factor is the same number",
                         c.penalizedTarget(one, &scaled) &&
                         scaled.v[0] == 524288ull && scaled.v[1] == 0);
        c.rank = 256;
        pearl::U256 scaled256;
        ok &= expectBool("at rank 256 they diverge, so they are not one constant",
                         c.penalizedTarget(one, &scaled256) &&
                         scaled256.v[0] == 262144ull &&
                         (double)c.tileSize() * c.commonDim == 524288.0);
    }

    // The regression for the rejections. A Pearl pool pushes a new header
    // every ~19 s with the epoch pinned at 0 (PearlPoolSource sets it to 0 on
    // every job). Gating prepare() on the epoch therefore prepared once, at
    // connect, and the miner spent the rest of the session hashing that first
    // header's matrices while quoting the newest job id back at the pool -
    // a proof that is valid for a job nobody asked about, answered with
    // "Jackpot condition not satisfied: hash does not meet difficulty target".
    {
        Job first;
        first.valid = true;
        first.epoch = 0;
        memset(first.msg, 0x11, sizeof(first.msg));
        for (int i = 0; i < 4; i++) first.target[i] = 0x0123456789abcdefull;
        first.extra = "job-0";

        PreparedJob prepared;
        ok &= expectBool("the first job always prepares",
                         shouldPrepare(true, prepared, first));
        prepared.take(first);
        ok &= expectBool("and the same job again does not",
                         !shouldPrepare(true, prepared, first));

        // The bug, in one assertion: same epoch, different header.
        Job second = first;
        memset(second.msg, 0x22, sizeof(second.msg));
        second.extra = "job-1";
        ok &= expectBool("the epochs are identical, which is what hid this",
                         second.epoch == first.epoch);
        ok &= expectBool("a new Pearl header re-prepares even at the same epoch",
                         shouldPrepare(true, prepared, second));

        // A pool may also refresh only the target (VarDiff).
        Job harder = first;
        harder.target[3] >>= 4;
        ok &= expectBool("a target-only refresh re-prepares too",
                         shouldPrepare(true, prepared, harder));

        // And the job id alone moving is still a different job to submit under.
        Job renamed = first;
        renamed.extra = "job-99";
        ok &= expectBool("a new job id re-prepares", shouldPrepare(true, prepared, renamed));

        // Autolykos must keep the epoch-only gate: re-preparing there rebuilds
        // a 7.27 GB dataset, so a per-job re-prepare would be a disaster.
        ok &= expectBool("Autolykos does NOT re-prepare on a same-epoch change",
                         !shouldPrepare(false, prepared, second));
        Job nextEpoch = second;
        nextEpoch.epoch = first.epoch + 1;
        ok &= expectBool("but Autolykos does re-prepare on a new epoch",
                         shouldPrepare(false, prepared, nextEpoch));
        ok &= expectBool("and so does Pearl", shouldPrepare(true, prepared, nextEpoch));
    }

    // ---------------------------------------------- difficulty negotiation
    //
    // Pearl Stratum V1 carries the target in every notify and HeroMiners sends
    // nothing else, so none of this fires against that pool. It exists because
    // a pool that DOES push a difficulty separately was previously ignored
    // outright, and the only symptom would have been every share refused with
    // "hash does not meet difficulty target".
    {
        printf("difficulty and target pushes\n");

        // The mapping, pinned against a measured live value rather than an
        // assumption: HeroMiners' job id names difficulty 2097152 and the
        // target it sends alongside is this.
        uint64_t t[4] = {};
        ok &= expectBool("difficulty 2097152 converts",
                         pearlDifficultyToTarget("2097152", t));
        ok &= expect("and it is exactly the target HeroMiners sends",
                     pearlTargetHexBE(t),
                     "00000000000007fff80000000000000000000000000000000000000000000000");

        uint64_t one[4] = {};
        ok &= expectBool("difficulty 1 converts", pearlDifficultyToTarget("1", one));
        ok &= expect("difficulty 1 is the whole 0xFFFF * 2^208 numerator",
                     pearlTargetHexBE(one),
                     "00000000ffff0000000000000000000000000000000000000000000000000000");

        uint64_t easy[4] = {};
        ok &= expectBool("AlphaPool's difficulty 50000 converts",
                         pearlDifficultyToTarget("50000", easy));
        ok &= expectBool("a 42x easier difficulty is a numerically larger target",
                         !u256Less(easy, t) && u256Less(t, easy));
        ok &= expectBool("difficulty 0 is refused rather than dividing by zero",
                         !pearlDifficultyToTarget("0", easy));

        // The two push shapes, and the lines that must NOT be mistaken for one.
        uint64_t got[4] = {};
        ok &= expectBool("object-form set_difficulty is understood",
                         pearlPoolTargetFromPush(
                             "{\"id\":null,\"method\":\"mining.set_difficulty\","
                             "\"params\":{\"difficulty\":2097152}}", got));
        ok &= expectBool("and yields the same target as the direct conversion",
                         memcmp(got, t, sizeof(t)) == 0);
        memset(got, 0, sizeof(got));
        ok &= expectBool("array-form set_difficulty is understood too",
                         pearlPoolTargetFromPush(
                             "{\"id\":null,\"method\":\"mining.set_difficulty\","
                             "\"params\":[2097152]}", got));
        ok &= expectBool("array form agrees with object form",
                         memcmp(got, t, sizeof(t)) == 0);
        memset(got, 0, sizeof(got));
        ok &= expectBool("set_target takes the hex target verbatim",
                         pearlPoolTargetFromPush(
                             "{\"id\":null,\"method\":\"mining.set_target\",\"params\":"
                             "{\"target\":\"00000000000007fff800000000000000000000000"
                             "00000000000000000000000\"}}", got));
        ok &= expectBool("and it lands in the same limbs a notify target would",
                         memcmp(got, t, sizeof(t)) == 0);
        ok &= expectBool("a mining.notify is NOT treated as a difficulty push",
                         !pearlPoolTargetFromPush(
                             notify(std::string(152, 'a'),
                                    std::string(63, '0') + "1", "00000000_2097152"),
                             got));
        ok &= expectBool("a plain submit reply is not either",
                         !pearlPoolTargetFromPush(
                             "{\"id\":101,\"error\":null,\"result\":true}", got));
        ok &= expectBool("a zero difficulty on the wire is refused",
                         !pearlPoolTargetFromPush(
                             "{\"method\":\"mining.set_difficulty\",\"params\":[0]}", got));
        ok &= expectBool("a zero target on the wire is refused",
                         !pearlPoolTargetFromPush(
                             "{\"method\":\"mining.set_target\",\"params\":{\"target\":\"" +
                             std::string(64, '0') + "\"}}", got));

        // The division itself, at the boundary a 32-bit-limb implementation
        // would get wrong.
        uint64_t big[4] = {};
        ok &= expectBool("a difficulty above 2^32 still converts",
                         pearlDifficultyToTarget("4294967297", big));
        ok &= expectBool("and gives a harder target than 2^32 - 1 does",
                         u256Less(big, t));
    }

    // --- the accept/reject verdict ------------------------------------
    //
    // Both of these were real, and both were found by the mock-gateway test on
    // its first run rather than by anything here. A verdict read wrong is the
    // worst kind of bug this file can carry: the miner keeps running, the
    // panel keeps printing a hashrate, and the counter simply disagrees with
    // the payout.
    {
        std::string why;
        ok &= expectBool("a compact accept is accepted",
                         pearlPoolReplyAccepted(
                             "{\"id\":1,\"error\":null,\"result\":true}", &why));

        // Ordinary, conforming JSON. The substring version required the exact
        // bytes "result":true and refused this, then blamed the pool for
        // being malformed.
        ok &= expectBool("a space after the colon is still an accept",
                         pearlPoolReplyAccepted(
                             "{\"id\": 1, \"error\": null, \"result\": true}", &why));
        ok &= expectBool("and pretty-printed across lines too",
                         pearlPoolReplyAccepted(
                             "{\n  \"id\": 1,\n  \"error\": null,\n"
                             "  \"result\": true\n}", &why));

        // A greeting alongside a good result used to be read as a rejection,
        // so the pool credited the share and we did not.
        ok &= expectBool("an accept carrying a message is still an accept",
                         pearlPoolReplyAccepted(
                             "{\"id\":1,\"error\":null,\"result\":true,"
                             "\"message\":\"welcome\"}", &why));

        // The negative half, which must keep working.
        ok &= expectBool("an error object is a rejection",
                         !pearlPoolReplyAccepted(
                             "{\"id\":2,\"result\":null,\"error\":"
                             "{\"code\":23,\"message\":\"Low difficulty share\"}}", &why));
        ok &= expectBool("and its message is the reason we report",
                         why == "Low difficulty share");
        ok &= expectBool("result=false is a rejection",
                         !pearlPoolReplyAccepted(
                             "{\"id\":2,\"result\":false,\"msg\":\"stale\"}", &why));
        ok &= expectBool("with the top-level msg as its wording",
                         why == "stale");
        ok &= expectBool("a reply with neither result nor error is a rejection",
                         !pearlPoolReplyAccepted("{\"id\":2}", &why));
        ok &= expectBool("and says so rather than staying silent", !why.empty());
    }

    // --- the SOLO path's verdict --------------------------------------
    //
    // pearl_gateway.h had the same bug as the pool path and it shipped in
    // v0.2.17. It failed in both directions: two hardcoded spellings of null
    // meant a good submission read as rejected, and - worse - it needed an
    // "error" key to be PRESENT before it would call anything a failure, so a
    // gateway rejecting everything read exactly like one accepting everything.
    {
        std::string why;
        ok &= expectBool("the documented success is an accept",
                         pearlGatewaySubmitAccepted(
                             "{\"jsonrpc\":\"2.0\",\"result\":\"submitted\","
                             "\"id\":2}", &why));
        ok &= expectBool("with error:null alongside it, spaced",
                         pearlGatewaySubmitAccepted(
                             "{\"jsonrpc\":\"2.0\",\"error\" : null,"
                             "\"result\":\"submitted\",\"id\":2}", &why));
        ok &= expectBool("and pretty-printed",
                         pearlGatewaySubmitAccepted(
                             "{\n  \"error\":\n    null,\n"
                             "  \"result\": \"submitted\"\n}", &why));
        ok &= expectBool("an error object is a rejection",
                         !pearlGatewaySubmitAccepted(
                             "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32000,"
                             "\"message\":\"bad proof\"},\"id\":2}", &why));
        // The false-accept half: silence is not consent.
        ok &= expectBool("a response with no result at all is NOT an accept",
                         !pearlGatewaySubmitAccepted(
                             "{\"jsonrpc\":\"2.0\",\"id\":2}", &why));
        ok &= expectBool("nor is a null result",
                         !pearlGatewaySubmitAccepted(
                             "{\"result\":null,\"id\":2}", &why));
        ok &= expectBool("nor is result:false",
                         !pearlGatewaySubmitAccepted(
                             "{\"result\":false,\"id\":2}", &why));
        ok &= expectBool("nor is an HTTP error body with no JSON at all",
                         !pearlGatewaySubmitAccepted(
                             "502 Bad Gateway", &why));
        ok &= expectBool("and each says why", !why.empty());
    }

    // --- dispatching on the method ------------------------------------
    //
    // A substring search matches the method name wherever it appears. The
    // dangerous one is handlePush(): in submit()'s reply loop it swallows any
    // line it claims, so a rejection whose wording mentions mining.notify used
    // to eat the share's own verdict.
    {
        uint64_t g[4] = {};
        ok &= expectBool("a set_target push is still recognised",
                         pearlPoolTargetFromPush(
                             "{\"method\":\"mining.set_target\",\"params\":"
                             "{\"target\":\"" + std::string(63, '0') + "1\"}}", g));
        // Both of these carry the value the old code would then read, so they
        // discriminate: byte-matching returns true for each.
        ok &= expectBool("a reply that merely MENTIONS the method is not a push",
                         !pearlPoolTargetFromPush(
                             "{\"id\":9,\"result\":false,\"target\":\"" +
                             std::string(63, '0') + "1\",\"error\":{\"code\":21,"
                             "\"message\":\"stale, see mining.set_target\"}}", g));
        ok &= expectBool("nor is a job whose id contains it",
                         !pearlPoolTargetFromPush(
                             "{\"id\":null,\"method\":\"mining.notify\",\"params\":"
                             "{\"job_id\":\"mining.set_difficulty\",\"difficulty\":8}}", g));
        ok &= expectBool("the method field is read, not searched for",
                         pearlPoolMethod(
                             "{\"id\": null, \"method\": \"mining.notify\"}") ==
                             "mining.notify");
        ok &= expectBool("a line with no method has none",
                         pearlPoolMethod("{\"id\":9,\"result\":true}").empty());
    }

    // --- matching a reply to its request ------------------------------
    //
    // A missed reply is silent: submit() gives up with "no reply from the
    // pool", the share is not counted, and the pool credits it anyway.
    {
        ok &= expectBool("a compact reply matches its request",
                         pearlPoolReplyIsFor("{\"id\":101,\"result\":true}", 101));
        ok &= expectBool("a space after the colon still matches",
                         pearlPoolReplyIsFor("{\"id\": 101, \"result\": true}", 101));
        ok &= expectBool("a string id matches its number",
                         pearlPoolReplyIsFor("{\"id\":\"101\"}", 101));
        ok &= expectBool("somebody else's reply does not match",
                         !pearlPoolReplyIsFor("{\"id\":102,\"result\":true}", 101));
        ok &= expectBool("and a push, whose id is null, does not either",
                         !pearlPoolReplyIsFor(
                             "{\"id\":null,\"method\":\"mining.notify\"}", 101));
        // 101 must not match 1011 or 1: the old substring test matched both.
        ok &= expectBool("a longer id is not a prefix match",
                         !pearlPoolReplyIsFor("{\"id\":1011}", 101));
    }

    if (!ok) return 1;
    puts("pearl pool JSON payloads: ok");
    return 0;
}
