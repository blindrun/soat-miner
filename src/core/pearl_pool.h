// Pearl pool mining.
//
// Pearl's own tooling has no pool path at all: pearl-gateway talks to a node
// you run yourself, holds the payout address, and turns a submitted PlainProof
// into the zero-knowledge certificate a block needs. That is fine if you run a
// node. Most people will not.
//
// Pools exist anyway, and they speak a stratum of their own. This is the wire,
// established by probing a live pool because nothing documents it:
//
//   -> {"id":1,"method":"mining.authorize","params":{"wallet":"prl1...",
//         "worker":"rig1","pass":"x","agent":"soat-miner"}}
//   <- {"id":1,"error":null,"result":true}
//   <- {"id":null,"method":"mining.notify","params":{
//         "job_id":"00000000_2097152",
//         "header":"<152 hex chars, the 76-byte incomplete header>",
//         "target":"<64 hex chars, big endian>",
//         "height":101009,"cert_version":3}}
//   -> {"id":9,"method":"mining.submit","params":{
//         "job_id":"...","plain_proof":"<base64 bincode PlainProof>"}}
//
// Three things are worth knowing before changing any of it:
//
//   * PARAMS ARE OBJECTS, not arrays. Sending stratum's usual positional array
//     gets {"code":20,"msg":"params must be an object"}.
//   * THE WALLET FIELD IS `wallet`. Not `login`, not `user` - both of those
//     authorize "successfully" and then fail with {"code":24,"Wallet is
//     missing"}, which is a confusing way to find out.
//   * The proof we already build for the gateway is EXACTLY what the pool
//     wants. `plain_proof` is the same base64 bincode PlainProof. Nothing
//     about the mining changes, only the transport.
//
// The pool PUSHES work rather than answering polls, so fetch() drains whatever
// has arrived and reports the newest job it has seen, instead of asking.

#pragma once

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <cctype>
#include <string>

#include "algo.h"
#include "http.h"
#include "json_lite.h"   // jsonRawValue/jsonUnquote: read the reply fields, not punctuation
#include "platform.h"
#include "pearl_gateway.h"   // packPearlExtra, unpackPearlExtra, base64/b3
#include "run.h"

namespace om {

/** Big-endian hex, as the pool sends targets, into the little-endian limbs
 *  Job::target carries. Getting this backwards yields a target that is either
 *  absurdly easy or unreachable, and both look like "the pool hates us". */
inline bool hexToLimbs(const std::string &hex, uint64_t out[4]) {
    uint8_t b[32];
    if (!hexToBytes(hex, b, 32)) return false;
    for (int limb = 0; limb < 4; limb++) {
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | b[(3 - limb) * 8 + i];
        out[limb] = v;
    }
    return true;
}

// ------------------------------------------------------- 256-bit division
//
// Only needed to turn a difficulty into a target. Bitwise long division rather
// than a limb algorithm on purpose: 256 iterations is free at the rate a pool
// changes difficulty, and it needs no `__int128`, which MSVC does not have and
// which `U256::mul` is already written in 32-bit limbs to avoid.

inline bool u256IsZero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

inline bool u256Less(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

inline void u256Sub(uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        const uint64_t bi = b[i] + borrow;
        // `b[i] + borrow` can wrap to 0 when b[i] is all ones; that is still a
        // borrow out, so test the addition itself rather than only a[i] < bi.
        borrow = (borrow && bi == 0) ? 1 : (a[i] < bi ? 1 : 0);
        a[i] -= bi;
    }
}

inline void u256Shl1(uint64_t a[4], uint64_t in) {
    for (int i = 0; i < 4; i++) {
        const uint64_t carry = a[i] >> 63;
        a[i] = (a[i] << 1) | in;
        in = carry;
    }
}

/** q = num / den. False when den is zero, which no pool difficulty can be. */
inline bool u256Div(const uint64_t num[4], const uint64_t den[4], uint64_t q[4]) {
    if (u256IsZero(den)) return false;
    uint64_t rem[4] = {0, 0, 0, 0};
    uint64_t quo[4] = {0, 0, 0, 0};
    for (int bit = 255; bit >= 0; bit--) {
        const uint64_t b = (num[bit / 64] >> (bit % 64)) & 1ULL;
        u256Shl1(rem, b);
        if (!u256Less(rem, den)) {
            u256Sub(rem, den);
            quo[bit / 64] |= 1ULL << (bit % 64);
        }
    }
    memcpy(q, quo, sizeof(quo));
    return true;
}

/**
 * Pearl Stratum's difficulty, as a share target.
 *
 * `target = 0xFFFF * 2^208 / difficulty`, the ordinary Bitcoin-style mapping.
 * That is not an assumption: HeroMiners names the difficulty in its job id
 * (`00000000_2097152`) and the `target` it sends alongside decodes to exactly
 * `0xFFFF * 2^187`, which is `0xFFFF * 2^208 / 2097152`. Measured against the
 * live pool on 2026-08-20.
 *
 * The result is the same LITTLE-endian limb layout `Job::target` carries, so it
 * is interchangeable with what hexToLimbs() produces from a notify.
 */
inline bool pearlDifficultyToTarget(const std::string &decDifficulty,
                                    uint64_t out[4]) {
    uint64_t den[4] = {0, 0, 0, 0};
    decimalToLimbs(decDifficulty, den);
    if (u256IsZero(den)) return false;
    // 0xFFFF * 2^208: bit 208 is limb 3, bit 16.
    const uint64_t num[4] = {0, 0, 0, 0xFFFFULL << 16};
    if (!u256Div(num, den, out)) return false;
    return !u256IsZero(out);
}

/** The job id rides in `extra` behind the header and cert version, so submit()
 *  can name the job the proof belongs to. unpackPearlExtra still parses the
 *  first two fields unchanged, so a gateway job and a pool job share a format. */
inline std::string packPoolExtra(const uint8_t header[76], int cert,
                                 const std::string &jobId) {
    return packPearlExtra(header, cert) + ":" + jobId;
}
inline std::string poolJobId(const std::string &extra) {
    const size_t first = extra.find(':');
    if (first == std::string::npos) return "";
    const size_t second = extra.find(':', first + 1);
    return second == std::string::npos ? "" : extra.substr(second + 1);
}

/** The leading integer of a positional `"params":[...]` array, for the
 *  array-shaped set_difficulty other stratums use. Empty when there is none. */
inline std::string pearlPoolFirstArrayNumber(const std::string &line) {
    // Take the params VALUE rather than searching the line for the bytes
    // `"params"`, which also matches the text inside a string value, and then
    // reads the next `[` from wherever that happened to be.
    std::string params;
    if (!jsonRawValue(line, "params", &params)) return std::string();
    if (params.empty() || params.front() != '[') return std::string();
    size_t a = 1;
    while (a < params.size() && (params[a] == ' ' || params[a] == '"')) a++;
    size_t b = a;
    while (b < params.size() && isdigit((unsigned char)params[b])) b++;
    return b == a ? std::string() : params.substr(a, b - a);
}

/** The JSON-RPC method this line names, or empty.
 *
 * Every dispatch here reads this rather than searching the line for the method
 * name, because a substring search matches the name wherever it appears -
 * inside a job id, inside a pool's error text, or as the prefix of a longer
 * method. handlePush() ate any line containing "mining.notify" and returned
 * true, which in the submit reply loop means the share's own verdict can be
 * swallowed as if it were a pushed job. Same rule as the reply parsing below:
 * parse the field, never match the bytes. */
inline std::string pearlPoolMethod(const std::string &line) {
    std::string m;
    if (!jsonString(line, "method", &m)) return std::string();
    return m;
}

/**
 * A pushed share-target change, from either notification a stratum can use.
 *
 * Pearl Stratum V1 has NO difficulty negotiation of its own: every
 * `mining.notify` carries the final `target`, and HeroMiners sends nothing
 * else. Ergo (`stratum.cpp`) handles `mining.set_difficulty` and
 * `mining.set_target`, and BC3 (`stratum_btc`) handles `set_difficulty`, so a
 * Pearl pool that borrows either idiom is plausible and used to be silently
 * ignored here - the miner would have kept mining the last notify's target
 * while the pool graded against a new one, which reads as
 * "hash does not meet difficulty target" and nothing else.
 *
 * Accepts the object form (`{"difficulty":N}` / `{"target":"<hex>"}`) and the
 * positional array form both. A fractional difficulty truncates, which is the
 * safe direction: a lower difficulty means an easier target, and rounding it
 * up would submit shares the pool refuses.
 *
 * Returns false for any other line, and for a value that cannot be used.
 */
inline bool pearlPoolTargetFromPush(const std::string &line, uint64_t out[4]) {
    const std::string method = pearlPoolMethod(line);
    if (method == "mining.set_target") {
        std::string hex;
        return jsonString(line, "target", &hex) && hexToLimbs(hex, out) &&
               !u256IsZero(out);
    }
    if (method == "mining.set_difficulty") {
        std::string dec;
        if (!jsonNumber(line, "difficulty", &dec)) dec = pearlPoolFirstArrayNumber(line);
        return !dec.empty() && pearlDifficultyToTarget(dec, out);
    }
    return false;
}

/** A proof is valid only for the exact pushed job that generated it.  Compare
 * all of the fields that affect proof verification, rather than treating an
 * unchanged header as permission to keep using an earlier share target. */
inline bool pearlPoolJobsMatch(const Job &a, const Job &b) {
    return a.valid && b.valid && a.epoch == b.epoch &&
           a.extra == b.extra &&
           memcmp(a.msg, b.msg, sizeof(a.msg)) == 0 &&
           memcmp(a.target, b.target, sizeof(a.target)) == 0;
}

/** JSON-RPC success is explicit.  In particular, `result:false` and a
 * structured `error` must not be reported as a submitted share. */
/** Did the pool accept, and if not, why.
 *
 * Read the fields rather than searching for punctuation. Two bugs lived in the
 * substring version, both found by the mock-gateway test on its first run:
 *
 *   - it required the literal `"result":true` and `"error":null`, so a server
 *     that puts a space after the colon - which is ordinary, conforming JSON -
 *     had every reply read as malformed. The login then failed with a message
 *     blaming the pool for being malformed.
 *   - the top-level msg/message check ran first and unconditionally, so a
 *     pool that sent a greeting or any informational `message` alongside a
 *     perfectly good `"result":true` had the ACCEPT counted as a rejection.
 *     That is a share credited by the pool and not by us, which is invisible
 *     until the payout does not match the counter.
 *
 * Order now: an explicit non-null `error` decides, then `result`, and the
 * top-level msg/message is only a source of WORDING for a failure - never the
 * verdict itself. */
inline bool pearlPoolReplyAccepted(const std::string &line, std::string *reason) {
    auto fallbackReason = [&](const char *dflt) {
        std::string msg;
        if (jsonString(line, "msg", &msg) && !msg.empty()) { *reason = msg; return; }
        if (jsonString(line, "message", &msg) && !msg.empty()) { *reason = msg; return; }
        *reason = dflt;
    };

    std::string err;
    if (jsonRawValue(line, "error", &err) && err != "null" && !err.empty()) {
        std::string msg;
        if (jsonString(err, "message", &msg) && !msg.empty()) *reason = msg;
        else if (jsonString(err, "msg", &msg) && !msg.empty()) *reason = msg;
        else *reason = jsonUnquote(err);
        return false;
    }

    std::string result;
    if (jsonRawValue(line, "result", &result)) {
        if (result == "true" || result == "1") return true;
        fallbackReason("pool answered result=false with no reason");
        return false;
    }

    fallbackReason("pool returned an unsuccessful or malformed JSON-RPC reply");
    return false;
}

/** Is this line the reply to request `id`?
 *
 * Read the field. Searching for the bytes `"id":N` misses `"id": N`, which is
 * ordinary JSON - and a missed reply is not a visible failure. submit() gives
 * up with "no reply from the pool", the share is not counted, and the pool
 * credits it anyway. The counter and the payout then disagree with no error
 * anywhere, which is the single hardest bug in this file to notice. */
inline bool pearlPoolReplyIsFor(const std::string &line, int id) {
    std::string raw;
    if (!jsonRawValue(line, "id", &raw)) return false;
    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"')
        raw = raw.substr(1, raw.size() - 2);
    if (raw.empty() || raw == "null") return false;
    return atoi(raw.c_str()) == id;
}

/** Why this string cannot be a Pearl payout address, or empty if it might be.
 *
 * The pool is the authority and rejects a bad address with `Invalid Pearl
 * address`, which does not say which part is wrong or that a template was
 * never edited. The Ergo path has checked its own address locally since the
 * beginning; Pearl only checked for empty, so the shipped placeholder reached
 * the pool and came back as that message.
 *
 * Deliberately no bech32m checksum here. A checksum this code gets subtly
 * wrong would refuse a real address, which is worse than a pool round trip.
 * These are the failures a local check can be certain about. */
inline std::string pearlWalletProblem(const std::string &w) {
    static const char *kBech32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    if (w.empty()) return "no address given";
    if (w.find("YOUR") != std::string::npos)
        return "this is the placeholder from the launcher, not an address - "
               "edit WALLET in mine_pearl_herominers.sh or .bat";
    if (w.rfind("tprl1", 0) == 0)
        return "this is a Pearl TESTNET address (tprl1). A mainnet pool cannot "
               "pay it";
    if (w.rfind("prl1", 0) != 0) {
        // The most likely wrong address is the one the user already has. Ergo
        // and Pearl are different chains: an Ergo address cannot be paid by a
        // Pearl pool, and no setting makes it work.
        if (w[0] == '9' && w.size() >= 40 && w.size() <= 60)
            return "that is an Ergo address. Pearl is a different chain and "
                   "pays its own prl1 address - you need a Pearl wallet, not a "
                   "setting change";
        return "a Pearl address starts with prl1";
    }
    if (w.find('.') != std::string::npos)
        return "the worker name belongs in --worker, not appended to the "
               "address after a dot";
    if (w.size() < 40 || w.size() > 90)
        return "wrong length for a Pearl address (" + std::to_string(w.size()) +
               " characters; a real one is about 63)";
    for (size_t i = 4; i < w.size(); i++) {
        if (!strchr(kBech32, w[i]))
            return std::string("'") + w[i] +
                   "' cannot appear in a Pearl address";
    }
    return std::string();
}

/** The two JSON-RPC payloads whose values can originate outside this process.
 * Kept as free functions so their escaping is testable without a pool socket. */
inline std::string pearlPoolAuthorizeRequest(const std::string &wallet,
                                             const std::string &worker,
                                             const std::string &pass = "x",
                                             const std::string &agent = "soat-miner") {
    std::string request =
        "{\"id\":1,\"method\":\"mining.authorize\",\"params\":{\"wallet\":" +
        jsonQuoted(wallet);
    if (!worker.empty()) request += ",\"worker\":" + jsonQuoted(worker);
    if (!pass.empty()) request += ",\"pass\":" + jsonQuoted(pass);
    if (!agent.empty()) request += ",\"agent\":" + jsonQuoted(agent);
    return request + "}}";
}

inline std::string pearlPoolSubmitRequest(int requestId, const std::string &jobId,
                                          const std::string &proof) {
    return "{\"id\":" + std::to_string(requestId) +
           ",\"method\":\"mining.submit\",\"params\":{\"job_id\":" +
           jsonQuoted(jobId) + ",\"plain_proof\":" + jsonQuoted(proof) + "}}";
}

/** The diagnostic transcript deliberately has a smaller data surface than the
 * wire protocol: no authorize request, wallet, worker, raw header, or proof
 * bytes are ever emitted. Defend the remaining free-text fields against a
 * pool echoing a PRL address in an error message or opaque job id. */
inline std::string pearlTranscriptRedact(const std::string &in) {
    std::string out;
    for (size_t p = 0; p < in.size();) {
        if (p + 4 <= in.size() && in.compare(p, 4, "prl1") == 0) {
            size_t end = p + 4;
            while (end < in.size() && std::isalnum((unsigned char)in[end])) end++;
            out += "[REDACTED_PRL_ADDRESS]";
            p = end;
        } else {
            out += in[p++];
        }
    }
    return out;
}

inline std::string pearlTargetHexBE(const uint64_t limbs[4]) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int limb = 3; limb >= 0; limb--)
        for (int byte = 7; byte >= 0; byte--) {
            const uint8_t v = (uint8_t)(limbs[limb] >> (byte * 8));
            out += hex[v >> 4];
            out += hex[v & 15];
        }
    return out;
}

inline std::string pearlTranscriptProofDigest(const std::string &proof) {
    uint8_t digest[32];
    pearl::b3::hash(nullptr, (const uint8_t *)proof.data(), proof.size(), digest);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t b : digest) { out += hex[b >> 4]; out += hex[b & 15]; }
    return out;
}

/**
 * A short fingerprint of the 76-byte header, via the digest the job already
 * carries. The header itself is deliberately never written to a transcript;
 * this is enough to tell two notifies apart and to prove which one a submitted
 * proof was mined against, which the job id alone cannot do - HeroMiners
 * advances the job id AND rewrites the header's timestamp on every push.
 */
inline std::string pearlHeaderFingerprint(const uint8_t msg[32]) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 8; i++) { out += hex[msg[i] >> 4]; out += hex[msg[i] & 15]; }
    return out;
}

inline std::string pearlTranscriptNotify(const Job &job) {
    return "{\"schema\":\"soat-pearl8-diagnostic-v1\",\"event\":\"mining.notify\",\"job_id\":" +
           jsonQuoted(pearlTranscriptRedact(poolJobId(job.extra))) +
           ",\"header_b3\":" + jsonQuoted(pearlHeaderFingerprint(job.msg)) +
           ",\"target_be\":" + jsonQuoted(pearlTargetHexBE(job.target)) +
           ",\"cert_version\":" + std::to_string(atoi(job.extra.c_str() + 153)) + "}";
}

/** A pushed share-target change, recorded so a transcript shows the bound the
 *  miner was actually using rather than only the one the last notify named. */
inline std::string pearlTranscriptTarget(const uint64_t limbs[4]) {
    return "{\"schema\":\"soat-pearl8-diagnostic-v1\",\"event\":\"mining.set_target\",\"target_be\":" +
           jsonQuoted(pearlTargetHexBE(limbs)) + "}";
}

/**
 * What the pool is about to recompute, written down before it answers.
 *
 * A rejection says only "hash does not meet difficulty target", which is the
 * same sentence for a stale header, a mis-scaled bound and a proof that
 * describes the wrong tile. These four fields separate them without another
 * run: `header_b3` ties the proof to a notify in the same file, `m`/`n`/`k`/
 * `rank` are read back out of the serialised proof rather than from the
 * miner's own variables, and `digest_be` against `bound_be` is the arithmetic
 * the pool is about to redo. A digest above the bound is our bug; a digest
 * below it that is still refused is not.
 */
inline std::string pearlTranscriptSubmit(int requestId, const std::string &jobId,
                                         const std::string &proof,
                                         const std::string &headerB3,
                                         const uint64_t digest[4],
                                         const uint64_t target[4]) {
    std::string shape = "null,\"n\":null,\"k\":null,\"rank\":null";
    std::string bound = "null";
    std::string within = "null";
    std::vector<uint8_t> raw;
    if (pearl::base64Decode(proof, &raw) && raw.size() >= 32) {
        uint64_t f[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++)
            for (int b = 7; b >= 0; b--) f[i] = (f[i] << 8) | raw[i * 8 + b];
        shape = std::to_string(f[0]) + ",\"n\":" + std::to_string(f[1]) +
                ",\"k\":" + std::to_string(f[2]) + ",\"rank\":" + std::to_string(f[3]);
        pearl::MiningConfig cfg;
        cfg.commonDim = (uint32_t)f[2];
        cfg.rank = (uint16_t)f[3];
        pearl::U256 scaled;
        if (f[2] && f[3] && cfg.penalizedTarget(pearl::U256::fromLimbs(target), &scaled)) {
            bound = "\"" + pearlTargetHexBE(scaled.v) + "\"";
            within = pearl::U256::fromLimbs(digest).le(scaled) ? "true" : "false";
        }
    }
    return "{\"schema\":\"soat-pearl8-diagnostic-v1\",\"event\":\"mining.submit\",\"request_id\":" +
           std::to_string(requestId) + ",\"job_id\":" +
           jsonQuoted(pearlTranscriptRedact(jobId)) +
           ",\"header_b3\":" + jsonQuoted(headerB3) +
           ",\"m\":" + shape +
           ",\"digest_be\":" + jsonQuoted(pearlTargetHexBE(digest)) +
           ",\"target_be\":" + jsonQuoted(pearlTargetHexBE(target)) +
           ",\"bound_be\":" + bound +
           ",\"digest_within_bound\":" + within +
           ",\"plain_proof_bytes\":" + std::to_string(proof.size()) +
           ",\"plain_proof_b3\":" + jsonQuoted(pearlTranscriptProofDigest(proof)) + "}";
}

inline std::string pearlTranscriptReply(int requestId, bool accepted,
                                        const std::string &reason) {
    return "{\"schema\":\"soat-pearl8-diagnostic-v1\",\"event\":\"mining.reply\",\"request_id\":" +
           std::to_string(requestId) + ",\"accepted\":" +
           (accepted ? "true" : "false") + ",\"reason\":" +
           jsonQuoted(pearlTranscriptRedact(reason)) + "}";
}

/** Parse a pushed job, including its server-selected share target, without a
 * socket.  This keeps target byte order and target-only refreshes testable. */
inline bool pearlPoolJobFromNotify(const std::string &line, Job *job) {
    std::string hdr, tgt, jid, certStr;
    if (!jsonString(line, "header", &hdr) || !jsonString(line, "target", &tgt) ||
        !jsonString(line, "job_id", &jid) || hdr.size() != 152)
        return false;
    uint8_t header[76];
    uint64_t target[4];
    if (!hexToBytes(hdr, header, sizeof(header)) || !hexToLimbs(tgt, target) ||
        (target[0] == 0 && target[1] == 0 && target[2] == 0 && target[3] == 0))
        return false;
    if (!jsonNumber(line, "cert_version", &certStr)) return false;
    const int cert = atoi(certStr.c_str());
    // The miner's proof encoder has explicit V1/V2/V3 paths.  Guessing V3
    // for an absent or unknown pool field makes a syntactically valid but
    // cryptographically different proof.
    if (cert < 1 || cert > 3) return false;
    pearl::b3::hash(nullptr, header, sizeof(header), job->msg);
    memcpy(job->target, target, sizeof(target));
    // Pearl Stratum V1 defines notify.target as the final, big-endian share
    // bound.  It is not the gateway's base block target and must not be
    // rank-penalized again in PearlPow.
    job->extra = packPoolExtra(header, cert, jid);
    job->epoch = 0;
    job->valid = true;
    return true;
}

class PearlPoolSource : public JobSource {
   public:
    PearlPoolSource(std::string host, int port, std::string wallet,
                    std::string worker, std::string transcriptPath = "")
        : host_(std::move(host)), port_(port), wallet_(std::move(wallet)),
          worker_(std::move(worker)), transcriptPath_(std::move(transcriptPath)) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pearl pool)";
        if (!transcriptPath_.empty()) {
            transcript_ = fopen(transcriptPath_.c_str(), "w");
            if (!transcript_)
                lastError_ = "could not open Pearl diagnostic transcript " + transcriptPath_;
        }
    }

    ~PearlPoolSource() override {
        disconnect();
        if (transcript_) fclose(transcript_);
    }

    const char *describe() const override { return desc_.c_str(); }
    const std::string &lastError() const { return lastError_; }

    bool fetch(Job *job) override {
        if (!transcriptPath_.empty() && !transcript_) return false;
        if (fd_ == OM_INVALID_SOCKET && !connectAndLogin()) return false;
        // ZERO, not 50 ms, once we have work. The core calls fetch() before
        // every search() batch, and a batch is about 10 ms of GPU time at
        // Pearl's rate - so a 50 ms wait here cost 83% of the hashrate and
        // showed up as 65 M candidates/s against 413 in the benchmark. Only
        // the FIRST job is worth blocking for.
        drain(haveJob_ ? 0 : 8000);
        if (!haveJob_) {
            if (lastError_.empty())
                lastError_ = "pool connected but sent no mining job within 8 seconds";
            return false;
        }

        pearl::b3::hash(nullptr, header_, 76, job->msg);
        for (int i = 0; i < 4; i++) job->target[i] = target_[i];
            job->extra = packPoolExtra(header_, cert_, jobId_);
        job->epoch = 0;
        job->valid = true;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override {
        if (sol.extra.empty()) {
            *err = "no proof attached to the solution";
            return false;
        }
        const std::string id = poolJobId(job.extra);
        if (id.empty()) {
            *err = "job is missing its pool job id";
            return false;
        }
        if (fd_ == OM_INVALID_SOCKET && !connectAndLogin()) {
            *err = "lost the pool connection";
            return false;
        }
        // A notify may have arrived while the last search batch was running.
        // Never send a proof for the old target/job id when it is already
        // queued locally.  A pool that invalidates work on refresh otherwise
        // sees this as a bad share.
        drainAvailable();
        Job current;
        if (!currentJob(&current) || !pearlPoolJobsMatch(job, current)) {
            *err = "pool job was replaced before submission; proof not sent";
            return false;
        }
        const std::string body = pearlPoolSubmitRequest(++id_, id, sol.extra);
        writeTranscript(pearlTranscriptSubmit(id_, id, sol.extra,
                                             pearlHeaderFingerprint(job.msg),
                                             sol.hit, job.target));
        if (!sendLine(body)) {
            *err = "could not send the share";
            disconnect();
            return false;
        }
        // The reply may arrive behind pushed jobs, so read until our id shows.
        //
        // A pushed job must NOT spend the budget. It used to: the loop ran a
        // fixed 12 iterations and counted pushes against them, so a pool that
        // sent a burst of notifies while we waited - which a busy pool does -
        // took the reply past the window. The share was then dropped as "no
        // reply from the pool" while the pool credited it. Pushes are now free
        // and only real replies are counted, under an overall time bound so a
        // pool that pushes forever still cannot wedge us here.
        const long long replyDeadline = nowMs() + 15000;
        for (int replies = 0; replies < 12 && nowMs() < replyDeadline;) {
            std::string line;
            if (!readLine(&line)) break;
            if (handlePush(line)) continue;
            replies++;
            if (!pearlPoolReplyIsFor(line, id_)) continue;
            const bool accepted = pearlPoolReplyAccepted(line, err);
            writeTranscript(pearlTranscriptReply(id_, accepted, *err));
            return accepted;
        }
        *err = "no reply from the pool";
        writeTranscript(pearlTranscriptReply(id_, false, *err));
        return false;
    }

   private:
    bool connectAndLogin() {
        if (!openSocket()) return false;
        const std::string auth = pearlPoolAuthorizeRequest(wallet_, worker_);
        if (!sendLine(auth)) {
            lastError_ = "TCP connection closed while sending mining.authorize";
            disconnect();
            return false;
        }
        std::string line;
        std::string replyError;
        // Pearl V1 may push the first notify before the authorize ack. Keep
        // that job instead of treating it as a malformed login response.
        for (;;) {
            if (!readLine(&line)) {
                lastError_ = "pool closed the TCP connection during mining.authorize";
                disconnect();
                return false;
            }
            if (handlePush(line)) continue;
            if (!pearlPoolReplyAccepted(line, &replyError)) {
                fprintf(stderr, "[pearl-pow] pool refused the login: %s\n", replyError.c_str());
                lastError_ = "pool refused the login: " + replyError;
                disconnect();
                return false;
            }
            break;
        }
        lastError_.clear();
        return true;
    }

    /** Read whatever has arrived, keeping the newest job. */
    void drain(int ms) {
        const long long deadline = nowMs() + ms;
        do {
            // Poll with no timeout when we already have work: take whatever
            // has arrived and get straight back to mining.
            if (!readable(fd_, haveJob_ ? 0 : 50)) {
                if (haveJob_) return;
                continue;
            }
            std::string line;
            if (!readLine(&line)) {
                lastError_ = "pool closed the TCP connection while waiting for work";
                disconnect();
                return;
            }
            handlePush(line);
        } while (nowMs() < deadline);
    }

    /** Consume every line already queued by the socket without waiting. */
    void drainAvailable() {
        while (fd_ != OM_INVALID_SOCKET &&
               (pending_.find('\n') != std::string::npos || readable(fd_, 0))) {
            std::string line;
            if (!readLine(&line)) {
                lastError_ = "pool closed the TCP connection while reading queued work";
                disconnect();
                return;
            }
            handlePush(line);
        }
    }

    bool currentJob(Job *job) const {
        if (!haveJob_) return false;
        pearl::b3::hash(nullptr, header_, sizeof(header_), job->msg);
        memcpy(job->target, target_, sizeof(target_));
            job->extra = packPoolExtra(header_, cert_, jobId_);
        job->epoch = 0;
        job->valid = true;
        return true;
    }

    /**
     * A line the SERVER started, consumed. False means it is a reply to one of
     * our own requests and the caller still has to look at it.
     *
     * Every read path funnels through here so a pushed share target can never
     * be seen by one of them and missed by another - the login loop, the two
     * drains and the submit reply loop each used to test for `mining.notify`
     * and drop everything else on the floor.
     */
    bool handlePush(const std::string &line) {
        if (pearlPoolMethod(line) == "mining.notify") {
            takeNotify(line);
            return true;
        }
        uint64_t pushed[4];
        if (!pearlPoolTargetFromPush(line, pushed)) return false;
        // Latest push wins, and keeps winning over later notifies: that is what
        // `set_difficulty` means everywhere else, and a pool that grades against
        // a target it pushed separately would otherwise refuse every share while
        // both sides believed they agreed. A pool that only ever puts the target
        // in its notify never reaches this line at all.
        memcpy(overrideTarget_, pushed, sizeof(overrideTarget_));
        haveOverride_ = true;
        // A live job is regraded immediately rather than at the next notify:
        // the pool applies a new bound to the very next share, not the next job.
        if (haveJob_) memcpy(target_, pushed, sizeof(target_));
        writeTranscript(pearlTranscriptTarget(pushed));
        return true;
    }

    void takeNotify(const std::string &line) {
        // A malformed replacement must retire the preceding job. Continuing
        // to mine it is worse than briefly waiting for a valid refresh.
        clearJob();
        Job fresh;
        if (!pearlPoolJobFromNotify(line, &fresh)) {
            writeTranscript("{\"schema\":\"soat-pearl8-diagnostic-v1\",\"event\":\"mining.notify_invalid\"}");
            return;
        }
        uint8_t header[76];
        if (!unpackPearlExtra(fresh.extra, header, &cert_)) return;
        memcpy(header_, header, sizeof(header_));
        memcpy(target_, haveOverride_ ? overrideTarget_ : fresh.target,
               sizeof(target_));
        memcpy(fresh.target, target_, sizeof(target_));  // so the transcript
                                                         // records the bound in
                                                         // force, not the one
                                                         // the notify named
        jobId_ = poolJobId(fresh.extra);
        haveJob_ = true;
        writeTranscript(pearlTranscriptNotify(fresh));
    }

    void clearJob() {
        haveJob_ = false;
        jobId_.clear();
        cert_ = 0;
        memset(header_, 0, sizeof(header_));
        memset(target_, 0, sizeof(target_));
    }

    void writeTranscript(const std::string &line) {
        if (!transcript_) return;
        fprintf(transcript_, "%s\n", line.c_str());
        fflush(transcript_);
    }

    bool openSocket() {
        disconnect();
        struct addrinfo hints {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo *res = nullptr;
        const std::string port = std::to_string(port_);
        if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &res) != 0) {
            lastError_ = "DNS lookup failed for " + host_ + ":" + port;
            return false;
        }
        fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd_ == OM_INVALID_SOCKET) {
            lastError_ = "could not create a TCP socket";
            freeaddrinfo(res);
            return false;
        }
        socketTimeout(fd_, 20);
        if (::connect(fd_, res->ai_addr, (int)res->ai_addrlen) != 0) {
            OM_CLOSESOCKET(fd_);
            fd_ = OM_INVALID_SOCKET;
            freeaddrinfo(res);
            lastError_ = "TCP connection to " + host_ + ":" + port + " failed";
            return false;
        }
        freeaddrinfo(res);
        pending_.clear();
        return true;
    }

    void disconnect() {
        if (fd_ != OM_INVALID_SOCKET) OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
        pending_.clear();
        // Jobs are scoped to a TCP session.  Reusing one after reconnect can
        // only produce a stale share, even if the server happens to repeat a
        // header.  So is a negotiated difficulty: the new session starts from
        // whatever the pool decides then, not from what the last one agreed.
        clearJob();
        haveOverride_ = false;
        memset(overrideTarget_, 0, sizeof(overrideTarget_));
    }

    bool sendLine(const std::string &body) {
        const std::string line = body + "\n";
        return socketSend(fd_, line.data(), line.size()) >= 0;
    }

    bool readLine(std::string *out) {
        for (;;) {
            const size_t nl = pending_.find('\n');
            if (nl != std::string::npos) {
                *out = pending_.substr(0, nl);
                pending_.erase(0, nl + 1);
                return true;
            }
            char buf[4096];
            const int n = socketRecv(fd_, buf, sizeof(buf));
            if (n <= 0) return false;
            pending_.append(buf, n);
        }
    }

    static bool readable(socket_t fd, int ms) {
        if (fd == OM_INVALID_SOCKET) return false;
        fd_set r;
        FD_ZERO(&r);
        FD_SET(fd, &r);
        struct timeval tv;
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        return select((int)fd + 1, &r, nullptr, nullptr, &tv) > 0;
    }

    /** Wall clock, deliberately. clock() measures CPU time, and this thread
     *  spends its wait blocked in select() consuming none of it, so a
     *  clock()-based deadline would never expire. */
    static long long nowMs() {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    std::string host_, wallet_, worker_, desc_, pending_, jobId_, lastError_,
        transcriptPath_;
    int port_ = 0;
    int id_ = 100;
    int cert_ = 3;
    bool haveJob_ = false;
    uint8_t header_[76] = {};
    uint64_t target_[4] = {};
    uint64_t overrideTarget_[4] = {};
    bool haveOverride_ = false;
    socket_t fd_ = OM_INVALID_SOCKET;
    FILE *transcript_ = nullptr;
};

}  // namespace om
