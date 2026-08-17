// Pearl's host side: the commitment chain, and turning a win into a proof.
//
// The kernels in noisy_gemm.cuh compute transcripts from matrices that are
// already noised, against a key and a target that arrive from somewhere. This
// is that somewhere. Everything here is a port of tests/pearl_job.py, which is
// checked against Pearl's own Rust in tests/pearl_oracle.py - including a
// proof their verifier accepts and one it rejects.
//
// The shape of a Pearl attempt, which is unlike every other PoW in this repo:
//
//   1. The job is a 76-byte header stub, a target and a certificate version.
//      There are no matrices in it. The miner picks A (m x k) and B (k x n),
//      so CHOOSING THE MATRICES IS THE NONCE.
//   2. job_key = blake3(header || mining_config). The config commits k, the
//      rank and the hash-tile shape, so it is fixed before any hashing.
//   3. Each matrix's Merkle root is keyed blake3 of it, zero-padded to 1024
//      bytes - the leaf size IS blake3's chunk length and the tree IS blake3's
//      tree, so mining never builds a tree. Only a win does, to open 16 rows.
//   4. Those roots, each salted with its own dimension under V3, fold into
//      commitment_B then commitment_A. commitment_A is the A-noise seed AND
//      the PoW key, which is what binds the matrices to the work.
//   5. One GEMM produces (m/16)*(n/16) transcripts, each an independent
//      candidate. The bound each is measured against is the block target
//      scaled by the work an attempt costs - that scaling is not free
//      difficulty, it is what makes thousands of candidates per GEMM fair.
//   6. A win is a PlainProof: the two Merkle openings, bincode, base64, sent
//      to pearl-gateway. The miner never builds a block. A block needs a
//      plonky2 certificate that only Pearl's own prover produces.
//
// No dynamic allocation in the hashing path and no CUDA here on purpose: this
// header is host C++ so the same code serves the CUDA and Vulkan backends.

#pragma once

#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

namespace om {
namespace pearl {

constexpr size_t kChunkLen = 1024;      // blake3 CHUNK_LEN, and MERKLE_LEAF_SIZE
constexpr size_t kBlockLen = 64;
constexpr size_t kDigestLen = 32;
constexpr int kPenaltyBaseRank = 128;

// ------------------------------------------------------------------ blake3
//
// Host-side, and complete rather than the single-block special case the device
// kernel needs: a Merkle proof needs the tree's INTERNAL chaining values, and
// no blake3 library hands those out.

namespace b3 {

constexpr uint32_t kIV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                             0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

constexpr uint32_t kChunkStart = 1, kChunkEnd = 2, kParent = 4, kRoot = 8,
                   kKeyedHash = 16;

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void g(uint32_t *s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    s[a] += s[b] + mx;  s[d] = rotr(s[d] ^ s[a], 16);
    s[c] += s[d];       s[b] = rotr(s[b] ^ s[c], 12);
    s[a] += s[b] + my;  s[d] = rotr(s[d] ^ s[a], 8);
    s[c] += s[d];       s[b] = rotr(s[b] ^ s[c], 7);
}

/** The compression function. `out` receives the 8-word chaining value. */
inline void compress(const uint32_t cv[8], const uint32_t block[16],
                     uint64_t counter, uint32_t blockLen, uint32_t flags,
                     uint32_t out[8]) {
    uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                      kIV[0], kIV[1], kIV[2], kIV[3],
                      (uint32_t)(counter & 0xFFFFFFFFu),
                      (uint32_t)(counter >> 32), blockLen, flags};
    uint32_t m[16];
    for (int i = 0; i < 16; i++) m[i] = block[i];

    static const int kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int r = 0; r < 7; r++) {
        g(s, 0, 4, 8, 12, m[0], m[1]);
        g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]);
        g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);
        g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]);
        g(s, 3, 4, 9, 14, m[14], m[15]);
        if (r < 6) {
            uint32_t t[16];
            for (int i = 0; i < 16; i++) t[i] = m[kPerm[i]];
            for (int i = 0; i < 16; i++) m[i] = t[i];
        }
    }
    for (int i = 0; i < 8; i++) out[i] = s[i] ^ s[i + 8];
}

inline void loadBlock(const uint8_t *p, size_t len, uint32_t out[16]) {
    uint8_t buf[kBlockLen] = {0};
    memcpy(buf, p, len);
    for (int i = 0; i < 16; i++)
        out[i] = (uint32_t)buf[4 * i] | ((uint32_t)buf[4 * i + 1] << 8) |
                 ((uint32_t)buf[4 * i + 2] << 16) | ((uint32_t)buf[4 * i + 3] << 24);
}

inline void storeCv(const uint32_t cv[8], uint8_t out[32]) {
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(cv[i]);
        out[4 * i + 1] = (uint8_t)(cv[i] >> 8);
        out[4 * i + 2] = (uint8_t)(cv[i] >> 16);
        out[4 * i + 3] = (uint8_t)(cv[i] >> 24);
    }
}

inline void loadCv(const uint8_t in[32], uint32_t cv[8]) {
    for (int i = 0; i < 8; i++)
        cv[i] = (uint32_t)in[4 * i] | ((uint32_t)in[4 * i + 1] << 8) |
                ((uint32_t)in[4 * i + 2] << 16) | ((uint32_t)in[4 * i + 3] << 24);
}

/**
 * Chaining value of one chunk (at most 1024 bytes).
 *
 * `key` is null for an unkeyed hash, in which case the IV stands in and the
 * KEYED_HASH flag is dropped - the only difference between the two modes.
 */
inline void chunkCv(const uint8_t *key, const uint8_t *data, size_t len,
                    uint64_t chunkIndex, bool root, uint8_t out[32]) {
    uint32_t cv[8];
    if (key) loadCv(key, cv);
    else for (int i = 0; i < 8; i++) cv[i] = kIV[i];

    const uint32_t base = key ? kKeyedHash : 0u;
    size_t blocks = (len + kBlockLen - 1) / kBlockLen;
    if (blocks == 0) blocks = 1;              // the empty input is one empty block

    for (size_t i = 0; i < blocks; i++) {
        const size_t off = i * kBlockLen;
        const size_t n = (len > off) ? ((len - off < kBlockLen) ? len - off : kBlockLen) : 0;
        uint32_t flags = base;
        if (i == 0) flags |= kChunkStart;
        if (i + 1 == blocks) {
            flags |= kChunkEnd;
            if (root) flags |= kRoot;
        }
        uint32_t block[16];
        loadBlock(data + off, n, block);
        compress(cv, block, chunkIndex, (uint32_t)n, flags, cv);
    }
    storeCv(cv, out);
}

inline void parentCv(const uint8_t *key, const uint8_t left[32],
                     const uint8_t right[32], bool root, uint8_t out[32]) {
    uint32_t cv[8];
    if (key) loadCv(key, cv);
    else for (int i = 0; i < 8; i++) cv[i] = kIV[i];

    uint8_t joined[64];
    memcpy(joined, left, 32);
    memcpy(joined + 32, right, 32);
    uint32_t block[16];
    loadBlock(joined, 64, block);

    uint32_t flags = (key ? kKeyedHash : 0u) | kParent | (root ? kRoot : 0u);
    compress(cv, block, 0, (uint32_t)kBlockLen, flags, cv);
    storeCv(cv, out);
}

/**
 * Pair a layer of chaining values, carrying an odd trailing node up unchanged.
 *
 * This is equivalent to blake3's "left subtree is the largest power of two"
 * split at every leaf count, not an approximation of it. That equivalence is
 * why the miner can treat a plain keyed hash as the Merkle root and only build
 * the real tree once, on a win.
 */
inline void combineLayer(const uint8_t *key, const std::vector<uint8_t> &in,
                         std::vector<uint8_t> *out) {
    const size_t n = in.size() / 32;
    out->clear();
    out->resize(((n + 1) / 2) * 32);
    size_t w = 0;
    for (size_t i = 0; i + 1 < n; i += 2, w++)
        parentCv(key, &in[i * 32], &in[(i + 1) * 32], false, &(*out)[w * 32]);
    if (n % 2) memcpy(&(*out)[w * 32], &in[(n - 1) * 32], 32);
}

/** blake3 of arbitrary data. `key` null means unkeyed. */
inline void hash(const uint8_t *key, const uint8_t *data, size_t len,
                 uint8_t out[32]) {
    if (len <= kChunkLen) {
        chunkCv(key, data, len, 0, true, out);
        return;
    }
    std::vector<uint8_t> cvs(((len + kChunkLen - 1) / kChunkLen) * 32);
    for (size_t i = 0, c = 0; i < len; i += kChunkLen, c++) {
        const size_t n = (len - i < kChunkLen) ? len - i : kChunkLen;
        chunkCv(key, data + i, n, c, false, &cvs[c * 32]);
    }
    std::vector<uint8_t> next;
    while (cvs.size() / 32 > 2) {
        combineLayer(key, cvs, &next);
        cvs.swap(next);
    }
    parentCv(key, &cvs[0], &cvs[32], true, out);
}

}  // namespace b3

// ------------------------------------------------------------------- u256
//
// Limb 0 is least significant, matching om::Job::target.

struct U256 {
    uint64_t v[4] = {0, 0, 0, 0};

    static U256 fromLimbs(const uint64_t limbs[4]) {
        U256 r;
        memcpy(r.v, limbs, sizeof(r.v));
        return r;
    }

    void toBytesLE(uint8_t out[32]) const {
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < 8; b++) out[i * 8 + b] = (uint8_t)(v[i] >> (8 * b));
    }

    static U256 fromBytesLE(const uint8_t in[32]) {
        U256 r;
        for (int i = 0; i < 4; i++) {
            uint64_t x = 0;
            for (int b = 7; b >= 0; b--) x = (x << 8) | in[i * 8 + b];
            r.v[i] = x;
        }
        return r;
    }

    /** true when this <= other. */
    bool le(const U256 &other) const {
        for (int i = 3; i >= 0; i--) {
            if (v[i] != other.v[i]) return v[i] < other.v[i];
        }
        return true;
    }

    /**
     * Multiply by a 64-bit factor. Returns false on overflow rather than
     * saturating: a bound that overflowed 256 bits is unusable, and saturating
     * to U256::MAX would silently make every hash a winner.
     */
    bool mul(uint64_t factor, U256 *out) const {
        // 32-bit limbs, because a 64x64 product needs 128 bits and this has to
        // build without __int128 on MSVC.
        uint32_t a[8], f[2] = {(uint32_t)factor, (uint32_t)(factor >> 32)};
        for (int i = 0; i < 4; i++) {
            a[2 * i] = (uint32_t)v[i];
            a[2 * i + 1] = (uint32_t)(v[i] >> 32);
        }
        uint32_t r[10] = {0};
        for (int i = 0; i < 8; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < 2; j++) {
                const uint64_t cur = (uint64_t)r[i + j] +
                                     (uint64_t)a[i] * (uint64_t)f[j] + carry;
                r[i + j] = (uint32_t)cur;
                carry = cur >> 32;
            }
            int at = i + 2;
            while (carry && at < 10) {
                const uint64_t cur = (uint64_t)r[at] + carry;
                r[at] = (uint32_t)cur;
                carry = cur >> 32;
                at++;
            }
            if (carry) return false;
        }
        if (r[8] || r[9]) return false;
        for (int i = 0; i < 4; i++)
            out->v[i] = (uint64_t)r[2 * i] | ((uint64_t)r[2 * i + 1] << 32);
        return true;
    }
};

// -------------------------------------------------------- mining config
//
// A PeriodicPattern is a 3D arithmetic progression in six bytes, because the
// chain allows strided hash tiles. A miner only needs the contiguous case: for
// a run of `len` indices the shape is (1,len),(len,1),(len,1), which encodes as
// {0, len-1, 0, 0, 0, 0}. 16x16 is the largest tile the rules allow, which
// means the fewest blake3 calls per unit of matmul.

inline void contiguousPattern(int len, uint8_t out[6]) {
    memset(out, 0, 6);
    out[1] = (uint8_t)(len - 1);
}

struct MiningConfig {
    uint32_t commonDim = 2048;   // k
    uint16_t rank = 128;
    int tileH = 16;
    int tileW = 16;

    static constexpr size_t kSerializedSize = 52;

    void toBytes(uint8_t out[kSerializedSize]) const {
        memset(out, 0, kSerializedSize);
        out[0] = (uint8_t)commonDim;
        out[1] = (uint8_t)(commonDim >> 8);
        out[2] = (uint8_t)(commonDim >> 16);
        out[3] = (uint8_t)(commonDim >> 24);
        out[4] = (uint8_t)rank;
        out[5] = (uint8_t)(rank >> 8);
        // out[6..8] is MMAType::Int7xInt7ToInt32 == 0, and the 32-byte trailer
        // at out[20..52] is the MoE config - zero means a dense job.
        contiguousPattern(tileH, out + 8);
        contiguousPattern(tileW, out + 14);
    }

    int tileSize() const { return tileH * tileW; }
    uint32_t dotProductLength() const { return commonDim - commonDim % rank; }

    /**
     * Every consensus rule that can be checked before any work is done.
     *
     * Each of these otherwise costs a full GEMM to discover, and a rejected
     * block is indistinguishable from bad luck at the miner. Returns an empty
     * string when the configuration is legal.
     */
    std::string check(uint32_t m, uint32_t n) const {
        const uint32_t r = rank, k = commonDim;
        if (r < kPenaltyBaseRank)
            return "rank below " + std::to_string(kPenaltyBaseRank) +
                   ": blocks are rejected outright";
        if ((r & (r - 1)) || r > 1024 || (r % 16))
            return "rank must be a power of two in [32,1024] and divisible by 16";
        const uint64_t kMax = (4ULL * r * r < 65536ULL) ? 4ULL * r * r : 65536ULL;
        if ((k % 64) || k < 1024 || k < 16u * r || k > kMax)
            return "k must be a multiple of 64, at least max(1024, 16*rank), "
                   "and at most min(2^16, 4*rank^2)";
        if ((tileH % 2) || (tileW % 2) || tileSize() < 32 || tileSize() > 256)
            return "hash tile sides must be even with 32 <= h*w <= 256";
        if (m > (1u << 24) || n > (1u << 24)) return "m and n must be <= 2^24";
        if ((uint32_t)tileH > m || (uint32_t)tileW > n)
            return "the hash tile does not fit inside m x n";
        return std::string();
    }

    /**
     * The bound one transcript is measured against: the block target scaled by
     * the work an attempt costs, normalised to rank 128.
     *
     * Returns false when that scaling overflows 256 bits, which the chain
     * treats as an unusable configuration rather than as "everything wins".
     */
    bool penalizedTarget(const U256 &target, U256 *out) const {
        const uint64_t factor = (uint64_t)tileSize() *
                                (dotProductLength() / rank) * kPenaltyBaseRank;
        if (factor == 0) return false;
        return target.mul(factor, out);
    }
};

// ------------------------------------------------------------ commitments

/** blake3(b"pearl/cert-v3/noise-seed/A"), computed once. */
inline const uint8_t *seedSaltA() {
    static uint8_t s[32];
    static bool done = false;
    if (!done) {
        const char *c = "pearl/cert-v3/noise-seed/A";
        b3::hash(nullptr, (const uint8_t *)c, strlen(c), s);
        done = true;
    }
    return s;
}

inline const uint8_t *seedSaltB() {
    static uint8_t s[32];
    static bool done = false;
    if (!done) {
        const char *c = "pearl/cert-v3/noise-seed/B";
        b3::hash(nullptr, (const uint8_t *)c, strlen(c), s);
        done = true;
    }
    return s;
}

inline void bindRoot(const uint8_t root[32], uint32_t dim, const uint8_t salt[32],
                     uint8_t out[32]) {
    // root(32) || dim u32 LE(4) || zero padding(28) is exactly one blake3 block.
    uint8_t msg[64] = {0};
    memcpy(msg, root, 32);
    msg[32] = (uint8_t)dim;
    msg[33] = (uint8_t)(dim >> 8);
    msg[34] = (uint8_t)(dim >> 16);
    msg[35] = (uint8_t)(dim >> 24);
    b3::hash(salt, msg, sizeof(msg), out);
}

/** job_key = blake3(header || mining_config). 76 + 52 = two blake3 blocks. */
inline void jobKey(const uint8_t header[76], const MiningConfig &cfg,
                   uint8_t out[32]) {
    uint8_t msg[76 + MiningConfig::kSerializedSize];
    memcpy(msg, header, 76);
    cfg.toBytes(msg + 76);
    b3::hash(nullptr, msg, sizeof(msg), out);
}

/**
 * commitment_A and commitment_B from the two matrix roots.
 *
 * commitment_A is both the A-noise seed and the PoW key, which is the single
 * value making every transcript depend on the matrices. `salted` selects the
 * V3 derivation; V1 and V2 skip it, and choosing wrong produces a proof that
 * looks perfectly valid here and is rejected by the node.
 */
inline void commitments(const uint8_t aRoot[32], const uint8_t btRoot[32],
                        const uint8_t key[32], uint32_t m, uint32_t n,
                        bool salted, uint8_t commitA[32], uint8_t commitB[32]) {
    uint8_t a[32], b[32];
    if (salted) {
        bindRoot(aRoot, m, seedSaltA(), a);
        bindRoot(btRoot, n, seedSaltB(), b);
    } else {
        memcpy(a, aRoot, 32);
        memcpy(b, btRoot, 32);
    }
    uint8_t buf[64];
    memcpy(buf, key, 32);
    memcpy(buf + 32, b, 32);
    b3::hash(nullptr, buf, 64, commitB);

    memcpy(buf, commitB, 32);
    memcpy(buf + 32, a, 32);
    b3::hash(nullptr, buf, 64, commitA);
}

// ----------------------------------------------------------- merkle proof

struct MerkleProof {
    std::vector<uint8_t> leafData;      // leafCount * 1024 bytes
    std::vector<uint64_t> leafIndices;
    uint64_t totalLeaves = 0;
    uint8_t root[32] = {0};
    std::vector<uint8_t> siblings;      // 32 bytes each
};

/** Which 1024-byte chunks a set of matrix rows lands in. */
inline std::vector<uint64_t> leafIndicesFromRows(uint64_t firstRow, uint64_t rowCount,
                                                 uint64_t cols) {
    std::vector<uint64_t> out;
    for (uint64_t row = firstRow; row < firstRow + rowCount; row++) {
        const uint64_t first = (row * cols) / kChunkLen;
        const uint64_t last = ((row + 1) * cols - 1) / kChunkLen;
        for (uint64_t i = first; i <= last; i++)
            if (out.empty() || out.back() < i) out.push_back(i);
    }
    return out;
}

/**
 * Build the tree over a zero-padded matrix and open the given leaves.
 *
 * This runs once per win, never in the mining loop, so it is written for
 * clarity: every layer is materialised rather than walking paths. `data` must
 * already be a multiple of 1024 bytes.
 */
inline bool buildProof(const uint8_t *data, size_t len, const uint8_t key[32],
                       const std::vector<uint64_t> &leaves, MerkleProof *out) {
    if (len == 0 || len % kChunkLen || leaves.empty()) return false;
    const size_t total = len / kChunkLen;
    if (leaves.back() >= total) return false;

    std::vector<std::vector<uint8_t> > layers;
    layers.push_back(std::vector<uint8_t>(total * 32));
    for (size_t c = 0; c < total; c++)
        b3::chunkCv(key, data + c * kChunkLen, kChunkLen, c, false,
                    &layers[0][c * 32]);

    while (layers.back().size() / 32 > 2) {
        std::vector<uint8_t> next;
        b3::combineLayer(key, layers.back(), &next);
        layers.push_back(next);
    }
    if (layers.back().size() / 32 == 2) {
        std::vector<uint8_t> root(32);
        b3::parentCv(key, &layers.back()[0], &layers.back()[32], true, &root[0]);
        layers.push_back(root);
    }

    out->leafIndices = leaves;
    out->totalLeaves = total;
    memcpy(out->root, &layers.back()[0], 32);
    out->leafData.resize(leaves.size() * kChunkLen);
    for (size_t i = 0; i < leaves.size(); i++)
        memcpy(&out->leafData[i * kChunkLen], data + leaves[i] * kChunkLen, kChunkLen);

    // Walk up, taking a sibling only when it is not itself being opened. That
    // omission is the whole point of a multi-leaf proof: 32 adjacent leaves out
    // of 4096 cost far fewer siblings than 32 separate paths would.
    out->siblings.clear();
    std::vector<uint64_t> current = leaves;
    size_t levelLen = total;
    size_t level = 0;
    while (levelLen > 1 && !current.empty()) {
        const std::vector<uint8_t> &nodes = layers[level];
        for (size_t at = 0; at < current.size(); at++) {
            const uint64_t i = current[at];
            const bool hasPrev = at > 0 && current[at - 1] == i - 1;
            const bool hasNext = at + 1 < current.size() && current[at + 1] == i + 1;
            if (i % 2 == 1) {
                if (!hasPrev) {
                    const size_t w = out->siblings.size();
                    out->siblings.resize(w + 32);
                    memcpy(&out->siblings[w], &nodes[(i - 1) * 32], 32);
                }
            } else if (!hasNext && (i + 1) < levelLen) {
                const size_t w = out->siblings.size();
                out->siblings.resize(w + 32);
                memcpy(&out->siblings[w], &nodes[(i + 1) * 32], 32);
            }
        }
        std::vector<uint64_t> parents;
        for (size_t at = 0; at < current.size(); at++) {
            const uint64_t p = current[at] / 2;
            if (parents.empty() || parents.back() != p) parents.push_back(p);
        }
        current.swap(parents);
        levelLen = (levelLen + 1) / 2;
        level++;
    }
    return true;
}

// ------------------------------------------------------------ plain proof
//
// bincode with fixint encoding, exactly as PlainProof::to_base64 writes it:
// usize is u64 little-endian, a Vec is its length then its elements, a fixed
// array is raw bytes with no length, and an Option is a single tag byte.

inline void putU64(std::vector<uint8_t> *out, uint64_t v) {
    for (int i = 0; i < 8; i++) out->push_back((uint8_t)(v >> (8 * i)));
}

inline void putBytes(std::vector<uint8_t> *out, const uint8_t *p, size_t n) {
    out->insert(out->end(), p, p + n);
}

inline void encodeMerkleProof(const MerkleProof &p, std::vector<uint8_t> *out) {
    const size_t leaves = p.leafIndices.size();
    putU64(out, leaves);
    for (size_t i = 0; i < leaves; i++) {
        putU64(out, kChunkLen);
        putBytes(out, &p.leafData[i * kChunkLen], kChunkLen);
    }
    putU64(out, leaves);
    for (size_t i = 0; i < leaves; i++) putU64(out, p.leafIndices[i]);
    putU64(out, p.totalLeaves);
    putBytes(out, p.root, 32);
    putU64(out, p.siblings.size() / 32);
    putBytes(out, p.siblings.data(), p.siblings.size());
}

/** The full dense PlainProof a winning tile is submitted as. */
inline std::vector<uint8_t> encodePlainProof(uint64_t m, uint64_t n, uint64_t k,
                                             uint64_t rank,
                                             const MerkleProof &aProof,
                                             uint64_t aFirstRow, uint64_t aRows,
                                             const MerkleProof &btProof,
                                             uint64_t bFirstCol, uint64_t bCols) {
    std::vector<uint8_t> out;
    putU64(&out, m);
    putU64(&out, n);
    putU64(&out, k);
    putU64(&out, rank);

    encodeMerkleProof(aProof, &out);
    putU64(&out, aRows);
    for (uint64_t i = 0; i < aRows; i++) putU64(&out, aFirstRow + i);

    encodeMerkleProof(btProof, &out);
    putU64(&out, bCols);
    for (uint64_t i = 0; i < bCols; i++) putU64(&out, bFirstCol + i);

    out.push_back(0);          // Option<MoEProofParams>::None - a dense proof
    return out;
}

/** Base64, for the gateway's JSON-RPC. */
inline std::string base64(const std::vector<uint8_t> &in) {
    static const char *kAlpha =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
        out += kAlpha[(v >> 18) & 63];
        out += kAlpha[(v >> 12) & 63];
        out += kAlpha[(v >> 6) & 63];
        out += kAlpha[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = (uint32_t)in[i] << 16;
        const bool two = (i + 1 < in.size());
        if (two) v |= (uint32_t)in[i + 1] << 8;
        out += kAlpha[(v >> 18) & 63];
        out += kAlpha[(v >> 12) & 63];
        out += two ? kAlpha[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

inline bool base64Decode(const std::string &in, std::vector<uint8_t> *out) {
    int8_t table[256];
    memset(table, -1, sizeof(table));
    static const char *kAlpha =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) table[(uint8_t)kAlpha[i]] = (int8_t)i;

    out->clear();
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < in.size(); i++) {
        const char c = in[i];
        if (c == '=' || c == '\n' || c == '\r') continue;
        const int8_t d = table[(uint8_t)c];
        if (d < 0) return false;
        acc = (acc << 6) | (uint32_t)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back((uint8_t)(acc >> bits));
        }
    }
    return true;
}

}  // namespace pearl
}  // namespace om
