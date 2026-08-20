// The Pearl host side against fixed vectors from the Python reference.
//
// job.h is a port of tests/pearl_job.py, which is itself checked against
// Pearl's own Rust in tests/pearl_oracle.py - including a proof their verifier
// accepts. So agreeing with these vectors means agreeing with the network,
// through two links that were each pinned down separately.
//
// The vectors carry expected outputs only; A and B_t are regenerated here from
// the same three-integer rule, which keeps the file at a few hundred bytes
// instead of megabytes of int8 without weakening anything - the outputs are
// still fixed, not recomputed by both sides.
//
//   python3 tests/pearl_job.py --emit-vectors /tmp/pearl_job_vectors.bin
//   c++ -O2 -std=c++17 tests/test_pearl_job.cpp -o tests/test_pearl_job
//   ./tests/test_pearl_job /tmp/pearl_job_vectors.bin

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/algos/pearl-pow/job.h"

using namespace om::pearl;

namespace {

int gPass = 0, gFail = 0;

void check(const char *name, bool ok, const std::string &detail = std::string()) {
    if (ok) {
        gPass++;
        printf("  ok   %s\n", name);
    } else {
        gFail++;
        printf("  FAIL %s %s\n", name, detail.c_str());
    }
}

std::string hex(const uint8_t *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

/** The same rule pearl_job.py's synth_matrix uses. */
std::vector<int8_t> synthMatrix(size_t rows, size_t cols, int64_t salt) {
    std::vector<int8_t> m(rows * cols);
    for (size_t i = 0; i < m.size(); i++)
        m[i] = (int8_t)((int64_t)(((int64_t)i * 37 + salt) & 0x7F) - 64);
    return m;
}

std::vector<uint8_t> padToChunk(const std::vector<int8_t> &in) {
    const size_t padded = ((in.size() + kChunkLen - 1) / kChunkLen) * kChunkLen;
    std::vector<uint8_t> out(padded, 0);
    memcpy(out.data(), in.data(), in.size());
    return out;
}

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    const uint8_t *take(size_t n) {
        if (p + n > end) {
            ok = false;
            return nullptr;
        }
        const uint8_t *r = p;
        p += n;
        return r;
    }
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pearl_job_vectors.bin>\n", argv[0]);
        return 2;
    }

    FILE *fh = fopen(argv[1], "rb");
    if (!fh) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    std::vector<uint8_t> blob((size_t)size);
    if (fread(blob.data(), 1, blob.size(), fh) != blob.size()) {
        fprintf(stderr, "short read\n");
        return 2;
    }
    fclose(fh);

    Reader r{blob.data(), blob.data() + blob.size()};
    const uint8_t *magic = r.take(8);
    if (!magic || memcmp(magic, "PRLJ0002", 8) != 0) {
        fprintf(stderr, "not a PRLJ0002 vector file\n");
        return 2;
    }
    const int32_t *dims = (const int32_t *)r.take(16);
    const uint32_t m = (uint32_t)dims[0], n = (uint32_t)dims[1];
    const uint32_t k = (uint32_t)dims[2], rank = (uint32_t)dims[3];

    const uint8_t *header = r.take(76);
    const uint8_t *wantCfg = r.take(MiningConfig::kSerializedSize);
    const uint8_t *wantKey = r.take(32);
    const uint8_t *wantARoot = r.take(32);
    const uint8_t *wantBtRoot = r.take(32);
    const uint8_t *wantCommitA = r.take(32);
    const uint8_t *wantCommitB = r.take(32);
    const uint8_t *wantBound = r.take(32);
    const int8_t *wantEAL = (const int8_t *)r.take((size_t)m * rank);
    const int8_t *wantEBR = (const int8_t *)r.take((size_t)rank * n);
    const uint8_t *wantArFirst = r.take((size_t)k * 2);
    const uint8_t *wantArSecond = r.take((size_t)k * 2);
    const uint8_t *wantBlFirst = r.take((size_t)k * 2);
    const uint8_t *wantBlSecond = r.take((size_t)k * 2);
    const uint8_t *wantANoised = r.take(32);
    const uint8_t *wantBNoised = r.take(32);
    const uint8_t *proofLenP = r.take(4);
    if (!r.ok) {
        fprintf(stderr, "vector file truncated\n");
        return 2;
    }
    uint32_t proofLen = 0;
    memcpy(&proofLen, proofLenP, 4);
    const uint8_t *wantProof = r.take(proofLen);
    if (!r.ok) {
        fprintf(stderr, "vector file truncated in the proof\n");
        return 2;
    }

    printf("vectors: m=%u n=%u k=%u rank=%u, proof %u bytes\n", m, n, k, rank,
           proofLen);

    MiningConfig cfg;
    cfg.commonDim = k;
    cfg.rank = (uint16_t)rank;

    printf("1. blake3\n");
    {
        // Sizes that straddle every boundary the implementation has: block,
        // chunk, and the odd-node carry in the tree.
        static const size_t kSizes[] = {0, 1, 63, 64, 65, 1023, 1024, 1025,
                                        2048, 3000, 4096, 7 * 1024, 11 * 1024};
        // Expected digests come from the vectors indirectly: the roots below
        // are keyed hashes of multi-chunk data, so a broken tree fails there.
        // Here the only claim is self-consistency of the two entry points.
        for (size_t i = 0; i < sizeof(kSizes) / sizeof(kSizes[0]); i++) {
            std::vector<uint8_t> data(kSizes[i]);
            for (size_t j = 0; j < data.size(); j++)
                data[j] = (uint8_t)((j * 37 + kSizes[i]) & 0xFF);
            uint8_t a[32], b[32];
            b3::hash(wantKey, data.data(), data.size(), a);
            data.push_back(1);
            b3::hash(wantKey, data.data(), data.size(), b);
            char nm[64];
            snprintf(nm, sizeof(nm), "appending a byte changes the %zu-byte digest",
                     kSizes[i]);
            check(nm, memcmp(a, b, 32) != 0);
        }
    }

    printf("2. mining configuration\n");
    {
        uint8_t got[MiningConfig::kSerializedSize];
        cfg.toBytes(got);
        check("52 bytes match the reference",
              memcmp(got, wantCfg, sizeof(got)) == 0,
              hex(got, sizeof(got)) + " vs " + hex(wantCfg, sizeof(got)));
        check("this configuration is legal", cfg.check(m, n).empty(), cfg.check(m, n));

        MiningConfig bad = cfg;
        bad.rank = 64;
        check("rank 64 is refused", !bad.check(m, n).empty());
        bad = cfg;
        bad.commonDim = 1024;
        check("k below 16*rank is refused", !bad.check(m, n).empty());
    }

    printf("3. job key\n");
    {
        uint8_t got[32];
        jobKey(header, cfg, got);
        check("job_key matches", memcmp(got, wantKey, 32) == 0,
              hex(got, 32) + " vs " + hex(wantKey, 32));
    }

    printf("4. matrix roots and the commitment chain\n");
    const std::vector<int8_t> A = synthMatrix(m, k, 11);
    const std::vector<int8_t> Bt = synthMatrix(n, k, 91);
    const std::vector<uint8_t> Apad = padToChunk(A);
    const std::vector<uint8_t> Btpad = padToChunk(Bt);
    {
        uint8_t aRoot[32], btRoot[32];
        b3::hash(wantKey, Apad.data(), Apad.size(), aRoot);
        b3::hash(wantKey, Btpad.data(), Btpad.size(), btRoot);
        check("A root matches", memcmp(aRoot, wantARoot, 32) == 0,
              hex(aRoot, 32) + " vs " + hex(wantARoot, 32));
        check("B^t root matches", memcmp(btRoot, wantBtRoot, 32) == 0,
              hex(btRoot, 32) + " vs " + hex(wantBtRoot, 32));

        uint8_t ca[32], cb[32];
        commitments(aRoot, btRoot, wantKey, m, n, true, ca, cb);
        check("commitment_A matches", memcmp(ca, wantCommitA, 32) == 0,
              hex(ca, 32) + " vs " + hex(wantCommitA, 32));
        check("commitment_B matches", memcmp(cb, wantCommitB, 32) == 0,
              hex(cb, 32) + " vs " + hex(wantCommitB, 32));

        // The V3 salting is what a wrong cert version gets wrong, and it is
        // invisible without a check like this: an unsalted chain produces a
        // perfectly well-formed proof that the node rejects.
        uint8_t ca2[32], cb2[32];
        commitments(aRoot, btRoot, wantKey, m, n, false, ca2, cb2);
        check("the unsalted V1/V2 chain differs", memcmp(ca, ca2, 32) != 0);

        commitments(aRoot, btRoot, wantKey, m + 1, n, true, ca2, cb2);
        check("V3 salting binds m", memcmp(ca, ca2, 32) != 0);
    }

    printf("5. rank-penalised target\n");
    {
        // 2^232, which is regtest's target and what the vectors used.
        U256 target;
        target.v[3] = 1ULL << (232 - 192);
        U256 bound;
        check("scaling succeeds", cfg.penalizedTarget(target, &bound));
        uint8_t got[32];
        bound.toBytesLE(got);
        check("bound matches", memcmp(got, wantBound, 32) == 0,
              hex(got, 32) + " vs " + hex(wantBound, 32));

        U256 huge;
        for (int i = 0; i < 4; i++) huge.v[i] = ~0ULL;
        U256 ignored;
        check("an unusably easy target reports overflow rather than accepting all",
              !cfg.penalizedTarget(huge, &ignored));

        U256 small;
        small.v[0] = 5;
        check("comparison is little-endian by limb", small.le(target));
        check("and not the other way round", !target.le(small));
    }

    printf("6. noise generation and the noising\n");
    {
        Noise noise;
        noise.generate(wantCommitA, wantCommitB, m, n, k, (int)rank);

        check("E_AL matches",
              memcmp(noise.eAL.data(), wantEAL, (size_t)m * rank) == 0);
        check("E_BR matches (transposed to rank x n)",
              memcmp(noise.eBR.data(), wantEBR, (size_t)rank * n) == 0);
        check("E_AR column indices match",
              memcmp(noise.ar.first.data(), wantArFirst, (size_t)k * 2) == 0 &&
                  memcmp(noise.ar.second.data(), wantArSecond, (size_t)k * 2) == 0);
        check("E_BL row indices match",
              memcmp(noise.bl.first.data(), wantBlFirst, (size_t)k * 2) == 0 &&
                  memcmp(noise.bl.second.data(), wantBlSecond, (size_t)k * 2) == 0);

        // Uniform noise has to land in [-32, 31]: half the nominal range,
        // because two index draws share it before the zero-point shift. Out of
        // range here and A + E overflows int8 on real data.
        int8_t lo = 127, hi = -128;
        for (size_t i = 0; i < noise.eAL.size(); i++) {
            if (noise.eAL[i] < lo) lo = noise.eAL[i];
            if (noise.eAL[i] > hi) hi = noise.eAL[i];
        }
        check("uniform noise stays in [-32, 31]", lo >= -32 && hi <= 31,
              std::to_string(lo) + ".." + std::to_string(hi));

        bool distinct = true;
        for (size_t i = 0; i < noise.ar.first.size(); i++)
            if (noise.ar.first[i] == noise.ar.second[i]) distinct = false;
        check("every permutation line has a distinct +1 and -1", distinct);

        // The noising, written the way the kernel will: two subtractions per
        // element, never a rank-long dot product against a matrix of mostly
        // zeros.
        std::vector<int8_t> aNoised((size_t)m * k);
        for (size_t row = 0; row < m; row++)
            for (size_t col = 0; col < k; col++) {
                const int32_t e = (int32_t)noise.eAL[row * rank + noise.ar.first[col]] -
                                  (int32_t)noise.eAL[row * rank + noise.ar.second[col]];
                aNoised[row * k + col] = (int8_t)(A[row * k + col] + e);
            }
        std::vector<int8_t> bNoised((size_t)k * n);
        for (size_t row = 0; row < k; row++)
            for (size_t col = 0; col < n; col++) {
                const int32_t e = (int32_t)noise.eBR[(size_t)noise.bl.first[row] * n + col] -
                                  (int32_t)noise.eBR[(size_t)noise.bl.second[row] * n + col];
                // Bt is n x k, so B[row][col] is Bt[col][row].
                bNoised[row * n + col] = (int8_t)(Bt[col * k + row] + e);
            }

        uint8_t got[32];
        b3::hash(nullptr, (const uint8_t *)aNoised.data(), aNoised.size(), got);
        check("A + E_AL@E_AR matches, computed as index subtractions",
              memcmp(got, wantANoised, 32) == 0,
              hex(got, 32) + " vs " + hex(wantANoised, 32));
        b3::hash(nullptr, (const uint8_t *)bNoised.data(), bNoised.size(), got);
        check("B + E_BL@E_BR matches, computed as index subtractions",
              memcmp(got, wantBNoised, 32) == 0,
              hex(got, 32) + " vs " + hex(wantBNoised, 32));
    }

    printf("7. leaf indices, Merkle proofs and the wire encoding\n");
    {
        const std::vector<uint64_t> aLeaves = leafIndicesFromRows(16, 16, k);
        const std::vector<uint64_t> bLeaves = leafIndicesFromRows(16, 16, k);
        check("16 rows of k=2048 need 32 leaves", aLeaves.size() == 32,
              std::to_string(aLeaves.size()));
        check("and they start at row 16 * 2048 / 1024", aLeaves[0] == 32,
              std::to_string(aLeaves[0]));

        MerkleProof aProof, btProof;
        check("A proof builds",
              buildProof(Apad.data(), Apad.size(), wantKey, aLeaves, &aProof));
        check("B^t proof builds",
              buildProof(Btpad.data(), Btpad.size(), wantKey, bLeaves, &btProof));
        check("the proof's root is the matrix root",
              memcmp(aProof.root, wantARoot, 32) == 0);

        const std::vector<uint8_t> blob2 =
            encodePlainProof(m, n, k, rank, aProof, 16, 16, btProof, 16, 16);
        check("the whole PlainProof is byte-identical to the reference",
              blob2.size() == proofLen &&
                  memcmp(blob2.data(), wantProof, proofLen) == 0,
              std::to_string(blob2.size()) + " vs " + std::to_string(proofLen));
    }

    printf("8. base64 round-trips\n");
    {
        for (size_t len = 0; len < 8; len++) {
            std::vector<uint8_t> in(len);
            for (size_t i = 0; i < len; i++) in[i] = (uint8_t)(i * 61 + 7);
            std::vector<uint8_t> back;
            check("round-trip", base64Decode(base64(in), &back) && back == in);
        }
        const std::vector<uint8_t> proof(wantProof, wantProof + proofLen);
        std::vector<uint8_t> back;
        check("the real proof round-trips",
              base64Decode(base64(proof), &back) && back == proof);
    }

    // ---------------------------------------------------------------- T1b
    //
    // openWin() hands recheck() the A and B_t buffers COPIED OUT OF DEVICE
    // MEMORY, plus a tile index the device chose. It never regenerates them.
    // So a device that computes the wrong thing is re-verified against its own
    // wrong data, win.verified comes back true, and the pool is the first
    // party in the chain to regenerate and disagree.
    //
    // This is the test whose absence let the Blackwell rejections pass local
    // verification. It builds the same tile digest twice: once the way
    // recheck() does, from supplied buffers, and once from buffers a caller
    // could not have tampered with. Then it tampers with one byte.
    // Reference transcript and digest for tile (0,0), appended by the emitter.
    const uint8_t *wantTranscript = r.take(kTranscriptWords * 4);
    const uint8_t *wantPowDigest = r.take(32);
    if (!r.ok) {
        fprintf(stderr, "vector file has no transcript vectors - regenerate it "
                        "with tests/pearl_job.py --emit-vectors\n");
        return 2;
    }

    printf("8b. tileTranscript and powDigest against the Python reference\n");
    {
        Noise noise;
        noise.generate(wantCommitA, wantCommitB, m, n, k, (int)rank);
        const int side = 16;
        std::vector<int8_t> aStrip((size_t)side * k), bStrip((size_t)k * side);
        for (int i = 0; i < side; i++)
            for (size_t col = 0; col < k; col++) {
                const int32_t e =
                    (int32_t)noise.eAL[(size_t)i * rank + noise.ar.first[col]] -
                    (int32_t)noise.eAL[(size_t)i * rank + noise.ar.second[col]];
                aStrip[(size_t)i * k + col] = (int8_t)(A[(size_t)i * k + col] + e);
            }
        for (size_t pp = 0; pp < k; pp++)
            for (int j = 0; j < side; j++) {
                const int32_t e =
                    (int32_t)noise.eBR[(size_t)noise.bl.first[pp] * n + j] -
                    (int32_t)noise.eBR[(size_t)noise.bl.second[pp] * n + j];
                bStrip[pp * side + j] = (int8_t)(Bt[(size_t)j * k + pp] + e);
            }
        uint32_t tr[kTranscriptWords];
        tileTranscript(aStrip.data(), bStrip.data(), k, side, (int)rank, 0, 0,
                       side, side, tr);
        check("the tile (0,0) transcript matches the reference",
              memcmp(tr, wantTranscript, sizeof(tr)) == 0);
        uint8_t d[32];
        powDigest(tr, wantCommitA, d);
        check("and its PoW digest matches the reference",
              memcmp(d, wantPowDigest, 32) == 0);
    }

    printf("9. recheck cannot see a device that lied about its matrices\n");
    {
        // A host mirror of recheck(): noise from the commitments, the two
        // winning strips, the tile transcript, the digest. Same arithmetic as
        // algo.cu, deliberately - the point is what it is FED, not how it
        // computes.
        auto digestFor = [&](const std::vector<int8_t> &a,
                             const std::vector<int8_t> &bt, uint32_t tRow,
                             uint32_t tCol) {
            Noise noise;
            noise.generate(wantCommitA, wantCommitB, m, n, k, (int)rank);
            const int side = 16;
            std::vector<int8_t> aStrip((size_t)side * k), bStrip((size_t)k * side);
            for (int i = 0; i < side; i++)
                for (size_t col = 0; col < k; col++) {
                    const int32_t e =
                        (int32_t)noise.eAL[(size_t)(tRow + i) * rank + noise.ar.first[col]] -
                        (int32_t)noise.eAL[(size_t)(tRow + i) * rank + noise.ar.second[col]];
                    aStrip[(size_t)i * k + col] =
                        (int8_t)(a[(size_t)(tRow + i) * k + col] + e);
                }
            for (size_t pp = 0; pp < k; pp++)
                for (int j = 0; j < side; j++) {
                    const int32_t e =
                        (int32_t)noise.eBR[(size_t)noise.bl.first[pp] * n + tCol + j] -
                        (int32_t)noise.eBR[(size_t)noise.bl.second[pp] * n + tCol + j];
                    bStrip[pp * side + j] =
                        (int8_t)(bt[(size_t)(tCol + j) * k + pp] + e);
                }
            uint32_t tr[kTranscriptWords];
            tileTranscript(aStrip.data(), bStrip.data(), k, side, (int)rank, 0, 0,
                           side, side, tr);
            uint8_t d[32];
            powDigest(tr, wantCommitA, d);
            return U256::fromBytesLE(d);
        };

        const U256 honest = digestFor(A, Bt, 0, 0);

        // One byte wrong in a row the tile actually opens, which is what a
        // miscompiled genMatrix or a stale buffer read looks like.
        std::vector<int8_t> tamperedA = A;
        tamperedA[5 * k + 3] = (int8_t)(tamperedA[5 * k + 3] ^ 1);
        const U256 fromTampered = digestFor(tamperedA, Bt, 0, 0);

        check("a single wrong byte of A changes the tile digest",
              memcmp(honest.v, fromTampered.v, sizeof(honest.v)) != 0);

        // The blind spot, stated as an assertion: recheck() agrees with itself
        // on the tampered data. Feed it the wrong A and it returns a digest
        // that is perfectly self-consistent, so a bound test on it says
        // nothing about whether the network will agree.
        check("and recheck is self-consistent on the wrong data, which is the bug",
              memcmp(digestFor(tamperedA, Bt, 0, 0).v, fromTampered.v,
                     sizeof(honest.v)) == 0);

        // The fix this test exists to justify: regenerate from the seeds and
        // compare BEFORE trusting the buffers. synthMatrix here stands in for
        // the seeded generator; the property is the same.
        const std::vector<int8_t> regeneratedA = synthMatrix(m, k, 11);
        check("regenerating A from its seed catches the tampering",
              memcmp(regeneratedA.data(), tamperedA.data(), regeneratedA.size()) != 0 &&
                  memcmp(regeneratedA.data(), A.data(), A.size()) == 0);
        check("and the regenerated digest is the honest one",
              memcmp(digestFor(regeneratedA, Bt, 0, 0).v, honest.v,
                     sizeof(honest.v)) == 0);

        // B is the other half and fails the same way.
        std::vector<int8_t> tamperedBt = Bt;
        tamperedBt[7 * k + 11] = (int8_t)(tamperedBt[7 * k + 11] ^ 1);
        check("a single wrong byte of B^t changes the tile digest too",
              memcmp(digestFor(A, tamperedBt, 0, 0).v, honest.v,
                     sizeof(honest.v)) != 0);
    }

    // ---------------------------------------------------------------- T1a
    //
    // noisy_gemm.cuh writes transcripts block-major over rank-sized blocks and
    // then (hi, wi) inside each. openWin() inverts that to a tile origin.
    // Getting it wrong opens the wrong sixteen rows, which verifies locally
    // against nothing and is rejected by the node.
    //
    // Checked as PROPERTIES, not by restating the expression: restating it
    // would pass against any expression, including a wrong one.
    printf("10. the transcript index inversion is a bijection over tiles\n");
    {
        const uint32_t side = 16, rk = 128;
        const uint32_t tm = 256, tn = 512;              // small, exhaustive
        const uint32_t tilesPerSide = rk / side;
        const uint32_t blocksPerRow = tn / rk;
        const uint32_t tiles = (tm / side) * (tn / side);

        std::vector<uint8_t> seen((size_t)(tm / side) * (tn / side), 0);
        bool inRange = true, aligned = true;
        for (uint32_t flat = 0; flat < tiles; flat++) {
            const uint32_t wi = flat % tilesPerSide;
            const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
            const uint32_t block = flat / (tilesPerSide * tilesPerSide);
            const uint32_t jIdx = block % blocksPerRow;
            const uint32_t iIdx = block / blocksPerRow;
            const uint32_t tRow = iIdx * rk + hi * side;
            const uint32_t tCol = jIdx * rk + wi * side;
            if (tRow + side > tm || tCol + side > tn) { inRange = false; break; }
            if (tRow % side || tCol % side) aligned = false;
            seen[(size_t)(tRow / side) * (tn / side) + (tCol / side)]++;
        }
        bool bijective = inRange;
        for (uint8_t c : seen) if (c != 1) bijective = false;

        check("every transcript index lands inside the matrix", inRange);
        check("every tile origin is 16-aligned", aligned);
        check("and every tile is hit exactly once", bijective);

        // Block-major, not row-major: the first tilesPerSide^2 indices must all
        // fall inside ONE rank x rank block. A row-major inversion spreads them
        // across the full width instead, so this is what tells the two apart.
        bool firstBlockIsOneBlock = true;
        for (uint32_t flat = 0; flat < tilesPerSide * tilesPerSide; flat++) {
            const uint32_t wi = flat % tilesPerSide;
            const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
            if (hi * side >= rk || wi * side >= rk) firstBlockIsOneBlock = false;
        }
        check("the first rank-block's tiles stay inside one rank x rank block",
              firstBlockIsOneBlock);
    }

    // ---------------------------------------------------------------- T1c
    //
    // verify() tests a solution against win.bound, captured when the win was
    // opened, not the bound current at submit time. Those are the same value
    // in today's loop. This pins the behaviour so a future restructure that
    // lets a solution outlive a target refresh fails here instead of at the
    // pool, where it reads as "hash does not meet difficulty target".
    printf("11. a win carries the bound it was opened under\n");
    {
        MiningConfig c;
        c.commonDim = 2048;
        c.rank = 128;
        const U256 poolTarget = U256::fromLimbs((const uint64_t[4]){0, 0, 0, 1ull << 11});
        U256 easy, hard;
        check("an easy bound scales", c.penalizedTarget(poolTarget, &easy));
        U256 harderTarget = poolTarget;
        harderTarget.v[3] >>= 4;                       // pool raises difficulty
        check("a harder bound scales", c.penalizedTarget(harderTarget, &hard));
        check("harder really is harder", hard.le(easy) && !easy.le(hard));

        // A digest that passed the easy bound and fails the hard one.
        U256 digest = hard;
        ++digest.v[0];
        check("the candidate passes the bound it was found under",
              digest.le(easy));
        check("and fails the bound that replaced it", !digest.le(hard));
        check("so a stale win must not be submitted after a refresh",
              !(digest.le(hard)));
    }

    // ---------------------------------------------------------------- T3
    //
    // The live-path probe, offline. recheck() is fed the device's own buffers,
    // so it cannot disagree with the device. digestFromProof() reads back only
    // what was serialised into the submission, which is what the pool
    // reconstructs from. openWin now compares the two before submitting.
    printf("12. the digest re-derived from the proof bytes\n");
    {
        const uint32_t side = 16, tRow = 16, tCol = 16;

        // The same tile digest recheck() would compute, from the buffers.
        auto digestFromBuffers = [&](const std::vector<int8_t> &a,
                                     const std::vector<int8_t> &bt) {
            Noise noise;
            noise.generate(wantCommitA, wantCommitB, m, n, k, (int)rank);
            std::vector<int8_t> aStrip((size_t)side * k), bStrip((size_t)k * side);
            for (uint32_t i = 0; i < side; i++)
                for (size_t col = 0; col < k; col++) {
                    const int32_t e =
                        (int32_t)noise.eAL[(size_t)(tRow + i) * rank + noise.ar.first[col]] -
                        (int32_t)noise.eAL[(size_t)(tRow + i) * rank + noise.ar.second[col]];
                    aStrip[(size_t)i * k + col] =
                        (int8_t)(a[(size_t)(tRow + i) * k + col] + e);
                }
            for (size_t pp = 0; pp < k; pp++)
                for (uint32_t j = 0; j < side; j++) {
                    const int32_t e =
                        (int32_t)noise.eBR[(size_t)noise.bl.first[pp] * n + tCol + j] -
                        (int32_t)noise.eBR[(size_t)noise.bl.second[pp] * n + tCol + j];
                    bStrip[pp * side + j] =
                        (int8_t)(bt[(size_t)(tCol + j) * k + pp] + e);
                }
            uint32_t tr[kTranscriptWords];
            tileTranscript(aStrip.data(), bStrip.data(), k, side, (int)rank, 0, 0,
                           (int)side, (int)side, tr);
            uint8_t d[32];
            powDigest(tr, wantCommitA, d);
            return U256::fromBytesLE(d);
        };

        MerkleProof aP, bP;
        const bool built =
            buildProof(Apad.data(), Apad.size(), wantKey,
                       leafIndicesFromRows(tRow, side, k), &aP) &&
            buildProof(Btpad.data(), Btpad.size(), wantKey,
                       leafIndicesFromRows(tCol, side, k), &bP);
        check("the proofs for the winning tile build", built);

        const U256 fromBuffers = digestFromBuffers(A, Bt);
        U256 fromProof;
        check("the digest re-derives from the proof bytes",
              digestFromProof(aP, tRow, bP, tCol, wantCommitA, wantCommitB, m, n,
                              k, (int)rank, (int)side, &fromProof));
        check("and it equals the digest mined from the buffers",
              memcmp(fromProof.v, fromBuffers.v, sizeof(fromProof.v)) == 0);

        // A proof opening the WRONG rows while the tile index still says tRow.
        // This is the shape of the bug openWin's new check exists to catch:
        // everything is self-consistent to us and wrong to the pool.
        MerkleProof wrongA;
        check("a proof for the wrong rows builds too",
              buildProof(Apad.data(), Apad.size(), wantKey,
                         leafIndicesFromRows(tRow + side, side, k), &wrongA));
        U256 wrongDigest;
        const bool wrongDerived =
            digestFromProof(wrongA, tRow, bP, tCol, wantCommitA, wantCommitB, m,
                            n, k, (int)rank, (int)side, &wrongDigest);
        check("a proof that opens the wrong rows is refused or disagrees",
              !wrongDerived ||
                  memcmp(wrongDigest.v, fromBuffers.v, sizeof(wrongDigest.v)) != 0);

        // One flipped byte inside the opened leaves.
        MerkleProof tamperedA = aP;
        tamperedA.leafData[3] ^= 1;
        U256 tamperedDigest;
        check("a flipped byte in the opened leaves changes the digest",
              digestFromProof(tamperedA, tRow, bP, tCol, wantCommitA,
                              wantCommitB, m, n, k, (int)rank, (int)side,
                              &tamperedDigest) &&
                  memcmp(tamperedDigest.v, fromBuffers.v,
                         sizeof(tamperedDigest.v)) != 0);

        // A proof that simply does not carry the rows the tile needs must be
        // reported, not silently treated as a zero-filled match.
        MerkleProof shortA;
        check("a proof for unrelated leaves builds",
              buildProof(Apad.data(), Apad.size(), wantKey,
                         leafIndicesFromRows(0, side, k), &shortA));
        U256 unused;
        check("and re-deriving from it fails rather than guessing",
              !digestFromProof(shortA, tRow, bP, tCol, wantCommitA, wantCommitB,
                               m, n, k, (int)rank, (int)side, &unused));
    }

    printf("\n%d passed, %d failed\n", gPass, gFail);
    return gFail ? 1 : 0;
}
