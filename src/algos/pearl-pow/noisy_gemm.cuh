// Pearl NoisyGEMM, CUDA.
//
// Correctness first, speed second. This is the naive form: one block per 16x16
// output tile, one thread per output element, plain int32 MACs. It exists to
// be checked against tests/pearl_reference.py before anything clever happens,
// because a subtly wrong miner finds nothing and reports no error.
//
// The tiling works out unusually well here. Each 16x16 output tile accumulates
// over k independently of every other tile, and the transcript for that tile
// only ever depends on its own running sum. So a tile is a self-contained unit
// of work with no cross-block communication at all.
//
// 16x16 is also exactly the int8 MMA fragment shape on both vendors (M16 N16
// K32 on Ada, M16 N16 K16 on RDNA2), which is what the fast path will use: the
// XOR reduction can run straight on the accumulator fragment in registers so
// the output never reaches memory. That is the whole optimisation, and it is
// deliberately NOT done yet.
//
// The "inner hash" is an XOR reduction, not a hash. blake3 runs once per
// 64-byte transcript and is left to the host in this pass.

#pragma once

#include <stdint.h>

// At file scope on purpose. Including this inside the namespace pulls every
// declaration in it - the warp intrinsics included - into om::pearl and makes
// calls like __shfl_xor_sync ambiguous against the global ones.
#if defined(__CUDACC__)
#include <mma.h>
#endif

namespace om {
namespace pearl {

constexpr int kHashTile = 16;              // both dimensions
constexpr int kTranscriptU32 = 16;         // 64 bytes, one blake3 block
constexpr int kRotation = 13;

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// ------------------------------------------------------------------ blake3
//
// Only the case Pearl needs: a keyed hash of exactly one 64-byte block. The
// transcript is 64 bytes, which is one block of one chunk, so the whole PoW
// check is a SINGLE compression - not a general blake3 implementation, and
// that is why it is cheap enough to sit in the mining loop.
//
// Ported from a pure-Python version checked against the reference blake3
// library on 200 random keyed inputs plus the short-input edge cases, so the
// flag and block-length handling is known good before it reached the device.

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void b3g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                    uint32_t mx, uint32_t my) {
    a += b + mx;  d = rotr32(d ^ a, 16);
    c += d;       b = rotr32(b ^ c, 12);
    a += b + my;  d = rotr32(d ^ a, 8);
    c += d;       b = rotr32(b ^ c, 7);
}

/**
 * Keyed blake3 of one block. `key` and `out` are 8 words, `block` is 16.
 * `blockLen` is the real byte count (64 for a full transcript).
 */
__device__ inline void blake3KeyedBlock(const uint32_t key[8], const uint32_t blockIn[16],
                                        uint32_t blockLen, uint32_t out[8]) {
    const uint32_t IV0 = 0x6A09E667u, IV1 = 0xBB67AE85u, IV2 = 0x3C6EF372u, IV3 = 0xA54FF53Au;
    // KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
    const uint32_t flags = 16u | 1u | 2u | 8u;

    uint32_t s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = key[i];
    s[8] = IV0; s[9] = IV1; s[10] = IV2; s[11] = IV3;
    s[12] = 0u;            // counter low
    s[13] = 0u;            // counter high
    s[14] = blockLen;
    s[15] = flags;

    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = blockIn[i];

    // The permutation is applied with literal indices on an unrolled loop so
    // the message words stay in registers. A table lookup here would be a
    // runtime index into a register array, which spills.
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        b3g(s[0], s[4], s[8],  s[12], m[0],  m[1]);
        b3g(s[1], s[5], s[9],  s[13], m[2],  m[3]);
        b3g(s[2], s[6], s[10], s[14], m[4],  m[5]);
        b3g(s[3], s[7], s[11], s[15], m[6],  m[7]);
        b3g(s[0], s[5], s[10], s[15], m[8],  m[9]);
        b3g(s[1], s[6], s[11], s[12], m[10], m[11]);
        b3g(s[2], s[7], s[8],  s[13], m[12], m[13]);
        b3g(s[3], s[4], s[9],  s[14], m[14], m[15]);
        if (r < 6) {
            const uint32_t t0 = m[2],  t1 = m[6],  t2 = m[3],  t3 = m[10];
            const uint32_t t4 = m[7],  t5 = m[0],  t6 = m[4],  t7 = m[13];
            const uint32_t t8 = m[1],  t9 = m[11], t10 = m[12], t11 = m[5];
            const uint32_t t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
            m[0] = t0;   m[1] = t1;   m[2] = t2;   m[3] = t3;
            m[4] = t4;   m[5] = t5;   m[6] = t6;   m[7] = t7;
            m[8] = t8;   m[9] = t9;   m[10] = t10; m[11] = t11;
            m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

/**
 * Hash each transcript and flag the ones at or below the target.
 *
 * Comparison is little-endian over the 32-byte digest, matching the
 * reference's int.from_bytes(digest, "little") <= pow_target, so the most
 * significant word is the LAST one.
 */
__global__ void powCheck(const uint32_t *__restrict__ transcripts, int count,
                         const uint32_t *__restrict__ key,
                         const uint32_t *__restrict__ target,
                         uint32_t *__restrict__ digests, uint32_t *__restrict__ hits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint32_t k[8], blk[16], out[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) k[j] = key[j];
#pragma unroll
    for (int j = 0; j < 16; ++j) blk[j] = transcripts[i * kTranscriptU32 + j];

    blake3KeyedBlock(k, blk, 64u, out);

    if (digests) {
#pragma unroll
        for (int j = 0; j < 8; ++j) digests[i * 8 + j] = out[j];
    }
    if (hits) {
        // Walk from the most significant limb down; first difference decides.
        int win = 1;
        for (int j = 7; j >= 0; --j) {
            if (out[j] < target[j]) break;
            if (out[j] > target[j]) { win = 0; break; }
        }
        hits[i] = win ? 1u : 0u;
    }
}

/**
 * One block computes one 16x16 output tile of A_noised @ B_noised, folding a
 * transcript as it goes.
 *
 * Launch with 256 threads and grid (n/16, m/16).
 *
 * A and B are row-major int8. cNoised is row-major int32, m x n.
 * transcripts is 16 uint32 per hash tile, ordered to match the layout
 * pearl_reference.py --emit-vectors writes: block-major over (m/rank, n/rank),
 * then (hi, wi) inside each block.
 */
__global__ void noisyGemmTile(const int8_t *__restrict__ A,
                              const int8_t *__restrict__ B,
                              int32_t *__restrict__ cNoised,
                              uint32_t *__restrict__ transcripts,
                              int m, int n, int k, int rank) {
    const int tileRow = blockIdx.y;        // in units of 16 rows
    const int tileCol = blockIdx.x;
    const int row = tileRow * kHashTile + (threadIdx.x / kHashTile);
    const int col = tileCol * kHashTile + (threadIdx.x % kHashTile);
    if (row >= m || col >= n) return;

    __shared__ uint32_t sRed[8];           // one slot per warp
    __shared__ uint32_t sTranscript[kTranscriptU32];
    if (threadIdx.x < kTranscriptU32) sTranscript[threadIdx.x] = 0;
    __syncthreads();

    int32_t acc = 0;
    int reduction = 0;

    for (int p = 0; p < k; p += rank) {
        const int pMax = min(p + rank, k);
        for (int kk = p; kk < pMax; ++kk) {
            acc += static_cast<int32_t>(A[row * k + kk]) *
                   static_cast<int32_t>(B[kk * n + col]);
        }

        // Only a full rank-sized reduction contributes, matching the reference.
        if (pMax - p == rank) {
            // XOR-reduce the tile's 256 accumulators to one uint32. XOR is
            // commutative and associative, so the order here cannot make the
            // device disagree with the reference - any mismatch is a real bug.
            uint32_t v = static_cast<uint32_t>(acc);
            for (int off = 16; off > 0; off >>= 1)
                v ^= __shfl_xor_sync(0xffffffffu, v, off);

            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            if (lane == 0) sRed[warp] = v;
            __syncthreads();

            if (threadIdx.x == 0) {
                uint32_t h = 0;
                for (int w = 0; w < 8; ++w) h ^= sRed[w];
                const int idx = reduction % kTranscriptU32;
                sTranscript[idx] = rotl32(sTranscript[idx], kRotation) ^ h;
            }
            __syncthreads();
            ++reduction;
        }
    }

    cNoised[row * n + col] = acc;

    if (threadIdx.x < kTranscriptU32) {
        // Rebuild the reference's ordering: which rank-block this tile sits in,
        // then its position inside that block.
        const int blocksPerRow = n / rank;
        const int iIdx = (tileRow * kHashTile) / rank;
        const int jIdx = (tileCol * kHashTile) / rank;
        const int hi = ((tileRow * kHashTile) % rank) / kHashTile;
        const int wi = ((tileCol * kHashTile) % rank) / kHashTile;
        const int tilesPerSide = rank / kHashTile;
        const int flat = ((iIdx * blocksPerRow + jIdx) * tilesPerSide + hi) * tilesPerSide + wi;
        transcripts[flat * kTranscriptU32 + threadIdx.x] = sTranscript[threadIdx.x];
    }
}

// ---------------------------------------------------------------- MMA path
//
// One WARP owns one 16x16 output tile, which is exactly the int8 WMMA shape.
// A block of 256 threads therefore covers 8 tiles.
//
// The win is that the XOR reduction runs on the accumulator fragment while it
// is still in registers. C never reaches shared or global memory during
// mining, which is the whole reason this algorithm suits tensor cores.
//
// A pleasant consequence of the reduction being XOR: it is commutative and
// associative, so the mapping of fragment elements to matrix positions does
// not matter and never has to be known. Every thread folds whatever elements
// it happens to hold, the warp folds those together, and the result is the XOR
// over the full tile regardless of layout. A sum would have needed the layout.

#if defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 720)

namespace wmma = nvcuda::wmma;

constexpr int kMmaK = 16;   // int8 WMMA is m16n16k16

/**
 * As noisyGemmTile, but tensor-core accumulated.
 *
 * Launch with 256 threads (8 warps, 8 tiles) and grid
 * ((n/16 * m/16 + 7) / 8). Requires k and rank to be multiples of 16.
 */
__global__ void noisyGemmMma(const int8_t *__restrict__ A,
                             const int8_t *__restrict__ B,
                             int32_t *__restrict__ cNoised,
                             uint32_t *__restrict__ transcripts,
                             int m, int n, int k, int rank, bool writeC) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tilesX = n / kHashTile;
    const int tile = blockIdx.x * (blockDim.x >> 5) + warp;
    if (tile >= tilesX * (m / kHashTile)) return;

    const int tileRow = tile / tilesX;
    const int tileCol = tile % tilesX;
    const int row = tileRow * kHashTile;
    const int col = tileCol * kHashTile;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cFrag;
    wmma::fill_fragment(cFrag, 0);

    // One transcript SLOT per lane rather than a 16-entry array per thread.
    //
    // The obvious form is `uint32_t transcript[16]` indexed by
    // `reduction % 16`, but that index is runtime-variable, which forces the
    // array out of registers into local memory - the same dynamic-indexing
    // trap that cost 75x on the Vulkan side of this repo. Every lane holds an
    // identical value after the butterfly reduction, so lane L can simply own
    // slot L and fold only on the steps that land on it. One register, no
    // array, no indexing.
    uint32_t slot = 0;

    int reduction = 0;
    for (int p = 0; p < k; p += rank) {
        if (k - p < rank) break;                 // partial tiles do not fold
        for (int kk = p; kk < p + rank; kk += kMmaK) {
            wmma::load_matrix_sync(aFrag, A + row * k + kk, k);
            wmma::load_matrix_sync(bFrag, B + kk * n + col, n);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        // Fold straight off the accumulator registers.
        uint32_t v = 0;
#pragma unroll
        for (int i = 0; i < cFrag.num_elements; ++i)
            v ^= static_cast<uint32_t>(cFrag.x[i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v ^= __shfl_xor_sync(0xffffffffu, v, off);

        if (lane == reduction % kTranscriptU32) slot = rotl32(slot, kRotation) ^ v;
        ++reduction;
    }

    if (writeC) {
        // Only for the correctness test; mining never needs C in memory.
        wmma::store_matrix_sync(cNoised + row * n + col, cFrag, n, wmma::mem_row_major);
    }

    if (lane < kTranscriptU32) {
        const int blocksPerRow = n / rank;
        const int iIdx = row / rank;
        const int jIdx = col / rank;
        const int hi = (row % rank) / kHashTile;
        const int wi = (col % rank) / kHashTile;
        const int tilesPerSide = rank / kHashTile;
        const int flat = ((iIdx * blocksPerRow + jIdx) * tilesPerSide + hi) * tilesPerSide + wi;
        transcripts[flat * kTranscriptU32 + lane] = slot;
    }
}

/**
 * As noisyGemmMma, but A is staged through shared memory.
 *
 * Every warp in a block works the same tile ROW, so they all want the identical
 * A fragment and were each fetching it separately. The unstaged kernel moves
 * about 1 GB per 2048^3 iteration, which at its measured runtime is ~3.2 TB/s -
 * far past this card's DRAM bandwidth, so it was being served by L2. Loading A
 * once per block and sharing it across the 8 warps cuts A traffic 8x.
 *
 * Launch with 256 threads and grid (tilesX / warpsPerBlock, tilesY).
 */
__global__ void noisyGemmMmaStaged(const int8_t *__restrict__ A,
                                   const int8_t *__restrict__ B,
                                   int32_t *__restrict__ cNoised,
                                   uint32_t *__restrict__ transcripts,
                                   int m, int n, int k, int rank, bool writeC) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;

    const int tileRow = blockIdx.y;
    const int tileCol = blockIdx.x * warps + warp;
    if (tileCol >= n / kHashTile || tileRow >= m / kHashTile) return;

    const int row = tileRow * kHashTile;
    const int col = tileCol * kHashTile;

    __shared__ int8_t sA[kHashTile * kMmaK];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cFrag;
    wmma::fill_fragment(cFrag, 0);

    uint32_t slot = 0;
    int reduction = 0;

    for (int p = 0; p < k; p += rank) {
        if (k - p < rank) break;
        for (int kk = p; kk < p + rank; kk += kMmaK) {
            // 256 threads, 256 bytes: one each, and each row of 16 is
            // contiguous so a group of 16 lanes makes one 16-byte transaction.
            __syncthreads();
            if (threadIdx.x < kHashTile * kMmaK) {
                const int i = threadIdx.x / kMmaK;
                const int j = threadIdx.x % kMmaK;
                sA[threadIdx.x] = A[(row + i) * k + kk + j];
            }
            __syncthreads();

            wmma::load_matrix_sync(aFrag, sA, kMmaK);
            wmma::load_matrix_sync(bFrag, B + kk * n + col, n);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        uint32_t v = 0;
#pragma unroll
        for (int i = 0; i < cFrag.num_elements; ++i)
            v ^= static_cast<uint32_t>(cFrag.x[i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v ^= __shfl_xor_sync(0xffffffffu, v, off);

        if (lane == reduction % kTranscriptU32) slot = rotl32(slot, kRotation) ^ v;
        ++reduction;
    }

    if (writeC)
        wmma::store_matrix_sync(cNoised + row * n + col, cFrag, n, wmma::mem_row_major);

    if (lane < kTranscriptU32) {
        const int blocksPerRow = n / rank;
        const int iIdx = row / rank;
        const int jIdx = col / rank;
        const int hi = (row % rank) / kHashTile;
        const int wi = (col % rank) / kHashTile;
        const int tilesPerSide = rank / kHashTile;
        const int flat = ((iIdx * blocksPerRow + jIdx) * tilesPerSide + hi) * tilesPerSide + wi;
        transcripts[flat * kTranscriptU32 + lane] = slot;
    }
}

/**
 * As noisyGemmMmaStaged, but each warp owns a 2x2 block of output tiles and
 * both operands come from shared memory.
 *
 * The staged kernel shares A across a block's warps and reads B per warp from
 * global. That leaves B as the whole traffic story: every 16x16 output tile
 * pulls 16 columns of B over the full k, so a 4096x32768 GEMM moves about
 * 16 GB of B - which only runs at all because B_noised at 64 MB still fits the
 * 4090's 72 MB L2. It is why the shape sweep falls off a cliff at 128 MB.
 *
 * Two changes, both aimed at that:
 *
 *   * A block of 8 warps is arranged 2 by 4 over a 64x128 output region, so a
 *     staged A row-strip serves 4 warps and a staged B column-strip serves 2.
 *   * Each warp computes 2x2 tiles, so one pair of A fragments and one pair of
 *     B fragments feeds four mma instructions instead of one feeding one.
 *
 * Together that is four fragment loads per four mmas rather than two per one,
 * and both loads come from shared memory rather than L2.
 *
 * The transcript trick from noisyGemmMma still holds and has to be repeated
 * per tile: four accumulators means four slots, but they are indexed by tile
 * on an unrolled loop rather than by the reduction counter, so they stay in
 * registers. Indexing by the counter is what spilled before.
 *
 * Templated on the warp grid and the per-warp tile count so the shape can be
 * swept rather than guessed - the right answer trades arithmetic intensity
 * against register pressure and is a property of the card, not of the
 * algorithm. Launch with kWarpsM*kWarpsN*32 threads and grid
 * (n / (kWarpsN*kTilesN*16), m / (kWarpsM*kTilesM*16)).
 */
template <int kWarpsM, int kWarpsN, int kTilesM, int kTilesN>
__global__ void noisyGemmMmaTiled(const int8_t *__restrict__ A,
                                  const int8_t *__restrict__ B,
                                  int32_t *__restrict__ cNoised,
                                  uint32_t *__restrict__ transcripts,
                                  int m, int n, int k, int rank, bool writeC) {
    constexpr int kBlockM = kWarpsM * kTilesM * kHashTile;
    constexpr int kBlockN = kWarpsN * kTilesN * kHashTile;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    const int rowBase = blockIdx.y * kBlockM;
    const int colBase = blockIdx.x * kBlockN;
    if (rowBase >= m || colBase >= n) return;

    __shared__ int8_t sA[kBlockM * kMmaK];      // 64 x 16
    __shared__ int8_t sB[kMmaK * kBlockN];      // 16 x 128

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cFrag[kTilesM][kTilesN];
#pragma unroll
    for (int i = 0; i < kTilesM; i++)
#pragma unroll
        for (int j = 0; j < kTilesN; j++) wmma::fill_fragment(cFrag[i][j], 0);

    uint32_t slot[kTilesM][kTilesN] = {};
    int reduction = 0;

    for (int p = 0; p < k; p += rank) {
        if (k - p < rank) break;                 // partial tiles do not fold
        for (int kk = p; kk < p + rank; kk += kMmaK) {
            __syncthreads();
            // Both stages move four bytes per thread per step. Consecutive
            // threads take consecutive four-byte groups of the same row, so a
            // row of the stage is one coalesced transaction.
            constexpr int kAWords = kBlockM * kMmaK / 4;
            constexpr int kBWords = kMmaK * kBlockN / 4;
#pragma unroll
            for (int idx = threadIdx.x; idx < kAWords; idx += kWarpsM * kWarpsN * 32) {
                const int row = idx / (kMmaK / 4);
                const int byte = (idx % (kMmaK / 4)) * 4;
                *(int32_t *)&sA[row * kMmaK + byte] =
                    *(const int32_t *)&A[(size_t)(rowBase + row) * k + kk + byte];
            }
#pragma unroll
            for (int idx = threadIdx.x; idx < kBWords; idx += kWarpsM * kWarpsN * 32) {
                const int row = idx / (kBlockN / 4);
                const int byte = (idx % (kBlockN / 4)) * 4;
                *(int32_t *)&sB[row * kBlockN + byte] =
                    *(const int32_t *)&B[(size_t)(kk + row) * n + colBase + byte];
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aFrag[kTilesM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> bFrag[kTilesN];
#pragma unroll
            for (int i = 0; i < kTilesM; i++) {
                const int r = (warpM * kTilesM + i) * kHashTile;
                wmma::load_matrix_sync(aFrag[i], sA + r * kMmaK, kMmaK);
            }
#pragma unroll
            for (int j = 0; j < kTilesN; j++) {
                const int c = (warpN * kTilesN + j) * kHashTile;
                wmma::load_matrix_sync(bFrag[j], sB + c, kBlockN);
            }
#pragma unroll
            for (int i = 0; i < kTilesM; i++)
#pragma unroll
                for (int j = 0; j < kTilesN; j++)
                    wmma::mma_sync(cFrag[i][j], aFrag[i], bFrag[j], cFrag[i][j]);
        }

        // Fold every tile this warp owns, straight off the accumulators.
#pragma unroll
        for (int i = 0; i < kTilesM; i++) {
#pragma unroll
            for (int j = 0; j < kTilesN; j++) {
                uint32_t v = 0;
#pragma unroll
                for (int e = 0; e < cFrag[i][j].num_elements; e++)
                    v ^= static_cast<uint32_t>(cFrag[i][j].x[e]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v ^= __shfl_xor_sync(0xffffffffu, v, off);
                if (lane == reduction % kTranscriptU32)
                    slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
            }
        }
        ++reduction;
    }

#pragma unroll
    for (int i = 0; i < kTilesM; i++) {
#pragma unroll
        for (int j = 0; j < kTilesN; j++) {
            const int row = rowBase + (warpM * kTilesM + i) * kHashTile;
            const int col = colBase + (warpN * kTilesN + j) * kHashTile;
            if (writeC)
                wmma::store_matrix_sync(cNoised + (size_t)row * n + col, cFrag[i][j],
                                        n, wmma::mem_row_major);
            if (lane < kTranscriptU32) {
                const int blocksPerRow = n / rank;
                const int iIdx = row / rank;
                const int jIdx = col / rank;
                const int hi = (row % rank) / kHashTile;
                const int wi = (col % rank) / kHashTile;
                const int tilesPerSide = rank / kHashTile;
                const int flat =
                    ((iIdx * blocksPerRow + jIdx) * tilesPerSide + hi) * tilesPerSide + wi;
                transcripts[flat * kTranscriptU32 + lane] = slot[i][j];
            }
        }
    }
}

#endif  // tensor cores available

}  // namespace pearl
}  // namespace om
