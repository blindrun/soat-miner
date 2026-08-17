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

/**
 * XOR-reduce a value across the warp.
 *
 * Ampere has this as a single instruction. The butterfly it replaces is five
 * shuffles, and the fold runs once per rank step for EVERY tile a warp owns -
 * with 4x4 tiles that is eighty shuffles per hundred and twenty-eight mma, on
 * the same ALU pipeline the profiler showed as the highest-utilised at 37%.
 *
 * The fallback is the original butterfly, so pre-Ampere is unaffected.
 */
/**
 * XOR three values in one instruction.
 *
 * The fold reduces eight accumulator registers per tile, which is seven
 * two-input XORs. The tensor pipeline tops out at 39% here and the arithmetic
 * says why: eight mma against about eleven ALU ops per tile per rank step caps
 * tensor utilisation near 8/19. LOP3 computes an arbitrary three-input boolean
 * function in one op, so the same eight values fold in four instructions
 * instead of seven, and the cap moves rather than the clock speed.
 *
 * 0x96 is the immLut for a ^ b ^ c.
 */
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    return a ^ b ^ c;
#endif
}

__device__ __forceinline__ uint32_t warpXor(uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_xor_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v ^= __shfl_xor_sync(0xffffffffu, v, off);
    return v;
#endif
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
        v = warpXor(v);

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
        v = warpXor(v);

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
                v = warpXor(v);
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

/**
 * As noisyGemmMmaTiled, but the shared stage is double buffered.
 *
 * The single-buffered kernel spends every k-step doing this:
 *
 *     __syncthreads(); load global -> shared; __syncthreads(); mma
 *
 * The tensor cores are idle across BOTH barriers and for the whole global
 * load, which at 18-24% of peak is where a lot of the missing time goes. With
 * two buffers the loads for step i+1 are issued before the mma for step i and
 * land in the buffer nobody is reading, so a step becomes:
 *
 *     issue global loads -> registers; mma from buf[i&1];
 *     store registers -> buf[(i+1)&1]; __syncthreads()
 *
 * One barrier per step instead of two, and the global latency sits underneath
 * the mma instead of in front of it. The store targets the buffer the compute
 * is not reading, and by the time that buffer is read again two barriers have
 * passed, so no extra synchronisation is needed.
 *
 * Costs two things: shared memory doubles (still only 4-12 KB), and a handful
 * of registers hold the in-flight slice. Both are cheap here - occupancy is
 * not what limits this kernel.
 */
template <int kWarpsM, int kWarpsN, int kTilesM, int kTilesN>
__global__ void noisyGemmMmaTiledDB(const int8_t *__restrict__ A,
                                    const int8_t *__restrict__ B,
                                    int32_t *__restrict__ cNoised,
                                    uint32_t *__restrict__ transcripts,
                                    int m, int n, int k, int rank, bool writeC) {
    constexpr int kBlockM = kWarpsM * kTilesM * kHashTile;
    constexpr int kBlockN = kWarpsN * kTilesN * kHashTile;
    constexpr int kThreads = kWarpsM * kWarpsN * 32;
    constexpr int kAWords = kBlockM * kMmaK / 4;
    constexpr int kBWords = kMmaK * kBlockN / 4;
    // Ceiling division: with an odd word count some threads carry one fewer.
    constexpr int kARegs = (kAWords + kThreads - 1) / kThreads;
    constexpr int kBRegs = (kBWords + kThreads - 1) / kThreads;
    // Same bank-conflict padding as the cp.async kernel - see there for why a
    // 128 or 256 byte row makes every row of a fragment load hit one bank set.
    constexpr int kBStride = kBlockN + 16;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    const int rowBase = blockIdx.y * kBlockM;
    const int colBase = blockIdx.x * kBlockN;
    if (rowBase >= m || colBase >= n) return;

    __shared__ int8_t sA[2][kBlockM * kMmaK];
    __shared__ int8_t sB[2][kMmaK * kBStride];

    int32_t regA[kARegs], regB[kBRegs];

    // Only whole rank blocks fold, so the step count is trimmed the same way
    // the single-buffered loop's `if (k - p < rank) break` trims it.
    const int dotLen = k - k % rank;
    const int totalSteps = dotLen / kMmaK;
    const int stepsPerRank = rank / kMmaK;
    if (totalSteps <= 0) return;

#define OM_PEARL_LOAD(step)                                                    \
    {                                                                          \
        const int kk_ = (step) * kMmaK;                                        \
        _Pragma("unroll") for (int r = 0; r < kARegs; r++) {                    \
            const int idx = threadIdx.x + r * kThreads;                        \
            if (idx < kAWords) {                                               \
                const int row = idx / (kMmaK / 4);                             \
                const int byte = (idx % (kMmaK / 4)) * 4;                      \
                regA[r] = *(const int32_t *)&A[(size_t)(rowBase + row) * k +   \
                                               kk_ + byte];                    \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll") for (int r = 0; r < kBRegs; r++) {                    \
            const int idx = threadIdx.x + r * kThreads;                        \
            if (idx < kBWords) {                                               \
                const int row = idx / (kBlockN / 4);                           \
                const int byte = (idx % (kBlockN / 4)) * 4;                    \
                regB[r] = *(const int32_t *)&B[(size_t)(kk_ + row) * n +       \
                                               colBase + byte];                \
            }                                                                  \
        }                                                                      \
    }

#define OM_PEARL_STORE(buf)                                                    \
    {                                                                          \
        _Pragma("unroll") for (int r = 0; r < kARegs; r++) {                    \
            const int idx = threadIdx.x + r * kThreads;                        \
            if (idx < kAWords) {                                               \
                const int row = idx / (kMmaK / 4);                             \
                const int byte = (idx % (kMmaK / 4)) * 4;                      \
                *(int32_t *)&sA[buf][row * kMmaK + byte] = regA[r];            \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll") for (int r = 0; r < kBRegs; r++) {                    \
            const int idx = threadIdx.x + r * kThreads;                        \
            if (idx < kBWords) {                                               \
                const int row = idx / (kBlockN / 4);                           \
                const int byte = (idx % (kBlockN / 4)) * 4;                    \
                *(int32_t *)&sB[buf][row * kBStride + byte] = regB[r];          \
            }                                                                  \
        }                                                                      \
    }

    OM_PEARL_LOAD(0)
    OM_PEARL_STORE(0)
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cFrag[kTilesM][kTilesN];
#pragma unroll
    for (int i = 0; i < kTilesM; i++)
#pragma unroll
        for (int j = 0; j < kTilesN; j++) wmma::fill_fragment(cFrag[i][j], 0);

    uint32_t slot[kTilesM][kTilesN] = {};
    int reduction = 0;

    for (int step = 0; step < totalSteps; step++) {
        const int buf = step & 1;
        // Issued before the mma so the memory latency hides under the tensor
        // cores rather than in front of them.
        if (step + 1 < totalSteps) OM_PEARL_LOAD(step + 1)

        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aFrag[kTilesM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> bFrag[kTilesN];
#pragma unroll
        for (int i = 0; i < kTilesM; i++) {
            const int r = (warpM * kTilesM + i) * kHashTile;
            wmma::load_matrix_sync(aFrag[i], sA[buf] + r * kMmaK, kMmaK);
        }
#pragma unroll
        for (int j = 0; j < kTilesN; j++) {
            const int c = (warpN * kTilesN + j) * kHashTile;
            wmma::load_matrix_sync(bFrag[j], sB[buf] + c, kBStride);
        }
#pragma unroll
        for (int i = 0; i < kTilesM; i++)
#pragma unroll
            for (int j = 0; j < kTilesN; j++)
                wmma::mma_sync(cFrag[i][j], aFrag[i], bFrag[j], cFrag[i][j]);

        // Written into the buffer the compute above did not read.
        if (step + 1 < totalSteps) OM_PEARL_STORE((step + 1) & 1)
        __syncthreads();

        if ((step + 1) % stepsPerRank == 0) {
#pragma unroll
            for (int i = 0; i < kTilesM; i++) {
#pragma unroll
                for (int j = 0; j < kTilesN; j++) {
                    uint32_t v = 0;
#pragma unroll
                    for (int e = 0; e < cFrag[i][j].num_elements; e++)
                        v ^= static_cast<uint32_t>(cFrag[i][j].x[e]);
                    v = warpXor(v);
                    if (lane == reduction % kTranscriptU32)
                        slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
                }
            }
            ++reduction;
        }
    }
#undef OM_PEARL_LOAD
#undef OM_PEARL_STORE

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

// ------------------------------------------------------------- cp.async
//
// Ampere added an instruction that copies global memory straight into shared
// without landing in registers first, and without occupying the thread while
// it is in flight. The double-buffered kernel above still pays for the trip:
// it loads to registers, then stores to shared, and both halves are ordinary
// instructions competing with the mma for issue slots.
//
// With cp.async the copy is fire-and-forget, so the pipeline gets deeper than
// two stages is worth. THREE stages is what removes the second barrier: with
// two buffers, the copy issued at step s+1 writes the buffer step s just read
// from, and nothing separates them; with three, the buffer being written was
// last read at s-1 and the barrier at s already stands between. One barrier
// per step and two copies always in flight.

#include <cuda_pipeline.h>

/**
 * As noisyGemmMmaTiledDB, but the stage is filled with cp.async and three
 * buffers deep.
 *
 * Copies are 16 bytes per thread, which is the widest cp.async takes and the
 * only width that can bypass L1. Alignment works out for free: a row of A is
 * exactly kMmaK = 16 bytes, k and n are multiples of 16, and cudaMalloc
 * returns 256-byte-aligned pointers, so every source address is 16-byte
 * aligned without any padding games.
 */
template <int kWarpsM, int kWarpsN, int kTilesM, int kTilesN>
__global__ void noisyGemmMmaTiledAsync(const int8_t *__restrict__ A,
                                       const int8_t *__restrict__ B,
                                       int32_t *__restrict__ cNoised,
                                       uint32_t *__restrict__ transcripts,
                                       int m, int n, int k, int rank,
                                       bool writeC) {
    constexpr int kBlockM = kWarpsM * kTilesM * kHashTile;
    constexpr int kBlockN = kWarpsN * kTilesN * kHashTile;
    constexpr int kThreads = kWarpsM * kWarpsN * 32;
    constexpr int kStages = 4;   // deeper lookahead; see the sweep in test_pearl_prepare
    constexpr int kACopies = kBlockM;                  // one 16-byte row each
    constexpr int kBCopies = kMmaK * (kBlockN / 16);

    // Pad B's row stride by one 16-byte group.
    //
    // Shared memory is 32 banks of 4 bytes, so a 128-byte row wraps the banks
    // exactly once and a 256-byte row exactly twice. Either way every row of a
    // fragment load lands on the SAME banks, and a 16-row load serialises into
    // sixteen conflicting accesses. The profiler saw it as L1/TEX throughput
    // at 71.6% with a 0.26% hit rate while DRAM sat at 3.4% - the kernel was
    // starved by its own shared-memory layout, not by memory bandwidth.
    //
    // Sixteen bytes of padding rotates each row by four banks and costs a few
    // hundred bytes of shared per stage.
    constexpr int kBStride = kBlockN + 16;
    constexpr int kAStride = kMmaK;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    const int rowBase = blockIdx.y * kBlockM;
    const int colBase = blockIdx.x * kBlockN;
    if (rowBase >= m || colBase >= n) return;

    __shared__ __align__(16) int8_t sA[kStages][kBlockM * kAStride];
    __shared__ __align__(16) int8_t sB[kStages][kMmaK * kBStride];

    const int dotLen = k - k % rank;
    const int totalSteps = dotLen / kMmaK;
    const int stepsPerRank = rank / kMmaK;
    if (totalSteps <= 0) return;

// Ampere and later get the real thing. Everything older gets a plain 16-byte
// store, which makes the kernel slower but not wrong - important, because the
// runtime picker measures whatever it can launch, and a kernel that silently
// did nothing on an old card would look like the fastest of the lot.
#if __CUDA_ARCH__ >= 800
#define OM_PEARL_CP(dst, src) __pipeline_memcpy_async((dst), (src), 16)
#define OM_PEARL_COMMIT() __pipeline_commit()
#define OM_PEARL_WAIT(n) __pipeline_wait_prior(n)
#else
#define OM_PEARL_CP(dst, src) \
    (*(int4 *)(dst) = *(const int4 *)(src))
#define OM_PEARL_COMMIT() ((void)0)
#define OM_PEARL_WAIT(n) ((void)0)
#endif

#define OM_PEARL_ISSUE(step, stage)                                            \
    {                                                                          \
        const int kk_ = (step) * kMmaK;                                        \
        for (int i = threadIdx.x; i < kACopies; i += kThreads) {               \
            OM_PEARL_CP(&sA[stage][i * kAStride],                               \
                        &A[(size_t)(rowBase + i) * k + kk_]);                  \
        }                                                                      \
        for (int i = threadIdx.x; i < kBCopies; i += kThreads) {               \
            const int row = i / (kBlockN / 16);                                \
            const int col = (i % (kBlockN / 16)) * 16;                         \
            OM_PEARL_CP(&sB[stage][row * kBStride + col],                       \
                        &B[(size_t)(kk_ + row) * n + colBase + col]);          \
        }                                                                      \
        OM_PEARL_COMMIT();                                                     \
    }

    // Fill the pipeline before entering the loop: two stages in flight.
#pragma unroll
    for (int pre = 0; pre < kStages - 1; pre++)
        if (pre < totalSteps) OM_PEARL_ISSUE(pre, pre)

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cFrag[kTilesM][kTilesN];
#pragma unroll
    for (int i = 0; i < kTilesM; i++)
#pragma unroll
        for (int j = 0; j < kTilesN; j++) wmma::fill_fragment(cFrag[i][j], 0);

    uint32_t slot[kTilesM][kTilesN] = {};
    int reduction = 0;

    for (int step = 0; step < totalSteps; step++) {
        const int stage = step % kStages;

        // Wait for THIS step's stage to land - one commit may still be in
        // flight - then barrier.
        //
        // The barrier has to come before the next issue, not after. The stage
        // about to be written, (step+kStages-1) % kStages, is the one the
        // PREVIOUS step read from, and without a barrier between that read and
        // this write they race. An earlier version issued first and passed the
        // correctness test anyway, because at k=256 there are only sixteen
        // steps and the timing never exposed it - exactly the kind of race
        // that ships.
        OM_PEARL_WAIT(kStages - 2);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aFrag[kTilesM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> bFrag[kTilesN];
#pragma unroll
        for (int i = 0; i < kTilesM; i++) {
            const int r = (warpM * kTilesM + i) * kHashTile;
            wmma::load_matrix_sync(aFrag[i], sA[stage] + r * kAStride, kAStride);
        }
#pragma unroll
        for (int j = 0; j < kTilesN; j++) {
            const int c = (warpN * kTilesN + j) * kHashTile;
            wmma::load_matrix_sync(bFrag[j], sB[stage] + c, kBStride);
        }
#pragma unroll
        for (int i = 0; i < kTilesM; i++)
#pragma unroll
            for (int j = 0; j < kTilesN; j++)
                wmma::mma_sync(cFrag[i][j], aFrag[i], bFrag[j], cFrag[i][j]);

        if ((step + 1) % stepsPerRank == 0) {
#pragma unroll
            for (int i = 0; i < kTilesM; i++) {
#pragma unroll
                for (int j = 0; j < kTilesN; j++) {
                    uint32_t v = 0;
#pragma unroll
                    for (int e = 0; e < cFrag[i][j].num_elements; e++)
                        v ^= static_cast<uint32_t>(cFrag[i][j].x[e]);
                    v = warpXor(v);
                    if (lane == reduction % kTranscriptU32)
                        slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
                }
            }
            ++reduction;
        }

        // Issued after the barrier and after this step's reads are underway,
        // into the stage nobody is reading now.
        if (step + kStages - 1 < totalSteps)
            OM_PEARL_ISSUE(step + kStages - 1, (step + kStages - 1) % kStages)
    }
#undef OM_PEARL_ISSUE
#undef OM_PEARL_CP
#undef OM_PEARL_COMMIT
#undef OM_PEARL_WAIT

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

// ------------------------------------------------------------- raw mma PTX
//
// WMMA's int8 fragment shape is m16n16k16. Ampere's actual instruction is
// m16n8k32, so every wmma::mma_sync becomes several of them and the k-loop
// runs twice as many iterations as it needs to. The profiler put the tensor
// pipeline at 37.4% - the highest of any pipeline, but far from saturated -
// with the rest going on fragment loads and loop overhead.
//
// This issues m16n8k32 directly. Half the k-steps for the same work, one
// instruction per half-tile, and the register layout is ours to arrange rather
// than the fragment API's.
//
// **B is read n-major here.** The mma's B fragment holds four CONSECUTIVE k
// for one n, which is a four-byte contiguous load out of an n-major tile and
// four separate strided bytes out of a k-major one. So this kernel takes
// B_noised from applyNoiseBt, not applyNoiseB - which is also the cheaper
// noising kernel, because B^t already arrives n-major and nothing has to
// transpose.
//
// Layouts, from the PTX ISA for mma.m16n8k32.row.col.s32.s8.s8.s32. Per thread
// t of a warp:
//
//   A (16x32): a0 = row t/4,   bytes (t%4)*4 .. +3
//              a1 = row t/4+8, bytes (t%4)*4 .. +3
//              a2 = row t/4,   bytes 16+(t%4)*4 .. +3
//              a3 = row t/4+8, bytes 16+(t%4)*4 .. +3
//   B (32x8):  b0 = col t%4,   k = (t/4)*4 .. +3
//              b1 = col t%4,   k = 16+(t/4)*4 .. +3
//   D (16x8):  d0,d1 = row t/4,   cols (t%4)*2, +1
//              d2,d3 = row t/4+8, cols (t%4)*2, +1

// Defined for every device pass so the host launcher links on a multi-arch
// build; mmaS8 itself falls back to a no-op below sm_80 and the host tuner
// refuses to select this kernel there (TileConfig::minArch).
__device__ __forceinline__ void mmaS8(uint32_t d[4], const uint32_t a[4],
                                      const uint32_t b[2], const uint32_t c[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    (void)a; (void)b;
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

/**
 * Raw-PTX NoisyGEMM. One 16x16 output tile is two m16n8k32 accumulators.
 *
 * Requires B in n-major order (applyNoiseBt). Launch with
 * kWarpsM*kWarpsN*32 threads and grid (n/kBlockN, m/kBlockM).
 */
template <int kWarpsM, int kWarpsN, int kTilesM, int kTilesN>
__global__ void noisyGemmPtx(const int8_t *__restrict__ A,
                             const int8_t *__restrict__ Bn,
                             int32_t *__restrict__ cNoised,
                             uint32_t *__restrict__ transcripts,
                             int m, int n, int k, int rank, bool writeC) {
    constexpr int kK32 = 32;                       // the instruction's k
    constexpr int kBlockM = kWarpsM * kTilesM * kHashTile;
    constexpr int kBlockN = kWarpsN * kTilesN * kHashTile;
    constexpr int kThreads = kWarpsM * kWarpsN * 32;
    constexpr int kStages = 3;
    constexpr int kAStride = kK32;                 // 32 bytes per row
    constexpr int kBStride = kK32 + 16;            // n-major, padded off a wrap

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    // Blocks are scheduled roughly in linear order, and the natural order is
    // the wrong one here. With grid (n/kBlockN, m/kBlockM), consecutive blocks
    // step along n, so a wave of concurrent blocks shares ONE A tile and pulls
    // a DIFFERENT B tile each - and B is the bigger of the two (kBlockN vs
    // kBlockM rows of k bytes). The profiler shows the cost: L2 throughput 88%
    // at a 98% hit rate with DRAM at 4.7%, which is the same bytes being
    // served out of L2 over and over.
    //
    // So walk down m in groups of kGroupM first. Within a group, kGroupM
    // consecutive blocks share one B tile, and a wave's distinct footprint
    // becomes one B tile plus kGroupM A tiles rather than the reverse.
    constexpr int kGroupM = 8;
    const int blocksN = gridDim.x, blocksM = gridDim.y;
    const int linear = blockIdx.x + blockIdx.y * blocksN;
    const int perGroup = kGroupM * blocksN;
    const int mFirst = (linear / perGroup) * kGroupM;
    const int inGroup = linear % perGroup;
    // The last group is short whenever blocksM is not a multiple of kGroupM.
    // Dividing by kGroupM there would alias two blocks onto one tile and leave
    // another tile never computed - a wrong product, not a slow one.
    const int groupRows = min(kGroupM, blocksM - mFirst);
    const int bIdxM = mFirst + inGroup % groupRows;
    const int bIdxN = inGroup / groupRows;

    const int rowBase = bIdxM * kBlockM;
    const int colBase = bIdxN * kBlockN;
    if (rowBase >= m || colBase >= n) return;

    __shared__ __align__(16) int8_t sA[kStages][kBlockM * kAStride];
    __shared__ __align__(16) int8_t sB[kStages][kBlockN * kBStride];

    const int dotLen = k - k % rank;
    const int totalSteps = dotLen / kK32;
    const int stepsPerRank = rank / kK32;
    if (totalSteps <= 0 || stepsPerRank <= 0) return;

    // Two accumulators per 16x16 tile: columns 0-7 and 8-15.
    uint32_t acc[kTilesM][kTilesN][2][4] = {};
    uint32_t slot[kTilesM][kTilesN] = {};
    int reduction = 0;

#define OM_PTX_ISSUE(step, stage)                                              \
    {                                                                          \
        const int kk_ = (step) * kK32;                                         \
        for (int i = threadIdx.x; i < kBlockM * 2; i += kThreads) {            \
            const int row = i >> 1;                                            \
            const int half = (i & 1) * 16;                                     \
            __pipeline_memcpy_async(&sA[stage][row * kAStride + half],         \
                                    &A[(size_t)(rowBase + row) * k + kk_ + half], 16); \
        }                                                                      \
        for (int i = threadIdx.x; i < kBlockN * 2; i += kThreads) {            \
            const int col = i >> 1;                                            \
            const int half = (i & 1) * 16;                                     \
            __pipeline_memcpy_async(&sB[stage][col * kBStride + half],         \
                                    &Bn[(size_t)(colBase + col) * k + kk_ + half], 16); \
        }                                                                      \
        __pipeline_commit();                                                   \
    }

#pragma unroll
    for (int pre = 0; pre < kStages - 1; pre++)
        if (pre < totalSteps) OM_PTX_ISSUE(pre, pre)

    // Counters, not modulo. stepsPerRank is a runtime value, so `step %
    // stepsPerRank` compiles to a full signed integer division - IMAD.HI,
    // IABS, SHF, ISETP, SEL, about twenty instructions - and it sat in the
    // hot loop executing once per k-slice. The SASS mix was dominated by
    // division sequences rather than by the fold or the mma.
    int stage = 0;                       // step % kStages
    int issueStage = (kStages - 1) % kStages;
    int sinceFold = 0;                   // counts up to stepsPerRank
    for (int step = 0; step < totalSteps; step++) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();

        uint32_t aReg[kTilesM][4], bReg[kTilesN][2][2];
#pragma unroll
        for (int i = 0; i < kTilesM; i++) {
            const int r0 = (warpM * kTilesM + i) * kHashTile + (lane >> 2);
            const int byte = (lane & 3) * 4;
            const int8_t *p = &sA[stage][r0 * kAStride + byte];
            aReg[i][0] = *(const uint32_t *)(p);
            aReg[i][1] = *(const uint32_t *)(p + 8 * kAStride);
            aReg[i][2] = *(const uint32_t *)(p + 16);
            aReg[i][3] = *(const uint32_t *)(p + 8 * kAStride + 16);
        }
#pragma unroll
        for (int j = 0; j < kTilesN; j++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {          // the tile's two n-halves
                // b0 and b1 differ by COLUMN, not by k: the fragment is eight
                // columns wide and threadID_in_group only spans four, so the
                // second register picks up columns 4-7. Getting this wrong
                // reads the right bytes in the wrong places and produces a
                // plausible, entirely incorrect product.
                // groupID picks the COLUMN and threadID_in_group picks k -
                // the opposite of A, and the opposite of what the operand
                // order suggests. Established by running a single mma against
                // a CPU product (scratchpad/mmaprobe.cu) rather than reasoning
                // about it; three plausible readings were all wrong.
                const int col = (warpN * kTilesN + j) * kHashTile + h * 8 + (lane >> 2);
                const int kOff = (lane & 3) * 4;
                const int8_t *pb = &sB[stage][col * kBStride + kOff];
                bReg[j][h][0] = *(const uint32_t *)(pb);
                bReg[j][h][1] = *(const uint32_t *)(pb + 16);
            }
        }
#pragma unroll
        for (int i = 0; i < kTilesM; i++)
#pragma unroll
            for (int j = 0; j < kTilesN; j++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                    mmaS8(acc[i][j][h], aReg[i], bReg[j][h], acc[i][j][h]);

        if (++sinceFold == stepsPerRank) {
            sinceFold = 0;
#pragma unroll
            for (int i = 0; i < kTilesM; i++) {
#pragma unroll
                for (int j = 0; j < kTilesN; j++) {
                    // Four instructions, not seven. The shape is fixed - two
                    // n-halves of four accumulator registers - so this is
                    // written out rather than looped.
                    uint32_t v = warpXor(
                        xor3(xor3(acc[i][j][0][0], acc[i][j][0][1], acc[i][j][0][2]),
                             xor3(acc[i][j][0][3], acc[i][j][1][0], acc[i][j][1][1]),
                             acc[i][j][1][2] ^ acc[i][j][1][3]));
                    if (lane == reduction % kTranscriptU32)
                        slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
                }
            }
            ++reduction;
        }

        if (step + kStages - 1 < totalSteps)
            OM_PTX_ISSUE(step + kStages - 1, issueStage)

        if (++stage == kStages) stage = 0;
        if (++issueStage == kStages) issueStage = 0;
    }
#undef OM_PTX_ISSUE

#pragma unroll
    for (int i = 0; i < kTilesM; i++) {
#pragma unroll
        for (int j = 0; j < kTilesN; j++) {
            const int row = rowBase + (warpM * kTilesM + i) * kHashTile;
            const int col = colBase + (warpN * kTilesN + j) * kHashTile;
            if (writeC) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = row + (lane >> 2);
                    const int c = col + h * 8 + (lane & 3) * 2;
                    cNoised[(size_t)r * n + c] = (int32_t)acc[i][j][h][0];
                    cNoised[(size_t)r * n + c + 1] = (int32_t)acc[i][j][h][1];
                    cNoised[(size_t)(r + 8) * n + c] = (int32_t)acc[i][j][h][2];
                    cNoised[(size_t)(r + 8) * n + c + 1] = (int32_t)acc[i][j][h][3];
                }
            }
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
