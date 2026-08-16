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

#endif  // tensor cores available

}  // namespace pearl
}  // namespace om
