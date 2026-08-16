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

}  // namespace pearl
}  // namespace om
