// The per-nonce hit computation and the search kernel.

#pragma once

#include "autolykos.cuh"

/** A candidate the kernel found; the host re-verifies before submitting. */
struct AlSolution {
    uint64_t nonce;
    uint64_t hit[4];  // limb[0] least significant
};

/** 256-bit compare: a < b, limbs little-endian order. */
__device__ __forceinline__ bool al_lt256(const uint64_t a[4], const uint64_t b[4]) {
#pragma unroll
    for (int i = 3; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;  // equal is not "less than"
}

/** acc += v, 256-bit with carry propagation. */
__device__ __forceinline__ void al_add256(uint64_t acc[4], const uint64_t v[4]) {
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t s = acc[i] + v[i];
        uint64_t c1 = (s < acc[i]) ? 1ULL : 0ULL;
        uint64_t s2 = s + carry;
        uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
        acc[i] = s2;
        carry = c1 + c2;
    }
}

/** Writes a 256-bit value as 32 big-endian bytes. */
__device__ __forceinline__ void al_to_be32(const uint64_t v[4], uint8_t out[32]) {
#pragma unroll
    for (int l = 0; l < 4; l++) {
        uint64_t x = v[3 - l];
#pragma unroll
        for (int j = 0; j < 8; j++) out[l * 8 + j] = (uint8_t)(x >> (8 * (7 - j)));
    }
}

/**
 * The 31 bytes of a dataset element, big-endian.
 * limb[3]'s low 56 bits are the top 7 bytes (the dropped byte is always 0).
 */
__device__ __forceinline__ void al_element_bytes31(const uint64_t limb[4],
                                                   uint8_t out[31]) {
#pragma unroll
    for (int j = 0; j < 7; j++) out[j] = (uint8_t)(limb[3] >> (8 * (6 - j)));
#pragma unroll
    for (int l = 0; l < 3; l++) {
        uint64_t x = limb[2 - l];
#pragma unroll
        for (int j = 0; j < 8; j++) out[7 + l * 8 + j] = (uint8_t)(x >> (8 * (7 - j)));
    }
}

/**
 * Computes the Autolykos v2 hit for one nonce.
 *
 * `msg` is the 32-byte header digest the pool or node hands out; `nonce` is
 * written big-endian, matching the wire format of a submitted solution.
 */
__device__ __forceinline__ void al_hit(const uint4 *dataset, const uint8_t msg[32],
                                       uint64_t nonce, uint32_t N, uint32_t height,
                                       uint64_t out_hit[4]) {
    uint8_t buf[71];

    // --- prei8 = H(msg || nonce)[24..32] ----------------------------------
#pragma unroll
    for (int i = 0; i < 32; i++) buf[i] = msg[i];
#pragma unroll
    for (int i = 0; i < 8; i++) buf[32 + i] = (uint8_t)(nonce >> (8 * (7 - i)));

    uint8_t d[32];
    b2b256_short<40>(buf, d);

    uint64_t prei8 = 0;
#pragma unroll
    for (int i = 24; i < 32; i++) prei8 = (prei8 << 8) | (uint64_t)d[i];

    uint32_t i0 = (uint32_t)(prei8 % (uint64_t)N);

    // --- seed = element(i0)[31] || msg || nonce ---------------------------
    uint64_t limb[4];
    al_load_element(dataset, i0, limb);

    uint8_t f[31];
    al_element_bytes31(limb, f);

#pragma unroll
    for (int i = 0; i < 31; i++) buf[i] = f[i];
#pragma unroll
    for (int i = 0; i < 32; i++) buf[31 + i] = msg[i];
#pragma unroll
    for (int i = 0; i < 8; i++) buf[63 + i] = (uint8_t)(nonce >> (8 * (7 - i)));

    b2b256_short<71>(buf, d);

    // --- 32 indices from H(seed) || H(seed)[0..3] -------------------------
    uint8_t ext[35];
#pragma unroll
    for (int i = 0; i < 32; i++) ext[i] = d[i];
    ext[32] = d[0];
    ext[33] = d[1];
    ext[34] = d[2];

    uint64_t acc[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < AL_K; j++) {
        uint32_t v = ((uint32_t)ext[j] << 24) | ((uint32_t)ext[j + 1] << 16) |
                     ((uint32_t)ext[j + 2] << 8) | (uint32_t)ext[j + 3];
        uint32_t idx = (uint32_t)(v % N);
        uint64_t e[4];
        al_load_element(dataset, idx, e);
        al_add256(acc, e);
    }

    // --- hit = H(sum as 32-byte BE) ---------------------------------------
    uint8_t sum_be[32];
    al_to_be32(acc, sum_be);
    b2b256_short<32>(sum_be, d);

#pragma unroll
    for (int l = 0; l < 4; l++) {
        uint64_t x = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) x = (x << 8) | (uint64_t)d[(3 - l) * 8 + j];
        out_hit[l] = x;
    }
}

/**
 * Searches nonces [base, base + gridDim*blockDim) and records any whose hit is
 * below `target`.
 */
__global__ __launch_bounds__(256, 2) void al_search(const uint4 *dataset, const uint8_t *msg_in,
                          uint64_t base_nonce, uint32_t N, uint32_t height,
                          const uint64_t *target_in, AlSolution *out,
                          uint32_t *out_count, uint32_t max_out) {
    __shared__ uint8_t s_msg[32];
    __shared__ uint64_t s_target[4];
    if (threadIdx.x < 32) s_msg[threadIdx.x] = msg_in[threadIdx.x];
    if (threadIdx.x < 4) s_target[threadIdx.x] = target_in[threadIdx.x];
    __syncthreads();

    uint64_t nonce = base_nonce + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t msg[32];
#pragma unroll
    for (int i = 0; i < 32; i++) msg[i] = s_msg[i];
    uint64_t target[4];
#pragma unroll
    for (int i = 0; i < 4; i++) target[i] = s_target[i];

    uint64_t hit[4];
    al_hit(dataset, msg, nonce, N, height, hit);

    if (al_lt256(hit, target)) {
        uint32_t slot = atomicAdd(out_count, 1u);
        if (slot < max_out) {
            out[slot].nonce = nonce;
#pragma unroll
            for (int i = 0; i < 4; i++) out[slot].hit[i] = hit[i];
        }
    }
}

/**
 * Recomputes the hit for a single nonce. Used by the host to re-verify a
 * candidate before submitting it: an unstable overclock produces wrong hits,
 * and this is what stops those going upstream as invalid shares.
 */
__global__ void al_verify_one(const uint4 *dataset, const uint8_t *msg_in,
                              uint64_t nonce, uint32_t N, uint32_t height,
                              uint64_t *out) {
    uint8_t msg[32];
#pragma unroll
    for (int i = 0; i < 32; i++) msg[i] = msg_in[i];
    uint64_t hit[4];
    al_hit(dataset, msg, nonce, N, height, hit);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = hit[i];
}
