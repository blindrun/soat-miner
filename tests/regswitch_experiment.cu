// Does a warp-uniform 32-way switch keep the ProgPoW mix in registers?
// Variant A: mix[32] indexed dynamically - what ships. Variant B: 32 named
// registers with switch selection - no runtime codegen.
#include <cstdint>
#define ROUNDS 64
#define NCACHE 12
#define NMATH  20

__device__ __forceinline__ uint32_t mrg(uint32_t a, uint32_t b, uint32_t s) {
    switch (s % 4u) {
        case 0: return a * 33u + b;
        case 1: return (a ^ b) * 33u;
        case 2: return ((a << ((s >> 16) % 31u + 1u)) | (a >> (32 - ((s >> 16) % 31u + 1u)))) ^ b;
        default: return ((a >> ((s >> 16) % 31u + 1u)) | (a << (32 - ((s >> 16) % 31u + 1u)))) ^ b;
    }
}

// ---- A: dynamic array indexing -------------------------------------------
__global__ void variantA(const uint32_t *__restrict__ l1, uint32_t *out,
                         const uint32_t *__restrict__ prog, uint32_t seed) {
    uint32_t mix[32];
    for (int i = 0; i < 32; i++) mix[i] = seed * (i + 1);
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < NCACHE; i++) {
            const uint32_t src = prog[i*3] & 31u, dst = prog[i*3+1] & 31u, sel = prog[i*3+2];
            mix[dst] = mrg(mix[dst], l1[mix[src] & 4095u], sel);
        }
        for (int i = 0; i < NMATH; i++) {
            const uint32_t a = prog[64+i*3] & 31u, b = prog[64+i*3+1] & 31u, d = prog[64+i*3+2] & 31u;
            mix[d] = mrg(mix[d], mix[a] + mix[b], (uint32_t)(r*31+i));
        }
    }
    uint32_t acc = 0;
    for (int i = 0; i < 32; i++) acc ^= mix[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---- B: named registers, switch selection --------------------------------
#define RDW(IDX, OUT) do { switch (IDX) {
#define ENDSW } } while (0)

__device__ __forceinline__ uint32_t rd(uint32_t i,
uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7, uint32_t m8, uint32_t m9, uint32_t m10, uint32_t m11, uint32_t m12, uint32_t m13, uint32_t m14, uint32_t m15, uint32_t m16, uint32_t m17, uint32_t m18, uint32_t m19, uint32_t m20, uint32_t m21, uint32_t m22, uint32_t m23, uint32_t m24, uint32_t m25, uint32_t m26, uint32_t m27, uint32_t m28, uint32_t m29, uint32_t m30, uint32_t m31) {
    switch (i) {
        case 0: return m0;
        case 1: return m1;
        case 2: return m2;
        case 3: return m3;
        case 4: return m4;
        case 5: return m5;
        case 6: return m6;
        case 7: return m7;
        case 8: return m8;
        case 9: return m9;
        case 10: return m10;
        case 11: return m11;
        case 12: return m12;
        case 13: return m13;
        case 14: return m14;
        case 15: return m15;
        case 16: return m16;
        case 17: return m17;
        case 18: return m18;
        case 19: return m19;
        case 20: return m20;
        case 21: return m21;
        case 22: return m22;
        case 23: return m23;
        case 24: return m24;
        case 25: return m25;
        case 26: return m26;
        case 27: return m27;
        case 28: return m28;
        case 29: return m29;
        case 30: return m30;
        case 31: return m31;
    }
    return 0;
}

__global__ void variantB(const uint32_t *__restrict__ l1, uint32_t *out,
                         const uint32_t *__restrict__ prog, uint32_t seed) {
    uint32_t m0=seed*1u; uint32_t m1=seed*2u; uint32_t m2=seed*3u; uint32_t m3=seed*4u; uint32_t m4=seed*5u; uint32_t m5=seed*6u; uint32_t m6=seed*7u; uint32_t m7=seed*8u; uint32_t m8=seed*9u; uint32_t m9=seed*10u; uint32_t m10=seed*11u; uint32_t m11=seed*12u; uint32_t m12=seed*13u; uint32_t m13=seed*14u; uint32_t m14=seed*15u; uint32_t m15=seed*16u; uint32_t m16=seed*17u; uint32_t m17=seed*18u; uint32_t m18=seed*19u; uint32_t m19=seed*20u; uint32_t m20=seed*21u; uint32_t m21=seed*22u; uint32_t m22=seed*23u; uint32_t m23=seed*24u; uint32_t m24=seed*25u; uint32_t m25=seed*26u; uint32_t m26=seed*27u; uint32_t m27=seed*28u; uint32_t m28=seed*29u; uint32_t m29=seed*30u; uint32_t m30=seed*31u; uint32_t m31=seed*32u;
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < NCACHE; i++) {
            const uint32_t s = prog[i*3] & 31u, d = prog[i*3+1] & 31u, sel = prog[i*3+2];
            const uint32_t sv = rd(s, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31);
            const uint32_t dv = rd(d, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31);
            const uint32_t v = mrg(dv, l1[sv & 4095u], sel);
            switch (d) {
        case 0: m0 = v; break;
        case 1: m1 = v; break;
        case 2: m2 = v; break;
        case 3: m3 = v; break;
        case 4: m4 = v; break;
        case 5: m5 = v; break;
        case 6: m6 = v; break;
        case 7: m7 = v; break;
        case 8: m8 = v; break;
        case 9: m9 = v; break;
        case 10: m10 = v; break;
        case 11: m11 = v; break;
        case 12: m12 = v; break;
        case 13: m13 = v; break;
        case 14: m14 = v; break;
        case 15: m15 = v; break;
        case 16: m16 = v; break;
        case 17: m17 = v; break;
        case 18: m18 = v; break;
        case 19: m19 = v; break;
        case 20: m20 = v; break;
        case 21: m21 = v; break;
        case 22: m22 = v; break;
        case 23: m23 = v; break;
        case 24: m24 = v; break;
        case 25: m25 = v; break;
        case 26: m26 = v; break;
        case 27: m27 = v; break;
        case 28: m28 = v; break;
        case 29: m29 = v; break;
        case 30: m30 = v; break;
        case 31: m31 = v; break;
            }
        }
        for (int i = 0; i < NMATH; i++) {
            const uint32_t a = prog[64+i*3] & 31u, b = prog[64+i*3+1] & 31u, dd = prog[64+i*3+2] & 31u;
            const uint32_t av = rd(a, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31);
            const uint32_t bv = rd(b, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31);
            const uint32_t dv = rd(dd, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31);
            const uint32_t v = mrg(dv, av + bv, (uint32_t)(r*31+i));
            switch (dd) {
        case 0: m0 = v; break;
        case 1: m1 = v; break;
        case 2: m2 = v; break;
        case 3: m3 = v; break;
        case 4: m4 = v; break;
        case 5: m5 = v; break;
        case 6: m6 = v; break;
        case 7: m7 = v; break;
        case 8: m8 = v; break;
        case 9: m9 = v; break;
        case 10: m10 = v; break;
        case 11: m11 = v; break;
        case 12: m12 = v; break;
        case 13: m13 = v; break;
        case 14: m14 = v; break;
        case 15: m15 = v; break;
        case 16: m16 = v; break;
        case 17: m17 = v; break;
        case 18: m18 = v; break;
        case 19: m19 = v; break;
        case 20: m20 = v; break;
        case 21: m21 = v; break;
        case 22: m22 = v; break;
        case 23: m23 = v; break;
        case 24: m24 = v; break;
        case 25: m25 = v; break;
        case 26: m26 = v; break;
        case 27: m27 = v; break;
        case 28: m28 = v; break;
        case 29: m29 = v; break;
        case 30: m30 = v; break;
        case 31: m31 = v; break;
            }
        }
    }
    uint32_t acc = m0 ^ m1 ^ m2 ^ m3 ^ m4 ^ m5 ^ m6 ^ m7 ^ m8 ^ m9 ^ m10 ^ m11 ^ m12 ^ m13 ^ m14 ^ m15 ^ m16 ^ m17 ^ m18 ^ m19 ^ m20 ^ m21 ^ m22 ^ m23 ^ m24 ^ m25 ^ m26 ^ m27 ^ m28 ^ m29 ^ m30 ^ m31;
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}


// ---- C: program as compile-time literals = what NVRTC emits --------------
__global__ void variantC(const uint32_t *__restrict__ l1, uint32_t *out,
                         const uint32_t *__restrict__ prog, uint32_t seed) {
    (void)prog;
    uint32_t m0=seed*1u; uint32_t m1=seed*2u; uint32_t m2=seed*3u; uint32_t m3=seed*4u; uint32_t m4=seed*5u; uint32_t m5=seed*6u; uint32_t m6=seed*7u; uint32_t m7=seed*8u; uint32_t m8=seed*9u; uint32_t m9=seed*10u; uint32_t m10=seed*11u; uint32_t m11=seed*12u; uint32_t m12=seed*13u; uint32_t m13=seed*14u; uint32_t m14=seed*15u; uint32_t m15=seed*16u; uint32_t m16=seed*17u; uint32_t m17=seed*18u; uint32_t m18=seed*19u; uint32_t m19=seed*20u; uint32_t m20=seed*21u; uint32_t m21=seed*22u; uint32_t m22=seed*23u; uint32_t m23=seed*24u; uint32_t m24=seed*25u; uint32_t m25=seed*26u; uint32_t m26=seed*27u; uint32_t m27=seed*28u; uint32_t m28=seed*29u; uint32_t m29=seed*30u; uint32_t m30=seed*31u; uint32_t m31=seed*32u;
    for (int r = 0; r < ROUNDS; r++) {
            m9 = mrg(m9, l1[m20 & 4095u], 207388624u);
            m23 = mrg(m23, l1[m6 & 4095u], 2503055453u);
            m2 = mrg(m2, l1[m13 & 4095u], 369140570u);
            m4 = mrg(m4, l1[m26 & 4095u], 1033639716u);
            m3 = mrg(m3, l1[m27 & 4095u], 4070378921u);
            m3 = mrg(m3, l1[m25 & 4095u], 4192983756u);
            m8 = mrg(m8, l1[m2 & 4095u], 1243862422u);
            m7 = mrg(m7, l1[m9 & 4095u], 2452055640u);
            m6 = mrg(m6, l1[m11 & 4095u], 2744112455u);
            m6 = mrg(m6, l1[m23 & 4095u], 884585951u);
            m20 = mrg(m20, l1[m27 & 4095u], 3966150535u);
            m19 = mrg(m19, l1[m23 & 4095u], 3349342752u);
            m31 = mrg(m31, m5 + m19, (uint32_t)(r*31+0));
            m18 = mrg(m18, m21 + m28, (uint32_t)(r*31+1));
            m26 = mrg(m26, m4 + m7, (uint32_t)(r*31+2));
            m9 = mrg(m9, m10 + m21, (uint32_t)(r*31+3));
            m2 = mrg(m2, m31 + m26, (uint32_t)(r*31+4));
            m21 = mrg(m21, m4 + m20, (uint32_t)(r*31+5));
            m29 = mrg(m29, m22 + m31, (uint32_t)(r*31+6));
            m17 = mrg(m17, m4 + m5, (uint32_t)(r*31+7));
            m3 = mrg(m3, m30 + m4, (uint32_t)(r*31+8));
            m18 = mrg(m18, m19 + m28, (uint32_t)(r*31+9));
            m1 = mrg(m1, m24 + m22, (uint32_t)(r*31+10));
            m10 = mrg(m10, m29 + m22, (uint32_t)(r*31+11));
            m3 = mrg(m3, m7 + m31, (uint32_t)(r*31+12));
            m8 = mrg(m8, m13 + m18, (uint32_t)(r*31+13));
            m25 = mrg(m25, m15 + m25, (uint32_t)(r*31+14));
            m10 = mrg(m10, m31 + m5, (uint32_t)(r*31+15));
            m17 = mrg(m17, m28 + m25, (uint32_t)(r*31+16));
            m17 = mrg(m17, m8 + m27, (uint32_t)(r*31+17));
            m24 = mrg(m24, m26 + m22, (uint32_t)(r*31+18));
            m5 = mrg(m5, m14 + m9, (uint32_t)(r*31+19));
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = m0 ^ m1 ^ m2 ^ m3 ^ m4 ^ m5 ^ m6 ^ m7 ^ m8 ^ m9 ^ m10 ^ m11 ^ m12 ^ m13 ^ m14 ^ m15 ^ m16 ^ m17 ^ m18 ^ m19 ^ m20 ^ m21 ^ m22 ^ m23 ^ m24 ^ m25 ^ m26 ^ m27 ^ m28 ^ m29 ^ m30 ^ m31;
}

// ---- host harness ---------------------------------------------------------
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    const int BLOCKS = 2048, THREADS = 128, ITERS = 20;
    uint32_t *l1, *out, *prog;
    cudaMalloc(&l1, 4096 * 4); cudaMemset(l1, 0x5a, 4096 * 4);
    cudaMalloc(&out, (size_t)BLOCKS * THREADS * 4);
    uint32_t hp[256];
    for (int i = 0; i < 256; i++) hp[i] = (uint32_t)(i * 2654435761u);
    cudaMalloc(&prog, sizeof hp); cudaMemcpy(prog, hp, sizeof hp, cudaMemcpyHostToDevice);

    for (int v = 0; v < 3; v++) {
        // warm up, then time - a first launch carries context setup
        if (v == 0) variantA<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u);
        else if (v == 1) variantB<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u);
        else variantC<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u);
        cudaDeviceSynchronize();

        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < ITERS; i++) {
            if (v == 0) variantA<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u + i);
            else if (v == 1) variantB<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u + i);
            else variantC<<<BLOCKS, THREADS>>>(l1, out, prog, 12345u + i);
        }
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        const double hashes = (double)BLOCKS * THREADS * ITERS;
        printf("  variant%c  %8.2f ms  %8.2f M mix/s\n",
               "ABC"[v], ms, hashes / (ms / 1000.0) / 1e6);
        // did it run? a zeroed result would mean the launch failed silently
        uint32_t probe = 0;
        cudaMemcpy(&probe, out, 4, cudaMemcpyDeviceToHost);
        printf("            result[0] = 0x%08x %s\n", probe,
               probe ? "" : "  <-- ZERO: launch did nothing, ignore the time");
    }
    return 0;
}
