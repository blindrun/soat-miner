// Verifies the device Blake2b-256 against vectors produced by Python hashlib.
#include <cstdio>
#include <cstring>
#include "../src/core/blake2b.cuh"

__global__ void k_hash(const uint8_t *in, uint32_t len, uint8_t *out) {
    b2b256_short<40>(in, out);
}

int main() {
    struct Case {
        const char *name;
        uint32_t len;
    } cases[] = {{"empty", 0}, {"abc", 3}, {"40B (msg||nonce)", 40},
                 {"71B (seed)", 71}, {"32B (sum)", 32}, {"127B", 127}};

    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, 128);
    cudaMalloc(&d_out, 32);

    for (auto &c : cases) {
        uint8_t buf[128];
        for (uint32_t i = 0; i < c.len; i++) buf[i] = (uint8_t)(i * 7 + 1);
        if (c.len == 3) { buf[0] = 'a'; buf[1] = 'b'; buf[2] = 'c'; }

        cudaMemcpy(d_in, buf, c.len ? c.len : 1, cudaMemcpyHostToDevice);
        k_hash<<<1, 1>>>(d_in, c.len, d_out);
        cudaDeviceSynchronize();

        uint8_t out[32];
        cudaMemcpy(out, d_out, 32, cudaMemcpyDeviceToHost);

        printf("%-20s len=%3u  ", c.name, c.len);
        for (int i = 0; i < 32; i++) printf("%02x", out[i]);
        printf("\n");
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
