// Verifies al_gen_element against the Python reference (tests/reference.py).
#include <cstdio>
#include "../src/algos/autolykos2/autolykos.cuh"

__global__ void k_elem(uint32_t base, uint32_t height, uint64_t *out) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t limb[4];
    al_gen_element(base + t, height, limb);
#pragma unroll
    for (int i = 0; i < 4; i++) out[t * 4 + i] = limb[i];
}

int main(int argc, char **argv) {
    uint32_t height = (argc > 1) ? (uint32_t)atoi(argv[1]) : 1851437u;
    const int COUNT = 8;
    const uint32_t idxs[COUNT] = {0, 1, 2, 42, 1023, 1024, 65536, 227251814u};

    uint64_t *d_out;
    cudaMalloc(&d_out, sizeof(uint64_t) * 4 * COUNT);

    uint64_t host[4 * COUNT];
    for (int c = 0; c < COUNT; c++) {
        k_elem<<<1, 1>>>(idxs[c], height, d_out + c * 4);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(host, d_out, sizeof(host), cudaMemcpyDeviceToHost);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    for (int c = 0; c < COUNT; c++) {
        // print as 64-hex-digit big-endian 256-bit value
        printf("%-12u ", idxs[c]);
        for (int l = 3; l >= 0; l--) printf("%016llx", (unsigned long long)host[c * 4 + l]);
        printf("\n");
    }
    cudaFree(d_out);
    return 0;
}
