// End-to-end check: build the real dataset for a real block's height, then
// confirm the GPU reproduces that block's hit for its known winning nonce.
//
// usage: test_hit <msg_hex64> <height> <nonce_hex16> [expected_hit_hex64]
//
// With an expected hit this EXITS NON-ZERO on any mismatch, which is the only
// thing that makes it a gate. Without one it just prints, which is fine for
// poking at a block by hand and useless as a test - `make test` always passes
// the expected hit.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "../src/algos/autolykos2/mine.cuh"

static uint32_t calc_n(uint32_t height) {
    const uint32_t NBase = 1u << 26;
    const uint32_t IncreaseStart = 600u * 1024u;
    const uint32_t IncreasePeriod = 50u * 1024u;
    const uint32_t MaxH = 4198400u;
    uint32_t h = height < MaxH ? height : MaxH;
    if (h < IncreaseStart) return NBase;
    uint32_t iters = (h - IncreaseStart) / IncreasePeriod + 1;
    uint32_t N = NBase;
    for (uint32_t i = 0; i < iters; i++) N = N / 100 * 105;
    return N;
}

static void hex2bin(const char *hex, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v;
        sscanf(hex + i * 2, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

__global__ void k_one_hit(const uint4 *ds, const uint8_t *msg, uint64_t nonce,
                          uint32_t N, uint32_t height, uint64_t *out) {
    uint8_t m[32];
    for (int i = 0; i < 32; i++) m[i] = msg[i];
    uint64_t hit[4];
    al_hit(ds, m, nonce, N, height, hit);
    for (int i = 0; i < 4; i++) out[i] = hit[i];
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s <msg_hex64> <height> <nonce_hex16>\n", argv[0]);
        return 1;
    }
    uint8_t msg[32];
    hex2bin(argv[1], msg, 32);
    uint32_t height = (uint32_t)strtoul(argv[2], NULL, 10);
    uint64_t nonce = strtoull(argv[3], NULL, 16);

    uint32_t N = calc_n(height);
    size_t bytes = (size_t)N * 32ULL;
    printf("height=%u  N=%u  dataset=%.2f GB\n", height, N, bytes / 1e9);

    uint4 *d_ds = nullptr;
    cudaError_t e = cudaMalloc(&d_ds, bytes);
    if (e != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // --- build ------------------------------------------------------------
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    int threads = 256;
    uint32_t blocks = (N + threads - 1) / threads;
    cudaEventRecord(t0);
    al_build_dataset<<<blocks, threads>>>(d_ds, N, height);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("build kernel failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("dataset built in %.2f s  (%.1f M elem/s)\n", ms / 1000.0, N / (ms * 1000.0));

    // --- one known nonce --------------------------------------------------
    uint8_t *d_msg;
    uint64_t *d_out;
    cudaMalloc(&d_msg, 32);
    cudaMalloc(&d_out, 32);
    cudaMemcpy(d_msg, msg, 32, cudaMemcpyHostToDevice);
    k_one_hit<<<1, 1>>>(d_ds, d_msg, nonce, N, height, d_out);
    cudaDeviceSynchronize();
    uint64_t hit[4];
    cudaMemcpy(hit, d_out, 32, cudaMemcpyDeviceToHost);

    char hitHex[65];
    for (int l = 3; l >= 0; l--)
        snprintf(hitHex + (3 - l) * 16, 17, "%016llx", (unsigned long long)hit[l]);
    printf("nonce=%016llx\nhit  =%s\n", (unsigned long long)nonce, hitHex);

    // The whole point of the test. A hit that does not match the reference
    // means this GPU disagrees with consensus, so it would mine forever and
    // find nothing - which reports as no error at all if nobody checks here.
    int failed = 0;
    if (argc >= 5) {
        char want[65];
        for (int i = 0; i < 64 && argv[4][i]; i++)
            want[i] = (char)tolower((unsigned char)argv[4][i]);
        want[64] = '\0';
        if (strncmp(hitHex, want, 64) != 0) {
            printf("FAIL: expected hit\n     =%s\n", want);
            failed = 1;
        } else {
            printf("PASS: hit matches the reference for block %u\n", height);
        }
    } else {
        printf("(no expected hit given - nothing was checked)\n");
    }

    // --- throughput -------------------------------------------------------
    uint64_t target[4] = {0, 0, 0, 0};  // impossible target: pure benchmark
    uint64_t *d_target;
    AlSolution *d_sol;
    uint32_t *d_cnt;
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_sol, sizeof(AlSolution) * 16);
    cudaMalloc(&d_cnt, 4);
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_cnt, 0, 4);

    uint64_t total = 1ULL << 24;  // 16.7M nonces
    uint32_t sblocks = (uint32_t)(total / threads);
    cudaEventRecord(t0);
    al_search<<<sblocks, threads>>>(d_ds, d_msg, 0, N, height, d_target, d_sol, d_cnt, 16);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("search kernel failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("search: %llu nonces in %.3f s = %.2f MH/s\n",
           (unsigned long long)total, ms / 1000.0, total / (ms * 1000.0));

    cudaFree(d_ds);
    return failed;
}
