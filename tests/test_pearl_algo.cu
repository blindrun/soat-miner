// The Pearl Algorithm object end to end: real work, real GPU, real node.
//
// Every other Pearl test is a link in a chain - kernels against vectors, host
// code against the Python, the Python against Pearl's own Rust, the transport
// against a gateway. This one runs the actual object the miner will use, on
// the actual card, against work fetched from a gateway, and submits what it
// finds. The node either takes the block or it does not.
//
// It is deliberately not a mock. Regtest is a real node validating real
// submissions, and its difficulty makes roughly one transcript in thirty-two a
// winner, so a single attempt produces plenty. A wrong commitment chain, wrong
// noise, a wrong transcript, a wrong tile inversion or a malformed proof all
// end the same way: rejected.
//
//   ~/pearl-regtest-gateway.sh tunnel      # in its own terminal
//   ~/pearl-regtest-gateway.sh gateway     # in its own terminal
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/test_pearl_algo.cu \
//        src/algos/pearl-pow/algo.cu -o tests/test_pearl_algo
//   ./tests/test_pearl_algo 127.0.0.1 8455
//
// Then check the height moved: ~/pearl-regtest-gateway.sh height

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "../src/algos/pearl-pow/gateway.h"
#include "../src/core/algo.h"

namespace om {
Algorithm *makePearlPow();
}

using namespace om;

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

}  // namespace

int main(int argc, char **argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const int port = argc > 2 ? atoi(argv[2]) : 8455;

    platformInit();

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s, %.1f GB\n", prop.name, prop.totalGlobalMem / 1e9);

    printf("1. the algorithm registers itself\n");
    std::unique_ptr<Algorithm> algo(makePearlPow());
    check("it is called pearl-pow", std::string(algo->name()) == "pearl-pow",
          algo->name());

    printf("2. fetching real work\n");
    PearlGatewaySource source(host, port);
    Job job;
    if (!source.fetch(&job)) {
        printf("  FAIL no answer from %s:%d\n", host.c_str(), port);
        printf("\nStart the regtest rig first: ~/pearl-regtest-gateway.sh\n");
        return 1;
    }
    gPass++;
    printf("  ok   fetched from %s\n", source.describe());

    uint8_t header[76];
    int cert = 0;
    check("the job carries a header and a certificate version",
          unpackPearlExtra(job.extra, header, &cert) && cert >= 1);

    printf("3. preparing\n");
    check("prepare succeeds", algo->prepare(job));
    printf("  ok   claims %.0f MB of device memory\n",
           algo->memoryBytes(job) / 1e6);
    gPass++;

    printf("4. searching\n");
    std::vector<Solution> sols;
    const uint64_t nonceBase = 0x1234;
    const auto t0 = std::chrono::steady_clock::now();
    // One attempt's worth. The algorithm reads `count` as a candidate budget
    // and rounds down to whole attempts, with a floor of one.
    check("search returns cleanly", algo->search(job, nonceBase, 1, &sols));
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Not a hashrate. This call also pays the once-per-job B setup and then,
    // because regtest difficulty means it wins immediately, the whole opening
    // path - reading both matrices back, regenerating the noise on the host
    // and building a Merkle tree over 64 MB. The steady-state rate is in
    // section 7 of tests/test_pearl_prepare.cu.
    printf("  ok   first attempt in %.1f ms, including per-job setup and the "
           "opening\n", secs * 1e3);
    gPass++;

    check("it found something at regtest difficulty", !sols.empty(),
          "one attempt is millions of candidates against a 2^232 target");
    if (sols.empty()) {
        printf("\n%d passed, %d failed\n", gPass, gFail);
        return 1;
    }

    printf("5. host verification\n");
    const Solution &sol = sols[0];
    check("the win re-verifies on the host", algo->verify(job, sol),
          "recomputing the tile from A and B must reproduce the digest");
    check("it carries a proof, not just a nonce", !sol.extra.empty(),
          std::to_string(sol.extra.size()) + " base64 chars");
    printf("  ok   proof is %zu base64 chars for nonce %llu\n", sol.extra.size(),
           (unsigned long long)sol.nonce);
    gPass++;

    // A solution the host cannot reproduce must never be submitted - that is
    // the whole point of verify(), and an unstable card is what it catches.
    Solution bogus = sol;
    bogus.nonce = sol.nonce + 999999;
    check("a solution the algorithm never produced does NOT verify",
          !algo->verify(job, bogus));

    printf("6. submitting\n");
    std::string err;
    check("the gateway accepted the submission", source.submit(job, sol, &err), err);

    algo->release();
    printf("\n%d passed, %d failed\n", gPass, gFail);
    printf("The gateway acknowledges, then proves and submits asynchronously.\n"
           "Check the node: ~/pearl-regtest-gateway.sh height\n");
    return gFail ? 1 : 0;
}
