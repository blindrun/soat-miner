// Offline Bitcoin-Stratum V1 fixture test. No socket, wallet, GPU, or pool.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/btc_job.h"
#include "core/btc_protocol.h"
#include "core/sha256.h"
#include "core/stratum_btc.h"

namespace om {
struct BitcoinStratumTestAccess {
    static void line(BitcoinStratumSource &source, const std::string &line) {
        source.handleLine(line);
    }
    static void reset(BitcoinStratumSource &source) { source.resetSession(); }
    static bool subscribed(const BitcoinStratumSource &source) { return source.subscribed_; }
    static bool haveJob(const BitcoinStratumSource &source) { return source.haveJob_; }
    static const Job &job(const BitcoinStratumSource &source) { return source.current_; }
    static bool rejected(const BitcoinStratumSource &source) { return source.loginRejected_.load(); }
    static std::string error(const BitcoinStratumSource &source) { return source.loginError_; }
    static bool hasDifficulty(const BitcoinStratumSource &source) { return source.haveDifficulty_; }
};
}  // namespace om

namespace {
int failures = 0;

void expect(bool ok, const char *what) {
    if (ok) std::printf("ok   %s\n", what);
    else { std::printf("FAIL %s\n", what); failures++; }
}

std::string hex(const uint8_t *p, size_t n) { return om::btc::vecToHex(p, n); }

void testPureProtocol() {
    uint8_t digest[32];
    om::sha256d((const uint8_t *)"abc", 3, digest);
    expect(hex(digest, sizeof(digest)) ==
           "4f8b42c22dd3729b519ba6f68d2da7cc5b2d606d05daed5ad5128cc03e6c6358",
           "SHA-256d known vector");

    std::vector<uint8_t> coinbase;
    om::btc::hexToVec("0100000001000000000000000000000000000000000000000000000000000000"
                      "00000000ffffffff0301e8030102030401000000", &coinbase);
    std::vector<std::string> branch(1, std::string(32, '\0'));
    uint8_t root[32];
    om::merkleRootFromBranch(coinbase, branch, root);
    expect(hex(root, sizeof(root)) ==
           "ef265fa50e19c1014be015df022fd01101734dc34fb9399e1b3edbf88299d328",
           "SHA-256d coinbase merkle root");

    uint64_t compact[4], diff1[4], fractional[4];
    expect(om::btc::compactToTarget(0x1d00ffff, compact), "compact target parses");
    om::btc::difficultyToTarget(1.0, diff1);
    om::btc::difficultyToTarget(0.01, fractional);
    expect(compact[0] == 0 && compact[1] == 0 && compact[2] == 0 &&
           compact[3] == 0x00000000ffff0000ULL, "compact target byte order");
    expect(std::memcmp(compact, diff1, sizeof(compact)) == 0,
           "difficulty one equals Bitcoin compact target");
    expect(fractional[3] > diff1[3] && fractional[3] != 0,
           "fractional difficulty has a nonzero easier target");

    expect(om::btc::subscribeRequest() ==
           "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"soat-miner/0.1\"]}",
           "subscribe request fixture");
    expect(om::btc::authorizeRequest("1abc.rig\"\\\n", "x") ==
           "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"1abc.rig\\\"\\\\\\n\",\"x\"]}",
           "authorize escapes login JSON");
    expect(om::btc::submitRequest(10, "1abc.rig\"\\\n", "job\"\\\n",
                                  std::string("\x01\x00", 2), 0x5f5e1000,
                                  0x1234) ==
           "{\"id\":10,\"method\":\"mining.submit\",\"params\":[\"1abc.rig\\\"\\\\\\n\",\"job\\\"\\\\\\n\",\"0100\",\"5f5e1000\",\"00001234\"]}",
           "submit escapes login and job id JSON");
}

void testFixture(const char *path) {
    om::BitcoinStratumSource source("offline.invalid", 1, "1offline", "rig", "x", 256);
    std::ifstream fixture(path);
    expect((bool)fixture, "open offline Stratum fixture");
    if (!fixture) return;
    std::string line;
    int step = 0;
    uint8_t firstHeader[80] = {};
    while (std::getline(fixture, line)) {
        om::BitcoinStratumTestAccess::line(source, line);
        ++step;
        if (step == 1) expect(om::BitcoinStratumTestAccess::subscribed(source),
                              "subscribe reply sets extranonce state");
        if (step == 2) expect(!om::BitcoinStratumTestAccess::rejected(source),
                              "authorize success reply accepted");
        if (step == 3) expect(om::BitcoinStratumTestAccess::hasDifficulty(source),
                              "difficulty notification parsed");
        if (step == 4) {
            expect(om::BitcoinStratumTestAccess::haveJob(source),
                   "notify builds a mineable offline job");
            expect(om::btcJobHeader(om::BitcoinStratumTestAccess::job(source).extra,
                                    firstHeader), "notify stores 80-byte header");
            expect(hex(firstHeader, 4) == "20001000" &&
                   hex(firstHeader + 4, 32) ==
                   "03020100070605040b0a09080f0e0d0c13121110171615141b1a19181f1e1d1c" &&
                   hex(firstHeader + 68, 12) == "00105e5fffff001d00000000",
                   "notify header scalar and prevhash byte order");
            expect(hex(firstHeader + 36, 32) ==
                   "60f25e3465890046776765a20327c70b67068b9224b931d75b4de40412aa942b",
                   "notify header uses SHA-256d merkle root");
        }
        if (step == 5) {
            std::string xn2;
            const om::Job &job = om::BitcoinStratumTestAccess::job(source);
            expect(om::btcJobExtranonce2(job.extra, &xn2) && xn2.size() == 8 &&
                   (uint8_t)xn2[0] == 2, "set_extranonce rolls an eight-byte extranonce2");
            uint8_t secondHeader[80];
            om::btcJobHeader(job.extra, secondHeader);
            expect(std::memcmp(firstHeader + 36, secondHeader + 36, 32) != 0,
                   "extranonce roll changes the merkle-root job identity");
        }
    }
    expect(step == 6 && om::BitcoinStratumTestAccess::rejected(source) &&
           om::BitcoinStratumTestAccess::error(source).find("bad worker") != std::string::npos,
           "authorize rejection fixture is surfaced");
    om::BitcoinStratumTestAccess::reset(source);
    expect(!om::BitcoinStratumTestAccess::subscribed(source) &&
           !om::BitcoinStratumTestAccess::haveJob(source) &&
           !om::BitcoinStratumTestAccess::hasDifficulty(source) &&
           !om::BitcoinStratumTestAccess::rejected(source),
           "reconnect reset drops prior session state before a new fixture");
    fixture.clear();
    fixture.seekg(0);
    for (int i = 0; i < 5 && std::getline(fixture, line); i++)
        om::BitcoinStratumTestAccess::line(source, line);
    expect(om::BitcoinStratumTestAccess::subscribed(source) &&
           om::BitcoinStratumTestAccess::haveJob(source) &&
           !om::BitcoinStratumTestAccess::rejected(source),
           "reconnect fixture rebuilds a clean authorized job");
}
}  // namespace

int main(int argc, char **argv) {
    testPureProtocol();
    testFixture(argc == 2 ? argv[1] : "tests/fixtures/btc_stratum_v1.jsonl");
    if (failures) std::printf("%d FAILURE(S)\n", failures);
    else std::puts("Bitcoin Stratum offline fixture checks passed");
    return failures ? 1 : 0;
}
