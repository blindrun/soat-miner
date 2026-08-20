// No secret values appear in this test. It checks the accepted *reference*
// grammar and confirms that literal recovery-material-shaped input is refused.
#include <cstdio>
#include <string>

#include "core/bc3_destination.h"

namespace {
int failures = 0;

void expect(bool ok, const char *what) {
    if (ok) std::printf("ok   %s\n", what);
    else { std::printf("FAIL %s\n", what); ++failures; }
}
}  // namespace

int main() {
    om::Bc3SweepDestination destination;
    expect(om::parseBc3SweepDestination(
               "address:bc1qexampledestinationonly0000000000000000000",
               &destination) &&
               destination.kind == om::Bc3DestinationKind::Address,
           "explicit BC3 address destination accepted");
    expect(om::parseBc3SweepDestination(
               "secret-ref:crypto-recovery/bc3-regtest-destination",
               &destination) &&
               destination.kind == om::Bc3DestinationKind::SecretReference,
           "opaque Crypto Recovery reference accepted without resolution");
    expect(!om::parseBc3SweepDestination("bc1qbare-address-is-not-a-schema", nullptr),
           "bare address is refused; address prefix is required");
    expect(!om::parseBc3SweepDestination("seed:<redacted>", nullptr),
           "literal seed form is refused");
    expect(!om::bc3AddressShape("Kprivate-key-shaped-placeholder"),
           "private-key-shaped input is not a BC3 payout address");
    expect(!om::parseBc3SweepDestination("word1 word2 word3 word4 word5 word6", nullptr),
           "space-separated recovery material is refused");
    expect(!om::parseBc3SweepDestination("secret-ref:other-store/record", nullptr),
           "non-Crypto-Recovery namespace is refused");
    if (failures) std::printf("%d FAILURE(S)\n", failures);
    else std::puts("BC3 destination schema checks passed");
    return failures ? 1 : 0;
}
