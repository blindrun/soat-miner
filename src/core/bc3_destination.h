// Safe schema for a future chain-only BC3 sweep-destination fixture.
//
// This deliberately does not resolve references, read a wallet, open a
// socket, or participate in pool login.  A caller can retain only either a
// payout address or an opaque Crypto Recovery reference.  Literal recovery
// material has no accepted grammar.
#pragma once

#include <string>

namespace om {

enum class Bc3DestinationKind { Address, SecretReference };

struct Bc3SweepDestination {
    Bc3DestinationKind kind;
    std::string value;
};

inline bool bc3AddressShape(const std::string &address) {
    if (address.size() < 14 || address.size() > 128) return false;
    if (address.rfind("bc1", 0) != 0 && address[0] != '1' && address[0] != '3')
        return false;
    for (unsigned char c : address) {
        if (c <= ' ' || c >= 0x7f) return false;
    }
    return true;
}

inline bool controlReferenceShape(const std::string &reference) {
    static const std::string prefix = "secret-ref:crypto-recovery/";
    if (reference.rfind(prefix, 0) != 0 || reference.size() == prefix.size() ||
        reference.size() > 256)
        return false;
    for (size_t i = prefix.size(); i < reference.size(); ++i) {
        const unsigned char c = reference[i];
        const bool allowed = (c >= 'a' && c <= 'z') ||
                             (c >= 'A' && c <= 'Z') ||
                             (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                             c == '-' || c == '/';
        if (!allowed) return false;
    }
    return true;
}

// Accepted forms are intentionally explicit:
//   address:<BC3 payout address>
//   secret-ref:crypto-recovery/<control-record>
// Nothing else is a destination specification, especially not a seed,
// mnemonic, private key, cookie, password, or an environment-variable value.
inline bool parseBc3SweepDestination(const std::string &spec,
                                     Bc3SweepDestination *out) {
    static const std::string addressPrefix = "address:";
    if (spec.rfind(addressPrefix, 0) == 0) {
        const std::string address = spec.substr(addressPrefix.size());
        if (!bc3AddressShape(address)) return false;
        if (out) *out = {Bc3DestinationKind::Address, address};
        return true;
    }
    if (controlReferenceShape(spec)) {
        if (out) *out = {Bc3DestinationKind::SecretReference, spec};
        return true;
    }
    return false;
}

}  // namespace om
