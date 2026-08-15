// Algorithm registry.
//
// Adding an algorithm is two lines here plus a directory under src/algos/.
// Kept as an explicit table rather than static-initialiser self-registration:
// with separate compilation units and CUDA, self-registration silently does
// nothing if the linker drops the object, and a miner that quietly supports
// nothing is a bad failure mode.

#include "algo.h"

namespace om {

Algorithm *makeAutolykos2();
// Algorithm *makeKawpow();     // <- add here
// Algorithm *makeEthash();

namespace {
struct Entry {
    const char *name;
    Algorithm *(*factory)();
};

const Entry kRegistry[] = {
    {"autolykos2", &makeAutolykos2},
};
}  // namespace

Algorithm *createAlgorithm(const std::string &name) {
    for (const auto &e : kRegistry) {
        if (name == e.name) return e.factory();
    }
    return nullptr;
}

std::vector<std::string> availableAlgorithms() {
    std::vector<std::string> out;
    for (const auto &e : kRegistry) out.push_back(e.name);
    return out;
}

}  // namespace om
