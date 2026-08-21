// The Vulkan algorithm registry.
//
// Deliberately its own translation unit rather than part of vk_common.cpp.
// The table names every algorithm's factory, so anything that links it links
// every backend and every embedded SPIR-V module with it - which is right for
// the miner and wrong for a test that wants one algorithm. Keeping the device
// plumbing separate means tests/test_sha3_vulkan pulls in the BC3 shader and
// not the 138 KB Autolykos one.
//
// An explicit table rather than static-initialiser self-registration, for the
// same reason src/core/registry.cu gives: with separate compilation units,
// self-registration silently does nothing if the linker drops the object, and
// a miner that quietly supports nothing is a bad failure mode.
//
// Adding a Vulkan algorithm is two lines here, plus its .comp, its
// algo_vk.cpp, and the build rules copied in the Makefile.

#include "vk_common.h"

namespace om {

Algorithm *makeAutolykos2VK(int deviceIndex);
Algorithm *makeSha3_256tVK(int deviceIndex);
// Registered only now that the backend really mines: prepare() runs the whole
// chain and checks it against job.h before a share is possible, search() opens
// a winning tile into a real proof, and the mock-pool test submits one that is
// accepted. Registering earlier would have made --algo pearl-pow start,
// allocate a device and then fail, which is worse for a user than the honest
// "Pearl is CUDA only" that miner_vk.cpp printed while this line was commented.
Algorithm *makePearlPowVK(int deviceIndex);

namespace {

struct Entry {
    const char *name;
    Algorithm *(*factory)(int);
};

const Entry kVulkanRegistry[] = {
    {"autolykos2", &makeAutolykos2VK},
    {"sha3-256t", &makeSha3_256tVK},
    {"pearl-pow", &makePearlPowVK},
};

}  // namespace

Algorithm *createVulkanAlgorithm(const std::string &name, int deviceIndex) {
    for (const auto &e : kVulkanRegistry) {
        if (name == e.name) return e.factory(deviceIndex);
    }
    return nullptr;
}

std::vector<std::string> availableVulkanAlgorithms() {
    std::vector<std::string> out;
    for (const auto &e : kVulkanRegistry) out.push_back(e.name);
    return out;
}

}  // namespace om
