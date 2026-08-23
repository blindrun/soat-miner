//! One CUDA header, as text, for NVRTC to compile against.
#pragma once

namespace om {
namespace kawpow {

/// A header NVRTC can `#include`: the name a generated kernel uses, and its
/// full text.
struct NvrtcHeader {
    /// The path a generated kernel writes in its `#include`.
    const char *name;
    /// The header's complete source.
    const char *source;
};

/// Every header a generated KawPow kernel needs, embedded at build time.
///
/// Embedded rather than read from disk because NVRTC has no filesystem and a
/// shipped binary must not require its source tree. Embedded rather than
/// rewritten as a string in the generator because that would be a second
/// definition of keccak and the mix primitives, and two implementations of a
/// hash function is how a miner produces shares that are confidently wrong.
const NvrtcHeader *nvrtcHeaders(int *count);

}  // namespace kawpow
}  // namespace om
