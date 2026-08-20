// SHA3-256t device kernels.
//
// There is no dataset and no epoch state, so this is the whole GPU side: one
// kernel that hashes a header with `count` consecutive nonces and reports the
// ones under target.

#pragma once

#include "sha3.h"

namespace om {

/** A nonce under target, with the digest that got it there. */
struct S3Solution {
    uint64_t nonce;
    uint64_t hit[4];
};

/**
 * The header and target go by value rather than through device buffers.
 * Together they are 112 bytes, well inside the 4 KB parameter limit, and
 * kernel parameters land in constant memory - so every thread reads the header
 * through the constant cache instead of a global load, and the host does not
 * copy anything per launch.
 *
 * `base` is 32 bits because BC3's header nonce is 32 bits. run.cpp is told
 * nonceBitsOwned()==32 and clamps launches to the subspace, so base + i never
 * silently wraps into a range we did not intend to mine.
 */
__global__ void s3_search(om::s3::Header hdr, uint32_t base, uint32_t count,
                          uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3,
                          S3Solution *out, uint32_t *found, uint32_t cap) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t nonce = base + i;
    uint64_t hit[4];
    om::s3::hash(hdr, nonce, hit);

    const uint64_t target[4] = {t0, t1, t2, t3};
    if (!om::s3::underTarget(hit, target)) return;

    // Count every hit even past capacity, so the host can say how many were
    // dropped rather than silently reporting a full buffer as the total.
    const uint32_t slot = atomicAdd(found, 1u);
    if (slot >= cap) return;
    out[slot].nonce = nonce;
    out[slot].hit[0] = hit[0];
    out[slot].hit[1] = hit[1];
    out[slot].hit[2] = hit[2];
    out[slot].hit[3] = hit[3];
}

}  // namespace om
