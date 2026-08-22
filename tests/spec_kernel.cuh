// GENERATED for block height 4501000 - do not edit.
// Program is a function of height/period, so this kernel is valid
// ONLY for that period. algo.cu checks before using it.
#pragma once
#include "../src/algos/kawpow/mine.cuh"

namespace om { namespace kawpow {

static const int kSpecHeight = 4501000;

__device__ __forceinline__ void specLane(uint32_t *mix,
        const uint32_t *l1, const uint32_t dagWords[4]) {
    om::progpow::randomMerge(&mix[24], l1[mix[2] % om::progpow::kL1CacheWords], 119290792u);
    om::progpow::randomMerge(&mix[11], om::progpow::randomMath(mix[9], mix[25], 3232152996u, 0), 2794290701u);
    om::progpow::randomMerge(&mix[12], l1[mix[16] % om::progpow::kL1CacheWords], 2388191108u);
    om::progpow::randomMerge(&mix[28], om::progpow::randomMath(mix[27], mix[30], 124212694u, 0), 2249140848u);
    om::progpow::randomMerge(&mix[16], l1[mix[15] % om::progpow::kL1CacheWords], 2917434019u);
    om::progpow::randomMerge(&mix[0], om::progpow::randomMath(mix[20], mix[5], 518703388u, 0), 2355248764u);
    om::progpow::randomMerge(&mix[29], l1[mix[1] % om::progpow::kL1CacheWords], 297522633u);
    om::progpow::randomMerge(&mix[26], om::progpow::randomMath(mix[6], mix[10], 4203576855u, 0), 733707977u);
    om::progpow::randomMerge(&mix[10], l1[mix[9] % om::progpow::kL1CacheWords], 1318282969u);
    om::progpow::randomMerge(&mix[5], om::progpow::randomMath(mix[28], mix[16], 837563738u, 0), 1867552681u);
    om::progpow::randomMerge(&mix[30], l1[mix[5] % om::progpow::kL1CacheWords], 728585170u);
    om::progpow::randomMerge(&mix[9], om::progpow::randomMath(mix[2], mix[6], 698353384u, 0), 1200752976u);
    om::progpow::randomMerge(&mix[18], l1[mix[18] % om::progpow::kL1CacheWords], 3053863422u);
    om::progpow::randomMerge(&mix[20], om::progpow::randomMath(mix[0], mix[29], 1437915722u, 0), 2882090119u);
    om::progpow::randomMerge(&mix[1], l1[mix[13] % om::progpow::kL1CacheWords], 3914780165u);
    om::progpow::randomMerge(&mix[4], om::progpow::randomMath(mix[14], mix[18], 1221128637u, 0), 22829256u);
    om::progpow::randomMerge(&mix[21], l1[mix[28] % om::progpow::kL1CacheWords], 1686030125u);
    om::progpow::randomMerge(&mix[2], om::progpow::randomMath(mix[17], mix[11], 4151614818u, 0), 3554054779u);
    om::progpow::randomMerge(&mix[15], l1[mix[0] % om::progpow::kL1CacheWords], 1976395340u);
    om::progpow::randomMerge(&mix[13], om::progpow::randomMath(mix[13], mix[23], 1690402974u, 0), 1771856962u);
    om::progpow::randomMerge(&mix[3], l1[mix[7] % om::progpow::kL1CacheWords], 1129379371u);
    om::progpow::randomMerge(&mix[19], om::progpow::randomMath(mix[2], mix[8], 1998088387u, 0), 2759411323u);
    om::progpow::randomMerge(&mix[8], om::progpow::randomMath(mix[10], mix[1], 1949815256u, 0), 799765336u);
    om::progpow::randomMerge(&mix[31], om::progpow::randomMath(mix[17], mix[5], 2245679693u, 0), 1161403440u);
    om::progpow::randomMerge(&mix[22], om::progpow::randomMath(mix[11], mix[31], 341352323u, 0), 3989947535u);
    om::progpow::randomMerge(&mix[27], om::progpow::randomMath(mix[9], mix[8], 2695523517u, 0), 2377857798u);
    om::progpow::randomMerge(&mix[6], om::progpow::randomMath(mix[24], mix[0], 2623339031u, 0), 321069297u);
    om::progpow::randomMerge(&mix[7], om::progpow::randomMath(mix[12], mix[13], 1225756801u, 0), 987350692u);
    om::progpow::randomMerge(&mix[14], om::progpow::randomMath(mix[3], mix[12], 3311075932u, 0), 2357321114u);
    om::progpow::randomMerge(&mix[0], dagWords[0], 2841581375u);
    om::progpow::randomMerge(&mix[17], dagWords[1], 433798607u);
    om::progpow::randomMerge(&mix[23], dagWords[2], 2404115684u);
    om::progpow::randomMerge(&mix[25], dagWords[3], 1565004418u);
}

}}  // namespace
