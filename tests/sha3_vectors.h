// Real BC3 mainnet blocks, shared by the CUDA and Vulkan gates.
//
// Shared rather than copied because two backends held to two different sets of
// vectors are not actually being held to the same reference - a mistake this
// repo already made once, where tests/test_vulkan carried its own hardcoded
// hit for a different block than the Makefile's.
//
// Each vector is a real block: its 80-byte header including the nonce that
// actually won it, and the SHA3-256t digest that nonce produces, little-endian
// as the comparison against the target reads it.
//
// Vectors regenerate from any explorer:
//   curl https://bc3mempool.codefalcon.dev/api/block/<hash>/header

#pragma once

struct Sha3Vector {
    const char *name;
    const char *header;  // 80 bytes; the last 4 are the winning nonce
    const char *hitLE;   // the SHA3-256t digest, little-endian, as hashed
};

// Post-fork blocks: 30240 is the fork block itself, the rest are spread over
// the months since. All have version bit 12 set.
static const Sha3Vector kSha3Vectors[] = {
    // The fork block itself: the first SHA3-256t block on the chain, mined at
    // the difficulty-1 reset chainparams.cpp:256 mandates (nbits ffff001d).
    {"30240 (fork block)",
     "0010002030d2767637fc55d1f70edc5b20a41fddd17d8a12854ba7150400000000000000"
     "18dcbe866321fcf914b8e4f187158d285d7a1e6966e14e2b57e7071ded479309a067106a"
     "ffff001d05417434",
     "ab75656190cd4814d075c09f1a73bae25983d3f5ea884876875e537c00000000"},
    {"38337",
     "0010002093b06b66e411dacc7117dc3f79c2d801e388e202f1d5b0c15d8c410000000000"
     "c0b8ce9df398623782eb0eab5fe75ce7523a0e94cddcedd30017e69162241a3463bb136a"
     "ffff001c8e8df150",
     "0cd925995b5792bbfaabb3419b87ee0a480eb7fd2ad413fae8a6910000000000"},
    {"43713",
     "001000205c3c2d23b5c04686450247e171d468fa2efd423f3033c0af9dd3060000000000"
     "c9bc4904d71e0455f5ce43d9799dc1b75eaca5b47fd4f398338f08a4220fc269cc9a256a"
     "1ea4121b3673bcc6",
     "991083632107a9096d34bbe49ce043992f949088607e8b30f0e7060000000000"},
    {"50204",
     "00100020fefeebfabe324c205c624d70ba649c488e46f73b32712940284e0100000000008d"
     "ba9c56417ab4c217aef6ecee23c5a998d0c673ac7743ab376301a514c557579492616a0556"
     "131b4636619e",
     "8ee77b21a567d28d0cf0a7775e836f29e443a13e18c3fa860d00000000000000"},
    {"58000",
     "00100020624d55b7e520e8c6f534bbba0becad5747ba16162c7555d154c0000000000000a"
     "badc54d73961d26b0ba40ba431116f7ae91d399fda1a2651f4b93d26644a981879c826a36"
     "a9011b6af38dcb",
     "ec7dc8a7c6bd521f15532958649f5307e210ba01152c658a4c11000000000000"},
    // Current-epoch difficulty: nbits 1a6a4d80, the same value the live pools
    // were handing out when this was written.
    {"58600",
     "00100020f5b0415715052f9fcf731e4fc8ff597df7884750a3005634ae2c000000000000"
     "d31a91a3cd4a579d68966d884f50f306db72b2af27a69a50a260d7987f149d3576d8836a"
     "804d6a1a439f3af3",
     "3d35a40fb46d0c3ca517e07d8b2e1589360c20baaf8c1f92a666000000000000"},
};

static const int kSha3VectorCount =
    (int)(sizeof(kSha3Vectors) / sizeof(kSha3Vectors[0]));
