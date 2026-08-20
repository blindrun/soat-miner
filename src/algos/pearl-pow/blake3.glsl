// blake3 on the device in GLSL - the twin of blake3.cuh.
//
// Included, not compiled on its own, which is why it is .glsl and not .comp:
// a Makefile rule matching %.comp would otherwise try to build it as a shader
// and fail on the missing main(). Include it with
// GL_GOOGLE_include_directive and -I on the algo directory.
//
// Ported function for function from src/algos/pearl-pow/blake3.cuh. The two
// must stay byte-identical, so keep the shapes recognisable: if a change here
// has no counterpart there, one of them is wrong.
//
// FOUR DELIBERATE DIFFERENCES FROM THE CUDA, all forced by GLSL:
//
//  * Buffers are uint, not uint8_t. blake3 reads each 64-byte block as sixteen
//    little-endian words and every chunk is 1024-byte aligned, so on a
//    little-endian device that assembly IS a direct word load. Byte extracts
//    would cost four times the loads for the same value. Every Vulkan
//    implementation this targets is little-endian; the roots are compared as
//    bytes against the CUDA vectors, so the assumption is self-checking.
//
//  * The blake3 counter is carried as two uints rather than a uint64_t, so
//    nothing here needs shaderInt64. Chunk counts top out at 131072 (a 128 MB
//    matrix), so the high word is always zero - but it is passed explicitly
//    rather than assumed, exactly as the CUDA does.
//
//  * Chaining values are eight uints at a word offset, not 32 bytes at a byte
//    offset. Same bytes in memory on a little-endian device.
//
//  * g() is a macro, not a function. An `inout uint s[16]` parameter copies
//    the whole state in and out on each of the 56 calls per compression;
//    __forceinline__ has no GLSL spelling and the macro is what the existing
//    autolykos2 shader does for blake2b.
//
// The message permutation uses literal indices, as the CUDA does and for the
// reason autolykos2/kernel.comp documents: a runtime index into a
// per-invocation array forces it out of registers into scratch, which measured
// as a 75x slowdown there.

#ifndef PEARL_BLAKE3_GLSL
#define PEARL_BLAKE3_GLSL

const uint B3_IV0 = 0x6A09E667u, B3_IV1 = 0xBB67AE85u,
           B3_IV2 = 0x3C6EF372u, B3_IV3 = 0xA54FF53Au,
           B3_IV4 = 0x510E527Fu, B3_IV5 = 0x9B05688Cu,
           B3_IV6 = 0x1F83D9ABu, B3_IV7 = 0x5BE0CD19u;

const uint B3_CHUNK_START = 1u, B3_CHUNK_END = 2u, B3_PARENT = 4u,
           B3_ROOT = 8u, B3_KEYED_HASH = 16u;

const uint B3_CHUNK_WORDS = 256u;   // 1024 bytes
const uint B3_BLOCK_WORDS = 16u;    // 64 bytes
const uint B3_CV_WORDS = 8u;

uint b3Rotr(uint x, int n) { return (x >> n) | (x << (32 - n)); }

#define B3_G(a, b, c, d, mx, my)       \
    s[a] += s[b] + mx;                 \
    s[d] = b3Rotr(s[d] ^ s[a], 16);    \
    s[c] += s[d];                      \
    s[b] = b3Rotr(s[b] ^ s[c], 12);    \
    s[a] += s[b] + my;                 \
    s[d] = b3Rotr(s[d] ^ s[a], 8);     \
    s[c] += s[d];                      \
    s[b] = b3Rotr(s[b] ^ s[c], 7);

/**
 * The compression function, chaining-value output only.
 *
 * Seven rounds, with the message permuted after each of the first six. The
 * `r < 6` guard only saves a permutation nobody reads, so dropping it would
 * not change the output - but it is kept because every deviation from
 * blake3.cuh is one more thing to rule out when a root does not match.
 */
void b3Compress(inout uint cv[8], uint m[16], uint counterLo, uint counterHi,
                uint blockLen, uint flags) {
    uint s[16];
    s[0] = cv[0]; s[1] = cv[1]; s[2] = cv[2]; s[3] = cv[3];
    s[4] = cv[4]; s[5] = cv[5]; s[6] = cv[6]; s[7] = cv[7];
    s[8] = B3_IV0; s[9] = B3_IV1; s[10] = B3_IV2; s[11] = B3_IV3;
    s[12] = counterLo;
    s[13] = counterHi;
    s[14] = blockLen;
    s[15] = flags;

    for (int r = 0; r < 7; r++) {
        B3_G(0, 4,  8, 12, m[0],  m[1])
        B3_G(1, 5,  9, 13, m[2],  m[3])
        B3_G(2, 6, 10, 14, m[4],  m[5])
        B3_G(3, 7, 11, 15, m[6],  m[7])
        B3_G(0, 5, 10, 15, m[8],  m[9])
        B3_G(1, 6, 11, 12, m[10], m[11])
        B3_G(2, 7,  8, 13, m[12], m[13])
        B3_G(3, 4,  9, 14, m[14], m[15])
        if (r < 6) {
            uint t0  = m[2],  t1  = m[6],  t2  = m[3],  t3  = m[10];
            uint t4  = m[7],  t5  = m[0],  t6  = m[4],  t7  = m[13];
            uint t8  = m[1],  t9  = m[11], t10 = m[12], t11 = m[5];
            uint t12 = m[9],  t13 = m[14], t14 = m[15], t15 = m[8];
            m[0]  = t0;  m[1]  = t1;  m[2]  = t2;  m[3]  = t3;
            m[4]  = t4;  m[5]  = t5;  m[6]  = t6;  m[7]  = t7;
            m[8]  = t8;  m[9]  = t9;  m[10] = t10; m[11] = t11;
            m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
        }
    }

    cv[0] = s[0] ^ s[8];  cv[1] = s[1] ^ s[9];
    cv[2] = s[2] ^ s[10]; cv[3] = s[3] ^ s[11];
    cv[4] = s[4] ^ s[12]; cv[5] = s[5] ^ s[13];
    cv[6] = s[6] ^ s[14]; cv[7] = s[7] ^ s[15];
}

/**
 * Parent of two chaining values. `root` adds the ROOT flag.
 *
 * Buffer-agnostic, so it lives here rather than in a shader: both merkle_reduce
 * (root false) and merkle_root (root true) call it with values already in
 * registers or shared memory.
 */
void b3ParentCv(uint key[8], uint left[8], uint right[8], bool root,
                out uint outCv[8]) {
    uint cv[8], m[16];
    for (int i = 0; i < 8; i++) {
        cv[i] = key[i];
        m[i] = left[i];
        m[i + 8] = right[i];
    }
    uint flags = B3_KEYED_HASH | B3_PARENT | (root ? B3_ROOT : 0u);
    b3Compress(cv, m, 0u, 0u, 64u, flags);
    for (int i = 0; i < 8; i++) outCv[i] = cv[i];
}

/**
 * Keyed hash of exactly one 64-byte message: one chunk, one block, root.
 *
 * Not used by the three Merkle kernels. It is here because the commitment
 * kernels need it and a second, divergent GLSL blake3 is exactly the failure
 * this file exists to prevent - see b3HashBlock64Unkeyed below for the same
 * reason.
 */
void b3HashBlock64(uint key[8], uint m[16], out uint outCv[8]) {
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    b3Compress(cv, m, 0u, 0u, 64u,
               B3_KEYED_HASH | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) outCv[i] = cv[i];
}

/** As b3HashBlock64, unkeyed - the IV stands in and the KEYED flag is dropped. */
void b3HashBlock64Unkeyed(uint m[16], out uint outCv[8]) {
    uint cv[8];
    cv[0] = B3_IV0; cv[1] = B3_IV1; cv[2] = B3_IV2; cv[3] = B3_IV3;
    cv[4] = B3_IV4; cv[5] = B3_IV5; cv[6] = B3_IV6; cv[7] = B3_IV7;
    b3Compress(cv, m, 0u, 0u, 64u, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) outCv[i] = cv[i];
}

#endif  // PEARL_BLAKE3_GLSL
