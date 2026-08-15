// Autolykos v2 in OpenCL C - the portable backend (AMD, Intel, NVIDIA).
//
// Ported from the CUDA implementation in this directory, which is itself
// verified against the Ergo node's AutolykosPowScheme.scala and against real
// mainnet blocks. Both backends must produce identical hits; tests/ checks it.
//
// The dataset is split across several buffers on purpose. OpenCL guarantees
// only CL_DEVICE_MAX_MEM_ALLOC_SIZE per allocation, and that is commonly a
// quarter of VRAM - 6.3 GB on a 24 GB RTX 4090, against a 7.27 GB dataset. A
// single clCreateBuffer would simply fail, so elements are addressed across
// DS_CHUNKS buffers.

#define AL_K 32

// ---------------------------------------------------------------- blake2b --
__constant ulong B2B_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL,
    0xa54ff53a5f1d36f1UL, 0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};

__constant uchar B2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

#define ROTR64(x, n) (rotate((ulong)(x), (ulong)(64 - (n))))

#define B2B_G(a, b, c, d, x, y)     \
    do {                            \
        a = a + b + (x);            \
        d = ROTR64(d ^ a, 32);      \
        c = c + d;                  \
        b = ROTR64(b ^ c, 24);      \
        a = a + b + (y);            \
        d = ROTR64(d ^ a, 16);      \
        c = c + d;                  \
        b = ROTR64(b ^ c, 63);      \
    } while (0)

static void b2b_compress(ulong *h, const ulong *m, ulong t, int last) {
    ulong v[16];
    for (int i = 0; i < 8; i++) v[i] = h[i];
    for (int i = 0; i < 8; i++) v[8 + i] = B2B_IV[i];
    v[12] ^= t;
    if (last) v[14] = ~v[14];

    for (int r = 0; r < 12; r++) {
        __constant uchar *s = B2B_SIGMA[r];
        B2B_G(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        B2B_G(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        B2B_G(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        B2B_G(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
        B2B_G(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        B2B_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        B2B_G(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        B2B_G(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

static void b2b256_init(ulong *h) {
    for (int i = 0; i < 8; i++) h[i] = B2B_IV[i];
    h[0] ^= 0x01010020UL;
}

/** Single-block hash of a message shorter than 128 bytes. */
static void b2b256_short(const uchar *in, uint len, uchar *out) {
    ulong h[8];
    b2b256_init(h);
    ulong m[16];
    for (int i = 0; i < 16; i++) {
        ulong w = 0;
        for (int j = 0; j < 8; j++) {
            uint idx = i * 8 + j;
            if (idx < len) w |= ((ulong)in[idx]) << (8 * j);
        }
        m[i] = w;
    }
    b2b_compress(h, m, (ulong)len, 1);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uchar)(h[i] >> (8 * j));
}

/** M's little-endian message word at index w is just bswap64(w). */
static ulong m_word(uint w) {
    ulong v = (ulong)w;
    return ((v & 0x00000000000000ffUL) << 56) | ((v & 0x000000000000ff00UL) << 40) |
           ((v & 0x0000000000ff0000UL) << 24) | ((v & 0x00000000ff000000UL) << 8) |
           ((v & 0x000000ff00000000UL) >> 8) | ((v & 0x0000ff0000000000UL) >> 24) |
           ((v & 0x00ff000000000000UL) >> 40) | ((v & 0xff00000000000000UL) >> 56);
}

static uint bswap32(uint v) {
    return ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
           ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
}

/** H(idx | height | M) with the leading byte dropped -> 4 limbs, limb[0] low. */
static void gen_element(uint idx, uint height, ulong *limb) {
    ulong h[8];
    b2b256_init(h);
    ulong m[16];
    ulong t = 0;

    // block 0: idx(4 BE) | height(4 BE) | M[0..119]
    m[0] = (ulong)bswap32(idx) | ((ulong)bswap32(height) << 32);
    for (int j = 1; j < 16; j++) m[j] = m_word((uint)(j - 1));
    t += 128;
    b2b_compress(h, m, t, 0);

    // blocks 1..63: pure M
    for (uint b = 1; b < 64; b++) {
        uint base = 15u + (b - 1u) * 16u;
        for (int j = 0; j < 16; j++) m[j] = m_word(base + j);
        t += 128;
        b2b_compress(h, m, t, 0);
    }

    // final block: M word 1023 (8 bytes)
    m[0] = m_word(1023u);
    for (int j = 1; j < 16; j++) m[j] = 0;
    b2b_compress(h, m, 8200UL, 1);

    uchar d[32];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) d[i * 8 + j] = (uchar)(h[i] >> (8 * j));

    // d[1] is the most significant byte, d[31] the least.
    for (int l = 0; l < 4; l++) {
        int off = (3 - l) * 8;
        ulong v = 0;
        for (int j = 0; j < 8; j++) {
            uchar bv = (off + j == 0) ? 0 : d[off + j];
            v = (v << 8) | (ulong)bv;
        }
        limb[l] = v;
    }
}

// --------------------------------------------------------- dataset build --
__kernel void build_dataset(__global uchar *chunk, uint chunkFirst,
                            uint chunkCount, uint height) {
    uint t = get_global_id(0);
    if (t >= chunkCount) return;

    ulong limb[4];
    gen_element(chunkFirst + t, height, limb);

    __global ulong *p = (__global ulong *)(chunk + (ulong)t * 32UL);
    p[0] = limb[0];
    p[1] = limb[1];
    p[2] = limb[2];
    p[3] = limb[3];
}

// ---------------------------------------------------------------- search --
// Elements are addressed across DS_CHUNKS buffers of DS_CHUNK_ELEMS each.
#ifndef DS_CHUNKS
#define DS_CHUNKS 4
#endif

static void load_element(__global const uchar *c0, __global const uchar *c1,
                         __global const uchar *c2, __global const uchar *c3,
                         uint chunkElems, uint i, ulong *limb) {
    uint which = i / chunkElems;
    uint off = i - which * chunkElems;
    __global const ulong *p;
    if (which == 0) p = (__global const ulong *)(c0 + (ulong)off * 32UL);
    else if (which == 1) p = (__global const ulong *)(c1 + (ulong)off * 32UL);
    else if (which == 2) p = (__global const ulong *)(c2 + (ulong)off * 32UL);
    else p = (__global const ulong *)(c3 + (ulong)off * 32UL);
    limb[0] = p[0];
    limb[1] = p[1];
    limb[2] = p[2];
    limb[3] = p[3];
}

static void add256(ulong *acc, const ulong *v) {
    ulong carry = 0;
    for (int i = 0; i < 4; i++) {
        ulong s = acc[i] + v[i];
        ulong c1 = (s < acc[i]) ? 1UL : 0UL;
        ulong s2 = s + carry;
        ulong c2 = (s2 < s) ? 1UL : 0UL;
        acc[i] = s2;
        carry = c1 + c2;
    }
}

static int lt256(const ulong *a, const ulong *b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return 0;
}

__kernel void search(__global const uchar *c0, __global const uchar *c1,
                     __global const uchar *c2, __global const uchar *c3,
                     uint chunkElems, __global const uchar *msg_in,
                     ulong baseNonce, uint N, uint height,
                     __global const ulong *target_in, __global ulong *out,
                     __global uint *outCount, uint maxOut) {
    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_in[i];
    ulong target[4];
    for (int i = 0; i < 4; i++) target[i] = target_in[i];

    ulong nonce = baseNonce + (ulong)get_global_id(0);

    uchar buf[71];
    for (int i = 0; i < 32; i++) buf[i] = msg[i];
    for (int i = 0; i < 8; i++) buf[32 + i] = (uchar)(nonce >> (8 * (7 - i)));

    uchar d[32];
    b2b256_short(buf, 40, d);

    ulong prei8 = 0;
    for (int i = 24; i < 32; i++) prei8 = (prei8 << 8) | (ulong)d[i];
    uint i0 = (uint)(prei8 % (ulong)N);

    ulong limb[4];
    load_element(c0, c1, c2, c3, chunkElems, i0, limb);

    // 31 bytes of the element, big-endian
    uchar f[31];
    for (int j = 0; j < 7; j++) f[j] = (uchar)(limb[3] >> (8 * (6 - j)));
    for (int l = 0; l < 3; l++) {
        ulong x = limb[2 - l];
        for (int j = 0; j < 8; j++) f[7 + l * 8 + j] = (uchar)(x >> (8 * (7 - j)));
    }

    for (int i = 0; i < 31; i++) buf[i] = f[i];
    for (int i = 0; i < 32; i++) buf[31 + i] = msg[i];
    for (int i = 0; i < 8; i++) buf[63 + i] = (uchar)(nonce >> (8 * (7 - i)));
    b2b256_short(buf, 71, d);

    uchar ext[35];
    for (int i = 0; i < 32; i++) ext[i] = d[i];
    ext[32] = d[0];
    ext[33] = d[1];
    ext[34] = d[2];

    ulong acc[4] = {0, 0, 0, 0};
    for (int j = 0; j < AL_K; j++) {
        uint v = ((uint)ext[j] << 24) | ((uint)ext[j + 1] << 16) |
                 ((uint)ext[j + 2] << 8) | (uint)ext[j + 3];
        uint idx = v % N;
        ulong e[4];
        load_element(c0, c1, c2, c3, chunkElems, idx, e);
        add256(acc, e);
    }

    uchar sum_be[32];
    for (int l = 0; l < 4; l++) {
        ulong x = acc[3 - l];
        for (int j = 0; j < 8; j++) sum_be[l * 8 + j] = (uchar)(x >> (8 * (7 - j)));
    }
    b2b256_short(sum_be, 32, d);

    ulong hit[4];
    for (int l = 0; l < 4; l++) {
        ulong x = 0;
        for (int j = 0; j < 8; j++) x = (x << 8) | (ulong)d[(3 - l) * 8 + j];
        hit[l] = x;
    }

    if (lt256(hit, target)) {
        uint slot = atomic_inc(outCount);
        if (slot < maxOut) {
            out[slot * 5 + 0] = nonce;
            for (int i = 0; i < 4; i++) out[slot * 5 + 1 + i] = hit[i];
        }
    }
}

/** Single-nonce hit, for host-side verification and cross-backend tests. */
__kernel void verify_one(__global const uchar *c0, __global const uchar *c1,
                         __global const uchar *c2, __global const uchar *c3,
                         uint chunkElems, __global const uchar *msg_in,
                         ulong nonce, uint N, uint height, __global ulong *out) {
    __global ulong *fakeTarget = out;  // unused; keeps signature simple
    (void)fakeTarget;

    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_in[i];

    uchar buf[71];
    for (int i = 0; i < 32; i++) buf[i] = msg[i];
    for (int i = 0; i < 8; i++) buf[32 + i] = (uchar)(nonce >> (8 * (7 - i)));
    uchar d[32];
    b2b256_short(buf, 40, d);

    ulong prei8 = 0;
    for (int i = 24; i < 32; i++) prei8 = (prei8 << 8) | (ulong)d[i];
    uint i0 = (uint)(prei8 % (ulong)N);

    ulong limb[4];
    load_element(c0, c1, c2, c3, chunkElems, i0, limb);

    uchar f[31];
    for (int j = 0; j < 7; j++) f[j] = (uchar)(limb[3] >> (8 * (6 - j)));
    for (int l = 0; l < 3; l++) {
        ulong x = limb[2 - l];
        for (int j = 0; j < 8; j++) f[7 + l * 8 + j] = (uchar)(x >> (8 * (7 - j)));
    }

    for (int i = 0; i < 31; i++) buf[i] = f[i];
    for (int i = 0; i < 32; i++) buf[31 + i] = msg[i];
    for (int i = 0; i < 8; i++) buf[63 + i] = (uchar)(nonce >> (8 * (7 - i)));
    b2b256_short(buf, 71, d);

    uchar ext[35];
    for (int i = 0; i < 32; i++) ext[i] = d[i];
    ext[32] = d[0]; ext[33] = d[1]; ext[34] = d[2];

    ulong acc[4] = {0, 0, 0, 0};
    for (int j = 0; j < AL_K; j++) {
        uint v = ((uint)ext[j] << 24) | ((uint)ext[j + 1] << 16) |
                 ((uint)ext[j + 2] << 8) | (uint)ext[j + 3];
        ulong e[4];
        load_element(c0, c1, c2, c3, chunkElems, v % N, e);
        add256(acc, e);
    }

    uchar sum_be[32];
    for (int l = 0; l < 4; l++) {
        ulong x = acc[3 - l];
        for (int j = 0; j < 8; j++) sum_be[l * 8 + j] = (uchar)(x >> (8 * (7 - j)));
    }
    b2b256_short(sum_be, 32, d);

    for (int l = 0; l < 4; l++) {
        ulong x = 0;
        for (int j = 0; j < 8; j++) x = (x << 8) | (ulong)d[(3 - l) * 8 + j];
        out[l] = x;
    }
}
