#!/usr/bin/env python3
"""
Autolykos v2 reference implementation, ported from the Ergo node's
AutolykosPowScheme.scala + HeaderSerializer.scala.

Purpose: prove the algorithm is understood correctly by reproducing the
validity of REAL mainnet blocks, before any of it is written in CUDA.
"""
import json
import urllib.request
from hashlib import blake2b

K = 32
NBASE = 2 ** 26
INCREASE_START = 600 * 1024
INCREASE_PERIOD = 50 * 1024
N_INCREASE_MAX = 4198400
Q = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# M = 1024 big-endian int64s, 0..1023  => 8192 bytes
M = b"".join(i.to_bytes(8, "big") for i in range(1024))
assert len(M) == 8192


def h256(b: bytes) -> bytes:
    return blake2b(b, digest_size=32).digest()


def calc_n(height: int) -> int:
    h = min(N_INCREASE_MAX, height)
    if h < INCREASE_START:
        return NBASE
    iters = (h - INCREASE_START) // INCREASE_PERIOD + 1
    n = NBASE
    for _ in range(iters):
        n = n // 100 * 105
    return n


def put_ulong(v: int) -> bytes:
    """Scorex VLQ putULong: unsigned LEB128."""
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def decode_compact_bits(nbits: int) -> int:
    """Bitcoin-style compact difficulty encoding."""
    size = (nbits >> 24) & 0xFF
    body = nbits & 0x007FFFFF
    if size <= 3:
        return body >> (8 * (3 - size))
    return body << (8 * (size - 3))


def serialize_without_pow(h: dict, unparsed: bytes = b"") -> bytes:
    """Exact field order from HeaderSerializer.serializeWithoutPow."""
    w = bytearray()
    w.append(h["version"])
    w += bytes.fromhex(h["parentId"])
    w += bytes.fromhex(h["adProofsRoot"])
    w += bytes.fromhex(h["transactionsRoot"])
    w += bytes.fromhex(h["stateRoot"])          # 33 bytes
    w += put_ulong(h["timestamp"])
    w += bytes.fromhex(h["extensionHash"])       # NOTE: before nBits
    w += h["nBits"].to_bytes(4, "big")
    w += put_ulong(h["height"])
    w += bytes(h["votes"]) if isinstance(h["votes"], list) else bytes.fromhex(h["votes"])
    if h["version"] > 1:
        w.append(len(unparsed))
        w += unparsed
    return bytes(w)


def gen_indexes(seed: bytes, n: int) -> list:
    hh = h256(seed)
    ext = hh + hh[:3]
    return [int.from_bytes(ext[i:i + 4], "big") % n for i in range(K)]


def gen_element(idx: int, height_bytes: bytes) -> int:
    """H(j | h | M) with the leading byte dropped -> 31-byte big int."""
    return int.from_bytes(
        h256(idx.to_bytes(4, "big") + height_bytes + M)[1:], "big"
    )


def hit_for_nonce(msg: bytes, nonce: bytes, height: int, n: int) -> int:
    height_bytes = height.to_bytes(4, "big")
    prei8 = int.from_bytes(h256(msg + nonce)[-8:], "big")
    i = (prei8 % n).to_bytes(4, "big")
    f = h256(i + height_bytes + M)[1:]              # 31 bytes
    seed = f + msg + nonce
    idxs = gen_indexes(seed, n)
    f2 = sum(gen_element(j, height_bytes) for j in idxs)
    return int.from_bytes(h256(f2.to_bytes(32, "big")), "big")


def verify_block(h: dict) -> dict:
    n = calc_n(h["height"])
    msg = h256(serialize_without_pow(h))
    nonce = bytes.fromhex(h["powSolutions"]["n"])
    hit = hit_for_nonce(msg, nonce, h["height"], n)
    d = decode_compact_bits(h["nBits"])
    b = Q // d
    return {
        "height": h["height"],
        "msg": msg.hex(),
        "N": n,
        "hit": hit,
        "b": b,
        "valid": hit < b,
        "id_matches": None,
    }


if __name__ == "__main__":
    def get(u):
        return json.loads(urllib.request.urlopen(u, timeout=60).read())

    blocks = get("https://api.ergoplatform.com/api/v1/blocks?limit=6")["items"]
    ok = 0
    for blk in blocks:
        full = get(f"https://api.ergoplatform.com/api/v1/blocks/{blk['id']}")
        h = full["block"]["header"]
        r = verify_block(h)
        mark = "VALID  " if r["valid"] else "INVALID"
        ratio = r["hit"] / r["b"]
        print(f"h={r['height']} {mark} hit/b={ratio:.6f}  N={r['N']:,}")
        ok += r["valid"]
    print(f"\n{ok}/{len(blocks)} real mainnet blocks verified")
