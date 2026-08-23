#!/usr/bin/env python3
"""Pin a real Ravencoin block as a KawPow test vector.

    python3 tests/kawpow_block.py            # a block ~1000 deep
    python3 tests/kawpow_block.py 4501000    # a specific height

Prints the arguments for tests/test_kawpow_block, and the Makefile lines that
pin them. It talks to a public Ravencoin RPC because the homelab runs no RVN
node; nothing here needs a wallet, a key or an authenticated call.

The interesting part is the byte order, which is where a miner silently
submits garbage:

  * The serialised header is 120 bytes for a KawPow block - the usual 80 plus
    nHeight (4), nNonce64 (8) and mix_hash (32). nHeight is inside the first
    80, which is exactly why it was added: the ProgPoW input stays 80 bytes.
  * The header hash handed to ProgPoW is the double-SHA256 of those first 80
    bytes, REVERSED, because Ravencoin passes it through uint256::GetHex().
  * mix_hash and the block hash come back in that same reversed order, so both
    can be compared directly against what an explorer shows.
"""
import hashlib
import json
import sys
import urllib.request

RPC = "https://rvn-rpc-mainnet.ting.finance/rpc"


def rpc(method, params):
    body = json.dumps({"jsonrpc": "1.0", "id": "soat", "method": method,
                       "params": params}).encode()
    # A default urllib User-Agent gets a 403 from this endpoint; curl does not.
    req = urllib.request.Request(
        RPC, data=body,
        headers={"content-type": "application/json",
                 "user-agent": "soat-miner-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    if d.get("error"):
        sys.exit(f"{method}: {d['error']}")
    return d["result"]


def main():
    if len(sys.argv) > 1:
        height = int(sys.argv[1])
    else:
        # Deep enough to be settled, not so deep the vector looks cherry-picked.
        height = rpc("getblockcount", []) - 1000

    block_hash = rpc("getblockhash", [height])
    raw = bytes.fromhex(rpc("getblockheader", [block_hash, False]))
    if len(raw) != 120:
        sys.exit(f"header is {len(raw)} bytes, not 120 - is height {height} "
                 f"before the KawPow fork at 1,219,736?")

    prefix = raw[:80]
    nonce = int.from_bytes(raw[80:88], "little")
    mix = raw[88:120][::-1].hex()
    bits = prefix[72:76][::-1].hex()
    header_hash = hashlib.sha256(hashlib.sha256(prefix).digest()).digest()[::-1].hex()

    print(f"height {height}  epoch {height // 7500}")
    print(f"block  {block_hash}")
    print()
    print(f"./tests/test_kawpow_block {header_hash} {height} {nonce:016x} \\")
    print(f"    {mix} {block_hash} {bits}")
    print()
    print("Makefile:")
    print(f"KAWPOW_BLK_HEADER = {header_hash}")
    print(f"KAWPOW_BLK_HEIGHT = {height}")
    print(f"KAWPOW_BLK_NONCE  = {nonce:016x}")
    print(f"KAWPOW_BLK_MIX    = {mix}")
    print(f"KAWPOW_BLK_HASH   = {block_hash}")
    print(f"KAWPOW_BLK_BITS   = {bits}")


if __name__ == "__main__":
    main()
