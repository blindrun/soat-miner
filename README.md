# SOAT Miner

An open-source CUDA miner for **Autolykos v2** (Ergo), built to be read rather
than trusted. Linux and Windows.

Every mainstream Autolykos miner is a closed-source binary. That is a poor fit
for a machine that also holds SSH keys, API tokens and real work — so this one
is source-available end to end, has **no developer fee**, and ships with the
test that proves it computes the right thing.

```
  SOAT Miner  autolykos2
  ────────────────────────────────────────────────────────────
  GPU        NVIDIA GeForce RTX 4090 (sm_89)
  Source     ergo node 127.0.0.1:9053 (solo)
  ────────────────────────────────────────────────────────────
  Hashrate     216.86 MH/s    avg 216.71 MH/s
  Power             167 W     efficiency 1.30 MH/W
  Temp               48 C     fan 31%
  Clocks           2340 MHz   mem 10251 MHz
  ────────────────────────────────────────────────────────────
  Epoch         1851444       dataset 7.27 GB
  Solutions  0 accepted
  Nonces        7.02 G
  Uptime      00:00:36
  ────────────────────────────────────────────────────────────
```

## Status

| | |
|---|---|
| Algorithm | Autolykos v2 (Ergo) |
| Hashrate | **217 MH/s** RTX 4090 (CUDA) · **82.9 MH/s** RX 6700 XT (Vulkan) |
| Dev fee | **none** |
| Correctness | verified against real mainnet blocks — see below |
| Backends | **CUDA** (NVIDIA) and **OpenCL** (AMD / Intel / NVIDIA) - both 217 MH/s |
| Work source | **pools (stratum)** and solo via your own node |
| Platforms | Linux (tested), Windows (builds; untested on hardware) |

For reference, lolMiner does ~265 MH/s on a 4090 and takes a 0.75–1% dev fee.
This is ~82% of that with none, and you can read it.

### Hashrate depends on the chain height, a lot

Autolykos v2's dataset grows over time, and this workload is memory bound, so
hashrate falls as the dataset outgrows cache. Measured on one RX 6700 XT, same
binary, only `--bench-height` changed:

| Epoch | Dataset | Hashrate |
|---|---|---|
| 2021 (h=614,400) | 2.25 GB | **235.2 MH/s** |
| 2023 (h=1,200,000) | 3.86 GB | **154.2 MH/s** |
| today (h=1,851,444) | **7.27 GB** | **82.9 MH/s** |

Most published Autolykos figures date from 2022, when the dataset was ~2.9 GB.
Compare like with like before concluding a miner is slow — including this one.

### Both paths are at hardware limits

The dataset build is ALU-saturated on both vendors, and the difference between
the cards is exactly the difference in their ALUs:

| GPU | Build | Throughput | % of ALU peak |
|---|---|---|---|
| RX 6700 XT | 9.86 s | 5.03 T ops/s | 81.3% |
| RTX 4090 | 1.47 s | 33.76 T ops/s | 81.8% |

Build-time ratio 6.71×, ALU capacity ratio 6.66×. The search path is memory
bound: removing its 33 random gathers takes it from 82.9 to 780 MH/s.

Optimisations tried and **rejected as measurably useless or harmful**, so
nobody repeats them: LDS-cached `M` table (cost 11 MH/s to occupancy),
`N` as a specialization constant (no change, and it baked `N` into the
pipeline so it would break at the next epoch), wave32 vs wave64 (no change),
8-wide batched loads for memory-level parallelism (71.4 vs 82.9), unrolling
the message-array fills (no change — the compiler already did it).

## Correctness

A miner that hashes slightly wrong finds nothing and tells you nothing is
wrong. So correctness is pinned to real chain data at three levels:

1. **`tests/reference.py`** — Autolykos v2 in plain Python, ported from the
   Ergo node's `AutolykosPowScheme.scala`. It fetches recent mainnet blocks,
   recomputes each header digest and hit from scratch, and asserts each block's
   own nonce satisfies its own target. Currently **6/6 blocks verified**.
2. **`tests/test_element.cu`** — the CUDA dataset elements must match the
   Python reference exactly, including index 0, the 1023/1024 block boundary,
   and the final element.
3. **`tests/test_hit.cu`** — builds the real 7.27 GB dataset for a real block's
   height and reproduces that block's hit from its winning nonce, byte for
   byte.

```bash
make test
```

Run these after any change. If they pass, the miner agrees with consensus.

## Build

**Linux**

```bash
make                    # builds BOTH backends
make cuda               # NVIDIA only
make opencl             # AMD / Intel / NVIDIA - no vendor toolchain needed
make package            # release tarball

make ARCH=sm_89         # defaults to sm_89 (Ada / RTX 40xx)
make ARCH=sm_86         # Ampere / RTX 30xx
make ARCH=sm_75         # Turing / RTX 20xx, 16xx
```

**Windows** (CUDA Toolkit + Visual Studio Build Tools)

```powershell
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

CMake works on Linux too if you prefer it.

## Run

```bash
./soat-miner --bench                 # benchmark, no node needed
./soat-miner --node 127.0.0.1        # solo mine against your node
./soat-miner --plain                 # one-line JSON per interval, for logs
./soat-miner --help
```

The readout auto-detects a terminal. Under systemd it emits one JSON object
per interval instead of redrawing a panel into your log file:

```json
{"ts":18,"algo":"autolykos2","mhs":216.83,"watts":183.0,"temp_c":49,
 "eff_mh_w":1.185,"epoch":1851444,"accepted":0,"rejected":0}
```

### Pool mining

Edit `config.txt`, then run the launcher:

```bash
./soat-miner.sh            # Linux
soat-miner.bat             # Windows
```

Or use a ready-made script — edit `WALLET` at the top and run it:

| Script | What it does |
|---|---|
| `mine_ergo_herominers.sh` / `.bat` | pool mine to HeroMiners |
| `mine_ergo_woolypooly.sh` / `.bat` | pool mine to WoolyPooly |
| `mine_ergo_solo.sh` / `.bat` | solo against your own node |
| `benchmark.sh` / `.bat` | benchmark, no pool or node needed |

Command-line flags override `config.txt`, so these work without editing it.

```ini
WALLET=9yourErgoAddress...
POOL=ergo.herominers.com:1180
WORKER=rig1
BACKEND=auto               # auto | cuda | opencl
```

Or directly:

```bash
./soat-miner --pool ergo.herominers.com:1180 --wallet 9yourAddr --worker rig1
```

The stratum protocol was captured from live pools rather than taken from
documentation, because Ergo stratum is a de-facto standard with no spec. Two
details that are easy to get wrong and are handled here: `mining.notify`
param 6 is the target in **decimal**, not a difficulty and not hex; and the
subscribe reply's extranonce is a **prefix** of the 8-byte nonce, so nonces
must be generated inside that subspace or every share is rejected.

### Solo mining

Leave `POOL` empty in `config.txt` and point it at your own node.

```bash
# on the node, ergo.conf:
ergo {
  node {
    mining = true
    useExternalMiner = true
  }
}
```

The **reward address is configured on the node, not here.** The miner asks the
node for work and hands back solutions; the node decides who gets paid.

Solo is the default on purpose. At 217 MH/s against a ~510 GH/s network you
expect a block roughly every **3 days** — high variance but not absurd — and
solo mining is the one thing that improves Ergo's mining centralisation rather
than worsening it. As of 2026-08-15 a single miner held **53–61% of network
hashrate for 40+ days straight**, so where you point hashrate matters.

## Guard: mining around real GPU work

`scripts/soat-miner-guard.py` starts and stops the miner so it never competes
with actual work. It stops mining when:

- **Ollama has a model loaded** (`/api/ps`) — i.e. Qwen was asked something
- **ComfyUI has anything running or queued** (`/queue`)
- **any other process appears on the GPU** — by name (`llama-server`, `ollama`)
  immediately, or by VRAM threshold for anything unrecognised
- it is **outside the allowed hours** (summer default avoids the heat of the day)
- the **GPU is too hot**

Stopping is immediate; starting waits for a short settle period, because
rebuilding the dataset costs ~4 s and thrashing between two prompts is worse
than losing a few seconds of mining.

**Why name-based detection matters.** On a 24 GB card, `qwen3:8k` needs
21.9 GB and the Autolykos dataset is 7.3 GB — 29.2 GB total, 4.6 GB over
capacity. They cannot coexist. Waiting for a VRAM threshold to trip means
reacting *after* the allocation has already failed, so known LLM processes are
treated as busy the moment they appear, at any size.

Install:

```bash
sudo install -Dm755 soat-miner            /opt/soat-miner/soat-miner
sudo install -Dm755 scripts/soat-miner-guard.py /opt/soat-miner/soat-miner-guard.py
sudo install -Dm644 scripts/guard.conf.example  /etc/soat-miner/guard.conf
sudo install -Dm644 scripts/soat-miner.service       /etc/systemd/system/
sudo install -Dm644 scripts/soat-miner-guard.service /etc/systemd/system/
sudo useradd --system --no-create-home --shell /usr/sbin/nologin soatminer
sudo mkdir -p /var/log/soat-miner && sudo chown soatminer /var/log/soat-miner
sudo systemctl enable --now soat-miner-guard
```

Only the guard is enabled. It starts the miner when appropriate; you never
start `soat-miner.service` directly.

Set `DRY_RUN=true` in `guard.conf` first and watch `/var/log/soat-miner/guard.log`
to confirm its decisions before letting it touch anything.

## Tuning

Measured on an RTX 4090:

| Power limit | Hashrate | Actual draw | Efficiency |
|---|---|---|---|
| 450 W | 217.5 MH/s | 221 W | 0.98 MH/W |
| **183 W** | **217.2 MH/s** | **167 W** | **1.30 MH/W** |
| 150 W | 170.0 MH/s | 134 W | 1.27 MH/W |

183 W costs **0.3 MH/s for 24% less power**. The guard applies this
automatically and restores the full limit the moment it stops.

## How it works

Per nonce, after the dataset exists:

```
prei8 = H(msg || nonce)[24..32]      1 compression
i     = prei8 mod N
f     = element(i)                   1 random read
seed  = f || msg || nonce            (31 + 32 + 8 = 71 bytes)
idx   = 32 indices from H(seed)      1 compression
sum   = SUM element(idx[j])          32 random reads
hit   = H(sum as 32-byte BE)         1 compression
```

Three Blake2b compressions and **33 random reads** into a 7.27 GB table. At
217 MH/s that is ~230 GB/s of random access against ~1008 GB/s of peak
bandwidth — this is a **memory-latency bound** workload, not a compute bound
one, which drives every optimisation decision here.

Two that mattered, and one that did not:

- **`M` is never stored.** The 8 KB constant is 1024 big-endian int64s, so
  message word *w* is just `bswap64(w)` — one instruction instead of 8 KB of
  traffic, 227 million times per dataset build.
- **Register pressure is the ceiling.** The obvious Blake2b stages its message
  through a `uint8_t buf[128]`, and those 128 bytes live in registers. Removing
  it, plus `__launch_bounds__(256, 2)`, took the kernel from 250 registers
  (~16% occupancy) to 128 with zero spills: **180 → 218 MH/s**.
- **Replacing the 33 integer modulos with a precomputed reciprocal did
  nothing** (218.2 → 216.7 MH/s). A useful negative result: it confirms the
  kernel is waiting on memory, not arithmetic. The code was reverted.

## Adding an algorithm

The core owns the nonce counter, job lifecycle, stop signals and reporting.
An algorithm only answers "test these nonces, tell me which won".

1. Create `src/algos/<name>/` and implement `om::Algorithm` (`src/core/algo.h`):
   `name`, `memoryBytes`, `prepare`, `search`, `verify`, `release`.
2. Add a factory line to `src/core/registry.cu`.
3. Add the object to `ALGO_OBJS` in the `Makefile` and to `CMakeLists.txt`.

`prepare()` is called whenever the job's epoch changes, which covers
DAG/dataset-per-epoch designs as well as algorithms needing no memory at all.
`Job` carries a 32-byte message plus a 256-bit target; anything chain-specific
rides in `Job::extra` and stays opaque to the core.

To add a pool instead of an algorithm, implement `JobSource` in
`src/core/miner.cu` — that is the only thing stratum support requires.

## Two backends

`soat-miner` is the CUDA build; `soat-miner-cl` is the OpenCL build. They are
verified to produce byte-identical hits, and on an RTX 4090 they benchmark the
same (217 MH/s), so OpenCL is not a slow fallback - it is a genuine portable
build that happens to also be the only one AMD can run.

OpenCL compiles its kernels at runtime, which is why an AMD build needs no AMD
machine and no ROCm/HIP toolchain to produce.

One OpenCL-specific constraint: `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is commonly a
quarter of VRAM (6.3 GB on a 24 GB card) while the dataset is 7.27 GB, so the
dataset is split across up to four buffers and addressed across them.

## Not done yet
- **Windows is built but untested on real hardware.** The platform shim covers
  Winsock, `_isatty`, and MSVC's lack of `__int128`; it compiles, but nobody
  has run it on Windows yet.
- Multi-GPU. Single device today.

## License

MIT. See [LICENSE](LICENSE).

Mining is not free money — read the economics before you run this. On the
hardware above it nets a few dollars a month at $0.115/kWh, declining ~25% per
quarter as Ergo's block reward steps down.
