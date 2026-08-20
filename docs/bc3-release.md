# Bitcoin III release preparation

Status: **Linux and Windows test packages built and verified on real hardware.**

This document is the bounded artifact plan for `sha3-256t`. It contains no
wallet data, pool credentials, node data, or recovery material.

## What BC3 actually is, and why that decides the gates

BC3 is **pool only**. `run.cpp` has no solo path for it: there is no node
call, no `getblocktemplate`, no coinbase built here, and no wallet read. The
only BC3 identity the miner handles is a payout address, checked for shape
before a socket is opened.

That matters because the earlier gate list blocked this work on a disposable
regtest node, a wallet-RPC-disabled proof, and a sweep-destination binding.
None of those exercise a code path BC3 has. They belong to a future solo or
sweep feature and are tracked in `bc3-readiness.md` as such. They are **not**
gates on a pool-mining test package, and treating them as gates is what kept
BC3 behind Ergo and Pearl.

## Gates for the pool-mining test package

| gate | state |
|---|---|
| `make -B test-bc3-host` on the candidate worktree | **passed**, 29 checks |
| CMake/CTest offline gate on a CMake host | **passed**, user-local CMake 4.4.2, both cases |
| One-GPU six-vector device gate, `gpulock` recorded | **passed**, see below |
| Clean commit boundary | **passed**, `73bca89`, 24 isolated files |
| Linux package built from that boundary | **passed**, see below |
| Windows CUDA build, run on a real card | **passed**, built on SOAT-MMICHELH, see below |
| Control review of the diff and manifest | outstanding |
| Publication approval | outstanding |

### Six-vector device gate, as run

- Claim `bc3-parity-20260819T163224Z` on the 4090, 15 minute window,
  `gpulock check` clean beforehand, released afterwards and verified free.
- `make -B tests/test_sha3_algo`, then the test under a 5 minute timeout.
- Revision `73bca89` plus the working-tree parity changes listed below.
- All six header vectors found a nonce, the hit matched, and `verify()`
  rejected both a corrupted hit and the wrong nonce. Exit 0.

### Linux package, as built

`make VERSION=0.2.11-bc3-test package` produced
`build/soat-miner_v0.2.11-bc3-test_Lin64.tar.gz`, SHA-256
`b84afb8c09d3fb30831b0adfab1d9d7235180fb66a450362d713043877ef310e`, with a
`.sha256` sidecar beside it. Rebuilt after the README and packaging edits, and
re-verified afterwards rather than assumed.

Verified by unpacking the archive and running it, not by reading the manifest:

1. the shipped binary lists `autolykos2`, `pearl-pow`, `sha3-256t`;
2. `./mine_bc3_pythonpool.sh` unedited selects CUDA, resolves the pool, and
   refuses `YOUR_BC3_ADDRESS_HERE` **before** opening a socket;
3. `BACKEND=vulkan` with `--algo sha3-256t` is overridden to CUDA;
4. `./soat-miner-vk --algo sha3-256t` refuses and names the CUDA binary.

No pool was contacted. The placeholder address is what makes that guaranteed
rather than promised.

## Windows, as built

Built by hand on **SOAT-MMICHELH** (RTX 4080, Windows 11), which is the fleet's
Windows CUDA build box. It already had the pairing the CI workflow pins: CUDA
12.6 and MSVC 14.44 from VS 2022 Build Tools. The stock pairing compiled
first time; `-allow-unsupported-compiler` was not needed.

The command mirrors the Linux one and the CI recipe: multi-arch fat binary
(sm_75/86/89 native plus compute_90 PTX) with `-cudart static`, so the `.exe`
carries no `cudart64` DLL. `Ws2_32.lib` comes from the `#pragma comment` in
`platform.h`, so sockets needed no extra flag.

`make VERSION=0.2.11-bc3-test windows-package` folded that build in and
produced `build/soat-miner_v0.2.11-bc3-test_Win64.zip`, SHA-256
`3701197a80526c81ac7fc181664c182e3d70e1820d44cca84f8c8a9af506329c`, containing
both binaries, the three VC++ runtime DLLs, and `mine_bc3_pythonpool.bat`.

Verified by shipping that zip back to the Windows box, extracting it clean and
running it:

1. `soat-miner.exe --list-algos` prints `autolykos2`, `pearl-pow`, `sha3-256t`;
2. `mine_bc3_pythonpool.bat` initialises the 4080 as `sm_89` (native code, not
   a PTX JIT), selects `sha3-256t`, resolves the pool, and refuses
   `YOUR_BC3_ADDRESS_HERE` before opening a socket.

**Caveat.** The bundled VC++ DLLs are in the zip but that box has Build Tools
installed, so its runtime is present system-wide. A clean Windows machine is
the only place that test means anything.

### The Windows cross-build was broken, and not only for BC3

`run.cpp` references `BitcoinStratumSource` unconditionally, but the Makefile's
`windows:` target never compiled `stratum_btc.cpp`. The Linux Vulkan target
already linked `stratum_btc_vk.o`; this one did not. So `make windows` failed
at link, which means **`windows-package` could not build any Windows archive at
all after the BC3 commit, Ergo included**.

Confirmed rather than reasoned about, by linking the old object list on purpose:

```
undefined reference to `om::BitcoinStratumSource::start(std::string*)'
undefined reference to `om::BitcoinStratumSource::takeJobWarning()'
undefined reference to `om::BitcoinStratumSource::stop()'
```

Fixed by compiling and linking `stratum_btc.o` in that target.

## Parity changes made in the working tree

- `packaging/mine_bc3_pythonpool.sh` and `.bat`, matching the Ergo and Pearl
  launcher pattern. Both go through `soat-miner.sh`/`.bat` so the backend is
  picked for the card.
- `packaging/soat-miner.sh`: the CUDA-only backend override was hardcoded to
  `pearl-pow`, so BC3 inherited the exact bug that once made Pearl mine Ergo
  and complain about an invalid Ergo address. It is a list now, and BC3 has
  since come off it - see below.
- `src/core/miner_vk.cpp`: the Vulkan main had no algorithm registry at all, so
  it is no longer possible for it to accept `--algo` and mine something else.
- `README.md`: BC3 sections, and it is named in "not done yet" including that
  no pool has ever accepted a BC3 share.
- `packaging/README-WINDOWS.txt`: the BC3 and Pearl launchers are listed, with
  why they bypass the backend picker.
- `Makefile`: the `windows:` target compiles and links `stratum_btc.o`, which
  is what unbroke the Windows cross-build.

## Publication boundary

Do not tag, create a GitHub release, upload an artifact, dispatch CI, contact
a pool, access a wallet, or resolve a destination reference without approval.
Pushing to GitHub additionally runs the `remove-ai-marks` inspection-only gate
first. Publication is a Control action, not evidence supplied by this document.


## Vulkan / AMD backend

BC3 shipped its first test builds CUDA-only, which does not meet the standing
rule that no algorithm ships without Vulkan and AMD. It does now.

- `src/algos/sha3-256t/kernel.comp`: Keccak-f[1600] and the triple-SHA3-256
  wrapper in GLSL, 64-bit lanes. The permutation body is generated from the
  same ROTC/PILN tables `sha3.h` uses, by `scripts/gen_keccak_round.py`, rather
  than hand-transcribed - 24 rho/pi assignments typed by hand is how a shader
  ends up producing plausible, wrong digests.
- `src/algos/sha3-256t/algo_vk.cpp`: the Algorithm. No dataset, so it is a
  fraction of the Autolykos backend. `verify()` runs on the HOST via `sha3.h`,
  as the CUDA backend does, which also makes every share the miner ever finds
  a host-vs-device cross-check of the GLSL against the reference.
- `src/core/vk_common.{h,cpp}` and `src/core/vk_registry.cpp`: the Vulkan main
  was structurally single-algorithm - `if (want != "autolykos2")` and a literal
  `--list-algos` string - and the device readout was read out of Autolykos's
  own instance pointer. Device handling and the registry are now shared, so a
  third Vulkan algorithm is a `.comp`, an `algo_vk.cpp` and two lines.
- `tests/test_sha3_vulkan.cpp`: the gate. Two halves - the raw shader against
  `sha3.h` over 4.7 M nonces per run in a dump mode that applies no target
  filter, and the real Algorithm against the same six mainnet blocks the CUDA
  gate uses, now shared via `tests/sha3_vectors.h` so the two backends cannot
  drift onto different references.
- `Makefile`: `$(WINDIR)/libvulkan-1.a` had no dependency on the Vulkan
  sources, so the generated `.def` went stale and any newly-used Vulkan symbol
  failed the cross-build at link. Fixed while adding
  `vkGetPhysicalDeviceFeatures` for the shaderInt64 check.

Measured, all with the correctness gate passing first:

| card | backend | MH/s | W |
|---|---|---|---|
| RTX 4090 | CUDA | 1543 | 432 |
| RTX 4090 | Vulkan | 1086 | 396 |
| RX 7900 XT | Vulkan | 540 | 289 |

NVIDIA users still want the CUDA binary; the Vulkan path exists so AMD has one
at all, and the launcher now routes to it instead of telling an AMD user that
BC3 does not exist for them.
