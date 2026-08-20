# Pearl development handoff

- 2026-08-19: `main` is at `c452e4a` (shape selection now uses a timed real
  attempt). The pre-existing worktree edit is `src/core/telemetry.h`; it
  changes the interactive Pearl rate and efficiency labels from hash units to
  candidate units and must be preserved.
- The telemetry regression was the first safe step. GPU work requires
  `gpulock` to report the target free, a clear purpose-specific claim, and an
  immediate release after the bounded test.
- `make test-telemetry` passed (host-only; 2026-08-19).
- Pool JSON-RPC payloads now quote wallet, worker, pool job id, and proof
  values. `make test-pearl-pool` passed; `python3 tests/pearl_job.py` also
  passed all 50 host-side checks (2026-08-19). Neither test opens a socket or
  uses CUDA.
- Leave the current Pearl testnet node and gateway running. The live gateway
  test is optional and should be run only if its endpoint is known healthy.

## Remaining

- CLI/help and README now document Pearl pool precedence and gateway defaults;
  malformed explicit `--gateway HOST[:PORT]` values fail before CUDA setup.
  Verified with a temporary `sm_89` build and `tests/test_pearl_cli.sh`, plus
  `make test-pearl-pool`, `make test-telemetry`, and 50/50 `pearl_job.py`
  checks (all host-only; 2026-08-19).
- Release-gate preflight: GPU01 4090, game01 Linux 5080, Windows 4080, and
  Windows 4070 SUPER are reachable. `soat-bamf01` does not resolve; game01's
  Windows SSH account is unavailable while the Linux boot is online. The
  Bazzite 6700 XT is Matt's active gaming card and is intentionally read-only.
- GPU01 release gate passed: `SOAT-UB-GPU01` / Linux / RTX 4090 / `c452e4a`.
  Claimed 4090 with `gpulock`, built `make ARCH=sm_89 tests/test_pearl`, then
  ran rank-64 synthetic vectors with `timeout 120 ./tests/test_pearl
  /tmp/pearl-gpu01-vectors.bin`; every listed product, transcript, BLAKE3,
  target, and negative-control check passed. The initial rank-8 vector command
  was invalid and failed before a kernel launch; it was corrected to rank 64.
  The 4090 lock was released and confirmed free immediately afterward.
- Exact target outcomes: game01 Linux 5080 is reachable but has only a non-git
  release directory and no `nvcc`; its 5080 claim was released without a test.
  The 7900 XT SSH hostname does not resolve. game01 Windows cannot authenticate
  while the Linux boot is online. The 4080 is reachable but has a stale
  `wiki-repo-pearl` reservation, which was not stolen. The 4070 SUPER is
  reachable but has only an older standalone binary (with stale help), no
  current source/test checkout, and no `nvcc`; its claim was released without
  a test. See `scratchpad/handoffs/pearl.md` for the full matrix.
- Next safe task: provision an isolated current-source checkout plus matching
  CUDA toolchain on the reachable 5080/4070 targets, or obtain explicit
  clearance for the 4080 stale lock; then repeat only the bounded rank-64
  synthetic test one claimed GPU at a time. Do not run anything on the 6700 XT.
- The partial manual pearl5 Linux archive is superseded and no longer linked.
  The approved current UX package is the full repository package:
  `soat-miner_v0.2.7-pearl6-full-ux-test_Lin64.tar.gz`, SHA-256
  `06253633c90c0486c91f6f05bce0ffa04717d3df3a15bab253fe2f78989c4ef7`.
  `make VERSION=0.2.7-pearl6-full-ux-test package` passed; local and wiki-host
  inspections confirmed 44 entries, both CUDA/Vulkan binaries, all current
  launchers/config/JSON/docs/status/guard/service assets, and executable modes
  where required. The wiki deployment is commit `7318fc8`; public HTTP 200
  verification passed. Full command/result log: `scratchpad/handoffs/pearl.md`.
- Published-artifact follow-up: the Linux 5080 was busy with an existing
  `soat-miner` process and desktop/Steam activity, so its temporary UX-artifact
  run was not started and the gpulock claim was released. The Windows 4070
  SUPER was idle and claimed, but the published tarball is Linux-only and its
  PowerShell failed before extraction because .NET Framework v4.0.30319 is
  missing; no miner started and the claim was released. Next safe boundary:
  wait for 5080 workload clearance, and provision a current Windows package
  plus working PowerShell/.NET (or approved equivalent) for the 4070. Do not
  steal the 4080 reservation or touch the 6700 XT.
- A final read-only 5080 check still found PID 9114 `./soat-miner` (400 MiB,
  21% GPU). No fresh reservation was taken.

## Windows Pearl6 full package (2026-08-19)

- User explicitly cleared the stale `wiki-repo-pearl` 4080 reservation for this
  bounded Windows build. Read-only preflight on `SOAT-MMICHELH`
  (`miner@192.168.1.173`, Windows 10.0.26200.9168, RTX 4080, driver 596.36)
  showed 0% GPU and only 98 MiB desktop use. `gpulock steal 4080 "Current
  Windows Pearl full-package build and bounded validation on SOAT-MMICHELH"`
  was used; the later `gpulock release 4080` reported no active claim, while
  status reported the device free. No other workload was interrupted.
- Recovered the proven MSVC/CUDA route from `C:\Users\miner\build-win-cuda.ps1`:
  CMake 3.x, VS 2022, CUDA 12.6.85, static CUDA/runtime, architectures
  `75-real;86-real;89-real;90-virtual`. The old `soat-src`, `cuda-pkg`, and
  logs were inspected but not overwritten. A new source snapshot of the dirty
  local `c452e4a` worktree was extracted as `soat-src-pearl6-r2`; CMake built
  `build-win\\Release\\soat-miner.exe` successfully (7,971,840 bytes).
- Initial bounded `--algo pearl-pow --bench --plain --interval 1` exposed a
  real host-side defect: benchmark jobs had no Pearl header. `src/core/run.cpp`
  now creates a fixed synthetic 76-byte header/certificate envelope for Pearl
  bench jobs. After rebuilding, the same benchmark ran with no pool, wallet,
  node, gateway, or shares, reporting 241.4 M candidates/s. The agent-owned
  process was stopped by PID and the GPU was left free.
- Built the complete package with `make VERSION=0.2.7-pearl6-full-ux-test
  windows-package`. It contains 27 entries: CUDA `soat-miner.exe`, Vulkan
  `soat-miner-vk.exe`, VC runtime DLLs, all current batch launchers, JSON/config/
  status files, README, and license. SHA-256:
  `06d0942dc85687d9f74887813f254f2aff30d537f5d6fc381eae08c23aca94a5`.
  On the 4080 host, the extracted ZIP had the same SHA-256 and its packaged
  CUDA binary passed `--help` and `--list-algos` (`autolykos2`, `pearl-pow`).
- The approved wiki upload is live at
  `https://wiki.sonofatech.com/downloads/soat-miner_v0.2.7-pearl6-full-ux-test_Win64.zip`.
  Host archive listing and SHA-256 were verified; wiki page commit `de36b67`
  deployed successfully. The old Pearl2 Windows ZIP is now explicitly stale
  and superseded. No GitHub push, tag, release, or CI run occurred.
- Next safe release-gate task: re-preflight the busy Linux 5080 and run only a
  bounded synthetic/current-artifact test when it is genuinely idle; retain the
  7900 XT unreachable record, do not test Matt's 6700 XT, and do not touch the
  dual-boot Windows 5080 while Linux is active. A Windows 4070 artifact run
  still needs a functional PowerShell/.NET or approved transfer harness.

## Follow-up published-package validation (2026-08-19)

- Linux 5080 re-preflight: `SOAT-GAME-01` is still not safe. RTX 5080 reported
  33% utilization / 1,467 MiB and existing `./soat-miner` PID 9114 held
  400 MiB, with its desktop workload. The 5080 lock was deliberately not
  claimed and no artifact, benchmark, or process change was made. This is the
  remaining live-workload boundary.
- Windows 4070 SUPER: preflight on `DESKTOP-DK7KI6S` was idle (0%, 78 MiB
  compositor-only); built-in CMD `tar.exe`, `certutil.exe`, and `timeout.exe`
  are available even though PowerShell/.NET is not. Claimed 4070s, copied and
  extracted the published Pearl6 Windows ZIP via CMD, and `certutil` matched
  SHA-256 `06d0942dc85687d9f74887813f254f2aff30d537f5d6fc381eae08c23aca94a5`.
  Packaged `soat-miner.exe --help` and `--list-algos` passed.
- The direct CMD package benchmark `--algo pearl-pow --bench --plain --interval
  1` displayed `BENCHMARK ONLY`, no pool/wallet/shares, and sustained about
  158 M candidates/s on the RTX 4070 SUPER (device probe 169.8 M candidates/s).
  Local SSH timeout did not propagate termination to the remote child; this was
  caught immediately. The only post-preflight `soat-miner.exe` (PID 9924,
  agent-owned) was terminated with `taskkill /im soat-miner.exe /t /f`, leaving
  78 MiB compositor-only memory. `gpulock release 4070s` succeeded and status
  showed every tracked GPU free. Do not reuse that timeout pattern without
  explicit remote process cleanup.
- Next safe task is blocked solely by the active game01 Linux 5080 workload:
  wait for PID 9114/Steam-desktop activity to be absent, then re-preflight and
  run the same bounded package benchmark under one fresh 5080 claim. No 6700 XT
  action, 7900 XT action, dual-boot Windows 5080 action, or live node/gateway
  change is authorized by this state.

## Pearl HeroMiners UX connectivity report (2026-08-19)

- The published Pearl6 launcher and README consistently use
  `pearl.herominers.com:1200`; its generic `config.txt` is Ergo-only and is
  bypassed by the Pearl launcher. GPU01 and game01 both resolved the name to
  `46.4.231.165` and completed a bounded TCP/1200 handshake. No miner or pool
  protocol was started. The endpoint is not currently stale/unavailable or
  blocked on those Linux hosts; a reporting user's separate network or login
  remains unproven.
- Root cause for the confusing report is likely diagnostic quality: Pearl's
  pool source currently collapses connection, protocol close, login, and
  no-work conditions into `cannot reach`. A local-only, unbuilt/unpublished
  Control-review change in `src/core/pearl_pool.h` + `src/core/run.cpp` reports
  the actual failed stage without changing endpoint/protocol/retry behavior.
  `make test-pearl-pool` and `git diff --check` passed. Proposed review test:
  invalid hostname and refused local TCP endpoint only; do not contact a pool.

## STOP: Pearl dynamic pool-target defect (2026-08-19)

- User-confirmed pool rejection after a valid wallet makes Pearl6 unsuitable
  for any further pool test. The Pearl launcher has no manual difficulty flag;
  target/difficulty must be negotiated from each pool `mining.notify` job.
- Offline trace found that target hex is correctly parsed big-endian into
  little-endian limbs and scaled into the CUDA search bound, but target-only
  refreshes were discarded by `run.cpp` (same header/epoch) and by the Pearl
  algorithm cache (same header/cert). An older easier bound could therefore
  produce a candidate the current pool rejects. The staged local fix refreshes
  on target/job-id changes, invalidates the CUDA target cache, and rechecks the
  saved candidate against its exact bound before submit.
- `tests/test_pearl_pool.cpp` now uses synthetic notify inputs to cover endian
  conversion, target-only refresh, and a candidate that passes only the stale
  target. `make test-pearl-pool`, `make cuda`, and `git diff --check` passed.
  No live endpoint/protocol interaction occurred. Control must review and make
  a new full `soat-miner_v0.2.7-pearl7-dynamic-target-fix` package before any
  user retest. Safe user guidance: stop Pearl6; do not alter any network,
  wallet, pool, node, or gateway setting while waiting.

## REVIEW_READY — Pearl9 reconciled, share-bound change must be reverted (2026-08-19)

- Sole writer is the Claude lane. Read-only inspection of the single existing
  Pearl9 capture on `game01`; no pool, GPU, wallet, endpoint, firewall,
  package or publication action.
- The capture holds five `mining.notify` and **no submit and no reply**.
  Its target decodes exactly as `0xFFFF * 2^208 / 2097152`, the ordinary
  difficulty target, matching the `_2097152` job-id suffix.
- **The staged `Job::targetIsShareBound` change is wrong and must be reverted
  or gated before any package.** Using the pool target directly is 2^19 too
  hard: predicted one submit per 419 days on the 5080, which is why Pearl9
  submitted nothing in 71 s. Five retained 4090 logs show **12 accepted, 0
  rejected** in 571 s against this same pool with the penalized bound — one
  share every 47.6 s against a 50.5 s prediction. The pool applies the
  rank/work penalty on its side whatever the v1 wording says.
- `tests/test_pearl_pool.cpp` asserts the share-bound semantics as correct and
  must be inverted with the same change.
- The Ada/Blackwell contrast survives and is now config-matched: 4090 sm_89 at
  4096x65536 ptx k64 4x4/2x4 accepted 5/5; 5080 sm_120 at the identical shape
  and config was rejected. Earlier "14 rejections" is unverified — only one
  Blackwell log exists.
- Full facts, hypotheses and the T1/T2 test plan are in
  `scratchpad/handoffs/pearl.md` under "Pearl9 capture reconciled with the
  Blackwell evidence". Evidence preserved in `scratchpad/pearl9-capture/`.
- Paused for Codex **review only**. Do not edit, publish, or run another
  capture.

## Pearl10 published and verified (2026-08-19)

- `targetIsShareBound` removed; pool and gateway targets both scale once
  through `penalizedTarget()`. Everything else from Pearl7/8/9 kept.
- Host tests: pool, telemetry, 52/52 job vectors, 39/39 upstream Pearl oracle,
  CLI. GPU: `make ARCH=sm_89 test-pearl` on a claimed-and-released 4090,
  28 passed 0 failed, tile checks 0/524288 differ.
- The inverted pool test first passed against a deliberately broken penalty
  factor because `tests/test_pearl_pool` did not depend on `job.h`. Makefile
  fixed; assertion re-proved by mutation.
- Published to wiki.sonofatech.com/downloads, hashes verified on the host and
  over cache-busted HTTPS, `miner-test.md` updated and deployed:
  `soat-miner_v0.2.10-pearl-bound-fix_Lin64.tar.gz`
  `d270429089b7b83b9861e9305207d66773659048e9934e79a680bfdb8a584778`
  `soat-miner_v0.2.10-pearl-bound-fix_Win64.zip`
  `52a22fa8bb882863eae1b1d9d7ec6c5bc1481550f50c4bb793480a510e9da775`
  Windows parity is a real MSVC/CUDA build of the current source. A stale
  pre-revert exe of identical byte size was already staged in
  `build/win/cuda/`; hash staged binaries before packaging them.
- One authorized capture on the 4090: **SOLUTION ACCEPTED**, transcript reply
  `accepted:true`, stopped at the first reply. The v0.2.9 direct-authorize
  handshake is validated live.
- **The Blackwell rejection is still open and untouched.** No GitHub push.
  README speed correction still unpushed and awaiting sign-off.
- Detail in `scratchpad/handoffs/pearl.md`. Paused.

## Pearl "Invalid Pearl address" — root-caused, staged, NOT published (2026-08-19)

- Root cause: an **Ergo address was used for Pearl**. Confirmed by the user's
  own evidence that the same address mines Ergo. Different chains, different
  encodings; no setting makes a Pearl pool pay an Ergo address.
- The package invited it: `config.txt` documents one WALLET as "Your Ergo
  payout address", the Pearl launchers carried a second WALLET with no note
  that it is a different coin, and the Pearl pool path validated nothing but
  emptiness while the Ergo path has always validated locally.
- Ruled out with evidence: worker concatenation (authorize sends `wallet` and
  `worker` as separate fields), wrong JSON field, stale config injection
  (`--pool` sets EXPLICIT_SOURCE, and the Windows Pearl .bat never sources
  config), and the unedited placeholder (a placeholder cannot mine Ergo).
- **Makefile bug found and fixed: `CUDA_DEPS` listed no Pearl headers**, so
  editing any of them did not rebuild `soat-miner` and `make cuda` reported
  success without compiling. Any past "rebuilt and verified" Pearl result that
  touched only headers is unproven unless the nvcc line was seen.
- Staged: `pearlWalletProblem()` in `pearl_pool.h`, a fail-fast call in
  `run.cpp` before any socket, launcher text in both Pearl scripts, and the
  Makefile dependency fix. Nine host-only tests, mutation-proved. All host
  tests green. Nothing packaged, nothing published.
- Detail and the rebuild/release plan: `scratchpad/handoffs/pearl.md`. Paused
  for Control review.

## Jackpot-condition rejection re-diagnosed; TH/s mapping parked (2026-08-19)

- The reported "Jackpot condition not satisfied" is the **already-open
  Blackwell defect**, not a new target-path regression. Endian/width and
  share-bound scaling are both ruled out by the Pearl10 capture and the 13
  accepted 4090 shares; no VarDiff movement appears in either capture. The
  discriminator is the card, and the report did not say which one.
- Latent trap recorded: `verify()` checks against the bound captured when the
  win opened, not the current one. Same value in today's loop; a live bug with
  this exact error message if the loop is ever restructured.
- **Two different 524,288s.** MACs per candidate is `tileSize * k`;
  `penalizedTarget`'s factor is `tileSize * (k/rank) * kPenaltyBaseRank`. Equal
  only because rank == 128 == kPenaltyBaseRank; they diverge at rank 256.
  Host-only test added and mutation-proved.
- Ask the user for their `--pearl-transcript` file, which settles stale-job and
  endian for their run. Do not request another capture or the address.
- TH/s work parked mid-change: only `Algorithm::macsPerUnit()` (default 0) is
  in. Conversion, full surface inventory and the Autolykos carve-out are in
  `scratchpad/handoffs/pearl.md`. Nothing packaged or published.

## 5080 log inspected: same file, and it clears the target path (2026-08-19)

- `game01-documents-pearl.log` is **byte-identical** to the log already on file
  (sha256 b1ebefc8...). No address text in it, nothing to redact.
- It cannot answer the notify-target / submit-job-id / reply comparison: it is
  console output, and its Pearl7 build predates `--pearl-transcript`.
- New inference from the file alone: 20.53G candidates in 112 s with **one**
  submit. The correct penalized bound predicts 1.195; unscaled predicts 0;
  double-scaled predicts 626,526. The scaling in that run was right.
- Still the open Blackwell defect. Next test is T1b, the host-only case where
  `recheck()` regenerates A and B from the seeds instead of trusting device
  buffers - the missing test that let this pass local verification.
- Ask for a Pearl8+ transcript from the failing run, not a new capture.

## Host-only Pearl tests T1b/T1a/T1c landed (2026-08-19)

- `tests/test_pearl_job` 52 -> **67 checks, 0 failed**. All host-only, no CUDA.
- T1b proves `recheck()`'s blind spot: fed a tampered A it returns a stable
  self-agreeing digest, and regenerating from the seed is what catches it.
- **Found by mutation: the 52 existing vector checks never called
  `tileTranscript` or `powDigest`.** The two functions that decide share
  validity had no fixed-vector coverage until now.
- T1a checks the index inversion as properties; only the block-major property
  distinguishes it from a row-major one, and it does.
- T1c pins `win.bound` so a future loop restructure fails in tests, not at the
  pool.
- Each mutation-proved individually. Full host suite green.
- **No production change made**: `recheck()` still trusts device buffers.
  Making it regenerate from seeds is a device-path change and waits for review.
  T2 (sm_120 cross-build for game01) not started.

## Pearl status: host side now cross-checked at the suspected layer (2026-08-19)

- T1b/T1a/T1c were already done; re-verified. The copied 5080 log re-hashed as
  `b1ebefc8...`, still identical to the one on file, still a Pearl7 console log
  with no notify/submit/reply fields.
- **Closed the coverage hole**: the vectors now pin `tileTranscript` and
  `powDigest` for tile (0,0) against the Python reference. 67 -> **69 checks**.
- Mutation used was a one-slot shift of the transcript write order - the
  leading Blackwell hypothesis. It is caught, and it passes on the real code,
  so the host ordering is right and the fault is device-side.
- Full host suite green. Root cause still the sm_120 divergence.
- **Next and only advancing step is T2**: cross-build `tests/test_pearl` for
  sm_120 and run it on game01 under a bounded claim. Not started - needs
  clearance. No device change made.

## T2 ran on the 5080: CLEAN. No fix, nothing published (2026-08-19)

- `test_pearl` cross-built for sm_120 (CUDA 13.1, sm_120 SASS only, sha256
  d39649f4...), staged and hash-verified on game01.
- **Both vector sets pass on the 5080**: product 0/65536 differ, transcripts
  0/4096 differ, blake3 0/2048, negative control fires. Every kernel variant,
  ranks 64 and 128. **The transcript-order hypothesis is dead.**
- `tests/test_pearl` only launches the `noisyGemm*` family and `powCheck`. It
  does NOT cover `genMatrix`, `chunkCvs`/`reduceTree`, the noise kernels,
  `transposeKtoN` or `powScan`. That untested half is where the fault must be.
- Strongest remaining suspect: the **Merkle/commitment path**. A wrong commitA
  makes the noise wrong and every later step self-consistently wrong, which is
  exactly the observed "host agrees, pool does not" signature.
- Separate 4080 Windows check: published Pearl10 exe (sha e13006ac...) verified
  on host, `--bench` only, 211-216 MC/s, no shares, process cleanup confirmed.
  Ada under Windows is fine; says nothing about Blackwell.
- **Slip recorded**: a shell trap released the 5080 claim early, so the
  rank-128 run was unclaimed. Card was free and idle, nothing collided.
- No fix found, so **no package and no publication**. Blocker: there is no
  synthetic test for the prepare/commit path. Write the `merkleRoot` vs
  `MerkleTree` case first.

## Prepare/commit path clean on Blackwell; blocker corrected (2026-08-19)

- **Correction: `tests/test_pearl_prepare.cu` already existed** and already
  covered genMatrix, Merkle roots at 7 chunk counts, commitments, noise and the
  noising. My earlier "no such harness exists" came from reading only
  `test_pearl.cu`. It needed running on the 5080, not writing.
- Built sm_89, validated on the 4090 (28/28), then sm_120
  (sha ca12ffe1..., sm_120 SASS only) and run on the 5080: **28 passed, 0
  failed**. Merkle reduction correct at every level on Blackwell.
- gpulock discipline corrected: preflight, run and post-run check all inside
  ONE claim/trap/release scope, trap on EXIT/INT/TERM.
- Remaining uncovered kernels: **`powScan`**, `applyNoiseBt`,
  `transcriptFingerprint`. `powScan` is the leading suspect by elimination.
- Next: a `powScan` vs host-scan case in the same harness; if clean, instrument
  `openWin` to re-derive the digest from the serialised proof bytes.
- No fix, so nothing built and nothing published.

## powScan clean on both cards; every share-deciding kernel now verified (2026-08-19)

- Section 10 added to `tests/test_pearl_prepare.cu`: 9 checks comparing
  powScan's indices and digests against a host scan of the same transcripts.
- **5080 first, as asked: 37 passed, 0 failed.** sm_120 binary
  sha 94cbcdc7..., hash-matched on target, one claim/trap/release scope,
  post-run state verified settled (a 100% sample was an end-of-run artifact).
- Mutation on the 4090 (comparison limbs walked least-significant-first):
  **1003 hits instead of 8**, 4 checks fail. Restored 37/0. The test detects
  exactly the failure signature being hunted; neither card has it.
- `powScan` was the last uncovered kernel on the winning path. Remaining
  uncovered: `applyNoiseBt`, `transcriptFingerprint` - neither on the proof
  path.
- Conclusion: no single kernel is wrong on Blackwell. What is left is the live
  pipeline - stream ordering, buffer reuse, or proof serialisation - which a
  single-shot harness cannot reproduce.
- Next probe (NOT taken, needs approval): make `openWin` re-derive the digest
  from the serialised proof bytes and assert equality before submit.
- Host suite green, all GPUs free, nothing built, nothing published.

## openWin now verifies the proof before submitting (2026-08-19)

- Added `rowsFromProof` and `digestFromProof` to `job.h` (host-only), and a
  check in `openWin` that re-derives the digest **from the serialised proof
  bytes** and refuses to submit on mismatch or on a proof that does not open
  the rows the tile needs.
- This is the gap that made every prior test come back clean: `recheck()` is
  fed the device's own buffers and cannot disagree with it; this reads back
  only what the pool will reconstruct from.
- Regression: section 12 of `tests/test_pearl_job.cpp`, 8 checks,
  **69 -> 77, 0 failed**. Mutation-proved twice - zero-filling a missing leaf
  fails 1 check, ignoring `firstRow` fails 3.
- Full host suite green; `make cuda` compiles with the instrumentation in.
- **Honest gap: the new path has been compiled but never executed** - reaching
  it needs a winning tile on a real device. If it ever fires on the 5080 that
  is the answer; if it never fires and shares still reject, the fault is
  downstream of the proof bytes.
- Next (not taken, needs approval): a bounded `--bench` run on a claimed card,
  no pool, to exercise the path on Ada and Blackwell.
- Nothing built for release, nothing published.

## Proof recheck exercised on the 5080: 0 fires (2026-08-19)

- Bench cannot reach `openWin` (zero target, "never hits"), so added
  `SOAT_PEARL_BENCH_ALL_HIT`, bench-only and off by default. First try used
  `~0ull` and **the penalty-scaling overflow guard correctly refused it**;
  corrected to 2^236.
- Native sm_120 build (sha 276ec4e0...), 75 s on the 5080, 155M nonces:
  **0 "proof digest disagrees", 0 "does not open the rows", 0 host-verification
  failures.** The 294 -> 2.28 MC/s collapse is the evidence the path ran, since
  only per-attempt Merkle proof building costs that.
- Eliminates proof construction, leaf indexing, tile opening and row/column
  reconstruction on Blackwell.
- **Limit stated:** bench uses a synthetic header, not a pool job's. Same code
  path, different matrices.
- **Asterisk:** the check has never been observed to fire on a device. A mutant
  binary feeding the wrong tile index would settle it in one bounded bench.
  Not taken - not covered by the current authorisation. An Ada control run is
  also still missing.
- Remaining hypothesis: the submission encoding (`encodePlainProof`/base64) at
  a large shape and arbitrary tile, which the oracle only covers at m=n=128
  rows 16-31. Cheapest next attack and it is host-only.
- Nothing built for release, nothing published. All GPUs free.

## Proof recheck proven live in situ; Ada control clean (2026-08-19)

- Mutant 1 (wrong tile index) fired 22x on the 5080 but tripped the **rows
  guard**, not the digest comparison - it names rows the proof never opened.
  That proves the call site is live and proves nothing about the branch that
  matters.
- Mutant 2 (commitA passed as commitB: rows found, noise wrong) fired the
  **digest branch** 22x on the 5080, 0 rows-guard fires. Both branches now
  proven live on the card that produced the clean result.
- **Ada control**: same unmutated binary on the 4090, 71.3M nonces, 0 fires,
  0 host-verification failures. Identical to the 5080's 155M-nonce result.
- The proof path is correct on both cards. Every stage from matrix generation
  to proof re-derivation is now verified on sm_120.
- Remaining: (1) `encodePlainProof`/base64 at a large shape and arbitrary tile,
  which the oracle only covers at m=n=128 rows 16-31 - host-only and cheap;
  (2) anything that only appears against a real job header, since every bench
  used the synthetic 0xab one.
- Mutants are local artefacts only; `algo.cu` restored after each build. All
  GPUs free. Nothing built for release, nothing published.

## Wire encoding cleared at a far tile: local testing is exhausted (2026-08-19)

- New `tests/pearl_oracle_large.py`: 256x16384, B^t 32768 leaves / 15 tree
  levels, opens the **last** of 538 winning tiles (row 192, col 16352, B^t leaf
  32704). **Pearl's own verifier accepts it**, and rejects the same proof moved
  one tile along. **10 passed, 0 failed.**
- Two false starts caught by the tools: the verifier wants an
  `IncompleteBlockHeader`, not bytes, and takes its target from that header's
  `nbits`, so an arbitrary mining target would have failed regardless.
- Honest limit: not the real 4096x65536 (pure-Python scan, ~0.5 GB of
  intermediates). It reaches 15 tree levels vs the small oracle's 9 and leaf
  32704 vs 47, but not >65536 leaves. `putU64` is fixed-width so there is no
  cliff, but that is reasoning, not measurement.
- **Every stage is now verified on the rejecting card.** Nothing local
  reproduces the failure.
- **Stop writing tests.** The only evidence that can separate a pool-side from
  a job-data cause is a `--pearl-transcript` from a run that actually gets
  rejected. Ask for that file.
- Nothing built for release, nothing published, all GPUs free.

## Alternate pool AlphaPool accepts our shares (2026-08-19)

- `us2.alphapool.tech:5571` chosen after read-only DNS/TCP checks. K1Pool and
  Kryptex excluded because they authenticate with an internal account username,
  not a Pearl address. pearlpow closed, f2pool NXDOMAIN.
- **Accepted on the first share, 6 s, on the 4090.** notify job
  `00018ca7-6b459c5313386642`, `cert_version 3`, target = difficulty **50,000**
  (42x easier than HeroMiners' 2,097,152); submit 90,180 proof bytes; reply
  `accepted: true`.
- Establishes our object-param dialect is **not** HeroMiners-specific and the
  penalized bound is right against a second pool. Note the published V1 spec
  actually specifies array params with dotted `prl1addr.worker`, so this was
  not guaranteed.
- **Test-design mistake, owned: I ran it on the 4090, which was never the
  problem.** It confounds card with pool and does not answer the question. It
  does establish the precondition that the endpoint speaks our contract.
- **Next: the same capture on the 5080.** Accepted -> fault is
  HeroMiners-specific; rejected -> fault follows the card and is finally
  reproducible on demand. Caveat: at difficulty 50,000 a marginal fault could
  pass here, so acceptance is suggestive, rejection decisive.
- Artifacts in `scratchpad/altpool-capture/`, redacted. Nothing published.

## The 5080 is accepted on the alternate pool: not a Blackwell fault (2026-08-19)

- Same binary (`b3a61fd1...`), same endpoint, same address/worker as the
  accepted 4090 run; only the card differed. **AlphaPool accepted the 5080's
  share**, first verdict at 11 s, `accepted: true`.
- Matrix: 4090 accepted on both pools; 5080 rejected on HeroMiners, **accepted
  on AlphaPool**. The fault does not follow the card. This is the first live
  evidence agreeing with all the local testing.
- Caveat kept: AlphaPool difficulty 50,000 vs HeroMiners 2,097,152, a 42x lower
  bar. Strong evidence, not proof.
- **Observed structural difference:** HeroMiners pushes a new job every ~14-19 s
  (5 job ids in 71 s; 3 in 58 s) while AlphaPool sent one job for the whole
  capture. Sequential job ids at an unchanging target mean the header changes,
  so a share found near a rotation could be verified against the wrong header.
  Leading hypothesis - but the arithmetic does not fully close it, since the
  4090 also straddles rotations and went 13 for 13.
- **Only outstanding question:** a `--pearl-transcript` from a HeroMiners run
  that gets rejected. If the submitted job id is not the newest notify, it is
  job staleness and the fix is ours. If they line up, it is HeroMiners-side.
- Nothing built, nothing published, all GPUs free.

## ROOT CAUSE: prepare() never re-runs on a new Pearl pool job (2026-08-19)

- HeroMiners rejection **reproduced on the 5080 with a full transcript**:
  5 notifies (00000000..00000004_2097152), submit cited **00000004** - the
  newest - and the reply was `accepted:false, "Jackpot condition not satisfied"`.
- **The submit was NOT against a superseded job id.** The proof was mined
  against a superseded job's *header*. Only the transcript could separate those.
- Cause: `run.cpp:442` calls `algo->prepare(job)` only when
  `job.epoch != preparedEpoch`, and `pearl_pool.h` sets `job->epoch = 0`
  unconditionally (lines 266, 311, 440). So prepare runs **once**, on the job
  current at startup, and never again. The submitted job id advances while the
  miner keeps mining the first job's matrices.
- Explains everything: no kernel is wrong, they were all applied to the wrong
  header. Looks card-dependent because it races time-to-share (4090 ~50 s,
  5080 ~69 s) against rotation (~19 s). AlphaPool never rotated, so both cards
  passed there.
- **Fix (not applied):** re-prepare on any material job change, not just epoch.
  `PearlPow::prepare()` already caches on header/target/cert so it is free on
  the common path. Keep the epoch gate for Autolykos (7.27 GB rebuild).
  Add a host-only regression: two same-epoch jobs with different headers must
  re-prepare.
- Nothing built, nothing published. **Pearl10 on the wiki still has the bug.**

## ROOT CAUSE FOUND AND FIXED: an unsynchronised shared-memory reduction in the Merkle tree (2026-08-20)

**A 5080 share is accepted by HeroMiners.** Same pool, same wallet, same fixed
difficulty 2,097,152, on the card that had never had one accepted.

### The defect

`reduceTree` in `prepare.cuh` reduced a level of chaining values in shared
memory with **one** `__syncthreads()` per level, at the bottom of the loop.
Thread `t` reads slots `2t` and `2t+1` and writes slot `t`, so slot `s < half`
is read by thread `s/2` and written by thread `s` **in the same level**. Inside
a warp the lockstep load-then-store hides it. Across warps nothing ordered
them: thread 100 could overwrite slot 100 before thread 50 read it.

It resolved differently per architecture (fine on Ada, not on Blackwell) and it
scaled with block count, which is why **A came out right and B came out wrong on
the same card in the same attempt**: at 4096x65536, A is 8192 chunks / 16 blocks
and B is 131072 chunks / 256 blocks.

Fixed by splitting the level into read -> `__syncthreads()` -> write ->
`__syncthreads()`. No measurable hashrate cost (4090 328.1 -> 332.7 MH/s).

### Why every local test passed

`recheck()` and `digestFromProof()` both take `commitA`/`commitB` **from the
device**. A corrupted Merkle root makes the noise, the transcript and the digest
all wrong *together and all agreeing*. The pool is the first party that
recomputes the root from the leaves the proof opens, and the only thing it can
say is "Jackpot condition not satisfied: hash does not meet difficulty target".

The two device values nothing downstream could check were the Merkle root and
the commitment derivation. Both now have a host-side guard in `openWin()`, which
refuses to submit and says it is a GPU-side defect rather than a pool problem.

### Why the test suite missed it

`test_pearl_prepare.cu` section 2b sweeps chunk counts `{2,4,8,64,1024,4096,
8192}`. **8192 is exactly what A needs at m=4096. B needs 131072.** The sweep
stopped one order of magnitude short of the size the miner actually runs at.
Extended to 32768/65536/131072.

### Evidence chain, all on the 5080, all against HeroMiners

| build | result |
|---|---|
| no guard, no fix | 1 submit, rejected, `digest_within_bound: true`, header fingerprint == newest notify |
| guard, no fix | guard fires: "the device's Merkle root for B disagrees with the host's over the same bytes" |
| guard + fix | **3 submits, 3 accepted, 0 rejected** (two runs) |

4090 control with the fix in: 3 accepted, 0 rejected, `make test-pearl`
19/50/77/40 passed with 0 failed and every tile check 0 differ. Unaffected.
The extended Merkle sweep passes on the 4090 at all ten counts including
131072. No measurable hashrate cost.

### What this closes, with evidence rather than reasoning

- **Difficulty negotiation is not the fault and cannot be.** Probed
  `pearl.herominers.com` live: port 1200 is the only open port, the pool never
  sends `set_difficulty` or `set_target`, and it ignored `worker+50000`,
  `pass=x+50000`, `pass=d=50000`, `pass=50000` and `worker.50000` alike - every
  job still arrived at 2,097,152. The notify `target` decodes exactly as
  `0xFFFF * 2^208 / 2097152`, so the target channel was always correct.
  `set_difficulty`/`set_target` handling was added anyway (a pool that pushed
  one was previously ignored outright), with tests, but it is defensive: it
  never fires against this pool.
- **Shape is not the fault.** The 4090 pinned to `2048x65536` - the 5080's own
  shape, and the tuner independently picked the 5080's `ptx k64 4x4/2x4` there -
  went **9 accepted, 0 rejected**. This is the reverse of the test that had been
  run before (the 4090's shape on the 5080) and it is the stronger direction.
- **The bound arithmetic is right.** Accepted 4090 digests and the rejected 5080
  digest sit under the identical `bound_be` `0xFFFF * 2^206`. The pool does
  apply the rank/work penalty.

### Diagnostics added to `--pearl-transcript`

`mining.notify` now carries `header_b3`, and `mining.submit` carries
`header_b3`, `m`/`n`/`k`/`rank` read back **out of the serialised proof**,
`digest_be`, `target_be`, `bound_be` and `digest_within_bound`. That is what
separated "stale header" from "wrong bound" from "wrong proof content" in one
run instead of three. The submit's `header_b3` matching the newest notify's is
what finally closed job staleness for good - the job id alone could not, because
HeroMiners rewrites the header timestamp on every push.

### Shared-machine slip, recorded

Stopping a run on `game01-win` used `taskkill /F /IM soat-miner.exe`, which kills
**every** `soat-miner.exe` on that box by image name, not just this lane's. The
BC3 lane was running one on the same card. Nothing was lost this time (its
process was started after the last such kill), but this is the Windows form of
the `pkill -f` hazard the Proxmox rules already warn about. Kill by PID, from
`Get-CimInstance Win32_Process`, which also shows the command line so the two
lanes' miners can be told apart.

### Not done

Nothing committed, pushed, tagged or published. Not yet re-tested: the 4060 Ti
and the Vulkan backend. `test_pearl_prepare` has not been run on the 5080 at the
extended chunk counts (the card was mining); that is the cheap durable proof and
should be run before any release.

## Pearl on Vulkan: shader proven on two vendors, bigger blocker found (2026-08-20)

- `src/algos/pearl-pow/kernel.comp` compiled for the first time (vulkan1.3,
  8,008 bytes) and `tests/test_pearl_vk` run for the first time. **Passes
  byte-identical to CUDA on both the RTX 4090 (M16N16K32, subgroup 32) and the
  RX 7900 XT (M16N16K16, subgroup 64)**: product 0/65536, transcripts 0/4096.
  RDNA3 has real cooperative matrix - extension present, feature bit true,
  configs enumerated, checked in that order.
- The two cards disagree on both spec constants, so making K and the subgroup
  size specialisation constants is now proven necessary, not just prudent.
- **RDNA2 confirmed to have none**, read-only and with no load on Matt's card:
  the Bazzite box has three Vulkan devices and the coopmat extension and
  feature bit both belong to **llvmpipe** (GPU2), not the 6700 XT (GPU0). A
  plain `grep -c` over `vulkaninfo` says the opposite.
- Landed: `Makefile` only. `$(BUILD)/kernel_pearl.spv` on its own vulkan1.3
  rule, `tests/test_pearl_vk` depends on it, and `test-pearl` runs it.
- **Blocker: the remaining work is not `algo_vk.cpp`.** Pearl has one `.comp`,
  covering the GEMM and transcript only. Thirteen further CUDA kernels have no
  GLSL - genMatrix, the blake3 Merkle trio, three commitment kernels, the two
  noise kernels, three applyNoise kernels, two transposes and powScan. The
  blake3 Merkle tree in GLSL is the uncosted piece.
- **Second discrepancy:** `sha3-256t/algo_vk.cpp` and `src/core/vk_registry.cpp`
  named in the handoff do not exist on `algo/pearl`; they are on `algo/bc3` in
  the `/home/blindrun/Projects/soat-bc3` worktree. Branch ordering is a Control
  decision, not something to fix by editing another lane's files.
- CUDA-only special-casing left in place deliberately: "once it is real", and
  it is not yet.
- Nothing pushed. All GPU work under `gpulock` with a trap; all cards released.

## Merged develop into algo/pearl; blake3-in-GLSL costed (2026-08-20)

- `gitea/develop` (46f0bd3) merged as `48d272b`. `make vulkan` and `make cuda`
  both build; telemetry, pearl-pool, pearl_job (77/77) and bc3-destination pass.
- Four conflicts. Three were unions. `telemetry.h` took develop's sparkline
  removal and pulse **without** its duplicate `rateUnit`/`effUnit` declaration -
  this tree already declares them where the JSON path needs them, and a second
  declaration in the same scope does not compile.
- **Pre-existing break found, not caused by the merge:** `tests/test_pearl_pool.cpp`
  did not compile at `ad827bd` (undeclared `secret`). Verified against that
  commit directly. Fixed with a synthetic `prl1…` address.
- `vk_registry.cpp:24` deliberately still commented: uncommenting without
  `makePearlPowVK` defined is a link error, so it lands with `algo_vk.cpp`.
- **blake3 in GLSL is the safe part, not the risk.** 184 lines of pure uint32;
  this repo already ports blake2b to GLSL and documents the one pitfall; a
  110-line probe compiles to 20,648 bytes of SPIR-V (unexecuted); no device
  limit on the 7900 XT is close to binding.
- **Estimate: 10-15 working days.** Two to three weeks, not one, not a month.
  The uncosted piece is `algo_vk.cpp` (about ten pipelines) rather than blake3.
  No shortcut: the matrices are the nonce, so prepare cannot move to the host.
- Decision handed back to Control: this is a parity feature against a stated
  non-earner. If it stops, the proven GEMM shader stands and `make test-pearl`
  runs it.

## CLOSED: Pearl Vulkan stopped at a costed 10-15 days (2026-08-20)

- Control decision, not started. **10-15 working days** estimated against an
  algorithm measured as a non-earner on every AMD card at 11.4 c/kWh. KawPow
  and ProgPowZ got the weeks instead.
- **Banked and verified post-merge:** the GEMM shader is byte-identical to CUDA
  on an RTX 4090 (K32/sg32) and an RX 7900 XT (K16/sg64), and `make test-pearl`
  runs it. Commits `c05daba` and `48d272b`.
- **Lesson recorded in `build-miner` SKILL.md:** cost a Vulkan port by counting
  **pipelines, not shaders**. Pearl needs 16 device kernels and the shader
  covered one; Autolykos's `algo_vk.cpp` is 715 lines for a single pipeline.
- Ship gate amended to name Pearl as an explicit reasoned exception rather than
  leaving it silently violated. Pearl stays NVIDIA-only and experimental.
- Reopening requires arguing with the estimate, which is recorded with its
  evidence in `scratchpad/handoffs/pearl-vulkan.md`.

## FREEZE: algo/pearl, no push to origin, squash-only (2026-08-20)

- This lane owns `algo/pearl` and accepts the freeze. No `git push origin` has
  been run from this repo and none will be. Gitea unaffected.
- **When `algo/pearl` lands on main it lands by SQUASH merge**, so the eight
  address-carrying commits never reach GitHub's history.
- Verified my own edits are clean with `t?prl1[a-z0-9]{20,}` - **no leading
  word boundary**, because `tprl1` contains `prl1` and an anchored pattern
  misses the testnet form. My `test_pearl_pool.cpp` fixture is a placeholder.
- **The scrub is still uncommitted.** `HEAD:tests/test_pearl_pool.cpp` lines
  183 and 185 still hold both real addresses, and all eight unpushed commits
  carry them. The `git commit -a` sweep hazard is live now, not hypothetical.
  Left for 70 SECURITY to land - one file, one writer.
- Blast-radius reduction from this lane: committed `src/core/vk_common.h` alone
  by explicit path (`459ecf5`). No `git add -A` or `commit -a` here during the
  freeze.
- Separate, and **not** a GitHub exposure: the mainnet address is also in
  `wiki-repo` (`miner-test.md`, `projects/soat-miner-gui-test-bundle.md`). That
  repo has only the internal `deploy` remote, no `origin`.
