# Bitcoin III release preparation

Status: **release preparation only; not release-ready for publication**.

This document is the bounded artifact plan for `sha3-256t`. It contains no
wallet data, pool credentials, node data, GPU result, or generated release
archive.

## Proposed release contents

After every gate below is complete and separately approved, a release review
may contain:

- the normal Linux and Windows miner binaries built from one recorded commit;
- SHA-256 checksums generated from those exact archives;
- the BC3 source/build parity changes and the offline protocol fixture suite;
- `docs/bc3-readiness.md` and this release checklist;
- a release note that identifies BC3 as pool-only, requires an address-shaped
  payout input, and does not imply node, wallet, pool, or GPU validation.

No archive or binary is staged by this plan. The existing Ergo release assets
are not BC3 evidence and must not be relabeled.

## Required gates before publication

1. `make -B test-bc3-host` passes on the exact candidate worktree.
2. A CMake-equipped host runs `make test-bc3-cmake`; both CTest cases pass.
3. Windows CUDA CMake build and offline CTest run on the existing workflow.
4. The separately approved one-GPU six-vector gate passes with a recorded
   `gpulock` reservation, command, device, revision, timeout, and result.
5. Control reviews the dirty-worktree diff and selects an explicit commit
   boundary; unrelated Pearl/Ergo changes must not be included.
6. Control approves the package manifest and all GitHub lifecycle actions.

## Current blockers

- This worktree is an uncommitted feature branch with pre-existing BC3 edits;
  there is no clean release commit or release branch.
- CMake/CTest has now passed locally through the user-local CMake 4.4.2
  installation; Windows parity is still unexecuted.
- The six-vector CUDA test has not run and no GPU may be claimed under this
  checkpoint.
- No disposable local BC3 regtest node and wallet-disabled proof are supplied;
  no node path is part of the current miner and no node validation is implied.
- No release archive, checksum, tag, GitHub release, or CI run exists for BC3.

## Publication boundary

Until all gates and the commit boundary are approved, do not commit, push,
tag, create a GitHub release, upload an artifact, dispatch CI, start a miner,
contact a pool/node, access a wallet, or claim a GPU. Publication is a later
Control action, not evidence supplied by this document.
