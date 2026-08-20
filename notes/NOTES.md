There is a window now. Pick a coin, pick a pool, paste your address, click Mine.

The miner itself still runs exactly as before from the terminal, and the `mine_*.sh` and `mine_*.bat` scripts are unchanged. Nothing about the command line has been taken away.

## A graphical miner, for Linux and Windows

Linux is an AppImage, Windows is an installer, and both carry the miner binaries inside them. No unpacking an archive, no editing `config.txt`.

It offers every pool the miner ships - Ergo, Bitcoin III and Pearl - each with its fee, its reward scheme and the operational notes that matter, like a solo endpoint that is advertised but refuses connections. The payout field validates against the coin you picked, so an Ergo address pasted into a Bitcoin III run is caught before you mine to nowhere rather than after.

Backend selection is per algorithm, not per vendor. Bitcoin III takes CUDA on NVIDIA because it is about 40% faster there; Ergo takes Vulkan on Blackwell for the same reason in the other direction. The window says in words what Auto is about to do.

## Multi-GPU

The window finds your cards, lets you pick which to mine on, and runs one miner process per card with a row each and one total. Stop stops all of them.

## Gaming mode

A button. Press it and every card drops to a set share of full speed - 30% by default - so you can play without stopping mining. Press it again and they go back.

Two things it does not do, both stated in the window rather than buried here. It does not free VRAM: the algorithm's tables stay allocated for the whole run, so a game short of memory is still short of memory. And it still costs frames; 30% is a compromise, not a full recovery.

From the terminal it is `--gaming-throttle N`, or write the number into `~/.config/soat-miner/gaming-mode` and any running miner picks it up.

## Two silent share-losing bugs in Pearl, fixed

Both were invisible while running. The miner kept working, the panel kept printing a hashrate, and the share counter simply disagreed with what the pool credited.

`pearlPoolReplyAccepted` matched the literal text `"result":true`. A pool that puts a space after the colon - ordinary, conforming JSON - had every reply read as malformed, and the login failed with a message blaming the pool. The same function checked the greeting field first, so a pool that sent a welcome alongside a good result had an accepted share counted as a rejection.

`submit()` looked for its own reply by searching for `"id":N`, missing `"id": N` the same way, and its read budget counted incoming job pushes against itself. A pool sending a burst of jobs while we waited pushed the reply out of the window, so the share was recorded as "no reply from the pool" - and the pool credited it anyway.

Both are now parsed as JSON fields rather than matched as bytes, with the read bounded by time instead of line count. If Pearl behaved differently for you on one pool than another, this is why.

## Also

- The share counter no longer depends on a hand-maintained list of stratum types. Bitcoin III once had 367 shares credited by a pool while the readout showed zero, for exactly that reason.
- Both Vulkan binaries build from one source list, so the Windows cross-build cannot drift from the Linux one.
- Pearl's Vulkan backend is not in this release. Every shader is written and byte-identical to the CUDA path on both vendors, but the driver loop is unfinished, so `pearl-pow` remains NVIDIA and CUDA only.

## Windows: Defender will flag this, and it is not signed

Windows Defender flags every GPU miner, including this one, as a coin miner and refuses to launch it. That is the heuristic doing its job on a real miner, not a compromise of this build - the source is right here.

The installer offers to add a Defender exclusion for its own folder, and explains what that means before you accept. You can decline and install anyway.

There is no code signing certificate yet, so Windows will also warn about an unknown publisher before the installer runs: choose More info, then Run anyway. Verify the SHA-256 below if you would rather check than trust. If you want to help pay for a certificate, the funding link is on the repository.

No dev fee.
