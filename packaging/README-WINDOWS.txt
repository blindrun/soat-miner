SOAT Miner - Windows
====================

WHAT IS IN HERE
  soat-miner.exe      CUDA build. NVIDIA only, faster on Ada and older.
  soat-miner-vk.exe   Vulkan build. Works on AMD, Intel and NVIDIA.
  msvcp140.dll        VC++ runtime, needed by the CUDA build only.
  vcruntime140.dll      "
  vcruntime140_1.dll    "
  soat-miner.bat      launcher, reads config.txt, picks the backend for you
  mine_ergo_*.bat     ready-made pool / solo scripts - edit WALLET and run
  ergo-pools.json     reviewed pool endpoints and their source URLs

  mine_pearl_*.bat    Pearl pool script - edit WALLET and run
  mine_bc3_*.bat      Bitcoin III pool scripts - edit WALLET and run.
                      One per pool, so you are not stuck with ours:
                        mine_bc3_pythonpool.bat        0%   solo
                        mine_bc3_axehub.bat            0%   solo
                        mine_bc3_btc3forge.bat         0.5% proportional
                        mine_bc3_hashbay.bat           0.5% proportional
                        mine_bc3_vexta.bat             0.5% PPLNS
                        mine_bc3_argfamining_solo.bat  1%   solo
                        mine_bc3_argfamining_prop.bat  1%   proportional
                        mine_bc3_rplant.bat            1%   proportional
                        mine_bc3_cryptoeire.bat        1%   solo
                        mine_bc3_hashforge.bat         1%   solo
                      pythonpool and axehub are the only 0% ones. Every
                      endpoint above was checked by hand: it answers
                      mining.subscribe, accepts a BC3 address and sends real
                      work on the live chain.
  benchmark.bat       benchmark, no pool or node needed

WHICH ONE RUNS
  All three algorithms have Vulkan shaders now, so every mine_*.bat goes
  through soat-miner.bat and it picks by GPU, not by brand. You do not choose:
    - AMD and Intel            -> Vulkan (soat-miner-vk.exe)
    - NVIDIA 50-series         -> Vulkan, ~22% faster than CUDA on Blackwell
    - NVIDIA 40-series / older -> CUDA (soat-miner.exe), ~34% faster on Ada
  Override it with BACKEND=cuda or BACKEND=vulkan in config.txt.

ABOUT THE CUDA BUILD
  It used to be missing from this archive because nvcc cannot cross-compile
  from Linux. It is built on a Windows host now (the windows-cuda CI job) and
  ships here.

  It is a fat binary: native code for Turing, Ampere and Ada, plus PTX so
  Blackwell and Hopper still run (the driver compiles the PTX at startup).
  The CUDA runtime is linked statically, so there is no cudart64 DLL to chase
  and no CUDA Toolkit to install - it runs on a clean Windows machine.

  The three VC++ DLLs above are the one dependency it cannot static-link
  cleanly, so they are bundled. Keep them next to soat-miner.exe.

REQUIREMENTS
  Windows 10 or 11, 64-bit, and a current GPU driver.

  The Vulkan build needs nothing else (vulkan-1.dll ships with all current
  AMD/NVIDIA/Intel drivers and the binary is statically linked).

  The CUDA build needs the three bundled VC++ DLLs, which are in this archive.
  Nothing to install.

  The dataset needs about 7.3 GB of VRAM at today's chain height, so a card
  with 8 GB or more.

QUICK START
  1. Double-click benchmark.bat to check it sees your GPU.
  2. Open mine_ergo_herominers.bat in Notepad, set WALLET to your Ergo address.
  3. Double-click it.

For another reviewed conventional pool, use the matching `mine_ergo_*.bat`
launcher. `ergo-pools.json` is the source of truth; its K1Pool launchers use a
K1Pool `Kr_` account instead of an Ergo wallet. `ergo-pool-status.json` is a
read-only MiningPoolStats snapshot, not a list of Stratum addresses.

PEARL
  Set WALLET in a mine_pearl_*.bat to a PEARL address - it starts prl1 and is
  about 63 characters. An Ergo address cannot be paid by a Pearl pool.

  These go through soat-miner.bat like every other script. They used to call
  soat-miner.exe directly, which was right while Pearl was CUDA-only and became
  wrong the day it got a Vulkan backend - an AMD user was told the binary was
  missing rather than being sent to the one that has it.

  On NVIDIA the launcher still picks CUDA for Pearl: its shape and tile tuner
  has no Vulkan equivalent yet. On AMD it picks Vulkan, which is the only
  backend there. The Vulkan path is new - the miner checks the whole chain
  against its own reference at startup and refuses to mine if a card computes
  it wrongly, and the end-to-end run has so far been confirmed on NVIDIA.

  There is one launcher per pool. Each file's header carries that pool's fee,
  payout scheme, alternate regions and ports, and the share difficulty it
  actually handed out when it was tested:

    mine_pearl_suprnova.bat     prl.suprnova.cc:3373            0%    diff 244
    mine_pearl_alphapool.bat    us2.alphapool.tech:5571         0%    diff 50000
    mine_pearl_rabbitminer.bat  nl.rabbitminer.cc:1902          1%    diff 232827
    mine_pearl_baikalmine.bat   pearl-eu.baikalmine.com:2010    0.5%  diff 262144
    mine_pearl_luckypool.bat    pearl-us-east.luckypool.io:3360 1%    diff 888888
    mine_pearl_k1pool_solo.bat  us.pearlsolo.k1pool.com:3362    1%    diff 1310720
    mine_pearl_k1pool.bat       us.pearl.k1pool.com:3360        0%    diff 1966080
    mine_pearl_jetskipool.bat   pearlski.jetskipool.ai:6970     1%    diff 2000000
    mine_pearl_kryptex.bat      prl.kryptex.network:7048        1%    diff 2097120
    mine_pearl_herominers.bat   pearl.herominers.com:1200       0%    diff 2097152
    mine_pearl_mkpool_solo.bat  pearl.mkpool.com:3411           2%    diff 2097184

  Pick by that last column, not by fee, if you are on a slower card. HeroMiners
  hands out a fixed 2097152 and negotiates nothing, so a slow card can run for
  hours before it has a single share to look at. Suprnova and AlphaPool start
  thousands of times easier and will show you a share quickly.

TESTED
  CUDA build, measured on real cards at the 7.27 GB dataset:
    RTX 4080        125 MH/s
    RTX 4070 SUPER   98 MH/s
    RTX 4060 Ti      64 MH/s
  Confirmed to run on a machine with no CUDA Toolkit installed, so the static
  link and the bundled DLLs do their job. The Vulkan path is tested on Linux on
  NVIDIA and AMD. Windows itself is lightly exercised, so please report what
  happens.
