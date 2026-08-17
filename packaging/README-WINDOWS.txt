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
  benchmark.bat       benchmark, no pool or node needed

WHICH ONE RUNS
  soat-miner.bat picks by GPU, not by brand, and you do not choose:
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

TESTED
  CUDA build, measured on real cards at the 7.27 GB dataset:
    RTX 4080        125 MH/s
    RTX 4070 SUPER   98 MH/s
    RTX 4060 Ti      64 MH/s
  Confirmed to run on a machine with no CUDA Toolkit installed, so the static
  link and the bundled DLLs do their job. The Vulkan path is tested on Linux on
  NVIDIA and AMD. Windows itself is lightly exercised, so please report what
  happens.
