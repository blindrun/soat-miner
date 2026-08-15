SOAT Miner - Windows
====================

WHAT IS IN HERE
  soat-miner-vk.exe   Vulkan build. Works on AMD, Intel and NVIDIA.
  soat-miner.bat      launcher, reads config.txt
  mine_ergo_*.bat     ready-made pool / solo scripts - edit WALLET and run
  benchmark.bat       benchmark, no pool or node needed

NO CUDA BUILD IN THIS ARCHIVE
  The CUDA binary cannot be cross-compiled from Linux (nvcc requires MSVC on a
  Windows host), so this archive ships the Vulkan build only.

  The Vulkan build is fine on NVIDIA now - 267.6 MH/s on an RTX 5080 and
  162.5 on a 4090. (It used to be about 3-6 MH/s: the dataset was not a
  dedicated allocation, so it never got large pages and the random lookups
  missed the TLB. Fixed.)

  CUDA is still faster on NVIDIA - 217 MH/s on a 4090 against 162.5 through
  Vulkan - so building it is worth it if you have the toolkit:

      cmake -B build
      cmake --build build --config Release

  That needs the CUDA Toolkit and Visual Studio Build Tools. RTX 50-series
  needs CUDA 12.8 or newer for native code; older toolkits still work through
  the PTX the build embeds for exactly this reason.

  On AMD, the Vulkan build IS the right one and there is nothing to miss.

QUICK START
  1. Double-click benchmark.bat to check it sees your GPU.
  2. Open mine_ergo_herominers.bat in Notepad, set WALLET to your Ergo address.
  3. Double-click it.

REQUIREMENTS
  Windows 10 or 11, 64-bit, and a GPU driver (vulkan-1.dll ships with all
  current AMD/NVIDIA/Intel drivers). No Visual C++ redistributable needed -
  the binary is statically linked.

  The dataset needs about 7.3 GB of VRAM at today's chain height, so a card
  with 8 GB or more.

TESTED
  This binary is cross-compiled from Linux and its imports verified (KERNEL32,
  WS2_32, msvcrt, vulkan-1 - nothing else). The Vulkan path is tested on Linux
  on both NVIDIA and AMD; Windows itself is only lightly exercised, so please
  report what happens.
