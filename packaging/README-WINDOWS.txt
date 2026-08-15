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

  That matters if you have an NVIDIA card: NVIDIA's Vulkan compute path is very
  slow for this workload - about 5.8 MH/s on an RTX 4090, versus 217 MH/s
  through CUDA. On NVIDIA, build the CUDA target from source:

      cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
      cmake --build build --config Release

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
  This binary is cross-compiled and its imports verified (KERNEL32, WS2_32,
  msvcrt, vulkan-1 - nothing else), but it has NOT been run on Windows.
  You are the first. Please report what happens.
