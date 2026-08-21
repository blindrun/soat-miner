@echo off
REM Pearl -> LuckyPool. 1% fee, proportional.
REM Three difficulty tiers on the same host, all vardiff:
REM   3360 starts at 2M   3361 starts at 4M   3362 starts at 8M
REM Observed 888888 on connect, so the pool moves it below the tier floor
REM for a new worker.
REM Other regions, same ports: pearl-eu1, pearl-eu2, pearl-pl, pearl-tr,
REM pearl-ru, pearl-us-west, pearl-us-central, pearl-us-ord, pearl-br,
REM pearl-ca1, pearl-ca2, pearl-sg1, pearl-sg2, pearl-id, pearl-hk,
REM pearl-in, pearl-jp, pearl-au .luckypool.io
REM Edit WALLET, then run.
REM WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
REM a different chain: an Ergo address cannot be paid by a Pearl pool, and the
REM pool answers one with "Invalid Pearl address". A Pearl address starts
REM with prl1 and is about 63 characters.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
REM Goes through soat-miner.bat so the backend is picked for the card, exactly
REM like the Ergo and Bitcoin III scripts. This used to call soat-miner.exe
REM directly, which was right while Pearl was CUDA-only and became wrong the
REM day it got a Vulkan backend: an AMD user was told the binary was missing
REM rather than being sent to the one that has it.
soat-miner.bat --algo pearl-pow --pool pearl-us-east.luckypool.io:3360 --wallet %WALLET% --worker %WORKER% %*
