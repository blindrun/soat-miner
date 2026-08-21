@echo off
REM Pearl -> K1Pool SOLO. 1% fee. You find the block, you keep it.
REM Handed out a fixed difficulty of 1310720.
REM Other regions, same port: eu.pearlsolo.k1pool.com, and
REM ru.pearlsolo.k1pool.org (note .org, not .com).
REM Their PPLNS pool is mine_pearl_k1pool.sh and charges nothing.
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
soat-miner.bat --algo pearl-pow --pool us.pearlsolo.k1pool.com:3362 --wallet %WALLET% --worker %WORKER% %*
