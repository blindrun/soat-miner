@echo off
REM Pearl -> K1Pool PPLNS. 0% fee.
REM Handed out a fixed difficulty of 1966080.
REM Other regions, same port: eu.pearl.k1pool.com, cn.pearl.k1pool.com
REM Their solo pool is a separate host and port - see
REM mine_pearl_k1pool_solo.sh.
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
soat-miner.bat --algo pearl-pow --pool us.pearl.k1pool.com:3360 --wallet %WALLET% --worker %WORKER% %*
