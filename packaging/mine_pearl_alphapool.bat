@echo off
REM Pearl -> AlphaPool. 0% fee, PPLNS.
REM Handed out 50000 on connect - 42x easier than HeroMiners' fixed
REM 2097152, and the easiest bound of any pool with real hashrate.
REM USE PORT 5571. It is their plain-stratum port. 5566 (their shim) and
REM 5573 (solo/lottery) both open with a pearl.challenge proof-of-work
REM handshake this miner does not implement, and never authorize.
REM Other regions, same port 5571: us1, eu1, ru1, sg1 .alphapool.tech
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
soat-miner.bat --algo pearl-pow --pool us2.alphapool.tech:5571 --wallet %WALLET% --worker %WORKER% %*
