@echo off
REM Pearl -> JetSkiPool (PearlSki). 1% pool fee, proportional.
REM They also charge a 2% miner fee on their own miner; this one is ours,
REM so only the pool fee applies.
REM Handed out a fixed difficulty of 2000000 on connect.
REM France only - the pool advertises no other region.
REM pearl.jetskipool.ai redirects to pearlski.jetskipool.ai for the site;
REM the stratum host is the pearlski one.
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
soat-miner.bat --algo pearl-pow --pool pearlski.jetskipool.ai:6970 --wallet %WALLET% --worker %WORKER% %*
