@echo off
REM Pearl -> RabbitMiner. 1% fee, proportional.
REM Vardiff; handed out 232827 on connect, the lowest of the big pools
REM apart from AlphaPool and Suprnova.
REM Other region, same port: fi.rabbitminer.cc (Finland).
REM Their web builder offers a static difficulty through a d=VALUE in the
REM password field. This miner always sends pass "x" on the Pearl path,
REM so that knob is not reachable from here yet.
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
soat-miner.bat --algo pearl-pow --pool nl.rabbitminer.cc:1902 --wallet %WALLET% --worker %WORKER% %*
