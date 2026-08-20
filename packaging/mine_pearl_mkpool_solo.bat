@echo off
REM Pearl -> mkpool SOLO. 2% fee. You find the block, you keep it.
REM Handed out a fixed difficulty of 2097184, essentially HeroMiners'.
REM 3411 is the working port. 3413 is advertised on the same host but
REM closes the connection without answering, so there is no launcher.
REM Edit WALLET, then run.
REM WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
REM a different chain: an Ergo address cannot be paid by a Pearl pool, and the
REM pool answers one with "Invalid Pearl address". A Pearl address starts
REM with prl1 and is about 63 characters.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool pearl.mkpool.com:3411 --wallet %WALLET% --worker %WORKER% %*
