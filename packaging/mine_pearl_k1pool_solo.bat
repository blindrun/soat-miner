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
soat-miner.exe --algo pearl-pow --pool us.pearlsolo.k1pool.com:3362 --wallet %WALLET% --worker %WORKER% %*
