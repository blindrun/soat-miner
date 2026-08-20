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
soat-miner.exe --algo pearl-pow --pool us.pearl.k1pool.com:3360 --wallet %WALLET% --worker %WORKER% %*
