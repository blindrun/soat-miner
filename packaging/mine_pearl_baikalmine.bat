@echo off
REM Pearl -> BaikalMine. 0.5% fee, PPLNS - the cheapest verified pool
REM that is not free.
REM Handed out a fixed difficulty of 262144, 8x easier than HeroMiners.
REM Other region, same port: pearl-ru2.baikalmine.com (Moscow).
REM The pool also publishes pearl.baikalmine.com:2010, but that host did
REM not answer a TCP connect from here; pearl-eu did, so use pearl-eu.
REM Edit WALLET, then run.
REM WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
REM a different chain: an Ergo address cannot be paid by a Pearl pool, and the
REM pool answers one with "Invalid Pearl address". A Pearl address starts
REM with prl1 and is about 63 characters.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool pearl-eu.baikalmine.com:2010 --wallet %WALLET% --worker %WORKER% %*
