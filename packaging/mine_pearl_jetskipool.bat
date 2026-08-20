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
soat-miner.exe --algo pearl-pow --pool pearlski.jetskipool.ai:6970 --wallet %WALLET% --worker %WORKER% %*
