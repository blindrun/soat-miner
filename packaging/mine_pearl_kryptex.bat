@echo off
REM Pearl -> Kryptex. 1% fee, PROP or SOLO (their PPS+ tier is 2%).
REM Handed out difficulty 2097120 on connect, the same fixed bound
REM HeroMiners uses.
REM 8048 is the same pool over SSL/TLS, which this miner does not speak.
REM Kryptex publishes eight regional servers; prl.kryptex.network routes
REM you to the nearest one, so there is no per-region launcher.
REM Edit WALLET, then run.
REM WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
REM a different chain: an Ergo address cannot be paid by a Pearl pool, and the
REM pool answers one with "Invalid Pearl address". A Pearl address starts
REM with prl1 and is about 63 characters.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool prl.kryptex.network:7048 --wallet %WALLET% --worker %WORKER% %*
