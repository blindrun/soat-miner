@echo off
REM Pearl -> Suprnova. PPLNS, currently 0% on a launch promotion.
REM 3373 is the vardiff port and the one to use. It opened at difficulty
REM 244 - four orders of magnitude below HeroMiners' fixed 2097152, which
REM is what a slow card needs to produce a testable share in minutes
REM rather than hours.
REM The fixed ports are lower still on connect. Measured, not advertised:
REM   3370 opened at 1.22    3371 opened at 488    3372 opened at 1953
REM 3374 is the same as 3373 over SSL/TLS, which this miner does not speak.
REM Suprnova labels difficulty in units 16384x ours, so its own job id
REM suffix reads 4000000 where the target decodes to 244. The target on
REM the wire is what the miner mines against, so ignore the label.
REM Other regions, same ports: stratum-eu2, stratum-us, stratum-apac
REM .suprnova.cc
REM Edit WALLET, then run.
REM WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
REM a different chain: an Ergo address cannot be paid by a Pearl pool, and the
REM pool answers one with "Invalid Pearl address". A Pearl address starts
REM with prl1 and is about 63 characters.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool prl.suprnova.cc:3373 --wallet %WALLET% --worker %WORKER% %*
