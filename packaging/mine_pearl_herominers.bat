@echo off
REM Pearl -> HeroMiners. Edit WALLET, then run.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool pearl.herominers.com:1200 --wallet %WALLET% --worker %WORKER% %*
