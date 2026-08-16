@echo off
REM Ergo -> WoolyPooly. Edit WALLET below, then double-click.
cd /d "%~dp0"
set WALLET=9YOUR_ERGO_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.bat --pool pool.woolypooly.com:3100 --wallet %WALLET% --worker %WORKER% --pass x %*
pause
