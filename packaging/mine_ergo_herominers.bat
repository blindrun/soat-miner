@echo off
REM Ergo -> HeroMiners. Edit WALLET below, then double-click.
cd /d "%~dp0"
set WALLET=9YOUR_ERGO_ADDRESS_HERE
set WORKER=rig1
soat-miner.bat --pool ergo.herominers.com:1180 --wallet %WALLET% --worker %WORKER% --pass x %*
pause
