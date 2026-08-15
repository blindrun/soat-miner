@echo off
REM Ergo -> HeroMiners. Edit WALLET below, then double-click.
cd /d "%~dp0"
set WALLET=9ea2QrXbTTmEhRA92qcnDVD98aeENUc4oJdDGxF7GKMBZ47wLTR
set WORKER=%COMPUTERNAME%
soat-miner.bat --pool ergo.herominers.com:1180 --wallet %WALLET% --worker %WORKER% --pass x %*
pause
