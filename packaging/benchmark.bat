@echo off
REM BENCHMARK ONLY - this measures hashrate. It does NOT mine:
REM no pool, no wallet, no shares, no payouts. That is why no address is
REM needed here. To actually mine, edit WALLET in mine_ergo_herominers.bat.
cd /d "%~dp0"
soat-miner.bat --bench %*
pause
