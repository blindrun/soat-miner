@echo off
REM Benchmark without a pool or node.
cd /d "%~dp0"
soat-miner.bat --bench %*
pause
