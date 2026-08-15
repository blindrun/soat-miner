@echo off
REM Ergo solo, against your own node. Payout address is set ON THE NODE.
cd /d "%~dp0"
soat-miner.bat --node 127.0.0.1 --port 9053 %*
pause
