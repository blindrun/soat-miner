@echo off
REM Ergo solo, against your own node. Payout address is set ON THE NODE.
cd /d "%~dp0"
soat-miner.bat --node 127.0.0.1 --port 9053 %*
pause

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
