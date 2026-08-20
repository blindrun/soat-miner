@echo off
REM Ergo -> WoolyPooly. Edit WALLET below, then double-click.
cd /d "%~dp0"
set WALLET=9YOUR_ERGO_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.bat --pool pool.woolypooly.com:3100 --wallet %WALLET% --worker %WORKER% --pass x %*
pause

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
