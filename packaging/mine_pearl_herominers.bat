@echo off
REM Pearl -> HeroMiners. Edit WALLET, then run.
cd /d "%~dp0"
set WALLET=prl1YOUR_PEARL_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo pearl-pow --pool pearl.herominers.com:1200 --wallet %WALLET% --worker %WORKER% %*

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
