@echo off
REM Bitcoin III -> PythonPool. Edit WALLET, then run.
cd /d "%~dp0"
set WALLET=YOUR_BC3_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
REM Goes through soat-miner.bat, like the Ergo scripts, so the backend is
REM picked for the card. This used to call soat-miner.exe directly and refuse
REM to start without it, because BC3 was CUDA only - which meant an AMD user
REM was told the algorithm did not exist. BC3 has a Vulkan shader now.
soat-miner.bat --algo sha3-256t --pool stratum.pythonpool.dev:3357 --wallet %WALLET% --worker %WORKER% --pass x %*
pause

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
