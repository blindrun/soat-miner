@echo off
REM Bitcoin III -> BTC3 Forge. 0.5% fee, proportional.
REM Their solo endpoint solo.btc3forge.com:3333 is advertised but refuses
REM connections, so there is no launcher for it.
REM Edit WALLET, then run.
cd /d "%~dp0"
set WALLET=YOUR_BC3_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
REM Goes through soat-miner.bat, not soat-miner.exe, so the backend is picked
REM for the card: CUDA on NVIDIA, Vulkan on AMD.
soat-miner.bat --algo sha3-256t --pool btc3forge.com:3337 --wallet %WALLET% --worker %WORKER% --pass x %*
pause

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
