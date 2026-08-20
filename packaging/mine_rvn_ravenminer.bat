@echo off
REM Ravencoin -> RavenMiner. Edit WALLET, then run.
REM
REM NOT YET PROVEN END TO END: this endpoint answers and speaks a protocol this
REM miner implements, but no share from this miner has been accepted by any
REM KawPow pool yet. Treat the first run as a test.
REM
REM KawPow needs about 7 GB of free VRAM and the launcher will pick the CUDA
REM binary on NVIDIA, which is much faster for this algorithm.
cd /d "%~dp0"
set WALLET=RYOUR_RAVENCOIN_ADDRESS_HERE
set WORKER=%COMPUTERNAME%
soat-miner.exe --algo kawpow --pool stratum.ravenminer.com:3838 --wallet %WALLET% --worker %WORKER% %*

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word.
echo.
