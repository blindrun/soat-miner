@echo off
REM SOAT Miner launcher for Windows. Edit config.txt first.
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist config.txt ( echo config.txt not found & exit /b 1 )
for /f "usebackq tokens=1,* delims==" %%A in ("config.txt") do (
  set "line=%%A"
  if not "!line:~0,1!"=="#" if not "%%A"=="" set "%%A=%%B"
)

REM Backend choice is per GPU architecture, not per vendor. Measured at 7.27 GB
REM on Windows: RTX 5080 Vulkan 259.0 vs natively compiled sm_120 CUDA 219.6,
REM so Blackwell wants Vulkan; RTX 4090 is the other way round (CUDA 217.5 vs
REM Vulkan 162.5). Compute capability 12.0 and up means Blackwell.
REM Done with labels rather than a parenthesised block: nvidia-smi's
REM --format=csv,noheader has to run outside a for /f, because the comma there
REM gets parsed as an argument separator and nvidia-smi fails with
REM "Option noheader is not recognized" - which then silently falls through to
REM whatever the default was.
if "%BACKEND%"=="" set BACKEND=auto
if /i not "%BACKEND%"=="auto" goto :backend_done

set BACKEND=vulkan
where nvidia-smi >nul 2>&1 || goto :backend_done

nvidia-smi --query-gpu=compute_cap --format=csv,noheader > "%TEMP%\soat_cap.txt" 2>nul
set "CAPMAJOR="
for /f "tokens=1 delims=." %%C in ('type "%TEMP%\soat_cap.txt" 2^>nul') do (
  if not defined CAPMAJOR set "CAPMAJOR=%%C"
)
del "%TEMP%\soat_cap.txt" >nul 2>&1

REM No capability reported means an older driver; CUDA is right for every
REM NVIDIA generation before Blackwell.
if not defined CAPMAJOR set BACKEND=cuda & goto :backend_done
if %CAPMAJOR% GEQ 12 (
  echo auto: compute capability %CAPMAJOR%.x ^(Blackwell^) - Vulkan, ~22%% faster than CUDA here
) else (
  set BACKEND=cuda
)
:backend_done
if /i "%BACKEND%"=="cuda" ( set BIN=soat-miner.exe ) else ( set BIN=soat-miner-vk.exe )

REM Both builds ship in the archive now. If the CUDA one is missing anyway
REM (deleted, or an old Vulkan-only archive), fall back rather than fail with a
REM confusing "not found" on an NVIDIA machine.
if not exist "%BIN%" (
  if exist "soat-miner-vk.exe" (
    echo [!] %BIN% not present, using soat-miner-vk.exe ^(Vulkan^)
    echo [!] On Blackwell ^(RTX 50-series^) that is the faster backend anyway.
    echo [!] On Ada and older, CUDA is about 34%% faster - re-download the full
    echo [!] archive, or build it from source with CMake.
    set BIN=soat-miner-vk.exe
  ) else (
    echo No miner binary found in this folder.
    exit /b 1
  )
)

REM Building the next block's table ahead is CUDA only so far.
if "%CACHE_DAG%"=="" set CACHE_DAG=auto
set CACHEARG=
if "%BIN%"=="soat-miner.exe" set CACHEARG=--cache-dag %CACHE_DAG%

REM Command-line args override config.txt, so the mine_ergo_*.bat wrappers
REM can pass their own --pool/--wallet.
echo %* | findstr /C:"--pool" /C:"--node" /C:"--bench" /C:"--lithos" /C:"--list-devices" /C:"--list-algos" /C:"--help" >nul
if %errorlevel%==0 (
  echo SOAT Miner [%BACKEND%] - using command-line settings
  %BIN% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
  goto :eof
)

REM Default set OUTSIDE the block below on purpose. cmd.exe expands %VAR% when
REM it parses a parenthesised block, before any line in it runs, so a `set`
REM inside the block is invisible to the lines that follow it there - and
REM enabledelayedexpansion does not change that, it only adds !VAR!. Setting it
REM here is what makes an upgraded config.txt with no LITHOS_ADDR line work
REM instead of passing an empty --pool.
if "%LITHOS_ADDR%"=="" set LITHOS_ADDR=127.0.0.1:4444

if /I "%LITHOS%"=="yes" (
  REM No WALLET check: on Lithos the stratum address is a label, and payment
  REM follows the node the Lithos client is attached to.
  echo SOAT Miner -^> Lithos client at %LITHOS_ADDR% [%BACKEND%]
  %BIN% --lithos --pool %LITHOS_ADDR% --worker %WORKER% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
  goto :eof
)

if not "%POOL%"=="" (
  REM Refuse the unedited placeholder so nobody mines to it by accident. The
  REM C++ guard catches it too, but check here so the message is clear on Windows.
  if "%WALLET%"=="9YOUR_ERGO_ADDRESS_HERE" (
    echo Set WALLET in config.txt to YOUR Ergo address before pool mining.
    echo   ^(or edit WALLET in one of the mine_ergo_*.bat scripts^)
    pause
    goto :eof
  )
  if "%WALLET%"=="" (
    echo Set WALLET in config.txt to your Ergo address before pool mining.
    pause
    goto :eof
  )
  echo SOAT Miner -^> pool %POOL% as %WORKER% paying %WALLET% [%BACKEND%]
  %BIN% --pool %POOL% --wallet %WALLET% --worker %WORKER% --pass %PASSWORD% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
) else (
  echo SOAT Miner -^> solo via node %NODE%:%NODE_PORT% [%BACKEND%]
  %BIN% --node %NODE% --port %NODE_PORT% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
)
