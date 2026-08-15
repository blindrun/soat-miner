@echo off
REM SOAT Miner launcher for Windows. Edit config.txt first.
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist config.txt ( echo config.txt not found & exit /b 1 )
for /f "usebackq tokens=1,* delims==" %%A in ("config.txt") do (
  set "line=%%A"
  if not "!line:~0,1!"=="#" if not "%%A"=="" set "%%A=%%B"
)

if "%BACKEND%"=="" set BACKEND=auto
if /i "%BACKEND%"=="auto" (
  where nvidia-smi >nul 2>&1 && ( set BACKEND=cuda ) || ( set BACKEND=vulkan )
)
if /i "%BACKEND%"=="cuda" ( set BIN=soat-miner.exe ) else ( set BIN=soat-miner-vk.exe )

REM The Windows archive ships the Vulkan build only - CUDA cannot be
REM cross-compiled from Linux (nvcc needs MSVC). Fall back rather than fail
REM with a confusing "not found" on an NVIDIA machine.
if not exist "%BIN%" (
  if exist "soat-miner-vk.exe" (
    echo [!] %BIN% not present, using soat-miner-vk.exe ^(Vulkan^)
    echo [!] NOTE: Vulkan is very slow on NVIDIA. Build the CUDA target from
    echo [!]       source with CMake for full speed on NVIDIA cards.
    set BIN=soat-miner-vk.exe
  ) else (
    echo No miner binary found in this folder.
    exit /b 1
  )
)

REM Command-line args override config.txt, so the mine_ergo_*.bat wrappers
REM can pass their own --pool/--wallet.
echo %* | findstr /C:"--pool" /C:"--node" >nul
if %errorlevel%==0 (
  echo SOAT Miner [%BACKEND%] - using command-line pool/node settings
  %BIN% --batch %BATCH% --interval %INTERVAL% %*
  goto :eof
)

if not "%POOL%"=="" (
  echo SOAT Miner -^> pool %POOL% as %WORKER% [%BACKEND%]
  %BIN% --pool %POOL% --wallet %WALLET% --worker %WORKER% --pass %PASSWORD% --batch %BATCH% --interval %INTERVAL% %*
) else (
  echo SOAT Miner -^> solo via node %NODE%:%NODE_PORT% [%BACKEND%]
  %BIN% --node %NODE% --port %NODE_PORT% --batch %BATCH% --interval %INTERVAL% %*
)
