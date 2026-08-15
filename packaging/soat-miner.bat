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

if not "%POOL%"=="" (
  echo SOAT Miner -^> pool %POOL% as %WORKER% [%BACKEND%]
  %BIN% --pool %POOL% --wallet %WALLET% --worker %WORKER% --pass %PASSWORD% --batch %BATCH% --interval %INTERVAL% %*
) else (
  echo SOAT Miner -^> solo via node %NODE%:%NODE_PORT% [%BACKEND%]
  %BIN% --node %NODE% --port %NODE_PORT% --batch %BATCH% --interval %INTERVAL% %*
)
