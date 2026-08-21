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
REM Backend choice is PER ALGORITHM. The Blackwell-prefers-Vulkan rule below was
REM measured on Autolykos and does not carry to BC3, which is faster on CUDA on
REM NVIDIA (4090: CUDA 1543 MH/s, Vulkan 1086). Without this an NVIDIA card
REM silently picks the slower backend for BC3.
REM Read the algorithm from BOTH config.txt and the command line. The
REM mine_bc3_*.bat scripts pass --algo on the command line and never set ALGO,
REM so testing %ALGO% alone missed every launcher-driven run and NVIDIA quietly
REM got Vulkan. Labels, not parenthesised blocks: cmd expands %VAR% when it
REM parses a block, so a `set` inside one is invisible to the lines after it -
REM the same trap this file already documents for LITHOS_ADDR.
set "WANT_BC3="
if /i "%ALGO%"=="sha3-256t" set "WANT_BC3=1"
echo %* | findstr /I /C:"sha3-256t" >nul && set "WANT_BC3=1"
if not defined WANT_BC3 goto :not_bc3
if not exist "soat-miner.exe" goto :not_bc3
where nvidia-smi >nul 2>&1 || goto :not_bc3
echo auto: BC3 on NVIDIA - CUDA, it beats Vulkan here
set BACKEND=cuda
goto :backend_done
:not_bc3
REM Pearl on NVIDIA prefers CUDA, and this is no longer a CUDA-ONLY rule.
REM
REM Pearl has a Vulkan backend now - ten shaders, each byte-identical to the
REM CUDA reference on an Ada and an RDNA3 card - so an AMD user is no longer
REM told the algorithm does not exist for them. Bitcoin III made exactly this
REM transition earlier for the same reason.
REM
REM CUDA still wins on NVIDIA, and not from a measurement: there is no Vulkan
REM Pearl throughput number yet. It is what the two paths ARE - the CUDA one
REM tunes shape and tile configuration per card at startup, the Vulkan one has
REM one fixed shape and an untuned GEMM. Revisit with a number, not a guess.
REM Note the nvidia-smi test: without it this sends an AMD card to a CUDA
REM binary it does not have, which is the bug this whole block exists to stop.
set "WANT_PEARL="
if /i "%ALGO%"=="pearl-pow" set "WANT_PEARL=1"
echo %* | findstr /I /C:"pearl-pow" >nul && set "WANT_PEARL=1"
if not defined WANT_PEARL goto :not_pearl
if not exist "soat-miner.exe" goto :not_pearl
where nvidia-smi >nul 2>&1 || goto :not_pearl
echo auto: Pearl on NVIDIA - CUDA, its shape tuner has no Vulkan equivalent
set BACKEND=cuda
goto :backend_done
:not_pearl
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
  REM Per algorithm: the placeholder and the address shape are different for
  REM each coin, and this used to test the Ergo one whatever you were mining.
  if "%ALGO%"=="" set ALGO=autolykos2
  set "COIN=Ergo"
  set "PLACEHOLDER=9YOUR_ERGO_ADDRESS_HERE"
  set "SCRIPTS=mine_ergo_*.bat"
  if /i "%ALGO%"=="pearl-pow" ( set "COIN=Pearl" & set "PLACEHOLDER=prl1YOUR_PEARL_ADDRESS_HERE" & set "SCRIPTS=mine_pearl_*.bat" )
  if /i "%ALGO%"=="sha3-256t" ( set "COIN=Bitcoin III" & set "PLACEHOLDER=YOUR_BC3_ADDRESS_HERE" & set "SCRIPTS=mine_bc3_*.bat" )
  if "%WALLET%"=="!PLACEHOLDER!" (
    echo Set WALLET in config.txt to YOUR !COIN! address before pool mining.
    echo   ^(or edit WALLET in one of the !SCRIPTS! scripts^)
    pause
    goto :eof
  )
  if "%WALLET%"=="" (
    echo Set WALLET in config.txt to your !COIN! address before pool mining.
    pause
    goto :eof
  )
  echo SOAT Miner -^> !COIN! on %POOL% as %WORKER% paying %WALLET% [%BACKEND%]
  %BIN% --algo %ALGO% --pool %POOL% --wallet %WALLET% --worker %WORKER% --pass %PASSWORD% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
) else (
  echo SOAT Miner -^> solo via node %NODE%:%NODE_PORT% [%BACKEND%]
  %BIN% --node %NODE% --port %NODE_PORT% --batch %BATCH% --interval %INTERVAL% %CACHEARG% %*
)

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
