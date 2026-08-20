@echo off
REM Ergo -^> Lithos, the decentralised pool protocol.
REM
REM Lithos is not a coin and not an algorithm: it is a pool protocol whose
REM reference client runs a stratum server on your own machine. You mine
REM ordinary Autolykos v2 into it, and it settles rewards on-chain from
REM Non-Interactive Share Proofs.
REM
REM Before this will work you need a fully synced Ergo node, Java 11, and the
REM Lithos client (github.com/Lithos-Protocol/Lithos-Client) running and
REM pointed at that node.
REM
REM The payout identity comes from the NODE, not from an address set here,
REM which is why no WALLET is needed below.
REM Named LITHOS_TARGET, not LITHOS: environment variables set here are
REM inherited by soat-miner.bat, which uses LITHOS as a yes/no switch and
REM LITHOS_ADDR as an address. Reusing either name here would collide.
cd /d "%~dp0"
set LITHOS_TARGET=127.0.0.1:4444
set WORKER=%COMPUTERNAME%
soat-miner.bat --lithos --pool %LITHOS_TARGET% --worker %WORKER% %*
pause

REM Ctrl+C is delivered to every process on this console, so cmd starts asking
REM "Terminate batch job (Y/N)?" while the miner is still printing its shutdown.
REM The miner flushes and clears its line first; this newline guarantees the
REM prompt begins on a fresh row instead of halfway through a word. Answering
REM either Y or N is safe - the miner has already stopped and put the card back.
echo.
