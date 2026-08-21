# SOAT Miner

Mines Ergo on your GPU. Autolykos v2, CUDA and Vulkan, Linux and Windows.

No dev fee. Nothing is skimmed off your hashrate. MIT licensed.

You will need an Ergo wallet address before you can mine to a pool.
Get one from Nautilus or the Ergo desktop wallet.

You will also need a GPU with 8GB or more.
The dataset is 7.27GB right now and it grows every few days.

## Get it

Download the release for your os.

https://github.com/blindrun/soat-miner/releases

Unpack it on Linux with command tar xzf soat-miner_v0.2.17_Lin64.tar.gz

On Windows just unzip it.

### Windows Defender will block it

Defender flags soat-miner.exe as a coin miner and refuses to run it.
It is not wrong about what the program is.
Every GPU miner gets flagged the same way.

You have to allow it yourself, or it will not start.
Open Windows Security.
Go to Virus and threat protection.
Open Manage settings, then Add or remove exclusions.
Add the folder you unpacked the miner into.

Do that for a folder you chose, never for your whole drive.
Build it yourself from source if you would rather not take our word for it.

## Mine to a pool

This is the part most people want.

Open mine_ergo_herominers.sh in a text editor.
Change WALLET to your own address.
Save it.
Run it with command ./mine_ergo_herominers.sh

On Windows open mine_ergo_herominers.bat instead, change WALLET, then double
click it.

That is it. You are mining.

There is a woolypooly script in there too if you prefer that pool.

### Other verified pools

Every release includes an `ergo-pools.json` catalog and a matching Linux
(`mine_ergo_*.sh`) and Windows (`mine_ergo_*.bat`) launcher for each verified
plain Stratum V1/TCP profile. Edit the credential placeholder in the launcher;
all of them use `--pass x` and pass the worker separately.

| Pool | Launcher stem | Endpoint | Credential |
|---|---|---|---|
| 2Miners | `mine_ergo_2miners` | `erg.2miners.com:8888` | Ergo wallet |
| 2Miners Solo | `mine_ergo_2miners_solo` | `solo-erg.2miners.com:9999` | Ergo wallet |
| Cruxpool | `mine_ergo_cruxpool` | `ergo.cruxpool.com:4444` | Ergo wallet |
| FenixPool Solo | `mine_ergo_fenixpool_solo` | `fenixpool.com:9052` | Ergo wallet |
| HeroMiners | `mine_ergo_herominers` | `ergo.herominers.com:1180` | Ergo wallet |
| K1Pool | `mine_ergo_k1pool` | `eu.erg.k1pool.com:3746` | K1Pool `Kr_` account |
| K1Pool Solo | `mine_ergo_k1pool_solo` | `eu.ergsolo.k1pool.com:3748` | K1Pool `Kr_` account |
| Kryptex | `mine_ergo_kryptex` | `erg.kryptex.network:7021` | Ergo wallet |
| Sigmanauts | `mine_ergo_sigmanauts` | `65.108.57.232:3052` | Ergo wallet |
| WoolyPooly | `mine_ergo_woolypooly` | `pool.woolypooly.com:3100` | Ergo wallet |

The source URL for every endpoint is recorded beside its profile. The developer
refresh command, `python3 scripts/refresh_ergo_pools.py`, updates only
`ergo-pool-status.json` from MiningPoolStats. It does **not** change a launcher
endpoint: that needs a review against the pool's own setup page. This avoids an
aggregator update silently redirecting a mining configuration. Run
`make test-ergo-pools` to validate the catalog and generated launchers without
starting a miner, contacting a pool, or using a GPU.

First start takes about 15 seconds. It builds the 7.27GB dataset before it
hashes anything. That is normal and it is not frozen.

## Check it is working

Look at the accepted count in the readout. It should climb every few minutes.

Then look yourself up on the pool.

https://ergo.herominers.com

Paste your wallet address in. Your worker should be listed with a hashrate.

BE WARNED: a miner that prints a hashrate is not proof you are getting paid.
Always confirm on the pool site. Shares can be rejected for reasons the miner
cannot see.

## Benchmark without mining

Run it with command ./soat-miner-vk --bench

No pool, no wallet, no shares, no payouts. It only measures speed.

## Which backend

Leave BACKEND=auto in config.txt and it picks for you.

It goes by GPU, not by brand.

- AMD and Intel get Vulkan.
- NVIDIA 50-series gets Vulkan. It is 22% faster than CUDA on Blackwell.
- Every older NVIDIA gets CUDA. It is 34% faster than Vulkan on Ada.

Force it with BACKEND=cuda or BACKEND=vulkan if you want.

See what it found with command ./soat-miner-vk --list-devices

## Get your memory clock back

Your card is probably running its memory slower than it is rated for, and it is
not your fault.

CUDA and Vulkan both put the GPU in performance state P2. P2 runs GDDR6X under
its rated speed. On a 4090 that is 10251 MHz against a rated 10501.

Autolykos is memory bound, so that is straight hashrate.

`nvidia-smi --lock-memory-clocks` does not fix it. It reports success and the
clock does not move.

Put it back with command ./soat-miner --mclk-offset 500

Measured on a 4090:

| | MH/s |
|---|---|
| CUDA, P2 | 218.1 |
| CUDA, clock restored | **223.8** |
| Vulkan, P2 | 162.7 |
| Vulkan, clock restored | **166.7** |

So about 2.5% on both backends.

This is not an overclock. 500 is the offset that gets a 4090 back to the speed
written on the box. The number is in MHz of transfer rate, so half of it lands
on the clock.

Needs root on Linux. Run it with sudo or it prints why it could not.

The miner sets the offset back to 0 when it stops.

You can push past stock if you want to, and that is a real overclock with real
risk. Bad memory produces bad hashes. This miner re-checks every solution on
the GPU before sending it, so you will see them rejected rather than sent, but
do not go hunting for numbers you cannot verify.

Command ./soat-miner --mem-oc does the safe thing for your card automatically,
chosen by memory type (the memory is the same across a type). It is off by
default and only touches memory it has been proven safe on. **Linux only** - the
clock control it uses does not exist on Windows.

On **AMD GDDR6** there is no downclock to undo, but the memory can go past stock:
it raises the clock a conservative amount over stock through amdgpu overdrive. A
7900 XT goes to 1325 MHz, worth about 5%. This needs overdrive enabled - add
`amdgpu.ppfeaturemask=0xffffffff` to the kernel command line, reboot, and run as
root - and it prints how to if it is off.

On **NVIDIA GDDR6X** (4090/4080/4070-family, 3090/3080/3070 Ti) it restores the
P2 downclock back to the rated speed, worth about 2.5%. That is undoing a
downclock, not an overclock. GDDR6 and GDDR7 NVIDIA cards have no such downclock,
so they are left alone. Use --mclk-offset by hand if you know your card.

Either way the clock is put back when the miner stops.

## Multiple GPUs

Pick the GPU with command ./soat-miner --device N, numbered as in --list-devices.

Run one process per GPU. Each gets its own --device and its own --mem-oc, which
is also how each card is tuned independently:

```
./soat-miner --device 0 --mem-oc --pool ... &
./soat-miner --device 1 --mem-oc --pool ... &
```

One process per card keeps a hung overclock on one GPU from taking the others
down with it. A single process that drives every GPU at once is not built yet.

## Building the next block ahead

The lookup table depends on the block height. Every new block means a new table.

That is consensus, not a bug. So every miner rebuilds 7.27GB every block.

On a 4090 that costs 1.5 seconds of every block. On a 6700 XT it costs 8.2.

Build it ahead of time instead with command ./soat-miner --cache-dag on

You know the next height already, so the next table can be built while you are
still mining the current one. A new block then starts instantly.

This needs two tables in memory at once, which is 14.5GB.

Leave it on auto and it checks your card first. It turns itself off if the
second table will not fit, and tells you why. It will never push your card into
an out of memory that it was not already in.

Auto is the default. Turn it off with command ./soat-miner --cache-dag off

Do not expect much. It is worth 0.73% on a 4090.

CUDA only. It was built for Vulkan too and it lost on both cards, so it is not
in that build. Numbers are further down.

It also has a shelf life. The table grows 5% every 51,200 blocks, so two of
them stop fitting a 16GB card around block 2,000,000 and a 24GB card around
block 2,364,000.

## Speeds

Measured at the current dataset size, 7.27GB.

| GPU | Backend | MH/s |
|---|---|---|
| RTX 5080 | Vulkan | 267.6 |
| RTX 4090 | CUDA | 235.5 |
| RTX 4080 | CUDA | 140.0 |
| RTX 4070 SUPER | CUDA | 117.6 |
| RTX 4060 Ti | CUDA | 76.9 |
| RX 7900 XT | Vulkan | 151.9 |
| RX 6700 XT | Vulkan | 82.9 |

The CUDA numbers went up 8-21% in v0.2.2 from a one-line change: the dataset
reads use streaming loads (`__ldcs`), not the read-only data cache (`__ldg`).
The dataset is 7.27GB and a read is reused about 0.5% of the time, so caching
every element only evicts others and starves the random reads that are the whole
bottleneck. Telling the hardware the data will not be reused takes the 4090 from
217.5 to 235.5, and it helps more on smaller cards where the cache is smaller
relative to the dataset. Vulkan is unchanged - GLSL has no clean equivalent.

The only like for like comparison here is SRBMiner 3.5.5 on the same 4090 at
the same 7.27GB. It does 237.7 gross and takes 1%, so it delivers 235.4 against
our 217.5. That is 92%, with no fee and source you can read.

lolMiner takes 1.5% on Autolykos V2. Their own algorithm table says so.

Do not trust the 265 MH/s figure the calculator sites list for a 4090. It is
not sourced to a run at this dataset size, and every old number is inflated for
the reason in the next paragraph.

Your number will drop over time and that is not a bug. Hashrate falls as the
dataset outgrows your cache. The same 6700 XT did 235 MH/s in 2021 at 2.25GB
and does 82.9 now at 7.27GB. Old numbers you find online are about double what
any miner gets today. Compare like with like.

## Money

Be honest with yourself about this.

A tuned 4090 nets a few dollars a month at $0.11/kWh. It drops about 25% a
quarter as the block reward steps down. This is not free money.

Power limit it to the efficiency knee. On a 4090 set 183W and you still get
217 MH/s while the card actually pulls 167W. That is 24% less power for 0.3
MH/s.

Set it with command sudo nvidia-smi -pl 183

## Solo mining

Solo needs your own Ergo node. Pool does not.

Your payout address is configured on the NODE, not in the miner. Put it in
ergo.conf.

```
ergo {
  node.mining = true
  wallet.secretStorage.secretDir = ...
}
```

Then run it with command ./mine_ergo_solo.sh

At 217 MH/s against a network around 510 GH/s you will find a block roughly
once every three years. Solo is in here because it is the one thing that
actually improves mining decentralisation, not because it pays.

## Lithos

> **Test network only.** The Lithos support here is tested end to end against a
> live Lithos client (v4.2.0) on the Ergo **test network**: the miner connects,
> receives real jobs, and its shares are accepted by the client (verified at 95
> accepted / 0 rejected). It has **not** been validated on mainnet yet, so run it
> on testnet only for now. The quickstart below defaults to testnet on purpose.
> The rest of the miner is unaffected either way.

[Lithos](https://docs.ergoplatform.com/eco/lithos/) is a decentralised pool
protocol for Ergo. It is not a coin and not an algorithm: the proof of work is
the same Autolykos v2, the same dataset and the same kernels. What changes is
who settles the rewards. Instead of a pool operator crediting shares in a
database, the Lithos client compresses your work into Non-Interactive Share
Proofs and settles them on-chain against collateral, so there is no operator to
trust and no operator to censor you.

Because there is no operator, there is no pool server to point at. Each miner
runs their own Ergo node and Lithos client, and payouts go to that miner's own
wallet. That is more to set up than a normal pool, so this download ships a
script that does all of it for you.

### Mine Lithos and get paid, from scratch

**1. Set up the node, client and wallet** (Debian/Ubuntu, one command):

```
sudo ./lithos-quickstart.sh --network testnet
```

This installs Java 11, an Ergo node and the Lithos client, creates the wallet
your payouts will go to, and starts everything. It handles the traps that
otherwise cost an afternoon (the testnet difficulty override, the node version
that is stable, creating the wallet the client needs before it will start).

**2. Wait for the node to sync**, and watch it plainly:

```
./lithos-status --watch
```

The node syncs the chain (this takes a while). When `lithos-status` shows
`SYNCED`, start the client:

```
systemctl start lithos-client
```

**3. Mine:**

```
./mine_ergo_lithos.sh          # or: ./soat-miner.sh --lithos
```

**4. Check you are actually earning.** `lithos-status` reads the client and
tells you your super-shares per hour against the payout bar (you need about
0.83/hour). If you are below it, lower the `diff` in the client config.

Payouts settle on-chain to the wallet the setup created. Back up its mnemonic
(the setup prints where it saved it).

If you already run your own node and client, skip the script. The client's
stratum server is on `127.0.0.1:4444` and you just mine into it:

```
./soat-miner.sh --lithos
```

`--lithos` defaults `--pool` to `127.0.0.1:4444` and stops requiring an Ergo
address. That second part is the bit worth knowing: on a conventional pool the
stratum address is who gets paid, so the miner refuses to start without a valid
one. On Lithos it is a label. Payment follows the node the Lithos client is
attached to, so demanding an address here would only reject working setups.

Two things the miner handles for Lithos specifically, both of which fail
silently if you get them wrong:

- **The extranonce moves.** When two rigs collide on the same prefix the client
  answers with `mining.set_extranonce` and hands one of them a new one. A miner
  that only reads the extranonce at connect time keeps hashing in a subspace it
  no longer owns.
- **The share target can be zero.** Lithos publishes `tau`, the share target, in
  `mining.notify` param 6, and it is legitimately `0` before the client has one.
  A zero target can never be beaten, so the miner would hash forever, submit
  nothing, and report no error at all. It refuses the job and says so instead.

`make test` runs both against a mock Lithos server transcribed from the client's
own Scala source, including the check that matters most: Lithos validates
`extraNonce1 ++ extraNonce2`, not the nonce field you send it, so if the miner
slices `extraNonce2` wrongly the server proof-checks a different nonce than the
one your GPU solved and every share dies as "low difficulty".

## Pearl (testing)

Mines Pearl on NVIDIA via CUDA and on AMD via Vulkan.

The Vulkan backend is new. Every one of its ten shader stages is byte-identical
to the CUDA reference on an RTX 4090 and an RX 7900 XT, and the miner runs the
whole chain against the host implementation at startup and refuses to mine if
anything differs - so a card that computes it wrongly stops with a message
rather than having every share rejected. **The end-to-end mine-and-submit has
so far been confirmed on NVIDIA only; the AMD run is pending.** On NVIDIA the
launcher still picks CUDA, whose shape and tile tuner has no Vulkan
equivalent.

**Pearl7 is withdrawn after a pool rejected a controlled share. Do not use a
normal Pearl launcher until Control approves a corrected release.**

This is the part most people want.

Open mine_pearl_herominers.sh in a text editor.
Change WALLET to your own address.
Save it.
Run it with command ./mine_pearl_herominers.sh

That is everything. You do not need a node and you do not need a gateway.

If you would rather type it out, it is command ./soat-miner --algo pearl-pow --pool pearl.herominers.com:1200 --wallet prl1yourwalletgoeshere --worker rig1

The pool is at pearl.herominers.com:1200 and takes no fee. `--worker` is
optional. Pool mode needs `--wallet`; it takes precedence over Pearl's default
local gateway, so do not remove `--pool` just because the startup banner also
mentions the gateway default.

### The other Pearl pools

There is a launcher for each pool below. They work the same way: edit WALLET,
run it. The withdrawal notice above covers all of them equally.

Every endpoint here was connected to and made to authorize, and the job it
pushed was checked field by field against what this miner's own notify parser
requires. Pools that authorize but push something we cannot read are listed
underneath, so nobody re-tests them by accident.

| launcher | endpoint | fee | payout | share difficulty on connect |
| --- | --- | --- | --- | --- |
| `mine_pearl_suprnova.sh` | prl.suprnova.cc:3373 | 0% (promo) | PPLNS | 244, vardiff |
| `mine_pearl_alphapool.sh` | us2.alphapool.tech:5571 | 0% | PPLNS | 50000 |
| `mine_pearl_rabbitminer.sh` | nl.rabbitminer.cc:1902 | 1% | PROP | 232827 |
| `mine_pearl_baikalmine.sh` | pearl-eu.baikalmine.com:2010 | 0.5% | PPLNS | 262144 |
| `mine_pearl_luckypool.sh` | pearl-us-east.luckypool.io:3360 | 1% | PROP | 888888 |
| `mine_pearl_k1pool_solo.sh` | us.pearlsolo.k1pool.com:3362 | 1% | SOLO | 1310720 |
| `mine_pearl_k1pool.sh` | us.pearl.k1pool.com:3360 | 0% | PPLNS | 1966080 |
| `mine_pearl_jetskipool.sh` | pearlski.jetskipool.ai:6970 | 1% | PROP | 2000000 |
| `mine_pearl_kryptex.sh` | prl.kryptex.network:7048 | 1% | PROP / SOLO | 2097120 |
| `mine_pearl_herominers.sh` | pearl.herominers.com:1200 | 0% | PROP / SOLO | 2097152 |
| `mine_pearl_mkpool_solo.sh` | pearl.mkpool.com:3411 | 2% | SOLO | 2097184 |

The difficulty column is the real reason to read this table rather than pick by
fee. HeroMiners hands out a fixed 2097152 and negotiates nothing, so a slow card
can mine for hours before it has one share to look at. Suprnova opened at 244 on
its vardiff port and lower still on its fixed ones - 1.22 on :3370 - and
AlphaPool opened at 50000. On a card that cannot reach HeroMiners' bound in
reasonable time, those are the pools that will actually produce a share to test
against. Each launcher's header records the alternate ports and regions.

Two things about difficulty that cost time to work out:

* The `target` in `mining.notify` is the truth. Pools label difficulty in their
  own units - Suprnova's job id says 4000000 where its target decodes to 244,
  a factor of 16384 - and the miner mines the target, not the label.
* Nothing here is negotiable from the miner's side yet. RabbitMiner and
  AlphaPool both accept a static difficulty as `d=VALUE` in the password field,
  but the Pearl login path always sends `pass "x"` (`pearlPoolAuthorizeRequest`
  defaults it and `run.cpp` never passes `opt.password` through), so `--pass`
  does nothing on Pearl. Wiring it through is a small change and would make
  those two pools tunable.

Pools deliberately without a launcher, and why:

| pool | what happened |
| --- | --- |
| pearlhash.xyz | Second largest by hashrate, and not this protocol. Its own instructions are `./pearl-miner --host pool.pearlhash.xyz:9000`; the port accepts TCP and then closes on our JSON authorize, on a plain, subscribe-first and array-login attempt alike. |
| 2miners (prl:1818, solo-prl:1919) | Authorizes, then pushes a notify with no `cert_version` field at all. The proof encoder has explicit V1/V2/V3 paths and guessing one builds a cryptographically different proof, so the parser refuses the job. |
| himpool (1480-1485) | Same: authorizes, no `cert_version`. |
| coin-miners.info:9337 | Same: authorizes, no `cert_version`. |
| linkrra.com:5567 | Same: authorizes, no `cert_version`. |
| pearlvolt.run:4141 | Header is 160 hex characters, an 80-byte header. Pearl's is 76. |
| grandpool.io:8070 | Agent whitelist. `soat-miner` gets `{"code":24,"msg":"miner agent not allowed"}`; `SRBMiner-MULTI/3.5.3` and `peakminer` are let straight in. Its notify shape looks right, so this pool is one hardcoded string away from working. |
| alphapool :5566 and :5573 | Open with a `pearl.challenge` proof-of-work handshake before login. Only their :5571 is plain stratum. |
| prlforge.com:3333/:3334 | Parses, but announces `"type":"v2"` and `cert_version: 2` while every other live pool is on 3, with no blocks recorded. Not shipped rather than ship a launcher that may mine unusable certificates. |
| pullsignal.co:3360 | Authorizes and then sends no job at all inside 12 seconds. |
| pearlfortune.org:443 | Closes the connection without answering. |
| mkpool :3413 | Advertised beside the working :3411, closes without answering. |
| akoyapool, tw-pool, kriptokyng, 1miner, nushypool, woolypooly, rplant, h9, f2pool, c3pool, meta2miner, pearlpool.cloud | No stratum endpoint published anywhere findable, and all report zero or no hashrate for Pearl. Not probed, so not shipped. |

### Pearl8 diagnostic capture (Control only)

This test-only build adds `--pearl-transcript FILE`. It records newline JSON
metadata for `mining.notify` (job id, target and certificate version), the
submit job id plus a BLAKE3 fingerprint/size of the proof, and the matching
pool reply. It never records `mining.authorize`, wallet addresses, worker
names, raw headers, or proof bytes; any PRL address echoed in a pool error is
redacted.

When Control explicitly authorizes one diagnostic run, add the following to
the existing Pearl launcher invocation and stop immediately after the first
reply:

`--pearl-transcript pearl8-control.jsonl`

Do not edit endpoint, wallet, node, gateway, or firewall settings. Give the
resulting local transcript only to Control; do not publish it.

## Pearl to your own node

Skip this unless you actually run a Pearl node.

A Pearl block needs a zero knowledge certificate. Your miner cannot build one.
pearl-gateway builds it, and it owns the node connection and the payout address.
So solo means running both.

Get the gateway from the Pearl repo.

https://github.com/pearl-research-labs/pearl

Then start the miner with command ./soat-miner --algo pearl-pow --gateway 127.0.0.1:8455

No wallet flag on that one. The gateway holds the address. Omitting `--gateway`
uses `127.0.0.1:8337`, the gateway's normal default; specify `HOST[:PORT]` to
use a different endpoint.

An 8GB card is enough. A 16GB card lets it pick a wider shape and go a bit faster.

## Pearl speeds

| GPU | Th/s | of the field |
|---|---|---|
| RTX 4090 | 167.9 | no published figure to compare to |
| RTX 5080 | 140.9 | 75% |

Th/s means tera multiply accumulates per second.
That is the unit hashrate.no reports, so these compare straight across.

hashrate.no lists 187.8 for a 5080.
They have no 4090 entry, so ignore anyone who quotes you one.

Both rows are measured end to end.
The number is the miner's own candidate counter, not a benchmark.
It ran against a live pool, at normal difficulty, on an idle card.

The 4080 and 4070 Super rows are gone for now.
Their old numbers came from the startup tuner.
The tuner times the matrix multiply and nothing else.
It never sees the work that runs beside it.
So it reads high, by 15 to 20 percent.
Both cards get re-measured before a number goes back here.

Do not compare watts against their table. We draw more than they list on the
same card and the same coin, and the same gap shows up on Ergo, so it is their
bench and not your card.

The miner times every shape and tile configuration on your gpu at startup and
keeps the fastest. A 4090 and a 5080 do not pick the same one. Hardcoding
either costs the other about 15%.

## Send me Ampere and Turing numbers

I do not have a 3000 series or a 2000 series card in the house right now.
So Pearl has never run on one.

If you have one, run it with command ./soat-miner --algo pearl-pow --bench
and open an issue with the line it prints.

https://github.com/blindrun/soat-miner/issues

What I want to know is whether it picks a sane configuration and what it gets.
Turing does not have the instruction the fast kernel uses, so it falls back to
a slower one. That path is tested for correctness but nobody has ever timed it.

## Pearl checks itself before it mines

The miner runs a small matrix multiply on your gpu at startup.
Then it runs the same one on your cpu and compares them.
If they disagree it refuses to mine and tells you why.

BE WARNED: a gpu or driver that computes this wrong does not look broken. It
mines happily and every share gets rejected. NVIDIA's Vulkan compiler was
caught doing exactly this to the Ergo kernel in August 2026, and it took an
outside contributor mining into a live pool to find it. That is why the check
is there, and it is why I am comfortable asking you to try this on a card I
have never tested.

## Bitcoin III (testing)

Mines Bitcoin III on NVIDIA via CUDA and on AMD, Intel and NVIDIA via Vulkan.
Both backends are checked against the same six real mainnet blocks, and the
Vulkan shader is additionally compared digest-for-digest against the host
implementation in `sha3.h` before it is trusted - 4.7 million nonces per run,
on each card. Measured: 1543 MH/s CUDA and 1086 MH/s Vulkan on an RTX 4090,
540 MH/s at 289 W on an RX 7900 XT.

Call this testing too. The protocol and the algorithm are covered by offline
tests and the six header vectors run on a real card, but nobody has mined a
share with it yet.

Open mine_bc3_pythonpool.sh in a text editor.
Change WALLET to your own address.
Save it.
Run it with command ./mine_bc3_pythonpool.sh

If you would rather type it out, it is command ./soat-miner --algo sha3-256t --pool stratum.pythonpool.dev:3357 --wallet yourbc3addresshere

The pool is at stratum.pythonpool.dev:3357.

### Other pools

You are not stuck with the one we happen to use. There is a script per pool,
and they all work the same way: edit WALLET, run it.

| script | pool | stratum | fee | payout |
|---|---|---|---|---|
| mine_bc3_pythonpool.sh | pythonpool.dev | stratum.pythonpool.dev:3357 | 0% | solo |
| mine_bc3_axehub.sh | axehub.app | pool.axehub.app:3338 | 0% | solo |
| mine_bc3_btc3forge.sh | btc3forge.com | btc3forge.com:3337 | 0.5% | proportional |
| mine_bc3_hashbay.sh | hashbay.io | stratum.hashbay.io:3344 | 0.5% | proportional |
| mine_bc3_vexta.sh | vexta-pool.co.uk | vexta-pool.co.uk:7333 | 0.5% | PPLNS |
| mine_bc3_argfamining_solo.sh | argfamining.com | stratum-us.argfamining.com:24152 | 1% | solo |
| mine_bc3_argfamining_prop.sh | argfamining.com | stratum-us.argfamining.com:24153 | 1% | proportional |
| mine_bc3_rplant.sh | rplant.xyz | stratum-na.rplant.xyz:7157 | 1% | proportional |
| mine_bc3_cryptoeire.sh | crypto-eire.com | stratum.crypto-eire.com:3362 | 1% | solo |
| mine_bc3_hashforge.sh | hashforge.online | stratum.hashforge.online:3341 | 1% | solo |

pythonpool and axehub are the only two that take nothing, which is a real
reason to prefer them. Fees are what each pool publishes, not what we measured.

Every endpoint in that list was opened by hand before the script was written:
it answers mining.subscribe, accepts a BC3 address, and pushes a mining.notify
whose prevhash matches the tip every other pool is building on. Several pools
run more than one port, and a few of them are for a different coin or a
different processor, so do not guess a port from another pool's number. Where a
pool has a second BC3 port worth knowing about, the script says so in a comment
at the top.

Some pools listed for BC3 on miningpoolstats have no script here on purpose.
zpool.ca sells hashrate by algorithm rather than by coin, so its sha3-256t port
was serving a different chain entirely when we looked, and it pays you in BTC
against a BTC address. ariabrain.com answers and authorises, but its
mining.notify carries 5 parameters where Bitcoin stratum has 9, so this miner
cannot read its work. btc3forge's advertised solo endpoint refuses connections.
A launcher pointing somewhere wrong is worse than no launcher.

BC3 pays to an ordinary Bitcoin-shaped address, so it starts with 1, 3 or bc1.
The miner checks the shape before it opens a socket and refuses anything that
is not one, so a seed or a private key pasted in by mistake never leaves your
machine.

## Bitcoin III is pool only

There is no solo path. Solo would mean calling getblocktemplate against a
Bitcoin III node and building the coinbase here, which is a different job from
the one the pool already does.

That also means nothing in the miner reads a wallet, a node, or a key. The
address you put in WALLET is a payout address the pool sends coins to, and it
is the only piece of BC3 identity the miner ever handles.

## Build from source

**Linux.** Install the CUDA toolkit and glslang, then run make.

Build both backends with command make

Build one with command make cuda or command make vulkan

**Windows.** You need the CUDA Toolkit and Visual Studio Build Tools with the
C++ workload. Then use cmake.

```
cmake -B build
cmake --build build --config Release
```

The binary lands in build\Release\soat-miner.exe

BE WARNED: RTX 50-series needs CUDA Toolkit 12.8 or newer to get native code.
Older toolkits still work. They fall back to PTX and the driver compiles it at
startup.

CUDA cannot be cross compiled from Linux. nvcc needs MSVC on a Windows host, so
the CUDA build is compiled on Windows in CI (`.github/workflows/windows-cuda.yml`).
The Windows download ships both builds and soat-miner.bat picks the right one.

## Correctness

A miner that hashes slightly wrong finds nothing and reports no error. So this
is checked against the real chain, not against itself.

Run the tests with command make test

It recomputes real mainnet blocks in Python from the Ergo node's own reference.
It compares the GPU dataset against that Python byte for byte. Then it rebuilds
the dataset on your card and reproduces a real block's hit from that block's
winning nonce.

Both backends are held to the same block and must agree exactly.

It also drives the miner's own search and verify against that block, instead of
just the kernels underneath them. A kernel can be right and the miner still
find nothing, if the solution count is read back before the kernel wrote it or
if building ahead hands mining the wrong table. Neither of those shows up as an
error anywhere.

That last one is checked by looking for the old block's nonce after the swap.
It has to be gone. If it is still there you are mining the previous height's
table and every share you send will be rejected.

The same reasoning covers the wire, because a correct hit still earns nothing if
the submit is malformed. The Lithos case runs the real binary against a mock
server transcribed from the Lithos client's own source, and checks the three
things that fail without any error: the submit carries the five params it
destructures, the nonce it reconstructs is the nonce the GPU actually solved,
and a reassigned extranonce is picked up rather than ignored. An earlier release
shipped a three-param submit to conventional pools and no share ever reached
one, while the panel reported every one of them as accepted.

## Mining around other GPU work

There is a guard script if you use the same card for Ollama or ComfyUI.

It stops mining when real work wants the GPU and starts it again after.

It watches process names, not VRAM. A 21.9GB model and a 7.3GB dataset cannot
share a 24GB card. By the time VRAM spikes it is already too late.

Copy guard.conf.example to /etc/soat-miner/guard.conf and edit it.
Install the two service files.
Start it with command systemctl enable --now soat-miner-guard

## How it works

Autolykos v2 is memory bound. Every nonce does 33 random lookups into that
7.27GB table. The hashing is not the slow part, the lookups are.

Proof: replace the lookups with arithmetic and a 6700 XT goes from 82.9 to 780
MH/s.

So hashrate tracks memory bandwidth and nothing else. Across three cards, two
brands and two APIs, every one of them lands at 20% to 23% of its own peak
bandwidth. The 7900 XT has 2.08x the bandwidth of a 6700 XT and gets 1.83x the
hashrate.

That also means most tuning does nothing.

All of these were measured. All of them got reverted.

- Caching the constant table in LDS.
- Baking the table size into the pipeline.
- wave32 against wave64.
- Batching the loads 8 wide.
- Unrolling the message fills.
- Replacing the 32 index divisions with a mask. 219.2 against 219.4.
- Splitting the hashing and the gathers into separate kernels.

Do not spend a weekend on them.

The last one is worth explaining. The kernel uses 128 registers, and that caps
it at 33% occupancy. The registers are all Blake2b. The gather loop that
actually stalls needs almost none, so it is stuck paying for hashing it is not
doing.

Splitting it in three fixed exactly that. The gather kernel came out at 64
registers with no spills, which is 67% occupancy, double what it had.

It made no difference. 217.3 against 217.5, and 12 more watts. Twice the warps
in flight bought nothing, so the card is not short of warps. It is waiting on
DRAM and more threads just wait alongside it.

Locality is the other thing people try. There is a real ceiling up there, but
you cannot reach it. Confining the lookups to a window and measuring a 4090:

| Window | MH/s |
|---|---|
| 64 MiB | 538.2 |
| 256 MiB | 231.3 |
| 1 GiB | 214.2 |
| 7.27 GiB, the real one | 217.5 |

So it is a cliff, not a curve. You only get it while the working set fits in
L2, which is 72 MB on this card.

Sorting the lookups into L2 sized bins would need 113 bins, and a bin only pays
if more lookups land in it than it holds elements. That means batching about
30 million nonces, and the index buffers for that come to 7.7GB on top of the
7.27GB dataset. It does not fit in 16GB at all. Not worth it.

Building the next block ahead is the same kind of trap, and it took a
measurement to see it.

Doing it the obvious way buys exactly zero. 202.6 against 202.6.

The reason is that the table costs the same amount of GPU work whether you
build it during the block or before it. Moving work around in time does not
create a card that can do more work. The stall you removed comes straight back
as a slower kernel.

What makes it pay is that the two kernels want different things. Mining waits
on DRAM. Building the table is Blake2b and barely touches memory. So they
should share the card better than they do.

They do not, because the build kernel launches 887,000 blocks and takes every
warp slot on every SM. Mining then cannot keep enough loads in flight.

Give the mining stream a high priority and the build stream a low one and that
stops happening. 215.1 against 216.6 at Ergo's real 2 minute blocks.

So 0.73%, and only because of the priorities. Without them it is nothing.

Two things to know if you build on this.

The deprioritized build takes about 21 seconds instead of 1.5. That is fine
against 2 minute blocks and it is not fine against 20 second ones. Test it at
the real block time. A short one flatters it.

Holding the second table costs nothing while it sits there. 218.1 against
218.1.

The cards that would gain most cannot run it. Build time goes up as the card
gets slower and memory goes up as the card gets more expensive. A 6700 XT
would save 8.2 seconds a block and has 12GB, so two tables never fit. The 4090
has the room and the least to gain.

None of this carried over to Vulkan on NVIDIA. It was built there too, with two
queues, two descriptor sets and an async fence, and it lost on both cards.

| GPU | off | on |
|---|---|---|
| RTX 5080 | 247.2 | 246.9 |
| RTX 4090 | 160.1 | 158.5 |

CUDA lets you say which stream matters. Vulkan only lets you hint it, with a
float at queue creation, and the driver is free to ignore it. NVIDIA's does.

You can see it in the readouts. With build-ahead off there is one clean dip per
block and then flat. With it on the dip is shallower but it smears across the
next several intervals, and the smear adds up to more than the dip it replaced.

Read that as Vulkan on NVIDIA, not as Vulkan. AMD was never tested and there is
good reason to think it goes the other way. AMD has real async compute engines,
its drivers honour queue priority, and lolMiner ships this same feature as an
AMD default called --ergo-prebuild. The card to try is a 7900 XT, since 20GB is
the smallest AMD card here that fits two 7.27GB tables at all.

The code is on the vulkan-build-ahead branch rather than deleted, for exactly
that test.

## Adding an algorithm

Drop it in src/algos/yourname/.
Add the object to ALGO_OBJS in the Makefile.
Add two lines to src/core/registry.cu.

An algorithm only has to answer four things. Prepare for this epoch. Test these
nonces. Say which won. Verify one.

The mining loop, the pool code and the readout are shared and you do not touch
them.

## Not done yet

- Multi GPU. One card at a time right now.
- More algorithms. Autolykos v2, Pearl and Bitcoin III so far.
- The Windows CUDA build is tested on a 5080 and nothing else.
- Bitcoin III on Windows. The code is there, the build is not: nvcc needs MSVC
  on a Windows host, so it comes from the windows-cuda job rather than from a
  Linux cross-build.
- Bitcoin III has never had a share accepted by a pool.

## Resources

https://github.com/blindrun/soat-miner

https://ergo.herominers.com

https://sonofatech.com

## License

MIT. Do what you want with it.
