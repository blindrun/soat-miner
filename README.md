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

Unpack it on Linux with command tar xzf soat-miner_v0.1.2_Lin64.tar.gz

On Windows just unzip it.

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
| RTX 4090 | CUDA | 217.5 |
| RX 7900 XT | Vulkan | 151.9 |
| RX 6700 XT | Vulkan | 82.9 |

lolMiner does about 265 MH/s on a 4090 and takes 0.75% to 1%. This is close to
that with none, and you can read the source.

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

CUDA cannot be cross compiled from Linux. nvcc needs MSVC on a Windows host.
That is why the Windows download ships the Vulkan build only.

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

None of this carried over to Vulkan. It was built there too, with two queues,
two descriptor sets and an async fence, and it lost on both cards.

| GPU | off | on |
|---|---|---|
| RTX 5080 | 247.2 | 246.9 |
| RTX 4090 | 160.1 | 158.5 |

CUDA lets you say which stream matters. Vulkan only lets you hint it, with a
float at queue creation, and the driver is free to ignore it. It does.

You can see it in the readouts. With build-ahead off there is one clean dip per
block and then flat. With it on the dip is shallower but it smears across the
next several intervals, and the smear adds up to more than the dip it replaced.

That code is not in the tree. It is a lot of Vulkan for a negative number.

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
- More algorithms. Only Autolykos v2 so far.
- The Windows CUDA build is tested on a 5080 and nothing else.

## Resources

https://github.com/blindrun/soat-miner

https://ergo.herominers.com

https://sonofatech.com

## License

MIT. Do what you want with it.
