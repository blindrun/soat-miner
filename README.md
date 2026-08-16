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

It does three things. It recomputes real mainnet blocks in Python from the Ergo
node's own reference. It compares the GPU dataset against that Python byte for
byte. Then it rebuilds the dataset on your card and reproduces a real block's
hit from that block's winning nonce.

Both backends are held to the same block and must agree exactly.

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
