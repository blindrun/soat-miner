// A REFERENCE ORACLE for ProgPoW 0.9.2 / ProgPowZ, transcribed line for line
// from chfast/ethash v0.4.1 lib/ethash/progpow.cpp - the file Zano's node
// vendors as its consensus implementation - and wired to THIS project's
// dataset functions.
//
// Why it exists: the end-to-end vectors do not reproduce, and the cheapest way
// to find out whether that is our loop or our data is to run a faithful copy
// of the reference loop over the same data. It is the same technique that
// found the KawPow raven-tail constant.
//
// WHAT IT ESTABLISHED: the oracle fails too, identically. So the difference is
// NOT in hashMixLight or runProgramLane - a verbatim transcription of the
// reference gets the same wrong answer over the same inputs. That eliminates
// the loop, which was the largest remaining suspect.
//
// What has since been verified individually, each against a published anchor,
// and all of which MATCH:
//   light cache and full dataset sizes for epoch 0 (16776896 / 1073739904)
//   epoch seeds (epoch 1 = 290decd9...e563, the known Ethash constant)
//   fnv1a, kiss99, keccak_f800, fill_mix, random_math, random_merge
//   the program's src/dst permutations and KISS99 seeding for period 600
//   the L1 build - calculate_dataset_item_2048 for i in 0..63
//   the seed being be::uint64 of the first keccak's output
//   the lane-hash and mix-hash reductions
//
// So every part verifies and the composite does not. That is the honest state;
// it is recorded without a proposed cause, because three causes have been
// proposed on this branch already and all three were withdrawn.
//
// Build: g++ -O2 --std=c++17 -Isrc tests/ref_progpow_092.cpp -o ref
// Not in `make test-kawpow` - it is a diagnostic, and it currently fails.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "algos/progpow/progpow.h"
#include "algos/progpow/ethash.h"
using namespace om::progpow;
static const uint32_t kRegs=32, kLanes=16;
struct RefState {
    Kiss99 rng; uint32_t dst_seq[32], src_seq[32];
    uint32_t dst_counter=0, src_counter=0;
    uint32_t next_dst(){ return dst_seq[(dst_counter++)%kRegs]; }
    uint32_t next_src(){ return src_seq[(src_counter++)%kRegs]; }
    explicit RefState(uint64_t seed){
        const uint32_t lo=(uint32_t)seed, hi=(uint32_t)(seed>>32);
        const uint32_t z=fnv1a(kFnvOffsetBasis,lo), w=fnv1a(z,hi);
        const uint32_t jsr=fnv1a(w,lo), jcong=fnv1a(jsr,hi);
        rng=Kiss99{z,w,jsr,jcong};
        for(uint32_t i=0;i<kRegs;i++){dst_seq[i]=i;src_seq[i]=i;}
        for(uint32_t i=kRegs;i>1;--i){
            std::swap(dst_seq[i-1],dst_seq[rng.next()%i]);
            std::swap(src_seq[i-1],src_seq[rng.next()%i]);
        }
    }
};
typedef uint32_t MixArr[16][32];
static void refRound(const uint32_t*cache,int nC,const uint32_t*l1,uint32_t nItems,
                     uint32_t r, MixArr mix, RefState state, int variant,
                     int cacheAcc, int mathOps){
    const uint32_t item_index = mix[r%kLanes][0] % nItems;
    uint32_t item[64];
    datasetItem2048(cache,nC,item_index,item);
    const int maxOps = cacheAcc>mathOps?cacheAcc:mathOps;
    for(int i=0;i<maxOps;i++){
        if(i<cacheAcc){
            const uint32_t src=state.next_src(), dst=state.next_dst(), sel=state.rng.next();
            for(uint32_t l=0;l<kLanes;l++){
                const uint32_t off = mix[l][src] % kL1CacheWords;
                randomMerge(&mix[l][dst], l1[off], sel);
            }
        }
        if(i<mathOps){
            const uint32_t src_rnd = state.rng.next() % (kRegs*(kRegs-1));
            const uint32_t src1 = src_rnd % kRegs;
            uint32_t src2 = src_rnd / kRegs; if(src2>=src1) ++src2;
            const uint32_t sel1=state.rng.next();
            const uint32_t dst=state.next_dst();
            const uint32_t sel2=state.rng.next();
            for(uint32_t l=0;l<kLanes;l++){
                const uint32_t data = randomMath(mix[l][src1],mix[l][src2],sel1,variant);
                randomMerge(&mix[l][dst],data,sel2);
            }
        }
    }
    uint32_t dsts[4],sels[4];
    for(int i=0;i<4;i++){ dsts[i]= i==0?0:state.next_dst(); sels[i]=state.rng.next(); }
    for(uint32_t l=0;l<kLanes;l++){
        const uint32_t off = ((l^r)%kLanes)*4;
        for(int i=0;i<4;i++) randomMerge(&mix[l][dsts[i]], item[off+i], sels[i]);
    }
}
int main(int argc,char**argv){
    int variant=kMath094, cacheAcc=12, mathOps=20, period=50;
    for(int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--math092") variant=kMath092; }
    // vector: block 0, header zeros, nonce 0
    const int block=0;
    uint32_t hdr[8]; memset(hdr,0,sizeof hdr);
    uint32_t st[25]; memset(st,0,sizeof st);
    for(int i=0;i<8;i++) st[i]=hdr[i];
    keccakf800(st);
    const uint32_t seed_hi=bswap32(st[0]), seed_lo=bswap32(st[1]);
    const uint64_t seed=((uint64_t)seed_hi<<32)|seed_lo;
    const int e=epochNumber(block,30000), nC=lightCacheNumItems(e);
    const uint32_t nItems=(uint32_t)(fullDatasetNumItems(e)/2);
    uint32_t sd[8]; epochSeed(e,sd);
    std::vector<uint32_t> cache((size_t)nC*16,0);
    buildLightCache(cache.data(),nC,sd);
    std::vector<uint32_t> l1(kL1CacheWords,0);
    for(uint32_t i=0;i<kL1CacheBytes/256;i++)
        datasetItem2048(cache.data(),nC,i,l1.data()+i*64);
    MixArr mix;
    { const uint32_t z=fnv1a(kFnvOffsetBasis,(uint32_t)seed);
      const uint32_t w=fnv1a(z,(uint32_t)(seed>>32));
      for(uint32_t l=0;l<kLanes;l++){
        const uint32_t jsr=fnv1a(w,l), jcong=fnv1a(jsr,l);
        Kiss99 rng{z,w,jsr,jcong};
        for(uint32_t i=0;i<kRegs;i++) mix[l][i]=rng.next(); } }
    RefState state{(uint64_t)(block/period)};
    for(uint32_t i=0;i<64;i++) refRound(cache.data(),nC,l1.data(),nItems,i,mix,state,variant,cacheAcc,mathOps);
    uint32_t lane[16];
    for(uint32_t l=0;l<kLanes;l++){ lane[l]=kFnvOffsetBasis;
        for(uint32_t i=0;i<kRegs;i++) lane[l]=fnv1a(lane[l],mix[l][i]); }
    uint32_t mh[8]; for(int i=0;i<8;i++) mh[i]=kFnvOffsetBasis;
    for(uint32_t l=0;l<kLanes;l++) mh[l%8]=fnv1a(mh[l%8],lane[l]);
    printf("mix  got  "); for(int i=0;i<8;i++) printf("%08x",mh[i]); printf("\n");
    printf("mix  want faeb1be51075b03a4ff44b335067951ead07a3b078539ace76fd56fc410557a3\n");
    return 0;
}
