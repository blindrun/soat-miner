// The GPU claim guard's own discrimination, proved against fixtures.
//
// This is the one tool in the tree whose entire job is to say no, and it said
// yes in two situations it existed to prevent: with gpulock off PATH it
// permitted an unclaimed card, and it matched the holder by substring so
// "wF:p1" would have passed against a claim held by "wF:p10". Both were found
// by a peer lane running a probe rather than by reading the code.
//
// Every case below runs against a fixture lock directory, so it needs no
// gpulock, no GPU and no claim. The guard exits(3) on refusal, so each case
// runs in a forked child and the parent reads the status.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#include "vk_claim_guard.h"

static int failures = 0, checks = 0;

static void writeClaim(const std::string &dir, const char *key,
                       const char *who, long pid, long long until) {
    std::string d = dir + "/" + key + ".lock";
    mkdir(dir.c_str(), 0755);
    mkdir(d.c_str(), 0755);
    FILE *f = fopen((d + "/meta").c_str(), "w");
    fprintf(f, "who=%s\npid=%ld\npurpose=fixture\nsince=0\nuntil=%lld\n", who,
            pid, until);
    fclose(f);
}

/** Run the guard in a child. Returns true if it PERMITTED. */
static bool permits(const char *device, const char *who, const char *dir) {
    // The guard refuses with exit(), which flushes stdio - and a forked child
    // inherits this process's unflushed stdout buffer, so every earlier line
    // was reprinted by each refusal. Flush before forking.
    fflush(stdout);
    pid_t p = fork();
    if (p == 0) {
        fclose(stderr);
        if (who) setenv("GPULOCK_WHO", who, 1); else unsetenv("GPULOCK_WHO");
        setenv("GPULOCK_DIR", dir, 1);
        unsetenv("SOAT_VK_NO_CLAIM_GUARD");
        om::requireGpuClaim(device);
        _exit(0);
    }
    int st = 0;
    waitpid(p, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

static void expect(const char *what, bool got, bool want) {
    checks++;
    if (got != want) {
        failures++;
        printf("  FAIL: %s -> %s, expected %s\n", what,
               got ? "PERMITTED" : "refused", want ? "PERMITTED" : "refused");
    } else {
        printf("  ok: %s -> %s\n", what, got ? "permitted" : "refused");
    }
}

int main() {
    char tmpl[] = "/tmp/claimguardXXXXXX";
    const char *dir = mkdtemp(tmpl);
    if (!dir) { printf("could not make a fixture dir\n"); return 1; }
    const long alive = (long)getpid();
    const long long far = 4102444800LL;   // 2100

    printf("claim guard:\n");

    // The two false permits that prompted this test.
    expect("no claim file at all is a refusal, not a shrug",
           permits("NVIDIA GeForce RTX 4090", "me", dir), false);

    writeClaim(dir, "4070s", "probe-holder-name", alive, far);
    expect("a holder that merely CONTAINS our name is not us",
           permits("NVIDIA GeForce RTX 4070 SUPER", "probe", dir), false);
    expect("and the lane-prefix case: wF:p1 against wF:p10",
           permits("NVIDIA GeForce RTX 4070 SUPER", "probe-holder-nam", dir),
           false);

    // The positive case, or the test proves only that it always refuses.
    expect("an exact holder with a live pid is permitted",
           permits("NVIDIA GeForce RTX 4070 SUPER", "probe-holder-name", dir),
           true);

    // Everything else that must refuse.
    expect("GPULOCK_WHO unset",
           permits("NVIDIA GeForce RTX 4070 SUPER", nullptr, dir), false);

    writeClaim(dir, "4080", "me", 999999999, far);
    expect("a claim whose process is gone is stale, not ours",
           permits("NVIDIA GeForce RTX 4080", "me", dir), false);

    // A malformed claim: the guard must not read it as consent.
    {
        std::string d = std::string(dir) + "/5080.lock";
        mkdir(d.c_str(), 0755);
        FILE *f = fopen((d + "/meta").c_str(), "w");
        fprintf(f, "holder=me\n");   // the format changed under us
        fclose(f);
    }
    expect("a claim it cannot parse is a refusal",
           permits("NVIDIA GeForce RTX 5080", "me", dir), false);

    // An unknown variant of a fleet family must not borrow another card's lock.
    writeClaim(dir, "4070s", "me", alive, far);
    expect("a plain RTX 4070 does not inherit the SUPER's claim",
           permits("NVIDIA GeForce RTX 4070", "me", dir), false);

    // And a card that is genuinely not contended still runs.
    expect("llvmpipe is not a fleet card",
           permits("llvmpipe (LLVM 15.0.7, 256 bits)", "me", dir), true);

    if (checks == 0) { printf("ZERO CHECKS RAN\n"); return 1; }
    if (failures) { printf("claim guard: %d failure(s)\n", failures); return 1; }
    printf("claim guard: ok (%d checks)\n", checks);
    return 0;
}
