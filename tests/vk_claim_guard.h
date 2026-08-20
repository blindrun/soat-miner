// Refuse to touch a GPU this session has not claimed.
//
// This lives in the test binary and not in a wrapper script on purpose. Every
// GPU run in this lane went through a runner that checked the result of
// `gpulock claim`, and twice the guard was bypassed anyway - once by piping the
// claim's output through a grep that hid the refusal, once by running the
// binary directly while debugging something else. Both times the card was
// already held by another lane and their measurement shared it.
//
// A guard you can step around is a convention. This one is in the same process
// as the work, so there is nothing to step around: the test aborts before it
// creates a Vulkan device.
//
// SOAT_VK_NO_CLAIM_GUARD=1 opts out, for a machine that has no gpulock - not
// for a hurry.
//
// CLAIM AND RUN IN THE SAME SHELL. gpulock records the claiming process's pid
// and treats a claim whose process is gone as stale - anyone may reap it - so
// this guard refuses on one too. A claim made in one shell and used from a
// later one is not protecting the card, and the refusal says so by name rather
// than letting the run proceed under a claim any lane may take.

#pragma once

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>

namespace om {

/// The gpulock key for a Vulkan device name, or "" if it is not a fleet card.
///
/// The needles are as specific as the fleet requires. "4070" alone used to map
/// every 4070 variant - Ti, SUPER, plain - onto the one key the fleet actually
/// has, so a guard running on a plain 4070 would have checked the SUPER's lock
/// and reported on a card it was not using. A recognised family with an
/// unrecognised variant returns kUnknownVariant, which the caller refuses on:
/// "I cannot tell which lock this is" must not read the same as "this card has
/// no lock".
inline const char *gpulockUnknownVariant() { return "?"; }

inline const char *gpulockKeyFor(const char *deviceName) {
    struct Map { const char *needle, *key; };
    static const Map kMap[] = {
        {"4090", "4090"},        {"5080", "5080"},   {"4080", "4080"},
        {"4070 SUPER", "4070s"}, {"7900 XT", "7900xt"},
        {"6700 XT", "6700xt"},
    };
    for (const auto &m : kMap)
        if (strstr(deviceName, m.needle)) return m.key;
    // Same family, variant we do not have a key for.
    static const char *kFamilies[] = {"4070", "7900", "6700"};
    for (const char *f : kFamilies)
        if (strstr(deviceName, f)) return gpulockUnknownVariant();
    return "";
}

/// One `key=value` line out of a gpulock claim's meta file.
inline bool gpulockMetaField(const std::string &meta, const char *key,
                             std::string *out) {
    const std::string pat = std::string(key) + "=";
    size_t p = 0;
    while (p <= meta.size()) {
        const size_t eol = meta.find('\n', p);
        const std::string line = meta.substr(p, eol == std::string::npos
                                                    ? std::string::npos
                                                    : eol - p);
        if (line.compare(0, pat.size(), pat) == 0) {
            *out = line.substr(pat.size());
            while (!out->empty() && (out->back() == '\r' || out->back() == ' '))
                out->pop_back();
            return true;
        }
        if (eol == std::string::npos) break;
        p = eol + 1;
    }
    return false;
}

/**
 * Aborts unless GPULOCK_WHO holds a claim on this device.
 *
 * Returns normally on a device that is not a fleet card - llvmpipe and
 * integrated parts are not in the lock table and never contended.
 */
inline void requireGpuClaim(const char *deviceName) {
    if (getenv("SOAT_VK_NO_CLAIM_GUARD")) return;

    const char *key = gpulockKeyFor(deviceName);
    if (!*key) return;   // llvmpipe, an iGPU: not in the lock table, never contended

    auto refuse = [&](const std::string &why) {
        fprintf(stderr,
                "\nREFUSING TO RUN on %s.\n  %s\n"
                "  Another lane's measurement shares the card otherwise, and "
                "both results become unusable.\n"
                "  export GPULOCK_WHO=<your session> && gpulock claim <gpu> "
                "\"why\" <minutes>\n\n",
                deviceName, why.c_str());
        exit(3);
    };

    if (strcmp(key, gpulockUnknownVariant()) == 0)
        refuse(std::string("this looks like a fleet GPU family but not a "
                           "variant with a lock key, so the guard cannot tell "
                           "which card it is. Set SOAT_VK_NO_CLAIM_GUARD=1 "
                           "only if this really is an uncontended card."));

    const char *who = getenv("GPULOCK_WHO");
    if (!who || !*who)
        refuse(std::string("GPULOCK_WHO is not set, so this cannot check "
                           "whether it holds ") + key + ".");

    // READ THE CLAIM, DO NOT PARSE THE REPORT.
    //
    // This used to run `gpulock status 2>/dev/null` and scan the output, which
    // failed in the one direction a guard must never fail. It refused only when
    // it POSITIVELY FOUND the card's row and positively failed to find the
    // holder; anything that made the output empty - gpulock missing, not on
    // PATH, crashed, renamed, or its column layout changed - left the search
    // finding nothing and the guard PERMITTED. The 2>/dev/null made a failing
    // gpulock indistinguishable from a working one that said nothing.
    // Reproduced: with gpulock off PATH the guard permitted an unclaimed 4090.
    //
    // The lock directory is the authority anyway - `status` is a rendering of
    // it - so read `who=` out of the claim's own meta file and require a
    // positive match. Every outcome that is not "this session demonstrably
    // holds this card" is now a refusal.
    const char *dir = getenv("GPULOCK_DIR");
    std::string path = (dir && *dir) ? dir : (std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.gpu-locks");
    path += "/" + std::string(key) + ".lock/meta";

    FILE *f = fopen(path.c_str(), "r");
    if (!f)
        refuse(std::string(key) + " has no claim at all (no " + path + ").");
    std::string meta;
    char chunk[512];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) meta.append(chunk, n);
    fclose(f);

    std::string holder;
    if (!gpulockMetaField(meta, "who", &holder))
        refuse(std::string("could not read who holds ") + key + " from " + path +
               " - the claim format may have changed. A guard that cannot tell "
               "must refuse.");

    // EXACT, not a substring. `strstr` used to be enough for "wF:p1" to match a
    // claim held by "wF:p10", and for a bare lane name to match any longer
    // holder string containing it. Reproduced with WHO=probe against a claim
    // held by probe-holder-name.
    if (holder != who)
        refuse(std::string(key) + " is held by \"" + holder + "\", not \"" +
               who + "\".");

    // A claim whose process is gone is stale: gpulock lets anyone reap it, so
    // it is not ours to rely on even if the name still matches.
    std::string pidStr;
    if (gpulockMetaField(meta, "pid", &pidStr)) {
        const long pid = atol(pidStr.c_str());
        if (pid > 0 && kill((pid_t)pid, 0) != 0)
            refuse(std::string(key) + " is claimed by \"" + holder +
                   "\" but that process (pid " + pidStr +
                   ") is gone, so the claim is stale and anyone may take it.");
    }

    // Expiry is a warning, not a refusal: the claim is still ours and another
    // lane is supposed to ask before stealing it. Say so, because a long run
    // that silently outlives its window is how a collision starts.
    std::string untilStr;
    if (gpulockMetaField(meta, "until", &untilStr)) {
        const long long until = atoll(untilStr.c_str());
        if (until > 0 && (long long)time(nullptr) > until)
            fprintf(stderr,
                    "[claim-guard] your claim on %s has expired but is still "
                    "yours; renew it with `gpulock renew %s <minutes>`\n",
                    key, key);
    }
}

}  // namespace om
