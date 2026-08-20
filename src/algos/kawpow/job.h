// What a KawPow job carries that the core Job cannot express.
//
// Job::epoch is the core's "rebuild your state when this changes" trigger, and
// for KawPow that is the Ethash epoch - height / 7500. But the random program
// changes on a THREE block period, so the algorithm also needs the actual
// block height, which is a different number changing 2500 times as often.
// Putting the height in epoch would rebuild a 6 GB DAG every minute; putting
// the epoch there and losing the height would mine the wrong program for up to
// 7499 blocks.
//
// So the height rides in Job::extra, which is exactly what that field is for.
// The format is deliberately plain text rather than JSON: it is written and
// read in this repository only, and a parser that cannot fail is worth more
// than a general one.
//
//   "h=4501000"

#pragma once

#include <stdlib.h>

#include <string>

namespace om {
namespace kawpow {

inline std::string encodeJobExtra(int height) {
    return "h=" + std::to_string(height);
}

/**
 * The block height this job is for.
 *
 * Falls back to treating Job::epoch as a height when there is no extra, which
 * is the benchmark path: run.cpp fills in msg and epoch and nothing else,
 * because Ergo's epoch IS its height. Reading it as a height here keeps
 * `--bench-height` meaning what it says.
 */
inline int jobHeight(const std::string &extra, uint64_t epochField) {
    const size_t p = extra.find("h=");
    if (p == std::string::npos) return (int)epochField;
    return (int)strtol(extra.c_str() + p + 2, nullptr, 10);
}

}  // namespace kawpow
}  // namespace om
