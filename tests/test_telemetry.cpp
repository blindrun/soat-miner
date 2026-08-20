// Host-only regression test for the live rate labels.
//
// Pearl candidates are tiles, not hashes. Keep that distinction visible in the
// interactive readout without needing a CUDA device or a running work source.

#include <cstdio>
#include <cstring>
#include <string>

#include <unistd.h>

#include "../src/core/telemetry.h"

using om::GpuTelemetry;
using om::MinerStats;

static std::string readoutFor(const char *algo, double macsPerUnit = 0,
                              double rate = 0, bool tty = true) {
    MinerStats stats;
    stats.algo = algo;
    stats.hashrate = rate > 0 ? rate : 123.45;
    // Deliberately NOT the same as the rate. Both are printed on one line, so
    // an assertion looking for the converted rate would be satisfied by the
    // average and pass a build where only the rate is wrong - which is how a
    // mutation of shownRate alone survived this test once.
    stats.hashrateAvg = rate > 0 ? rate / 2.0 : 120.0;
    stats.macsPerUnit = macsPerUnit;
    stats.history = {110.0, 123.45};

    GpuTelemetry telemetry;
    telemetry.valid = true;
    telemetry.powerMilliwatts = 200000;
    telemetry.temperatureC = 60;
    telemetry.fanPercent = 40;

    FILE *capture = tmpfile();
    if (!capture) return "";
    const int saved = dup(fileno(stdout));
    if (saved < 0 || dup2(fileno(capture), fileno(stdout)) < 0) return "";

    om::g_asciiOnly = true;
    om::printReadout(stats, telemetry, tty);
    fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);

    rewind(capture);
    std::string output;
    char buffer[256];
    while (const size_t n = fread(buffer, 1, sizeof(buffer), capture))
        output.append(buffer, n);
    fclose(capture);
    return output;
}

int main() {
    // The shipped Pearl configuration: a 16x16 tile of a product whose every
    // element is a length-2048 dot product.
    const double kPearlMacs = 16.0 * 16.0 * 2048.0;   // 524,288

    const std::string pearl = readoutFor("pearl-pow", kPearlMacs);
    const std::string ergo = readoutFor("autolykos2");

    const bool pearlUnits = pearl.find("T MAC/s") != std::string::npos &&
                            pearl.find("T MAC/W") != std::string::npos &&
                            pearl.find("MH/s") == std::string::npos &&
                            pearl.find("MC/s") == std::string::npos;
    const bool hashUnits = ergo.find("MH/s") != std::string::npos &&
                           ergo.find("MH/W") != std::string::npos &&
                           ergo.find("T MAC") == std::string::npos;

    if (!pearlUnits || !hashUnits) {
        fprintf(stderr, "telemetry rate unit regression\n");
        return 1;
    }

    // The label alone is not the point. A wrong multiplier prints a
    // confident, comparable-looking number that is off by orders of
    // magnitude, which is worse than an honest MC/s. So check the arithmetic.
    //
    // 340 M candidates/s on a 4090 is 340e6 * 524288 = 1.783e14 MAC/s
    // = 178.26 T MAC/s. That figure sits against hashrate.no's 187.8 for a
    // 5080, which is the right order for these two cards.
    const std::string fast = readoutFor("pearl-pow", kPearlMacs, 340.0);
    if (fast.find("178.26") == std::string::npos) {
        fprintf(stderr, "T MAC/s conversion wrong: expected 178.26 in\n%s",
                fast.c_str());
        return 1;
    }

    // And the JSON, which is what the GUI and any monitoring actually read.
    const std::string json = readoutFor("pearl-pow", kPearlMacs, 340.0, false);
    const bool jsonOk =
        json.find("\"unit\":\"T MAC/s\"") != std::string::npos &&
        json.find("\"eff_unit\":\"T MAC/W\"") != std::string::npos &&
        json.find("\"rate\":178.26") != std::string::npos &&
        json.find("\"rate_avg\":89.13") != std::string::npos &&
        // The legacy keys stay for one release and must agree with the new
        // ones, or a consumer reading the old name gets a different number
        // from one reading the new name.
        json.find("\"mhs\":178.26") != std::string::npos &&
        json.find("\"eff_mh_w\":") != std::string::npos &&
        // The raw counter survives for diagnosis: 340 M/s = 3.4e8.
        json.find("\"candidates_per_s\":340000000") != std::string::npos;
    if (!jsonOk) {
        fprintf(stderr, "telemetry JSON shape regression:\n%s", json.c_str());
        return 1;
    }

    // A hashing algorithm must not grow a MAC unit or the raw candidate key.
    const std::string ergoJson = readoutFor("autolykos2", 0, 217.0, false);
    if (ergoJson.find("\"unit\":\"MH/s\"") == std::string::npos ||
        ergoJson.find("candidates_per_s") != std::string::npos ||
        ergoJson.find("\"rate\":217.00") == std::string::npos) {
        fprintf(stderr, "hashing algorithms must stay MH/s:\n%s", ergoJson.c_str());
        return 1;
    }

    puts("telemetry unit labels: ok");
    return 0;
}
