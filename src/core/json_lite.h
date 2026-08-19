// The minimal JSON reading the stratum clients need.
//
// These three functions started life inside stratum.cpp's anonymous namespace,
// where the Ergo client still has its own copy. They are duplicated here rather
// than lifted out of that file because the Ergo stratum client is under active
// work in another branch and a gratuitous refactor of it would collide. If both
// clients survive, delete the copies in stratum.cpp and include this instead -
// two transcriptions of a hand-rolled parser is exactly the kind of thing that
// drifts.

#pragma once

#include <cctype>
#include <string>
#include <vector>

namespace om {

/** Extracts the raw text of a JSON value by key, without a JSON library. */
inline bool jsonRawValue(const std::string &j, const char *key,
                         std::string *out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && isspace((unsigned char)j[p])) p++;
    if (p >= j.size()) return false;

    const size_t start = p;
    if (j[p] == '[' || j[p] == '{') {
        const char open = j[p], close = (open == '[') ? ']' : '}';
        int depth = 0;
        bool inStr = false;
        for (; p < j.size(); p++) {
            const char c = j[p];
            if (inStr) {
                if (c == '\\') p++;
                else if (c == '"') inStr = false;
                continue;
            }
            if (c == '"') inStr = true;
            else if (c == open) depth++;
            else if (c == close) {
                depth--;
                if (depth == 0) { p++; break; }
            }
        }
    } else if (j[p] == '"') {
        p++;
        while (p < j.size() && j[p] != '"') {
            if (j[p] == '\\') p++;
            p++;
        }
        p++;
    } else {
        while (p < j.size() && j[p] != ',' && j[p] != '}' && j[p] != ']') p++;
        size_t b = start, e = p;
        auto ws = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        };
        while (b < e && ws(j[b])) b++;
        while (e > b && ws(j[e - 1])) e--;
        *out = j.substr(b, e - b);
        return true;
    }
    *out = j.substr(start, p - start);
    return true;
}

/** Splits a JSON array's top-level elements into raw strings. */
inline std::vector<std::string> jsonSplitArray(const std::string &arr) {
    std::vector<std::string> out;
    if (arr.size() < 2) return out;
    size_t p = arr.find('[');
    if (p == std::string::npos) return out;
    p++;
    int depth = 0;
    bool inStr = false;
    size_t start = p;
    for (; p < arr.size(); p++) {
        const char c = arr[p];
        if (inStr) {
            if (c == '\\') p++;
            else if (c == '"') inStr = false;
            continue;
        }
        if (c == '"') inStr = true;
        else if (c == '[' || c == '{') depth++;
        else if (c == '}' || (c == ']' && depth > 0)) depth--;
        else if ((c == ',' && depth == 0) || (c == ']' && depth == 0)) {
            std::string e = arr.substr(start, p - start);
            while (!e.empty() && isspace((unsigned char)e.front())) e.erase(e.begin());
            while (!e.empty() && isspace((unsigned char)e.back())) e.pop_back();
            if (!e.empty()) out.push_back(e);
            start = p + 1;
            if (c == ']') break;
        }
    }
    return out;
}

inline std::string jsonUnquote(const std::string &s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

}  // namespace om
