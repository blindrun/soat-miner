// Load a SPIR-V module, and refuse anything that is not one.
//
// Every gate here rolled its own read-the-file-into-a-vector, and none of them
// checked what they had read. That is not a theoretical gap: a Makefile line
// passed a 24 MB vector file where the .spv belonged, vkCreateShaderModule
// returned VK_ERROR_INVALID_SHADER, and the error reads as "the shader is
// broken" - so it points at whoever last touched the shader rather than at
// whoever last touched the wiring. Two minutes went into suspecting a shader
// that was fine.
//
// Four bytes settle it before Vulkan is involved.
#pragma once

#include <stdint.h>
#include <stdio.h>

#include <vector>
#include <string.h>

namespace om {

/// SPIR-V's magic number, little-endian, from the spec's first word.
static constexpr uint32_t kSpirvMagic = 0x07230203u;

/** Reads `path` and returns true only if it really is a SPIR-V module.
 *  Prints what is wrong; the caller only has to check the bool. */
inline bool loadSpirv(const char *path, std::vector<uint8_t> *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "%s is empty\n", path);
        fclose(f);
        return false;
    }
    out->resize((size_t)sz);
    const bool read = fread(out->data(), 1, (size_t)sz, f) == (size_t)sz;
    fclose(f);
    if (!read) {
        fprintf(stderr, "could not read all of %s\n", path);
        return false;
    }
    if (out->size() % 4 != 0) {
        fprintf(stderr,
                "%s is %zu bytes, not a multiple of 4, so it is not SPIR-V\n",
                path, out->size());
        return false;
    }
    uint32_t magic = 0;
    memcpy(&magic, out->data(), 4);
    if (magic != kSpirvMagic) {
        fprintf(stderr,
                "%s does not start with the SPIR-V magic number "
                "(got 0x%08x, want 0x%08x) - this is not a compiled shader.\n"
                "  Check the argument order: these gates take the .spv path.\n",
                path, magic, kSpirvMagic);
        return false;
    }
    return true;
}

/** The same, as the uint32_t words vkCreateShaderModule wants. */
inline bool loadSpirvWords(const char *path, std::vector<uint32_t> *out) {
    std::vector<uint8_t> bytes;
    if (!loadSpirv(path, &bytes)) return false;
    out->resize(bytes.size() / 4);
    memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

}  // namespace om
