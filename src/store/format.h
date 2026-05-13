#pragma once

#include <cstdint>

namespace vexdb {

// VEX0 binary format for vectors.bin
// Layout: [magic 4 bytes] [version 4 bytes] [dim 4 bytes] [count 4 bytes] [float data...]

constexpr uint32_t kVex0Magic = 0x30584556;  // "VEX0" in little-endian
constexpr uint32_t kVex0Version = 1;

struct Vex0Header {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t count;
};

static_assert(sizeof(Vex0Header) == 16);

}  // namespace vexdb