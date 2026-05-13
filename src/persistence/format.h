#pragma once

#include <cstdint>

namespace vexdb {

// VEX0 format for vectors.bin
// Layout: [header 16 bytes] [float data...]
constexpr uint32_t kVex0Magic = 0x30584556;  // "VEX0" as bytes on little-endian
constexpr uint32_t kVex0Version = 1;

struct Vex0Header {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t count;
};
static_assert(sizeof(Vex0Header) == 16);

// HNSW format for hnsw.bin
// Layout: [header 24 bytes] [per-node: level(4) + layer0 neighbors + upper neighbors]
constexpr uint32_t kHnswMagic = 0x57534E48;  // "HNSW" as bytes on little-endian
constexpr uint32_t kHnswVersion = 1;

struct HnswHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t m;
    uint32_t max_level;
    uint32_t entry_point;
    uint32_t node_count;
};
static_assert(sizeof(HnswHeader) == 24);

// IDS format for ids.bin
// Layout: [header 12 bytes] [uint64_t array: offset -> user_id]
constexpr uint32_t kIdsMagic = 0x00534449;  // "IDS\0" as bytes on little-endian
constexpr uint32_t kIdsVersion = 1;

struct IdsHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
};
static_assert(sizeof(IdsHeader) == 12);

}  // namespace vexdb