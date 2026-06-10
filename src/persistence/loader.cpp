#include "persistence/loader.h"

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "persistence/format.h"
#include "store/in_memory_store.h"
#include "store/mmap_store.h"

namespace vextor {

// Bound for per-node HNSW level from disk (defense against corrupt hnsw.bin).
constexpr int k_max_plausible_hnsw_node_level = 64;

static HnswGraph load_hnsw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open " + path);

    HnswHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw std::runtime_error("load: truncated HNSW header");
    if (header.magic != kHnswMagic) throw std::runtime_error("load: invalid HNSW magic");
    if (header.version != kHnswVersion) throw std::runtime_error("load: unsupported HNSW version");

    HnswGraph graph;
    graph.m = static_cast<int>(header.m);
    graph.max_level = static_cast<int>(header.max_level);
    graph.entry_point = header.entry_point;

    int stride = graph.layer0_stride();
    uint32_t node_count = header.node_count;

    for (uint32_t i = 0; i < node_count; i++) {
        int32_t level = 0;
        in.read(reinterpret_cast<char*>(&level), sizeof(level));
        if (!in) throw std::runtime_error("load: truncated HNSW node data");

        if (level < 0 || level > k_max_plausible_hnsw_node_level) {
            throw std::runtime_error("load: implausible HNSW level");
        }

        graph.levels.push_back(level);

        int32_t l0_count = 0;
        in.read(reinterpret_cast<char*>(&l0_count), sizeof(l0_count));
        if (!in) throw std::runtime_error("load: truncated HNSW layer0 count");
        if (l0_count < 0 || l0_count > graph.max_m0() + 1) {
            throw std::runtime_error("load: invalid HNSW layer0 neighbor count");
        }

        std::size_t base = graph.layer0.size();
        graph.layer0.resize(base + stride, 0);
        in.read(reinterpret_cast<char*>(&graph.layer0[base]), l0_count * sizeof(Offset));
        if (!in) throw std::runtime_error("load: truncated HNSW layer0 neighbors");
        graph.layer0_count.push_back(l0_count);

        graph.upper.emplace_back();
        if (level > 0) {
            graph.upper.back().resize(level);
            for (int layer = 1; layer <= level; layer++) {
                int32_t count = 0;
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                if (!in) throw std::runtime_error("load: truncated HNSW upper layer count");
                if (count < 0 || count > graph.m) {
                    throw std::runtime_error("load: invalid HNSW upper layer neighbor count");
                }
                graph.upper.back()[layer - 1].resize(count);
                in.read(reinterpret_cast<char*>(graph.upper.back()[layer - 1].data()),
                        count * sizeof(Offset));
                if (!in) throw std::runtime_error("load: truncated HNSW upper layer neighbors");
            }
        }
    }

    graph.empty = (graph.size() == 0);
    return graph;
}

static IdMapping load_ids(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open " + path);

    IdsHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw std::runtime_error("load: truncated IDS header");
    if (header.magic != kIdsMagic) throw std::runtime_error("load: invalid IDS magic");
    if (header.version != kIdsVersion) throw std::runtime_error("load: unsupported IDS version");

    std::vector<VectorId> ids(header.count);
    in.read(reinterpret_cast<char*>(ids.data()), header.count * sizeof(VectorId));
    if (!in) throw std::runtime_error("load: truncated IDS data");

    IdMapping mapping;
    for (VectorId id : ids) {
        mapping.insert(id);
    }
    return mapping;
}

static InMemoryStore load_vectors_memory(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open " + path);

    Vex0Header header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw std::runtime_error("load: truncated VEX0 header");
    if (header.magic != kVex0Magic) throw std::runtime_error("load: invalid VEX0 magic");
    if (header.version != kVex0Version) throw std::runtime_error("load: unsupported VEX0 version");

    InMemoryStore store(header.dim);
    std::vector<float> buf(header.dim);
    for (uint32_t i = 0; i < header.count; i++) {
        in.read(reinterpret_cast<char*>(buf.data()), header.dim * sizeof(float));
        if (!in) throw std::runtime_error("load: truncated vector data");
        store.add_vector(buf.data());
    }
    return store;
}

SealedSegment load_segment_mmap(const std::string& dir) {
    MmapStore store((dir + "/vectors.bin").c_str());
    HnswGraph graph = load_hnsw(dir + "/hnsw.bin");
    IdMapping ids = load_ids(dir + "/ids.bin");
    return SealedSegment::from_mmap(std::move(store), std::move(graph), std::move(ids));
}

SealedSegment load_segment_memory(const std::string& dir) {
    InMemoryStore store = load_vectors_memory(dir + "/vectors.bin");
    HnswGraph graph = load_hnsw(dir + "/hnsw.bin");
    IdMapping ids = load_ids(dir + "/ids.bin");
    return SealedSegment::from_memory(std::move(store), std::move(graph), std::move(ids));
}

}  // namespace vextor
