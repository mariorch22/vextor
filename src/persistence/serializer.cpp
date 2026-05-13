#include "persistence/serializer.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "persistence/format.h"

namespace vexdb {

static void write_vectors(const InMemoryStore& store, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serialize: cannot open " + path);

    Vex0Header header{};
    header.magic = kVex0Magic;
    header.version = kVex0Version;
    header.dim = store.dimensions();
    header.count = static_cast<uint32_t>(store.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (std::size_t i = 0; i < store.size(); i++) {
        out.write(reinterpret_cast<const char*>(store.get_vector(static_cast<Offset>(i))),
                  store.dimensions() * sizeof(float));
    }
}

static void write_hnsw(const HnswGraph& graph, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serialize: cannot open " + path);

    HnswHeader header{};
    header.magic = kHnswMagic;
    header.version = kHnswVersion;
    header.m = static_cast<uint32_t>(graph.m);
    header.max_level = static_cast<uint32_t>(graph.max_level);
    header.entry_point = graph.entry_point;
    header.node_count = static_cast<uint32_t>(graph.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    int stride = graph.layer0_stride();
    for (std::size_t i = 0; i < graph.size(); i++) {
        // Write level.
        int32_t level = graph.levels[i];
        out.write(reinterpret_cast<const char*>(&level), sizeof(level));

        // Write layer 0 neighbor count + neighbors.
        int32_t l0_count = graph.layer0_count[i];
        out.write(reinterpret_cast<const char*>(&l0_count), sizeof(l0_count));
        out.write(reinterpret_cast<const char*>(&graph.layer0[i * stride]),
                  l0_count * sizeof(Offset));

        // Write upper layer neighbors.
        for (int layer = 1; layer <= level; layer++) {
            int32_t count = static_cast<int32_t>(graph.upper[i][layer - 1].size());
            out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            out.write(reinterpret_cast<const char*>(graph.upper[i][layer - 1].data()),
                      count * sizeof(Offset));
        }
    }
}

static void write_ids(const IdMapping& mapping, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serialize: cannot open " + path);

    IdsHeader header{};
    header.magic = kIdsMagic;
    header.version = kIdsVersion;
    header.count = static_cast<uint32_t>(mapping.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const auto& ids = mapping.offset_to_id();
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(VectorId));
}

void serialize_segment(const ActiveSegment& segment, const std::string& dir) {
    std::filesystem::create_directories(dir);
    write_vectors(segment.store(), dir + "/vectors.bin");
    write_hnsw(segment.index().graph(), dir + "/hnsw.bin");
    write_ids(segment.id_mapping(), dir + "/ids.bin");
}

}  // namespace vexdb