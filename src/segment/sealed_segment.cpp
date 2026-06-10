#include "segment/sealed_segment.h"

#include <stdexcept>
#include <utility>

namespace vextor {

static void validate_components(std::size_t store_size, std::size_t graph_size,
                                std::size_t mapping_size) {
    if (store_size != graph_size || store_size != mapping_size) {
        throw std::invalid_argument("SealedSegment: store/graph/id_mapping size mismatch");
    }
}

SealedSegment SealedSegment::from_memory(InMemoryStore store, HnswGraph graph,
                                         IdMapping id_mapping) {
    validate_components(store.size(), graph.size(), id_mapping.size());
    SealedSegment seg;
    seg.id_mapping_ = std::move(id_mapping);
    seg.backend_ =
        std::make_unique<TypedBackend<InMemoryStore>>(std::move(store), std::move(graph));
    return seg;
}

SealedSegment SealedSegment::from_mmap(MmapStore store, HnswGraph graph, IdMapping id_mapping) {
    validate_components(store.size(), graph.size(), id_mapping.size());
    SealedSegment seg;
    seg.id_mapping_ = std::move(id_mapping);
    seg.backend_ = std::make_unique<TypedBackend<MmapStore>>(std::move(store), std::move(graph));
    return seg;
}

std::vector<QueryResult> SealedSegment::search(const float* query, std::size_t k,
                                               int ef_search) const {
    auto internal = backend_->search(query, k, ef_search);
    std::vector<QueryResult> results;
    results.reserve(internal.size());
    for (const auto& r : internal) {
        results.push_back({.user_id = id_mapping_.get_user_id(r.offset), .distance = r.distance});
    }
    return results;
}

std::size_t SealedSegment::size() const {
    return backend_->size();
}

Dim SealedSegment::dimensions() const {
    return backend_->dimensions();
}

}  // namespace vextor