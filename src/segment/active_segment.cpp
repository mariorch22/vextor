#include "segment/active_segment.h"

#include <stdexcept>

namespace vextor {

ActiveSegment::ActiveSegment(Dim dim, std::size_t capacity, HnswBuildParams params)
    : store_(dim), index_(store_, params), capacity_(capacity) {}

void ActiveSegment::insert(VectorId user_id, const float* data) {
    if (is_full()) {
        throw std::runtime_error("ActiveSegment: segment is full");
    }
    if (id_mapping_.get_offset(user_id).has_value()) {
        throw std::invalid_argument("ActiveSegment: duplicate user_id");
    }

    store_.add_vector(data);
    try {
        index_.insert();
    } catch (...) {
        store_.rollback_last();
        throw;
    }
    id_mapping_.insert(user_id);
}

std::vector<QueryResult> ActiveSegment::search(const float* query, std::size_t k,
                                               HnswSearchParams params) const {
    auto internal = index_.search(query, k, params);
    std::vector<QueryResult> results;
    results.reserve(internal.size());
    for (const auto& r : internal) {
        results.push_back({.user_id = id_mapping_.get_user_id(r.offset), .distance = r.distance});
    }
    return results;
}

std::size_t ActiveSegment::size() const {
    return store_.size();
}

std::size_t ActiveSegment::capacity() const {
    return capacity_;
}

bool ActiveSegment::is_full() const {
    return store_.size() >= capacity_;
}

Dim ActiveSegment::dimensions() const {
    return store_.dimensions();
}

}  // namespace vextor
