#include "store/in_memory_store.h"

#include <cassert>

namespace vextor {

InMemoryStore::InMemoryStore(Dim dim) : dim_(dim) {
    assert(dim > 0);
}

Offset InMemoryStore::add_vector(const float* data) {
    Offset offset = static_cast<Offset>(data_.size() / dim_);
    data_.insert(data_.end(), data, data + dim_);
    return offset;
}

void InMemoryStore::rollback_last() {
    assert(data_.size() >= dim_);
    data_.resize(data_.size() - dim_);
}

const float* InMemoryStore::get_vector(Offset offset) const {
    assert(offset < size());
    return data_.data() + static_cast<std::size_t>(offset) * dim_;
}

std::size_t InMemoryStore::size() const {
    return data_.size() / dim_;
}

Dim InMemoryStore::dimensions() const {
    return dim_;
}

}  // namespace vextor