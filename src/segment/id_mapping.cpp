#include "segment/id_mapping.h"

#include <cassert>
#include <stdexcept>

namespace vextor {

Offset IdMapping::insert(VectorId user_id) {
    if (id_to_offset_.count(user_id)) {
        throw std::invalid_argument("IdMapping: duplicate user_id");
    }
    Offset offset = static_cast<Offset>(offset_to_id_.size());
    offset_to_id_.push_back(user_id);
    id_to_offset_[user_id] = offset;
    return offset;
}

VectorId IdMapping::get_user_id(Offset offset) const {
    assert(offset < offset_to_id_.size());
    return offset_to_id_[offset];
}

std::optional<Offset> IdMapping::get_offset(VectorId user_id) const {
    auto it = id_to_offset_.find(user_id);
    if (it == id_to_offset_.end()) return std::nullopt;
    return it->second;
}

std::size_t IdMapping::size() const {
    return offset_to_id_.size();
}

}  // namespace vextor