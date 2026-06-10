#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/types.h"

namespace vextor {

class IdMapping {
   public:
    Offset insert(VectorId user_id);
    VectorId get_user_id(Offset offset) const;
    std::optional<Offset> get_offset(VectorId user_id) const;
    std::size_t size() const;

    // Exposes offset-ordered IDs for the persistence layer's binary ids.bin format.
    const std::vector<VectorId>& offset_to_id() const { return offset_to_id_; }

   private:
    std::vector<VectorId> offset_to_id_;
    std::unordered_map<VectorId, Offset> id_to_offset_;
};

}  // namespace vextor