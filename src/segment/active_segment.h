#pragma once

#include <cstddef>
#include <vector>

#include "core/query_result.h"
#include "core/types.h"
#include "index/hnsw_index.h"
#include "segment/id_mapping.h"
#include "store/in_memory_store.h"

namespace vexdb {

class ActiveSegment {
   public:
    explicit ActiveSegment(Dim dim, std::size_t capacity, int m = 16, int ef_construction = 200);

    void insert(VectorId user_id, const float* data);
    [[nodiscard]] std::vector<QueryResult> search(const float* query, std::size_t k,
                                                  int ef_search = 128) const;

    std::size_t size() const;
    std::size_t capacity() const;
    bool is_full() const;
    Dim dimensions() const;

    const InMemoryStore& store() const { return store_; }
    const HnswIndex<InMemoryStore>& index() const { return index_; }
    const IdMapping& id_mapping() const { return id_mapping_; }

   private:
    friend class SegmentManager;

    InMemoryStore take_store() { return std::move(store_); }
    HnswGraph take_graph() { return index_.take_graph(); }
    IdMapping take_id_mapping() { return std::move(id_mapping_); }

    InMemoryStore store_;
    HnswIndex<InMemoryStore> index_;
    IdMapping id_mapping_;
    std::size_t capacity_;
};

}  // namespace vexdb