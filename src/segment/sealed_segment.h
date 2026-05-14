#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/query_result.h"
#include "core/search_result.h"
#include "core/types.h"
#include "index/hnsw_index.h"
#include "segment/id_mapping.h"
#include "store/in_memory_store.h"
#include "store/mmap_store.h"

namespace vexdb {

class SealedSegment {
   public:
    static SealedSegment from_memory(InMemoryStore store, HnswGraph graph, IdMapping id_mapping);
    static SealedSegment from_mmap(MmapStore store, HnswGraph graph, IdMapping id_mapping);

    [[nodiscard]] std::vector<QueryResult> search(const float* query, std::size_t k,
                                                  int ef_search = 128) const;

    std::size_t size() const;
    Dim dimensions() const;
    const IdMapping& id_mapping() const { return id_mapping_; }

   private:
    SealedSegment() = default;

    struct Backend {
        virtual ~Backend() = default;
        virtual std::vector<SearchResult> search(const float* query, std::size_t k,
                                                 int ef_search) const = 0;
        virtual std::size_t size() const = 0;
        virtual Dim dimensions() const = 0;
    };

    template <typename Store>
    struct TypedBackend : Backend {
        Store store;
        HnswIndex<Store> index;

        TypedBackend(Store s, HnswGraph graph)
            : store(std::move(s)), index(store, std::move(graph)) {}

        std::vector<SearchResult> search(const float* query, std::size_t k,
                                         int ef_search) const override {
            return index.search(query, k, ef_search);
        }
        std::size_t size() const override { return store.size(); }
        Dim dimensions() const override { return store.dimensions(); }
    };

    std::unique_ptr<Backend> backend_;
    IdMapping id_mapping_;
};

}  // namespace vexdb