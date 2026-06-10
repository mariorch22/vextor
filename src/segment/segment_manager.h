#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/types.h"
#include "segment/active_segment.h"
#include "segment/sealed_segment.h"

namespace vextor {

class SegmentManager {
   public:
    // Create a new database at the given path (or in-memory if path is empty).
    explicit SegmentManager(Dim dim, std::size_t segment_capacity, const std::string& db_path = "",
                            int m = 16, int ef_construction = 200);

    void insert(VectorId user_id, std::span<const float> data);
    [[nodiscard]] std::vector<QueryResult> search(std::span<const float> query, std::size_t k,
                                                  int ef_search = 128) const;

    // Flush: persist the active segment and write segments.json.
    void save();

    // Load an existing database from disk.
    [[nodiscard]] static SegmentManager load(const std::string& path);

    std::size_t total_vectors() const;
    std::size_t segment_count() const;
    Dim dimensions() const;

   private:
    Dim dim_;
    std::size_t segment_capacity_;
    int m_;
    int ef_construction_;
    std::string db_path_;
    std::unique_ptr<ActiveSegment> active_;
    std::vector<SealedSegment> sealed_;
    std::unordered_set<VectorId> all_ids_;

    void seal_active();
    void write_segments_json(std::size_t total) const;
};

}  // namespace vextor