#include "segment/segment_manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>

#include "persistence/loader.h"
#include "persistence/serializer.h"

namespace vextor {

SegmentManager::SegmentManager(Dim dim, std::size_t segment_capacity, const std::string& db_path,
                               int m, int ef_construction)
    : dim_(dim),
      segment_capacity_(segment_capacity),
      m_(m),
      ef_construction_(ef_construction),
      db_path_(db_path),
      active_(std::make_unique<ActiveSegment>(dim, segment_capacity, m, ef_construction)) {
    if (segment_capacity == 0) {
        throw std::invalid_argument("SegmentManager: segment_capacity must be > 0");
    }
}

void SegmentManager::insert(VectorId user_id, std::span<const float> data) {
    if (data.size() != dim_) {
        throw std::invalid_argument("SegmentManager: expected " + std::to_string(dim_) +
                                    " dimensions, got " + std::to_string(data.size()));
    }
    if (all_ids_.count(user_id)) {
        throw std::invalid_argument("SegmentManager: duplicate user_id");
    }
    if (active_->is_full()) {
        seal_active();
    }
    active_->insert(user_id, data.data());
    all_ids_.insert(user_id);
}

std::vector<QueryResult> SegmentManager::search(std::span<const float> query, std::size_t k,
                                                int ef_search) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("SegmentManager: expected " + std::to_string(dim_) +
                                    " dimensions, got " + std::to_string(query.size()));
    }
    std::vector<QueryResult> all;

    if (active_->size() > 0) {
        auto r = active_->search(query.data(), k, ef_search);
        all.insert(all.end(), r.begin(), r.end());
    }

    for (const auto& seg : sealed_) {
        auto r = seg.search(query.data(), k, ef_search);
        all.insert(all.end(), r.begin(), r.end());
    }

    std::sort(all.begin(), all.end(),
              [](const QueryResult& a, const QueryResult& b) { return a.distance < b.distance; });
    if (all.size() > k) all.resize(k);
    return all;
}

// B1: seal_active — split take_* into local variables to enforce evaluation order
void SegmentManager::seal_active() {
    if (!db_path_.empty()) {
        std::string seg_dir = db_path_ + "/segment_" + std::to_string(sealed_.size());
        serialize_segment(*active_, seg_dir);
        sealed_.push_back(load_segment_mmap(seg_dir));
    } else {
        auto store = active_->take_store();
        auto graph = active_->take_graph();
        auto ids = active_->take_id_mapping();
        sealed_.push_back(
            SealedSegment::from_memory(std::move(store), std::move(graph), std::move(ids)));
    }
    active_ = std::make_unique<ActiveSegment>(dim_, segment_capacity_, m_, ef_construction_);
}

void SegmentManager::save() {
    if (db_path_.empty()) {
        throw std::runtime_error("save: no db_path configured");
    }

    std::filesystem::create_directories(db_path_);

    // Serialize the active segment as the next segment directory.
    std::size_t total = sealed_.size();
    if (active_->size() > 0) {
        std::string seg_dir = db_path_ + "/segment_" + std::to_string(total);
        serialize_segment(*active_, seg_dir);
        total++;
    }

    // Write segments.json last (crash atomicity).
    write_segments_json(total);
}

void SegmentManager::write_segments_json(std::size_t total) const {
    std::ofstream out(db_path_ + "/segments.json");
    if (!out) throw std::runtime_error("save: cannot write segments.json");

    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"dim\": " << dim_ << ",\n";
    out << "  \"segment_capacity\": " << segment_capacity_ << ",\n";
    out << "  \"m\": " << m_ << ",\n";
    out << "  \"ef_construction\": " << ef_construction_ << ",\n";
    out << "  \"segment_count\": " << total << "\n";
    out << "}\n";
}

SegmentManager SegmentManager::load(const std::string& path) {
    std::ifstream in(path + "/segments.json");
    if (!in) throw std::runtime_error("load: cannot open segments.json");

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    auto extract_uint = [&](const std::string& key) -> unsigned long long {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) throw std::runtime_error("load: missing key " + key);
        pos = content.find(':', pos);
        try {
            return std::stoull(content.substr(pos + 1));
        } catch (const std::exception& e) {
            throw std::runtime_error("load: invalid value for key '" + key +
                                     "' in segments.json: " + e.what());
        }
    };

    Dim dim = static_cast<Dim>(extract_uint("dim"));
    auto capacity = static_cast<std::size_t>(extract_uint("segment_capacity"));
    auto m = static_cast<int>(extract_uint("m"));
    auto ef = static_cast<int>(extract_uint("ef_construction"));

    const unsigned long long seg_count_raw = extract_uint("segment_count");
    if (seg_count_raw >
        static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())) {
        throw std::runtime_error("load: segment_count out of range");
    }
    const std::size_t seg_count = static_cast<std::size_t>(seg_count_raw);

    SegmentManager mgr(dim, capacity, path, m, ef);

    for (std::size_t i = 0; i < seg_count; i++) {
        mgr.sealed_.push_back(load_segment_mmap(path + "/segment_" + std::to_string(i)));
        for (VectorId id : mgr.sealed_.back().id_mapping().offset_to_id()) {
            mgr.all_ids_.insert(id);
        }
    }

    return mgr;
}

std::size_t SegmentManager::total_vectors() const {
    std::size_t total = active_->size();
    for (const auto& seg : sealed_) total += seg.size();
    return total;
}

std::size_t SegmentManager::segment_count() const {
    return sealed_.size() + 1;
}

Dim SegmentManager::dimensions() const {
    return dim_;
}

}  // namespace vextor