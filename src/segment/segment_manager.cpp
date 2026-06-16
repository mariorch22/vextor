#include "segment/segment_manager.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>

#include "persistence/loader.h"
#include "persistence/serializer.h"

namespace vextor {

namespace {

constexpr unsigned long long k_segments_json_format_version = 1;

unsigned long long extract_json_uint(const std::string& content, const std::string& key) {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) throw std::runtime_error("load: missing key " + key);

    pos = content.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("load: missing ':' for key " + key);
    pos++;

    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;

    if (pos == content.size()) throw std::runtime_error("load: missing value for key " + key);
    if (content[pos] == '-') throw std::runtime_error("load: negative value for key " + key);
    if (!std::isdigit(static_cast<unsigned char>(content[pos]))) {
        throw std::runtime_error("load: invalid value for key " + key);
    }

    unsigned long long value = 0;
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
        const unsigned digit = static_cast<unsigned>(content[pos] - '0');
        if (value > ((std::numeric_limits<unsigned long long>::max)() - digit) / 10) {
            throw std::runtime_error("load: value out of range for key " + key);
        }
        value = value * 10 + digit;
        pos++;
    }

    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
    if (pos < content.size() && content[pos] != ',' && content[pos] != '}') {
        throw std::runtime_error("load: invalid trailing characters for key " + key);
    }

    return value;
}

}  // namespace

SegmentManager::SegmentManager(Dim dim, std::size_t segment_capacity, const std::string& db_path,
                               HnswBuildParams params)
    : dim_(dim),
      segment_capacity_(segment_capacity),
      build_params_(params),
      db_path_(db_path),
      active_(std::make_unique<ActiveSegment>(dim, segment_capacity, params)) {
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
                                                HnswSearchParams params) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("SegmentManager: expected " + std::to_string(dim_) +
                                    " dimensions, got " + std::to_string(query.size()));
    }
    std::vector<QueryResult> all;

    if (active_->size() > 0) {
        auto r = active_->search(query.data(), k, params);
        all.insert(all.end(), r.begin(), r.end());
    }

    for (const auto& seg : sealed_) {
        auto r = seg.search(query.data(), k, params);
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
        auto sealed = load_segment_mmap(seg_dir);
        // Register the sealed segment immediately: without this, a crash before the
        // next save() would leave the segment directory on disk but unreferenced,
        // and load() would silently drop it.
        write_segments_json(sealed_.size() + 1);
        sealed_.push_back(std::move(sealed));
    } else {
        auto store = active_->take_store();
        auto graph = active_->take_graph();
        auto ids = active_->take_id_mapping();
        sealed_.push_back(
            SealedSegment::from_memory(std::move(store), std::move(graph), std::move(ids)));
    }
    active_ = std::make_unique<ActiveSegment>(dim_, segment_capacity_, build_params_);
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
    // Write to a temp file and rename over segments.json so a crash mid-write
    // cannot corrupt the existing registry.
    const std::string final_path = db_path_ + "/segments.json";
    const std::string tmp_path = final_path + ".tmp";

    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out) throw std::runtime_error("write_segments_json: cannot write " + tmp_path);

        out << "{\n";
        out << "  \"format_version\": " << k_segments_json_format_version << ",\n";
        out << "  \"dim\": " << dim_ << ",\n";
        out << "  \"segment_capacity\": " << segment_capacity_ << ",\n";
        out << "  \"m\": " << build_params_.m << ",\n";
        out << "  \"ef_construction\": " << build_params_.ef_construction << ",\n";
        out << "  \"segment_count\": " << total << "\n";
        out << "}\n";

        out.flush();
        if (!out) throw std::runtime_error("write_segments_json: write failed for " + tmp_path);
    }

    std::filesystem::rename(tmp_path, final_path);
}

SegmentManager SegmentManager::load(const std::string& path) {
    std::ifstream in(path + "/segments.json");
    if (!in) throw std::runtime_error("load: cannot open segments.json");

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const auto format_version = extract_json_uint(content, "format_version");
    if (format_version != k_segments_json_format_version) {
        throw std::runtime_error("load: unsupported segments.json format_version");
    }

    auto extract_dim = [&](const std::string& key) -> Dim {
        const auto value = extract_json_uint(content, key);
        if (value > static_cast<unsigned long long>((std::numeric_limits<Dim>::max)())) {
            throw std::runtime_error("load: value out of range for key " + key);
        }
        return static_cast<Dim>(value);
    };

    auto extract_size = [&](const std::string& key) -> std::size_t {
        const auto value = extract_json_uint(content, key);
        if (value > static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())) {
            throw std::runtime_error("load: value out of range for key " + key);
        }
        return static_cast<std::size_t>(value);
    };

    auto extract_int = [&](const std::string& key) -> int {
        const auto value = extract_json_uint(content, key);
        if (value > static_cast<unsigned long long>((std::numeric_limits<int>::max)())) {
            throw std::runtime_error("load: value out of range for key " + key);
        }
        return static_cast<int>(value);
    };

    Dim dim = extract_dim("dim");
    auto capacity = extract_size("segment_capacity");
    HnswBuildParams params;
    params.m = extract_int("m");
    params.ef_construction = extract_int("ef_construction");

    const unsigned long long seg_count_raw = extract_json_uint(content, "segment_count");
    if (seg_count_raw >
        static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())) {
        throw std::runtime_error("load: segment_count out of range");
    }
    const std::size_t seg_count = static_cast<std::size_t>(seg_count_raw);

    SegmentManager mgr(dim, capacity, path, params);

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
