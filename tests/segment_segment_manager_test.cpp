#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "segment/segment_manager.h"

namespace {

std::string make_temp_dir(const std::string& name) {
    auto path = std::filesystem::temp_directory_path() / ("vextor_" + name);
    std::filesystem::remove_all(path);
    return path.string();
}

void write_registry(const std::string& dir, const std::string& content) {
    std::filesystem::create_directories(dir);
    std::ofstream out(std::filesystem::path(dir) / "segments.json");
    out << content;
}

}  // namespace

TEST(SegmentManager, InsertAndSearch) {
    vextor::SegmentManager mgr(4, 100);

    mgr.insert(1, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    mgr.insert(2, std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    mgr.insert(3, std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f});

    EXPECT_EQ(mgr.total_vectors(), 3);

    auto results = mgr.search(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].user_id, 1);
}

TEST(SegmentManager, SealOnCapacity) {
    vextor::SegmentManager mgr(2, 3);

    mgr.insert(1, std::vector<float>{0.0f, 0.0f});
    mgr.insert(2, std::vector<float>{1.0f, 0.0f});
    mgr.insert(3, std::vector<float>{2.0f, 0.0f});
    EXPECT_EQ(mgr.segment_count(), 1);

    mgr.insert(4, std::vector<float>{3.0f, 0.0f});
    EXPECT_EQ(mgr.segment_count(), 2);
    EXPECT_EQ(mgr.total_vectors(), 4);
}

TEST(SegmentManager, SearchAcrossSegments) {
    vextor::SegmentManager mgr(2, 3);

    mgr.insert(10, std::vector<float>{0.0f, 0.0f});
    mgr.insert(20, std::vector<float>{1.0f, 0.0f});
    mgr.insert(30, std::vector<float>{2.0f, 0.0f});
    // Triggers seal.
    mgr.insert(40, std::vector<float>{10.0f, 0.0f});
    mgr.insert(50, std::vector<float>{11.0f, 0.0f});

    EXPECT_EQ(mgr.segment_count(), 2);

    auto results = mgr.search(std::vector<float>{0.0f, 0.0f}, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].user_id, 10);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST(SegmentManager, SaveAndLoadRoundTrip) {
    auto dir = make_temp_dir("mgr_roundtrip");

    {
        vextor::SegmentManager mgr(8, 50, dir);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < 30; i++) {
            std::vector<float> v(8);
            for (auto& x : v) x = dist(rng);
            mgr.insert(static_cast<vextor::VectorId>(i), v);
        }

        mgr.save();
    }

    auto mgr = vextor::SegmentManager::load(dir);
    EXPECT_EQ(mgr.total_vectors(), 30);
    EXPECT_EQ(mgr.dimensions(), 8);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> query(8);
    for (auto& x : query) x = dist(rng);

    auto results = mgr.search(query, 5, vextor::HnswSearchParams{.ef_search = 128});
    ASSERT_EQ(results.size(), 5);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, SaveAndLoadWithSealedSegments) {
    auto dir = make_temp_dir("mgr_sealed_roundtrip");

    {
        vextor::SegmentManager mgr(4, 5, dir);

        for (int i = 0; i < 12; i++) {
            std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
            mgr.insert(static_cast<vextor::VectorId>(i), v);
        }

        // Should have sealed twice: 5 + 5 sealed, 2 active.
        EXPECT_EQ(mgr.segment_count(), 3);
        mgr.save();
    }

    auto mgr = vextor::SegmentManager::load(dir);
    // 2 sealed (from seal_active) + active serialized = 3 segments loaded as sealed.
    EXPECT_EQ(mgr.total_vectors(), 12);

    auto results = mgr.search(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].user_id, 0);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, SealedSegmentsSurviveCrashWithoutSave) {
    auto dir = make_temp_dir("mgr_crash_no_save");

    {
        vextor::SegmentManager mgr(4, 5, dir);

        // 12 inserts with capacity 5: two seals, 2 vectors left in the active segment.
        for (int i = 0; i < 12; i++) {
            std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
            mgr.insert(static_cast<vextor::VectorId>(i), v);
        }
        EXPECT_EQ(mgr.segment_count(), 3);
        // Simulated crash: no save().
    }

    // Only the active segment is lost; both sealed segments must be recovered.
    auto mgr = vextor::SegmentManager::load(dir);
    EXPECT_EQ(mgr.total_vectors(), 10);

    auto results = mgr.search(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].user_id, 0);

    // IDs from sealed segments are known after load.
    EXPECT_THROW(mgr.insert(9, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}), std::invalid_argument);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, RegistryWriteIsAtomic) {
    auto dir = make_temp_dir("mgr_registry_atomic");
    const auto registry = std::filesystem::path(dir) / "segments.json";
    const auto tmp = std::filesystem::path(dir) / "segments.json.tmp";

    {
        vextor::SegmentManager mgr(4, 5, dir);
        for (int i = 0; i < 7; i++) {
            std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
            mgr.insert(static_cast<vextor::VectorId>(i), v);
        }
        mgr.save();
    }

    // The temp file must never survive a completed registry write.
    EXPECT_TRUE(std::filesystem::exists(registry));
    EXPECT_FALSE(std::filesystem::exists(tmp));

    // Simulate a crash mid-write: a torn temp file is left behind, the registry
    // itself is untouched. load() must only ever read segments.json.
    {
        std::ofstream torn(tmp);
        torn << "{ \"version\": 1, \"dim\": 4, \"segment_ca";
    }

    auto mgr = vextor::SegmentManager::load(dir);
    EXPECT_EQ(mgr.total_vectors(), 7);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, RegistryIncludesFormatVersion) {
    auto dir = make_temp_dir("mgr_registry_format_version");
    const auto registry = std::filesystem::path(dir) / "segments.json";

    {
        vextor::SegmentManager mgr(4, 5, dir);
        mgr.insert(1, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
        mgr.save();
    }

    std::ifstream in(registry);
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("\"format_version\": 1"), std::string::npos);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, LoadRejectsUnknownRegistryFormatVersion) {
    auto dir = make_temp_dir("mgr_registry_unknown_version");
    write_registry(dir, R"({
  "format_version": 2,
  "dim": 4,
  "segment_capacity": 5,
  "m": 16,
  "ef_construction": 200,
  "segment_count": 0
})");

    EXPECT_THROW((void)vextor::SegmentManager::load(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, LoadRejectsMissingRegistryKey) {
    auto dir = make_temp_dir("mgr_registry_missing_key");
    write_registry(dir, R"({
  "format_version": 1,
  "dim": 4,
  "segment_capacity": 5,
  "m": 16,
  "ef_construction": 200
})");

    EXPECT_THROW((void)vextor::SegmentManager::load(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, LoadRejectsNegativeRegistryValue) {
    auto dir = make_temp_dir("mgr_registry_negative_value");
    write_registry(dir, R"({
  "format_version": 1,
  "dim": 4,
  "segment_capacity": -5,
  "m": 16,
  "ef_construction": 200,
  "segment_count": 0
})");

    EXPECT_THROW((void)vextor::SegmentManager::load(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, LoadRejectsOverflowRegistryValue) {
    auto dir = make_temp_dir("mgr_registry_overflow_value");
    write_registry(dir, R"({
  "format_version": 1,
  "dim": 4,
  "segment_capacity": 5,
  "m": 18446744073709551616,
  "ef_construction": 200,
  "segment_count": 0
})");

    EXPECT_THROW((void)vextor::SegmentManager::load(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, DuplicateIdAcrossSegmentsThrows) {
    vextor::SegmentManager mgr(2, 3);

    mgr.insert(42, std::vector<float>{0.0f, 0.0f});
    mgr.insert(43, std::vector<float>{1.0f, 0.0f});
    mgr.insert(44, std::vector<float>{2.0f, 0.0f});
    // Triggers seal.
    mgr.insert(45, std::vector<float>{3.0f, 0.0f});

    EXPECT_EQ(mgr.segment_count(), 2);

    // ID 42 is in a sealed segment — inserting again should throw.
    EXPECT_THROW(mgr.insert(42, std::vector<float>{4.0f, 0.0f}), std::invalid_argument);
}

TEST(SegmentManager, RecallAcrossSegments) {
    const int n = 300;
    const vextor::Dim dim = 32;

    vextor::SegmentManager mgr(dim, 100);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        mgr.insert(static_cast<vextor::VectorId>(i), v);
    }

    EXPECT_GE(mgr.segment_count(), 3);

    std::vector<float> query(dim);
    for (auto& x : query) x = dist(rng);

    auto results = mgr.search(query, 10, vextor::HnswSearchParams{.ef_search = 128});
    ASSERT_EQ(results.size(), 10);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST(SegmentManager, DuplicateIdAfterLoadThrows) {
    auto dir = make_temp_dir("mgr_dup_after_load");

    {
        vextor::SegmentManager mgr(2, 10, dir);
        mgr.insert(42, std::vector<float>{1.0f, 0.0f});
        mgr.save();
    }

    auto mgr = vextor::SegmentManager::load(dir);
    EXPECT_THROW(mgr.insert(42, std::vector<float>{2.0f, 0.0f}), std::invalid_argument);

    std::filesystem::remove_all(dir);
}

TEST(SegmentManager, InsertWrongDimensionThrows) {
    vextor::SegmentManager mgr(4, 100);

    EXPECT_THROW(mgr.insert(1, std::vector<float>{1.0f, 0.0f}), std::invalid_argument);
    EXPECT_THROW(mgr.insert(1, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
                 std::invalid_argument);
    EXPECT_THROW(mgr.insert(1, std::span<const float>{}), std::invalid_argument);
}

TEST(SegmentManager, SearchWrongDimensionThrows) {
    vextor::SegmentManager mgr(4, 100);
    mgr.insert(1, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});

    EXPECT_THROW((void)mgr.search(std::vector<float>{1.0f, 0.0f}, 1), std::invalid_argument);
    EXPECT_THROW((void)mgr.search(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1),
                 std::invalid_argument);
}
