#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <set>
#include <vector>

#include "persistence/loader.h"
#include "persistence/serializer.h"
#include "segment/active_segment.h"

namespace {

std::string temp_dir() {
    auto path = std::filesystem::temp_directory_path() / "vextor_persist_test";
    std::filesystem::create_directories(path);
    return path.string();
}

void cleanup(const std::string& dir) {
    std::filesystem::remove_all(dir);
}

}  // namespace

TEST(Persistence, RoundTripMemory) {
    vextor::ActiveSegment seg(32, 100);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<vextor::VectorId> inserted_ids;
    for (int i = 0; i < 50; i++) {
        std::vector<float> v(32);
        for (auto& x : v) x = dist(rng);
        vextor::VectorId id = 1000 + i;
        seg.insert(id, v.data());
        inserted_ids.push_back(id);
    }

    auto dir = temp_dir();
    vextor::serialize_segment(seg, dir + "/seg0");

    auto sealed = vextor::load_segment_memory(dir + "/seg0");
    EXPECT_EQ(sealed.size(), 50);
    EXPECT_EQ(sealed.dimensions(), 32);

    // Search and verify we get valid user IDs back.
    std::vector<float> query(32);
    for (auto& x : query) x = dist(rng);

    auto results = sealed.search(query.data(), 10, 128);
    ASSERT_EQ(results.size(), 10);

    std::set<vextor::VectorId> id_set(inserted_ids.begin(), inserted_ids.end());
    for (const auto& r : results) {
        EXPECT_TRUE(id_set.count(r.user_id))
            << "Unexpected user_id " << r.user_id << " in search results";
    }

    // Results should be sorted by distance.
    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }

    cleanup(dir);
}

TEST(Persistence, RoundTripMmap) {
    vextor::ActiveSegment seg(16, 100);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < 30; i++) {
        std::vector<float> v(16);
        for (auto& x : v) x = dist(rng);
        seg.insert(static_cast<vextor::VectorId>(i), v.data());
    }

    auto dir = temp_dir();
    vextor::serialize_segment(seg, dir + "/seg_mmap");

    auto sealed = vextor::load_segment_mmap(dir + "/seg_mmap");
    EXPECT_EQ(sealed.size(), 30);
    EXPECT_EQ(sealed.dimensions(), 16);

    std::vector<float> query(16);
    for (auto& x : query) x = dist(rng);

    auto results = sealed.search(query.data(), 5, 128);
    ASSERT_EQ(results.size(), 5);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }

    cleanup(dir);
}

TEST(Persistence, SearchResultsMatchBeforeAndAfterSeal) {
    const int n = 200;
    const vextor::Dim dim = 64;

    vextor::ActiveSegment seg(dim, n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        seg.insert(static_cast<vextor::VectorId>(i), v.data());
    }

    // Search before seal.
    std::vector<float> query(dim);
    for (auto& x : query) x = dist(rng);
    auto before = seg.search(query.data(), 10, 128);

    // Seal and reload.
    auto dir = temp_dir();
    vextor::serialize_segment(seg, dir + "/seg_match");
    auto sealed = vextor::load_segment_memory(dir + "/seg_match");

    auto after = sealed.search(query.data(), 10, 128);

    // Results should be identical.
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); i++) {
        EXPECT_EQ(before[i].user_id, after[i].user_id) << "Mismatch at position " << i;
        EXPECT_FLOAT_EQ(before[i].distance, after[i].distance);
    }

    cleanup(dir);
}

TEST(Persistence, LoadInvalidDirectoryThrows) {
    EXPECT_THROW(([] {
                     [[maybe_unused]] const auto _ =
                         vextor::load_segment_memory("/nonexistent/path");
                 }()),
                 std::runtime_error);
    EXPECT_THROW(
        ([] { [[maybe_unused]] const auto _ = vextor::load_segment_mmap("/nonexistent/path"); }()),
        std::runtime_error);
}
