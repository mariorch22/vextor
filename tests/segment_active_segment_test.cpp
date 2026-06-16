#include <gtest/gtest.h>

#include <random>
#include <set>
#include <vector>

#include "segment/active_segment.h"

TEST(ActiveSegment, InsertAndSearchRoundTrip) {
    vextor::ActiveSegment seg(4, 100);

    std::vector<float> v0 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 0.0f, 1.0f, 0.0f};

    seg.insert(100, v0.data());
    seg.insert(200, v1.data());
    seg.insert(300, v2.data());

    EXPECT_EQ(seg.size(), 3);

    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    auto results = seg.search(query.data(), 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].user_id, 100);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(ActiveSegment, SearchReturnsUserIds) {
    vextor::ActiveSegment seg(2, 100);

    seg.insert(42, std::vector<float>{0.0f, 0.0f}.data());
    seg.insert(99, std::vector<float>{1.0f, 0.0f}.data());
    seg.insert(77, std::vector<float>{3.0f, 4.0f}.data());

    std::vector<float> query = {0.0f, 0.0f};
    auto results = seg.search(query.data(), 3);

    ASSERT_EQ(results.size(), 3);

    std::set<vextor::VectorId> ids;
    for (const auto& r : results) ids.insert(r.user_id);

    EXPECT_TRUE(ids.count(42));
    EXPECT_TRUE(ids.count(99));
    EXPECT_TRUE(ids.count(77));
}

TEST(ActiveSegment, LargeUserIdPreserved) {
    vextor::ActiveSegment seg(2, 100);

    vextor::VectorId big_id = 1ULL << 40;
    seg.insert(big_id, std::vector<float>{1.0f, 0.0f}.data());

    auto results = seg.search(std::vector<float>{1.0f, 0.0f}.data(), 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].user_id, big_id);
}

TEST(ActiveSegment, CapacityAndIsFull) {
    vextor::ActiveSegment seg(2, 3);

    EXPECT_EQ(seg.capacity(), 3);
    EXPECT_FALSE(seg.is_full());

    seg.insert(1, std::vector<float>{0.0f, 0.0f}.data());
    seg.insert(2, std::vector<float>{1.0f, 0.0f}.data());
    seg.insert(3, std::vector<float>{2.0f, 0.0f}.data());

    EXPECT_TRUE(seg.is_full());
    EXPECT_EQ(seg.size(), 3);
}

TEST(ActiveSegment, InsertWhenFullThrows) {
    vextor::ActiveSegment seg(2, 1);
    seg.insert(1, std::vector<float>{0.0f, 0.0f}.data());

    EXPECT_THROW(seg.insert(2, std::vector<float>{1.0f, 0.0f}.data()), std::runtime_error);
}

TEST(ActiveSegment, DuplicateUserIdThrows) {
    vextor::ActiveSegment seg(2, 100);
    seg.insert(42, std::vector<float>{0.0f, 0.0f}.data());

    EXPECT_THROW(seg.insert(42, std::vector<float>{1.0f, 0.0f}.data()), std::invalid_argument);

    // Segment state should be unchanged after failed insert.
    EXPECT_EQ(seg.size(), 1);
    auto results = seg.search(std::vector<float>{0.0f, 0.0f}.data(), 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].user_id, 42);
}

TEST(ActiveSegment, RecallWithRandomVectors) {
    const int n = 1000;
    const vextor::Dim dim = 32;

    vextor::ActiveSegment seg(dim, n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        seg.insert(static_cast<vextor::VectorId>(i), v.data());
        vecs.push_back(std::move(v));
    }

    std::vector<float> query(dim);
    for (auto& x : query) x = dist(rng);

    auto results = seg.search(query.data(), 10, vextor::HnswSearchParams{.ef_search = 128});
    ASSERT_EQ(results.size(), 10);

    // Results should be sorted by distance.
    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}
