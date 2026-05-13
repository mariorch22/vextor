#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/distance.h"
#include "index/flat_index.h"
#include "store/in_memory_store.h"

TEST(FlatIndex, EmptyStoreReturnsEmpty) {
    vexdb::InMemoryStore store(3);
    vexdb::FlatIndex index(store);

    auto results = index.search(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), 5);
    EXPECT_TRUE(results.empty());
}

TEST(FlatIndex, SingleVector) {
    vexdb::InMemoryStore store(3);
    std::vector<float> vec = {1.0f, 0.0f, 0.0f};
    store.add_vector(vec.data());

    vexdb::FlatIndex index(store);
    std::vector<float> query = {0.0f, 0.0f, 0.0f};
    auto results = index.search(query.data(), 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].offset, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 1.0f);
}

TEST(FlatIndex, HandComputedDistances) {
    vexdb::InMemoryStore store(2);
    // v0 at (0,0), v1 at (3,4), v2 at (1,0)
    store.add_vector(std::vector<float>{0.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{3.0f, 4.0f}.data());
    store.add_vector(std::vector<float>{1.0f, 0.0f}.data());

    vexdb::FlatIndex index(store);
    std::vector<float> query = {0.0f, 0.0f};

    auto results = index.search(query.data(), 3);

    ASSERT_EQ(results.size(), 3);
    // Closest first: v0 (dist=0), v2 (dist=1), v1 (dist=25)
    EXPECT_EQ(results[0].offset, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
    EXPECT_EQ(results[1].offset, 2);
    EXPECT_FLOAT_EQ(results[1].distance, 1.0f);
    EXPECT_EQ(results[2].offset, 1);
    EXPECT_FLOAT_EQ(results[2].distance, 25.0f);
}

TEST(FlatIndex, TopKSmallerThanStore) {
    vexdb::InMemoryStore store(2);
    store.add_vector(std::vector<float>{0.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{1.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{2.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{3.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{4.0f, 0.0f}.data());

    vexdb::FlatIndex index(store);
    std::vector<float> query = {0.0f, 0.0f};

    auto results = index.search(query.data(), 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].offset, 0);
    EXPECT_EQ(results[1].offset, 1);
    EXPECT_EQ(results[2].offset, 2);
}

TEST(FlatIndex, KLargerThanStore) {
    vexdb::InMemoryStore store(2);
    store.add_vector(std::vector<float>{1.0f, 0.0f}.data());
    store.add_vector(std::vector<float>{2.0f, 0.0f}.data());

    vexdb::FlatIndex index(store);
    std::vector<float> query = {0.0f, 0.0f};

    auto results = index.search(query.data(), 10);

    // Should return all vectors, not crash.
    ASSERT_EQ(results.size(), 2);
}

TEST(FlatIndex, ResultsMatchDirectComputation) {
    // Random vectors, verify FlatIndex matches manual distance computation.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    vexdb::Dim dim = 64;
    int n = 200;

    vexdb::InMemoryStore store(dim);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        store.add_vector(v.data());
        vecs.push_back(std::move(v));
    }

    std::vector<float> query(dim);
    for (auto& x : query) x = dist(rng);

    vexdb::FlatIndex index(store);
    auto results = index.search(query.data(), 10);

    ASSERT_EQ(results.size(), 10);

    // Verify results are sorted by distance.
    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }

    // Verify distances match direct computation.
    for (const auto& r : results) {
        float expected = vexdb::l2_distance(query.data(), vecs[r.offset].data(), dim);
        EXPECT_FLOAT_EQ(r.distance, expected);
    }

    // Verify these are actually the closest: no un-returned vector should be closer
    // than the worst returned result.
    float worst_returned = results.back().distance;
    for (int i = 0; i < n; i++) {
        float d = vexdb::l2_distance(query.data(), vecs[i].data(), dim);
        bool in_results = false;
        for (const auto& r : results) {
            if (r.offset == static_cast<vexdb::Offset>(i)) {
                in_results = true;
                break;
            }
        }
        if (!in_results) {
            EXPECT_GE(d, worst_returned)
                << "Vector " << i << " with distance " << d << " should have been in top-k";
        }
    }
}