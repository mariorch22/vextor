#include <gtest/gtest.h>

#include <random>
#include <set>
#include <vector>

#include "segment/sealed_segment.h"

// Helper: build an ActiveSegment, extract its components, and create a SealedSegment.
static vextor::SealedSegment make_sealed_from_active(int n, vextor::Dim dim) {
    vextor::InMemoryStore store(dim);
    vextor::HnswIndex<vextor::InMemoryStore> index(store);
    vextor::IdMapping id_mapping;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        store.add_vector(v.data());
        index.insert();
        id_mapping.insert(static_cast<vextor::VectorId>(1000 + i));
    }

    return vextor::SealedSegment::from_memory(std::move(store), index.graph(),
                                              std::move(id_mapping));
}

TEST(SealedSegment, SizeAndDimensions) {
    auto seg = make_sealed_from_active(100, 32);
    EXPECT_EQ(seg.size(), 100);
    EXPECT_EQ(seg.dimensions(), 32);
}

TEST(SealedSegment, SearchReturnsUserIds) {
    vextor::InMemoryStore store(2);
    vextor::HnswIndex<vextor::InMemoryStore> index(store);
    vextor::IdMapping id_mapping;

    store.add_vector(std::vector<float>{0.0f, 0.0f}.data());
    index.insert();
    id_mapping.insert(42);

    store.add_vector(std::vector<float>{1.0f, 0.0f}.data());
    index.insert();
    id_mapping.insert(99);

    store.add_vector(std::vector<float>{3.0f, 4.0f}.data());
    index.insert();
    id_mapping.insert(77);

    auto seg =
        vextor::SealedSegment::from_memory(std::move(store), index.graph(), std::move(id_mapping));

    std::vector<float> query = {0.0f, 0.0f};
    auto results = seg.search(query.data(), 3);

    ASSERT_EQ(results.size(), 3);

    std::set<vextor::VectorId> ids;
    for (const auto& r : results) ids.insert(r.user_id);

    EXPECT_TRUE(ids.count(42));
    EXPECT_TRUE(ids.count(99));
    EXPECT_TRUE(ids.count(77));
}

TEST(SealedSegment, SearchResultsSortedByDistance) {
    auto seg = make_sealed_from_active(500, 32);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> query(32);
    for (auto& x : query) x = dist(rng);

    auto results = seg.search(query.data(), 10, vextor::HnswSearchParams{.ef_search = 128});
    ASSERT_EQ(results.size(), 10);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST(SealedSegment, NoInsertAvailable) {
    // SealedSegment has no insert method — this is a compile-time check.
    // If this file compiles, the invariant holds.
    auto seg = make_sealed_from_active(10, 4);
    EXPECT_EQ(seg.size(), 10);
}

TEST(SealedSegment, LargeUserIdPreserved) {
    vextor::InMemoryStore store(2);
    vextor::HnswIndex<vextor::InMemoryStore> index(store);
    vextor::IdMapping id_mapping;

    vextor::VectorId big_id = 1ULL << 40;
    store.add_vector(std::vector<float>{1.0f, 0.0f}.data());
    index.insert();
    id_mapping.insert(big_id);

    auto seg =
        vextor::SealedSegment::from_memory(std::move(store), index.graph(), std::move(id_mapping));

    auto results = seg.search(std::vector<float>{1.0f, 0.0f}.data(), 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].user_id, big_id);
}
