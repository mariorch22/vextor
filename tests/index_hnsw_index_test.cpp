#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "index/flat_index.h"
#include "index/hnsw_index.h"
#include "store/in_memory_store.h"

// --- Helper: build a store with random vectors ---

static vextor::InMemoryStore make_random_store(int n, vextor::Dim dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vextor::InMemoryStore store(dim);
    for (int i = 0; i < n; i++) {
        std::vector<float> vec(dim);
        for (auto& x : vec) x = dist(rng);
        store.add_vector(vec.data());
    }
    return store;
}

// --- Basic functionality ---

TEST(HnswIndex, InsertSingleVector) {
    vextor::InMemoryStore store(4);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    store.add_vector(vec.data());

    vextor::HnswIndex index(store);
    index.insert();

    EXPECT_EQ(index.graph().size(), 1);
    EXPECT_FALSE(index.graph().empty);
    EXPECT_EQ(index.graph().entry_point, 0);
}

TEST(HnswIndex, InsertMultipleVectors) {
    auto store = make_random_store(100, 32);
    vextor::HnswIndex index(store);

    for (vextor::Offset i = 0; i < 100; i++) {
        index.insert();
    }

    EXPECT_EQ(index.graph().size(), 100);
}

TEST(HnswIndex, InsertWithoutAvailableVectorThrows) {
    vextor::InMemoryStore store(4);
    vextor::HnswIndex index(store);

    EXPECT_THROW(index.insert(), std::out_of_range);
}

TEST(HnswIndex, InsertPastStoreSizeThrows) {
    vextor::InMemoryStore store(4);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    store.add_vector(vec.data());

    vextor::HnswIndex index(store);
    index.insert();

    EXPECT_THROW(index.insert(), std::out_of_range);
}

TEST(HnswIndex, SearchEmptyIndex) {
    vextor::InMemoryStore store(4);
    vextor::HnswIndex index(store);

    std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
    auto results = index.search(query.data(), 10);
    EXPECT_TRUE(results.empty());
}

TEST(HnswIndex, SearchSingleVector) {
    vextor::InMemoryStore store(3);
    std::vector<float> vec = {1.0f, 0.0f, 0.0f};
    store.add_vector(vec.data());

    vextor::HnswIndex index(store);
    index.insert();

    std::vector<float> query = {0.0f, 0.0f, 0.0f};
    auto results = index.search(query.data(), 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].offset, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 1.0f);
}

TEST(HnswIndex, SearchKZeroReturnsEmpty) {
    auto store = make_random_store(10, 8);
    vextor::HnswIndex index(store);
    for (int i = 0; i < 10; i++) index.insert();

    std::vector<float> query(8, 0.0f);
    auto results = index.search(query.data(), 0);
    EXPECT_TRUE(results.empty());
}

TEST(HnswIndex, ResultsSortedByDistance) {
    auto store = make_random_store(500, 32);
    vextor::HnswIndex index(store);
    for (int i = 0; i < 500; i++) index.insert();

    std::vector<float> query(32, 0.0f);
    auto results = index.search(query.data(), 20);

    for (std::size_t i = 1; i < results.size(); i++) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

// --- Graph invariants ---

TEST(HnswIndex, BidirectionalEdges) {
    auto store = make_random_store(500, 32);
    vextor::HnswIndex index(store);
    for (int i = 0; i < 500; i++) index.insert();

    const auto& g = index.graph();
    int stride = g.layer0_stride();

    for (vextor::Offset node = 0; node < static_cast<vextor::Offset>(g.size()); node++) {
        // Check layer 0.
        int count = g.layer0_count[node];
        for (int j = 0; j < count; j++) {
            vextor::Offset nb = g.layer0[node * stride + j];
            // nb should have node as a neighbor too.
            bool found = false;
            int nb_count = g.layer0_count[nb];
            for (int k = 0; k < nb_count; k++) {
                if (g.layer0[nb * stride + k] == node) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Node " << node << " → " << nb
                               << " at layer 0, but reverse edge missing";
        }

        // Check upper layers.
        for (int layer = 1; layer <= g.levels[node]; layer++) {
            for (vextor::Offset nb : g.upper[node][layer - 1]) {
                ASSERT_GT(g.levels[nb], layer - 1)
                    << "Node " << nb << " doesn't exist at layer " << layer;
                bool found = std::find(g.upper[nb][layer - 1].begin(), g.upper[nb][layer - 1].end(),
                                       node) != g.upper[nb][layer - 1].end();
                EXPECT_TRUE(found) << "Node " << node << " → " << nb << " at layer " << layer
                                   << ", but reverse edge missing";
            }
        }
    }
}

TEST(HnswIndex, MaxNeighborConstraint) {
    auto store = make_random_store(500, 32);
    vextor::HnswIndex index(store);
    for (int i = 0; i < 500; i++) index.insert();

    const auto& g = index.graph();
    int m0 = 2 * g.m;

    for (vextor::Offset node = 0; node < static_cast<vextor::Offset>(g.size()); node++) {
        // Layer 0: max 2M neighbors.
        EXPECT_LE(g.layer0_count[node], m0) << "Node " << node << " has " << g.layer0_count[node]
                                            << " neighbors at layer 0 (max " << m0 << ")";

        // Upper layers: max M neighbors.
        for (int layer = 1; layer <= g.levels[node]; layer++) {
            EXPECT_LE(static_cast<int>(g.upper[node][layer - 1].size()), g.m)
                << "Node " << node << " has " << g.upper[node][layer - 1].size()
                << " neighbors at layer " << layer << " (max " << g.m << ")";
        }
    }
}

TEST(HnswIndex, LayerDistribution) {
    auto store = make_random_store(10000, 32);
    vextor::HnswIndex index(store);
    for (int i = 0; i < 10000; i++) index.insert();

    const auto& g = index.graph();

    // Count nodes per layer.
    std::vector<int> layer_counts(g.max_level + 1, 0);
    for (int level : g.levels) {
        for (int l = 0; l <= level; l++) {
            layer_counts[l]++;
        }
    }

    // Layer 0 should have all nodes.
    EXPECT_EQ(layer_counts[0], 10000);

    // Each higher layer should have fewer nodes than the one below (exponential decay).
    for (int l = 1; l <= g.max_level; l++) {
        EXPECT_LT(layer_counts[l], layer_counts[l - 1])
            << "Layer " << l << " has " << layer_counts[l] << " nodes, but layer " << l - 1
            << " has " << layer_counts[l - 1];
    }

    // Top layer should have very few nodes.
    EXPECT_LT(layer_counts[g.max_level], 100)
        << "Top layer has too many nodes: " << layer_counts[g.max_level];
}

// --- Recall test ---

TEST(HnswIndex, RecallAt10Above85Percent) {
    const int n = 10000;
    const vextor::Dim dim = 128;
    const int k = 10;
    const int num_queries = 100;

    auto store = make_random_store(n, dim, 42);

    // Build HNSW index.
    vextor::HnswIndex hnsw(store);
    for (int i = 0; i < n; i++) hnsw.insert();

    // Build flat index as ground truth.
    vextor::FlatIndex flat(store);

    // Generate queries.
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int total_correct = 0;
    for (int q = 0; q < num_queries; q++) {
        std::vector<float> query(dim);
        for (auto& x : query) x = dist(rng);

        auto hnsw_results =
            hnsw.search(query.data(), k, vextor::HnswSearchParams{.ef_search = 128});
        auto flat_results = flat.search(query.data(), k);

        // Count how many of HNSW's top-k are in the true top-k.
        std::set<vextor::Offset> true_set;
        for (const auto& r : flat_results) true_set.insert(r.offset);

        for (const auto& r : hnsw_results) {
            if (true_set.count(r.offset)) total_correct++;
        }
    }

    float recall = static_cast<float>(total_correct) / (num_queries * k);
    EXPECT_GT(recall, 0.85f) << "Recall@10 = " << recall << " (expected > 0.85)";
}
