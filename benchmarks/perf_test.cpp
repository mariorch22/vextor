#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "segment/segment_manager.h"

int main() {
    constexpr int dim = 128;
    constexpr int n = 100000;
    constexpr int n_queries = 100;
    constexpr int k = 10;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vexdb::SegmentManager db(dim, n);

    // bulk insert
    std::vector<float> vec(dim);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        for (auto& v : vec) v = dist(rng);
        db.insert(static_cast<uint64_t>(i), vec);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inserted " << n << " vectors in " << insert_ms << " ms\n";
    std::cout << "  " << (insert_ms / n) << " ms/insert\n";

    // queries
    std::vector<float> query(dim);
    auto t2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_queries; i++) {
        for (auto& v : query) v = dist(rng);
        auto results = db.search(query, k);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Ran " << n_queries << " searches in " << search_ms << " ms\n";
    std::cout << "  " << (search_ms / n_queries) << " ms/query\n";

    return 0;
}