#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "index/flat_index.h"
#include "store/in_memory_store.h"

static void BM_FlatIndexSearch_1K_128d(benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vextor::InMemoryStore store(128);
    for (int i = 0; i < 1000; i++) {
        std::vector<float> v(128);
        for (auto& x : v) x = dist(rng);
        store.add_vector(v.data());
    }

    std::vector<float> query(128);
    for (auto& x : query) x = dist(rng);

    vextor::FlatIndex index(store);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10));
    }
}

static void BM_FlatIndexSearch_10K_128d(benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vextor::InMemoryStore store(128);
    for (int i = 0; i < 10000; i++) {
        std::vector<float> v(128);
        for (auto& x : v) x = dist(rng);
        store.add_vector(v.data());
    }

    std::vector<float> query(128);
    for (auto& x : query) x = dist(rng);

    vextor::FlatIndex index(store);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10));
    }
}

BENCHMARK(BM_FlatIndexSearch_1K_128d);
BENCHMARK(BM_FlatIndexSearch_10K_128d);