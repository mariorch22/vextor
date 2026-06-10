#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "index/flat_index.h"
#include "index/hnsw_index.h"
#include "store/in_memory_store.h"

// --- Helpers ---

static vextor::InMemoryStore make_store(int n, vextor::Dim dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vextor::InMemoryStore store(dim);
    for (int i = 0; i < n; i++) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        store.add_vector(v.data());
    }
    return store;
}

static std::vector<float> make_query(vextor::Dim dim, int seed = 99) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(dim);
    for (auto& x : q) x = dist(rng);
    return q;
}

// --- Insert benchmarks ---

static void BM_HnswInsert_10K_128d(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto store = make_store(10000, 128);
        vextor::HnswIndex index(store, 16, 200);
        state.ResumeTiming();

        for (vextor::Offset i = 0; i < 10000; i++) {
            index.insert();
        }
    }
    state.SetItemsProcessed(state.iterations() * 10000);
}

static void BM_HnswInsert_10K_128d_EF64(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto store = make_store(10000, 128);
        vextor::HnswIndex index(store, 16, 64);
        state.ResumeTiming();

        for (vextor::Offset i = 0; i < 10000; i++) {
            index.insert();
        }
    }
    state.SetItemsProcessed(state.iterations() * 10000);
}

// --- Search benchmarks ---

static void BM_HnswSearch_10K_128d(benchmark::State& state) {
    auto store = make_store(10000, 128);
    vextor::HnswIndex index(store, 16, 200);
    for (int i = 0; i < 10000; i++) index.insert();

    auto query = make_query(128);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10, 64));
    }
}

static void BM_HnswSearch_10K_128d_EF200(benchmark::State& state) {
    auto store = make_store(10000, 128);
    vextor::HnswIndex index(store, 16, 200);
    for (int i = 0; i < 10000; i++) index.insert();

    auto query = make_query(128);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10, 200));
    }
}

// --- Flat vs HNSW comparison ---

static void BM_FlatSearch_10K_128d(benchmark::State& state) {
    auto store = make_store(10000, 128);
    vextor::FlatIndex index(store);
    auto query = make_query(128);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10));
    }
}

// --- 100K benchmarks ---

static void BM_HnswInsert_100K_128d(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto store = make_store(100000, 128);
        vextor::HnswIndex index(store, 16, 200);
        state.ResumeTiming();

        for (vextor::Offset i = 0; i < 100000; i++) {
            index.insert();
        }
    }
    state.SetItemsProcessed(state.iterations() * 100000);
}

static void BM_HnswInsert_100K_128d_EF64(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto store = make_store(100000, 128);
        vextor::HnswIndex index(store, 16, 64);
        state.ResumeTiming();

        for (vextor::Offset i = 0; i < 100000; i++) {
            index.insert();
        }
    }
    state.SetItemsProcessed(state.iterations() * 100000);
}

static void BM_HnswSearch_100K_128d(benchmark::State& state) {
    auto store = make_store(100000, 128);
    vextor::HnswIndex index(store, 16, 200);
    for (int i = 0; i < 100000; i++) index.insert();

    auto query = make_query(128);

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.search(query.data(), 10, 128));
    }
}

BENCHMARK(BM_HnswInsert_10K_128d)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HnswInsert_10K_128d_EF64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HnswSearch_10K_128d);
BENCHMARK(BM_HnswSearch_10K_128d_EF200);
BENCHMARK(BM_FlatSearch_10K_128d);
BENCHMARK(BM_HnswInsert_100K_128d)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HnswInsert_100K_128d_EF64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HnswSearch_100K_128d);