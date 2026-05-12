#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "core/distance.h"

static void BM_L2Scalar128d(benchmark::State& state) {
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(128, 2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vexdb::l2_distance_scalar(a.data(), b.data(), 128));
    }
}

static void BM_L2Scalar768d(benchmark::State& state) {
    std::vector<float> a(768, 1.0f);
    std::vector<float> b(768, 2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vexdb::l2_distance_scalar(a.data(), b.data(), 768));
    }
}

static void BM_L2AVX2_128d(benchmark::State& state) {
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(128, 2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vexdb::l2_distance(a.data(), b.data(), 128));
    }
}

static void BM_L2AVX2_768d(benchmark::State& state) {
    std::vector<float> a(768, 1.0f);
    std::vector<float> b(768, 2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vexdb::l2_distance(a.data(), b.data(), 768));
    }
}

static void BM_L2AVX2_128d_Batch(benchmark::State& state) {
    const int n = 10000;
    std::vector<float> vecs(n * 128);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : vecs) v = dist(rng);

    std::vector<float> query(128);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(vexdb::l2_distance(query.data(), &vecs[i * 128], 128));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_L2AVX2_768d_Batch(benchmark::State& state) {
    const int n = 10000;
    std::vector<float> vecs(n * 768);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : vecs) v = dist(rng);

    std::vector<float> query(768);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(vexdb::l2_distance(query.data(), &vecs[i * 768], 768));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_L2Scalar128d);
BENCHMARK(BM_L2AVX2_128d);
BENCHMARK(BM_L2Scalar768d);
BENCHMARK(BM_L2AVX2_768d);
BENCHMARK(BM_L2AVX2_128d_Batch);
BENCHMARK(BM_L2AVX2_768d_Batch);