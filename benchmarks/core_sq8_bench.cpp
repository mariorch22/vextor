#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "core/distance.h"
#include "core/sq8.h"

static void BM_L2Float32_128d(benchmark::State& state) {
    const int n = 10000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vecs(n * 128);
    for (auto& v : vecs) v = dist(rng);
    std::vector<float> query(128);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(vextor::l2_distance(query.data(), &vecs[i * 128], 128));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_SQ8Asymmetric_128d(benchmark::State& state) {
    const int n = 10000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vecs(n * 128);
    for (auto& v : vecs) v = dist(rng);

    auto params = vextor::sq8_compute_params(vecs.data(), n, 128);
    std::vector<uint8_t> encoded(n * 128);
    for (int i = 0; i < n; i++) {
        vextor::sq8_encode(&vecs[i * 128], &encoded[i * 128], params);
    }

    std::vector<float> query(128);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(
                vextor::sq8_asymmetric_l2(query.data(), &encoded[i * 128], params));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_L2Float32_768d(benchmark::State& state) {
    const int n = 10000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vecs(n * 768);
    for (auto& v : vecs) v = dist(rng);
    std::vector<float> query(768);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(vextor::l2_distance(query.data(), &vecs[i * 768], 768));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_SQ8Asymmetric_768d(benchmark::State& state) {
    const int n = 10000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vecs(n * 768);
    for (auto& v : vecs) v = dist(rng);

    auto params = vextor::sq8_compute_params(vecs.data(), n, 768);
    std::vector<uint8_t> encoded(n * 768);
    for (int i = 0; i < n; i++) {
        vextor::sq8_encode(&vecs[i * 768], &encoded[i * 768], params);
    }

    std::vector<float> query(768);
    for (auto& v : query) v = dist(rng);

    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            benchmark::DoNotOptimize(
                vextor::sq8_asymmetric_l2(query.data(), &encoded[i * 768], params));
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_L2Float32_128d);
BENCHMARK(BM_SQ8Asymmetric_128d);
BENCHMARK(BM_L2Float32_768d);
BENCHMARK(BM_SQ8Asymmetric_768d);