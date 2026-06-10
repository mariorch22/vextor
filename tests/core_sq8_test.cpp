#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/distance.h"
#include "core/sq8.h"

// --- Encoding ---

TEST(SQ8, EncodeDecodeRoundTrip) {
    std::vector<float> vec = {0.1f, -0.3f, 0.5f, 0.8f};
    vextor::Dim dim = 4;

    auto params = vextor::sq8_compute_params(vec.data(), 1, dim);

    std::vector<uint8_t> encoded(dim);
    vextor::sq8_encode(vec.data(), encoded.data(), params);

    std::vector<float> decoded(dim);
    vextor::sq8_decode(encoded.data(), decoded.data(), params);

    for (vextor::Dim d = 0; d < dim; d++) {
        EXPECT_NEAR(vec[d], decoded[d], params.scale[d] / 255.0f + 1e-6f);
    }
}

TEST(SQ8, EncodeMapsMinToZeroMaxTo255) {
    std::vector<float> vecs = {
        0.0f, 0.0f,  // vec 0: both min
        1.0f, 2.0f,  // vec 1: both max
    };
    vextor::Dim dim = 2;

    auto params = vextor::sq8_compute_params(vecs.data(), 2, dim);

    // Encode vec 0 (min values)
    std::vector<uint8_t> enc0(dim);
    vextor::sq8_encode(vecs.data(), enc0.data(), params);
    EXPECT_EQ(enc0[0], 0);
    EXPECT_EQ(enc0[1], 0);

    // Encode vec 1 (max values)
    std::vector<uint8_t> enc1(dim);
    vextor::sq8_encode(vecs.data() + dim, enc1.data(), params);
    EXPECT_EQ(enc1[0], 255);
    EXPECT_EQ(enc1[1], 255);
}

TEST(SQ8, ConstantDimensionHandled) {
    // All vectors have the same value in dim 0
    std::vector<float> vecs = {
        5.0f, 1.0f, 5.0f, 2.0f, 5.0f, 3.0f,
    };
    vextor::Dim dim = 2;

    auto params = vextor::sq8_compute_params(vecs.data(), 3, dim);

    // scale[0] should be non-zero (we set it to 1.0 for constant dims)
    EXPECT_GT(params.scale[0], 0.0f);

    // Encode and decode should not crash or produce NaN
    std::vector<uint8_t> enc(dim);
    vextor::sq8_encode(vecs.data(), enc.data(), params);

    std::vector<float> dec(dim);
    vextor::sq8_decode(enc.data(), dec.data(), params);
    EXPECT_FALSE(std::isnan(dec[0]));
}

// --- Asymmetric distance ---

TEST(SQ8, AsymmetricDistanceCloseToFloat32) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    vextor::Dim dim = 128;
    size_t n = 100;

    // Generate random vectors
    std::vector<float> vecs(n * dim);
    for (auto& v : vecs) v = dist(rng);

    auto params = vextor::sq8_compute_params(vecs.data(), n, dim);

    // Encode all vectors
    std::vector<uint8_t> encoded(n * dim);
    for (size_t i = 0; i < n; i++) {
        vextor::sq8_encode(vecs.data() + i * dim, encoded.data() + i * dim, params);
    }

    // Query
    std::vector<float> query(dim);
    for (auto& v : query) v = dist(rng);

    // Compare asymmetric SQ8 distance vs full float32 distance
    for (size_t i = 0; i < n; i++) {
        float exact = vextor::l2_distance(query.data(), vecs.data() + i * dim, dim);
        float approx = vextor::sq8_asymmetric_l2(query.data(), encoded.data() + i * dim, params);

        // SQ8 error should be small relative to the true distance
        float rel_error = std::abs(exact - approx) / (exact + 1e-6f);
        EXPECT_LT(rel_error, 0.05f)
            << "Vector " << i << ": exact=" << exact << " approx=" << approx;
    }
}

// --- AVX2 vs Scalar consistency ---

class SQ8ConsistencyTest : public ::testing::TestWithParam<vextor::Dim> {};

TEST_P(SQ8ConsistencyTest, AVX2MatchesScalar) {
    vextor::Dim dim = GetParam();
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    size_t n = 50;
    std::vector<float> vecs(n * dim);
    for (auto& v : vecs) v = dist(rng);

    auto params = vextor::sq8_compute_params(vecs.data(), n, dim);

    std::vector<uint8_t> encoded(n * dim);
    for (size_t i = 0; i < n; i++) {
        vextor::sq8_encode(vecs.data() + i * dim, encoded.data() + i * dim, params);
    }

    std::vector<float> query(dim);
    for (auto& v : query) v = dist(rng);

    for (size_t i = 0; i < n; i++) {
        float scalar =
            vextor::sq8_asymmetric_l2_scalar(query.data(), encoded.data() + i * dim, params);
        float avx2 = vextor::sq8_asymmetric_l2(query.data(), encoded.data() + i * dim, params);

        EXPECT_NEAR(scalar, avx2, scalar * 1e-5f + 1e-6f);
    }
}

INSTANTIATE_TEST_SUITE_P(Dimensions, SQ8ConsistencyTest,
                         ::testing::Values(1, 7, 8, 9, 16, 17, 127, 128, 129, 768));