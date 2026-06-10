#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/distance.h"

// --- Helper: generate deterministic test vectors ---

static std::vector<float> make_vector(vextor::Dim dim, float start, float step) {
    std::vector<float> v(dim);
    for (vextor::Dim i = 0; i < dim; i++) {
        v[i] = start + static_cast<float>(i) * step;
    }
    return v;
}

// --- Scalar correctness against hand-computed values ---

TEST(Distance, ScalarIdenticalVectorsIsZero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    EXPECT_FLOAT_EQ(vextor::l2_distance_scalar(a.data(), a.data(), 3), 0.0f);
}

TEST(Distance, ScalarKnownResult) {
    // (0,0) to (3,4): squared distance = 9 + 16 = 25
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f};
    EXPECT_FLOAT_EQ(vextor::l2_distance_scalar(a.data(), b.data(), 2), 25.0f);
}

TEST(Distance, ScalarSingleDimension) {
    float a = 3.0f;
    float b = 7.0f;
    EXPECT_FLOAT_EQ(vextor::l2_distance_scalar(&a, &b, 1), 16.0f);
}

// --- AVX2 vs Scalar consistency across dimensions ---

class DistanceConsistencyTest : public ::testing::TestWithParam<vextor::Dim> {};

TEST_P(DistanceConsistencyTest, AVX2MatchesScalar) {
    vextor::Dim dim = GetParam();
    auto a = make_vector(dim, 0.0f, 0.1f);
    auto b = make_vector(dim, 1.0f, -0.05f);

    float scalar = vextor::l2_distance_scalar(a.data(), b.data(), dim);
    float avx2 = vextor::l2_distance(a.data(), b.data(), dim);

    // Allow small floating-point divergence from different accumulation order
    EXPECT_NEAR(scalar, avx2, scalar * 1e-5f + 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, DistanceConsistencyTest,
                         ::testing::Values(1, 2, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 768));

// --- Edge cases ---

TEST(Distance, ZeroDimension) {
    EXPECT_FLOAT_EQ(vextor::l2_distance(nullptr, nullptr, 0), 0.0f);
    EXPECT_FLOAT_EQ(vextor::l2_distance_scalar(nullptr, nullptr, 0), 0.0f);
}

TEST(Distance, LargeIdenticalVectors) {
    std::vector<float> a(768, 42.0f);
    EXPECT_FLOAT_EQ(vextor::l2_distance(a.data(), a.data(), 768), 0.0f);
    EXPECT_FLOAT_EQ(vextor::l2_distance_scalar(a.data(), a.data(), 768), 0.0f);
}