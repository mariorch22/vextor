#include "core/sq8.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef VEXTOR_AVX2
#include <immintrin.h>
#endif

namespace vextor {

SQ8Params sq8_compute_params(const float* vectors, std::size_t n, Dim dim) {
    if (n == 0) throw std::invalid_argument("sq8_compute_params: n must be > 0");

    SQ8Params params;
    params.dim = dim;
    params.min.resize(dim);
    params.scale.resize(dim);

    // Initialize min/max per dimension
    std::vector<float> max_vals(dim);
    for (Dim d = 0; d < dim; d++) {
        params.min[d] = std::numeric_limits<float>::max();
        max_vals[d] = std::numeric_limits<float>::lowest();
    }

    // Scan all vectors
    for (std::size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        for (Dim d = 0; d < dim; d++) {
            params.min[d] = std::min(params.min[d], vec[d]);
            max_vals[d] = std::max(max_vals[d], vec[d]);
        }
    }

    // Compute scale, avoid division by zero
    for (Dim d = 0; d < dim; d++) {
        params.scale[d] = max_vals[d] - params.min[d];
        if (params.scale[d] < 1e-10f) {
            params.scale[d] = 1.0f;  // constant dimension, scale doesn't matter
        }
    }

    return params;
}

void sq8_encode(const float* src, uint8_t* dst, const SQ8Params& params) {
    for (Dim d = 0; d < params.dim; d++) {
        float normalized = (src[d] - params.min[d]) / params.scale[d];
        normalized = std::clamp(normalized, 0.0f, 1.0f);
        dst[d] = static_cast<uint8_t>(std::round(normalized * 255.0f));
    }
}

void sq8_decode(const uint8_t* src, float* dst, const SQ8Params& params) {
    for (Dim d = 0; d < params.dim; d++) {
        dst[d] = (static_cast<float>(src[d]) / 255.0f) * params.scale[d] + params.min[d];
    }
}

// --- Scalar asymmetric L2 ---

float sq8_asymmetric_l2_scalar(const float* query, const uint8_t* encoded,
                               const SQ8Params& params) {
    float sum = 0.0f;
    for (Dim d = 0; d < params.dim; d++) {
        float decoded = (static_cast<float>(encoded[d]) / 255.0f) * params.scale[d] + params.min[d];
        float diff = query[d] - decoded;
        sum += diff * diff;
    }
    return sum;
}

// --- AVX2 asymmetric L2 ---

#ifdef VEXTOR_AVX2
float sq8_asymmetric_l2(const float* query, const uint8_t* encoded, const SQ8Params& params) {
    const float* min = params.min.data();
    const float* scale = params.scale.data();
    const float inv255 = 1.0f / 255.0f;
    __m256 v_inv255 = _mm256_set1_ps(inv255);
    __m256 v_sum = _mm256_setzero_ps();

    Dim d = 0;
    for (; d + 8 <= params.dim; d += 8) {
        // Load 8 uint8 values and convert to float32
        __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(encoded + d));
        __m256i ints = _mm256_cvtepu8_epi32(bytes);
        __m256 floats = _mm256_cvtepi32_ps(ints);

        // Dequantize: value = (float / 255) * scale + min
        __m256 v_scale = _mm256_loadu_ps(scale + d);
        __m256 v_min = _mm256_loadu_ps(min + d);
        __m256 decoded = _mm256_fmadd_ps(_mm256_mul_ps(floats, v_inv255), v_scale, v_min);

        // L2: (query - decoded)^2
        __m256 v_query = _mm256_loadu_ps(query + d);
        __m256 diff = _mm256_sub_ps(v_query, decoded);
        v_sum = _mm256_fmadd_ps(diff, diff, v_sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(v_sum, 1);
    __m128 lo = _mm256_castps256_ps128(v_sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Tail elements
    for (; d < params.dim; d++) {
        float decoded = (static_cast<float>(encoded[d]) / 255.0f) * scale[d] + min[d];
        float diff = query[d] - decoded;
        result += diff * diff;
    }

    return result;
}
#else
float sq8_asymmetric_l2(const float* query, const uint8_t* encoded, const SQ8Params& params) {
    return sq8_asymmetric_l2_scalar(query, encoded, params);
}
#endif

}  // namespace vextor