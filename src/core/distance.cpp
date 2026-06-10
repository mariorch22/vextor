#include "core/distance.h"

#ifdef VEXTOR_AVX2
#include <immintrin.h>
#endif

namespace vextor {

float l2_distance_scalar(const float* a, const float* b, Dim dim) {
    float sum = 0.0f;
    for (Dim i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#ifdef VEXTOR_AVX2
float l2_distance(const float* a, const float* b, Dim dim) {
    __m256 sum = _mm256_setzero_ps();
    Dim i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}
#else
float l2_distance(const float* a, const float* b, Dim dim) {
    return l2_distance_scalar(a, b, dim);
}
#endif

}  // namespace vextor