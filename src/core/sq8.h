#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/types.h"

namespace vextor {

struct SQ8Params {
    std::vector<float> min;    // per-dimension minimum
    std::vector<float> scale;  // per-dimension (max - min)
    Dim dim;
};

// Compute encoding parameters from a batch of float32 vectors.
// vectors: flat array of n * dim floats, row-major.
SQ8Params sq8_compute_params(const float* vectors, std::size_t n, Dim dim);

// Encode a single float32 vector to uint8.
void sq8_encode(const float* src, uint8_t* dst, const SQ8Params& params);

// Decode a single uint8 vector back to float32 (for testing).
void sq8_decode(const uint8_t* src, float* dst, const SQ8Params& params);

// Asymmetric L2 squared distance: float32 query vs uint8 database vector.
// Dequantizes on the fly — query is NOT quantized.
float sq8_asymmetric_l2(const float* query, const uint8_t* encoded, const SQ8Params& params);

// Scalar-only version (always available, used as ground truth in tests).
float sq8_asymmetric_l2_scalar(const float* query, const uint8_t* encoded, const SQ8Params& params);

}  // namespace vextor