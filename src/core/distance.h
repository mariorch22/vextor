#pragma once

#include "core/types.h"

namespace vextor {

float l2_distance(const float* a, const float* b, Dim dim);
float l2_distance_scalar(const float* a, const float* b, Dim dim);

}  // namespace vextor