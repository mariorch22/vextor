#pragma once

#include "core/types.h"

namespace vexdb {

struct QueryResult {
    VectorId user_id;
    float distance;
};

}  // namespace vexdb