#pragma once

// Public API of vexdb — the only header users need to include.
// Everything else under src/ is internal and may change between versions.

#include "core/query_result.h"
#include "core/types.h"
#include "segment/segment_manager.h"

namespace vexdb {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

// Primary user-facing entry point.
using Database = SegmentManager;

}  // namespace vexdb
