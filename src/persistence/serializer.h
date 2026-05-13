#pragma once

#include <string>

#include "segment/active_segment.h"

namespace vexdb {

// Writes an ActiveSegment to a directory as vectors.bin, hnsw.bin, ids.bin.
void serialize_segment(const ActiveSegment& segment, const std::string& dir);

}  // namespace vexdb