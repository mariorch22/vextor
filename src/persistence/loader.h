#pragma once

#include <string>

#include "segment/sealed_segment.h"

namespace vexdb {

// Load a segment directory as a SealedSegment with MmapStore.
[[nodiscard]] SealedSegment load_segment_mmap(const std::string& dir);

// Load a segment directory as a SealedSegment with InMemoryStore.
[[nodiscard]] SealedSegment load_segment_memory(const std::string& dir);

}  // namespace vexdb