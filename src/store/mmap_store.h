#pragma once

#include <cstddef>

#include "core/types.h"
#include "store/concept.h"

namespace vexdb {

class MmapStore {
   public:
    // Open and mmap a VEX0 vectors.bin file.
    explicit MmapStore(const char* path);

    // RAII: munmap in destructor.
    ~MmapStore();

    // Non-copyable, movable. Rule of 5.
    MmapStore(const MmapStore&) = delete;
    MmapStore& operator=(const MmapStore&) = delete;
    MmapStore(MmapStore&& other) noexcept;
    MmapStore& operator=(MmapStore&& other) noexcept;

    // VectorStore interface.
    const float* get_vector(Offset offset) const;
    std::size_t size() const;
    Dim dimensions() const;

   private:
    void* mapped_ = nullptr;
    std::size_t mapped_size_ = 0;
    const float* data_ = nullptr;  // points past the header
    Dim dim_ = 0;
    uint32_t count_ = 0;
};

// assert that Mmap-store is compatible with the VectorStore-concept
static_assert(VectorStore<MmapStore>);

}  // namespace vexdb