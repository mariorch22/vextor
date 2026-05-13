#include "store/mmap_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "persistence/format.h"

namespace vexdb {

MmapStore::MmapStore(const char* path) {
    // open read-only file. Returns -1 if it cannot open the file.
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(std::string("MmapStore: cannot open file: ") + path);
    }

    // Get file size.
    struct stat st {};
    if (fstat(fd, &st) < 0) {
        close(fd);
        throw std::runtime_error("MmapStore: fstat failed");
    }
    // off_t is signed, size_t is unsigned. Explicit cast to suppress warning.
    mapped_size_ = static_cast<std::size_t>(st.st_size);

    if (mapped_size_ < sizeof(Vex0Header)) {
        close(fd);
        throw std::runtime_error("MmapStore: file too small for VEX0 header");
    }

    // Map the file.
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // fd can be closed after mmap

    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: mmap failed");
    }

    // Parse header. Unmap if verification fails.
    const auto* header = static_cast<const Vex0Header*>(mapped_);
    if (header->magic != kVex0Magic) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: invalid VEX0 magic bytes");
    }
    if (header->version != kVex0Version) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: unsupported VEX0 version");
    }

    dim_ = header->dim;
    count_ = header->count;

    if (dim_ == 0) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: dim cannot be zero");
    }

    // Check for overflow before computing expected size.
    std::size_t vectors_size = static_cast<std::size_t>(count_) * dim_ * sizeof(float);
    if (count_ != 0 && vectors_size / dim_ / sizeof(float) != count_) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: header values would overflow");
    }
    if (vectors_size > std::numeric_limits<std::size_t>::max() - sizeof(Vex0Header)) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: header values would overflow");
    }

    // Verify file size matches header.
    std::size_t expected =
        sizeof(Vex0Header) + static_cast<std::size_t>(count_) * dim_ * sizeof(float);
    if (mapped_size_ < expected) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
        throw std::runtime_error("MmapStore: file size does not match header");
    }

    // Data starts right after the header.
    data_ = reinterpret_cast<const float*>(static_cast<const char*>(mapped_) + sizeof(Vex0Header));
}

MmapStore::~MmapStore() {
    if (mapped_) {
        munmap(mapped_, mapped_size_);
    }
}

// Move constructor: transfer ownership of the mapping, leave other empty.
MmapStore::MmapStore(MmapStore&& other) noexcept
    : mapped_(other.mapped_),
      mapped_size_(other.mapped_size_),
      data_(other.data_),
      dim_(other.dim_),
      count_(other.count_) {
    other.mapped_ = nullptr;
    other.mapped_size_ = 0;
    other.data_ = nullptr;
    other.dim_ = 0;
    other.count_ = 0;
}

// Move assignment: release our mapping, take ownership of other's.
MmapStore& MmapStore::operator=(MmapStore&& other) noexcept {
    if (this != &other) {
        if (mapped_) {
            munmap(mapped_, mapped_size_);
        }
        mapped_ = other.mapped_;
        mapped_size_ = other.mapped_size_;
        data_ = other.data_;
        dim_ = other.dim_;
        count_ = other.count_;
        other.mapped_ = nullptr;
        other.mapped_size_ = 0;
        other.data_ = nullptr;
        other.dim_ = 0;
        other.count_ = 0;
    }
    return *this;
}

const float* MmapStore::get_vector(Offset offset) const {
    assert(offset < count_);
    // Pointer arithmetic: skip offset * dim_ floats into the flat array.
    return data_ + static_cast<std::size_t>(offset) * dim_;
}

std::size_t MmapStore::size() const {
    return count_;
}

Dim MmapStore::dimensions() const {
    return dim_;
}

}  // namespace vexdb