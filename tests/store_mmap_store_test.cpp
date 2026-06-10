#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "persistence/format.h"
#include "store/mmap_store.h"

namespace {

// Helper: write a valid VEX0 file with given vectors. Each test passes a
// unique name so tests stay independent under `ctest --parallel` — a shared
// file gets rewritten while another test still has it mmap'ed (SIGBUS).
std::string write_vex0_file(const std::string& name, const std::vector<std::vector<float>>& vectors,
                            vextor::Dim dim) {
    auto path = std::filesystem::temp_directory_path() / ("vextor_" + name + ".vex0");
    std::string path_str = path.string();

    std::ofstream out(path_str, std::ios::binary);

    vextor::Vex0Header header{};
    header.magic = vextor::kVex0Magic;
    header.version = vextor::kVex0Version;
    header.dim = dim;
    header.count = static_cast<uint32_t>(vectors.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const auto& vec : vectors) {
        out.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    out.close();

    return path_str;
}

}  // namespace

TEST(MmapStore, LoadAndRetrieveVectors) {
    vextor::Dim dim = 4;
    std::vector<std::vector<float>> vecs = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
    };

    auto path = write_vex0_file("load_and_retrieve", vecs, dim);

    vextor::MmapStore store(path.c_str());
    EXPECT_EQ(store.size(), 3);
    EXPECT_EQ(store.dimensions(), 4);

    const float* v0 = store.get_vector(0);
    EXPECT_EQ(v0[0], 1.0f);
    EXPECT_EQ(v0[3], 4.0f);

    const float* v2 = store.get_vector(2);
    EXPECT_EQ(v2[0], 9.0f);
    EXPECT_EQ(v2[3], 12.0f);

    std::filesystem::remove(path);
}

TEST(MmapStore, DataIntegrityManyVectors) {
    vextor::Dim dim = 128;
    const int n = 500;

    std::vector<std::vector<float>> vecs;
    vecs.reserve(n);
    for (int i = 0; i < n; i++) {
        vecs.emplace_back(dim, static_cast<float>(i));
    }

    auto path = write_vex0_file("data_integrity", vecs, dim);

    vextor::MmapStore store(path.c_str());
    EXPECT_EQ(store.size(), n);

    for (int i = 0; i < n; i++) {
        const float* vec = store.get_vector(static_cast<vextor::Offset>(i));
        for (vextor::Dim d = 0; d < dim; d++) {
            EXPECT_EQ(vec[d], static_cast<float>(i)) << "Mismatch at vector " << i << " dim " << d;
        }
    }

    std::filesystem::remove(path);
}

TEST(MmapStore, InvalidMagicThrows) {
    auto path = std::filesystem::temp_directory_path() / "vextor_bad_magic.vex0";
    std::string path_str = path.string();

    vextor::Vex0Header header{};
    header.magic = 0xDEADBEEF;  // wrong magic
    header.version = vextor::kVex0Version;
    header.dim = 4;
    header.count = 0;

    std::ofstream out(path_str, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.close();

    EXPECT_THROW(vextor::MmapStore store(path_str.c_str()), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(MmapStore, FileTooSmallThrows) {
    auto path = std::filesystem::temp_directory_path() / "vextor_truncated.vex0";
    std::string path_str = path.string();

    // Write only half a header.
    std::ofstream out(path_str, std::ios::binary);
    uint32_t partial = 0;
    out.write(reinterpret_cast<const char*>(&partial), sizeof(partial));
    out.close();

    EXPECT_THROW(vextor::MmapStore store(path_str.c_str()), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(MmapStore, MoveSemantics) {
    vextor::Dim dim = 3;
    std::vector<std::vector<float>> vecs = {{1.0f, 2.0f, 3.0f}};
    auto path = write_vex0_file("move_semantics", vecs, dim);

    vextor::MmapStore store1(path.c_str());

    // Move construct.
    vextor::MmapStore store2(std::move(store1));
    EXPECT_EQ(store2.size(), 1);
    EXPECT_EQ(store2.get_vector(0)[0], 1.0f);

    // Move assign.
    vextor::MmapStore store3(path.c_str());
    store3 = std::move(store2);
    EXPECT_EQ(store3.size(), 1);
    EXPECT_EQ(store3.get_vector(0)[0], 1.0f);

    std::filesystem::remove(path);
}