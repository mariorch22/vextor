#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "store/in_memory_store.h"

TEST(InMemoryStore, StartsEmpty) {
    vextor::InMemoryStore store(4);
    EXPECT_EQ(store.size(), 0);
    EXPECT_EQ(store.dimensions(), 4);
}

TEST(InMemoryStore, AddAndRetrieveSingleVector) {
    vextor::InMemoryStore store(3);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};

    vextor::Offset offset = store.add_vector(vec.data());

    EXPECT_EQ(offset, 0);
    EXPECT_EQ(store.size(), 1);

    const float* retrieved = store.get_vector(0);
    EXPECT_EQ(retrieved[0], 1.0f);
    EXPECT_EQ(retrieved[1], 2.0f);
    EXPECT_EQ(retrieved[2], 3.0f);
}

TEST(InMemoryStore, AddMultipleVectors) {
    vextor::InMemoryStore store(2);
    std::vector<float> v0 = {1.0f, 2.0f};
    std::vector<float> v1 = {3.0f, 4.0f};
    std::vector<float> v2 = {5.0f, 6.0f};

    EXPECT_EQ(store.add_vector(v0.data()), 0);
    EXPECT_EQ(store.add_vector(v1.data()), 1);
    EXPECT_EQ(store.add_vector(v2.data()), 2);
    EXPECT_EQ(store.size(), 3);

    const float* r1 = store.get_vector(1);
    EXPECT_EQ(r1[0], 3.0f);
    EXPECT_EQ(r1[1], 4.0f);
}

TEST(InMemoryStore, DataIntegrityAfterManyInserts) {
    vextor::Dim dim = 128;
    vextor::InMemoryStore store(dim);
    const int n = 1000;

    // Insert n vectors with known patterns
    for (int i = 0; i < n; i++) {
        std::vector<float> vec(dim, static_cast<float>(i));
        store.add_vector(vec.data());
    }

    EXPECT_EQ(store.size(), n);

    // Verify each vector
    for (int i = 0; i < n; i++) {
        const float* vec = store.get_vector(static_cast<vextor::Offset>(i));
        for (vextor::Dim d = 0; d < dim; d++) {
            EXPECT_EQ(vec[d], static_cast<float>(i)) << "Mismatch at vector " << i << " dim " << d;
        }
    }
}

TEST(InMemoryStore, OffsetsValidAfterReallocation) {
    // After many inserts, earlier get_vector calls should still return valid data.
    // Note: pointers may be invalidated by realloc, but offsets always work.
    vextor::InMemoryStore store(4);

    for (int i = 0; i < 10000; i++) {
        std::vector<float> vec(4, static_cast<float>(i));
        store.add_vector(vec.data());
    }

    // Verify first and last
    const float* first = store.get_vector(0);
    const float* last = store.get_vector(9999);
    EXPECT_EQ(first[0], 0.0f);
    EXPECT_EQ(last[0], 9999.0f);
}