#include <gtest/gtest.h>

#include <algorithm>
#include <queue>
#include <vector>

#include "core/search_result.h"
#include "core/types.h"

// --- Type size guarantees ---

TEST(CoreTypes, TypeSizes) {
    static_assert(sizeof(vexdb::VectorId) == 8);
    static_assert(sizeof(vexdb::Offset) == 4);
    static_assert(sizeof(vexdb::Dim) == 4);
}

// --- SearchResult ordering ---

TEST(SearchResult, SmallerDistanceIsLess) {
    vexdb::SearchResult a{.offset = 0, .distance = 1.0f};
    vexdb::SearchResult b{.offset = 1, .distance = 2.0f};
    EXPECT_LT(a, b);
    EXPECT_GT(b, a);
}

TEST(SearchResult, EqualDistanceIsEqual) {
    vexdb::SearchResult a{.offset = 0, .distance = 1.5f};
    vexdb::SearchResult b{.offset = 0, .distance = 1.5f};
    EXPECT_EQ(a, b);
}

TEST(SearchResult, SortByDistance) {
    std::vector<vexdb::SearchResult> results = {
        {.offset = 0, .distance = 3.0f},
        {.offset = 1, .distance = 1.0f},
        {.offset = 2, .distance = 2.0f},
    };

    std::sort(results.begin(), results.end());

    EXPECT_EQ(results[0].offset, 1);
    EXPECT_EQ(results[1].offset, 2);
    EXPECT_EQ(results[2].offset, 0);
}

TEST(SearchResult, PriorityQueueClosestOnTop) {
    // std::greater makes it a min-heap: closest result on top
    std::priority_queue<vexdb::SearchResult, std::vector<vexdb::SearchResult>,
                        std::greater<vexdb::SearchResult>>
        pq;

    pq.push({.offset = 0, .distance = 5.0f});
    pq.push({.offset = 1, .distance = 1.0f});
    pq.push({.offset = 2, .distance = 3.0f});

    EXPECT_EQ(pq.top().offset, 1);
    EXPECT_FLOAT_EQ(pq.top().distance, 1.0f);
}

TEST(SearchResult, PriorityQueueMaxHeapFarthestOnTop) {
    // Default max-heap: farthest result on top (useful for HNSW candidate pruning)
    std::priority_queue<vexdb::SearchResult> pq;

    pq.push({.offset = 0, .distance = 5.0f});
    pq.push({.offset = 1, .distance = 1.0f});
    pq.push({.offset = 2, .distance = 3.0f});

    EXPECT_EQ(pq.top().offset, 0);
    EXPECT_FLOAT_EQ(pq.top().distance, 5.0f);
}

TEST(SearchResult, EqualDistanceDifferentOffsetIsNotEqual) {
    vexdb::SearchResult a{.offset = 0, .distance = 1.5f};
    vexdb::SearchResult b{.offset = 1, .distance = 1.5f};
    EXPECT_NE(a, b);      // different offset → not equal
    EXPECT_FALSE(a < b);  // same distance → neither is less
    EXPECT_FALSE(b < a);
}