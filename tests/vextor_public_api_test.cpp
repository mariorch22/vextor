#include <gtest/gtest.h>
#include <vextor/vextor.h>

#include <vector>

// The public header must be self-contained and expose the documented API
// surface: Database, QueryResult, VectorId, Dim, version constants.

TEST(PublicApi, VersionConstants) {
    EXPECT_EQ(vextor::kVersionMajor, 0);
    EXPECT_EQ(vextor::kVersionMinor, 1);
    EXPECT_EQ(vextor::kVersionPatch, 0);
}

TEST(PublicApi, DatabaseInsertSearchRoundTrip) {
    vextor::Database db(/*dim=*/4, /*segment_capacity=*/100);

    std::vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    vextor::VectorId id = 42;
    db.insert(id, vec);

    auto results = db.search(vec, /*k=*/1);
    ASSERT_EQ(results.size(), 1u);
    vextor::QueryResult top = results.front();
    EXPECT_EQ(top.user_id, id);
    EXPECT_FLOAT_EQ(top.distance, 0.0f);
}
