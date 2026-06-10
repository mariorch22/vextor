#include <gtest/gtest.h>
#include <vexdb/vexdb.h>

#include <vector>

// The public header must be self-contained and expose the documented API
// surface: Database, QueryResult, VectorId, Dim, version constants.

TEST(PublicApi, VersionConstants) {
    EXPECT_EQ(vexdb::kVersionMajor, 0);
    EXPECT_EQ(vexdb::kVersionMinor, 1);
    EXPECT_EQ(vexdb::kVersionPatch, 0);
}

TEST(PublicApi, DatabaseInsertSearchRoundTrip) {
    vexdb::Database db(/*dim=*/4, /*segment_capacity=*/100);

    std::vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    vexdb::VectorId id = 42;
    db.insert(id, vec);

    auto results = db.search(vec, /*k=*/1);
    ASSERT_EQ(results.size(), 1u);
    vexdb::QueryResult top = results.front();
    EXPECT_EQ(top.user_id, id);
    EXPECT_FLOAT_EQ(top.distance, 0.0f);
}
