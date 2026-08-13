#include <gtest/gtest.h>

#include <algorithm>

#include "facade_parser/edges.hpp"
#include "facade_parser/periodicity.hpp"
#include "synthetic_facade.hpp"

TEST(Periodicity, SmoothProfilePreservesLength) {
  const std::vector<float> profile(100, 1.0F);
  facade_parser::Config config;

  const auto smoothed = facade_parser::smoothProfile(profile, config);

  EXPECT_EQ(smoothed.size(), profile.size());
}

TEST(Periodicity, AnalyzePeriodicityReturnsAscendingBoundaries) {
  const std::vector<float> profile(200, 1.0F);
  facade_parser::Config config;

  const auto result = facade_parser::analyzePeriodicity(profile, config);

  ASSERT_GE(result.boundary_positions_px.size(), 2U);
  EXPECT_TRUE(std::is_sorted(result.boundary_positions_px.begin(),
                              result.boundary_positions_px.end()));
}

TEST(Periodicity, AnalyzePeriodicityOnFlatProfileFindsNoSplit) {
  const std::vector<float> profile(200, 0.0F);
  facade_parser::Config config;

  const auto result = facade_parser::analyzePeriodicity(profile, config);

  EXPECT_EQ(result.boundary_positions_px, (std::vector<int>{0, 200}));
  EXPECT_TRUE(result.low_confidence);
}

// Ground-truth-backed: the column profile of a synthetic regular facade
// has known wall-gap-center positions (see synthetic_facade.cpp); assert
// analyzePeriodicity recovers them within a small pixel tolerance, per
// docs/PLAN.md's testing strategy.
TEST(Periodicity, RecoversColumnBoundariesOnSyntheticFacade) {
  const auto facade = facade_parser::test::makeRegularFacade(1, 4, 55, 75, 22);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const auto profile = facade_parser::computeColumnProfile(edges.gradient_magnitude, bbox);
  const auto result = facade_parser::analyzePeriodicity(profile, config);

  const auto& expected = facade.ground_truth_grid.col_boundaries_px.front();
  ASSERT_EQ(result.boundary_positions_px.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.boundary_positions_px[i], expected[i], 3) << "boundary index " << i;
  }
  EXPECT_FALSE(result.low_confidence);
}
