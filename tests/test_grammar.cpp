#include <gtest/gtest.h>

#include "facade_parser/edges.hpp"
#include "facade_parser/grammar.hpp"
#include "synthetic_facade.hpp"

namespace {

// Ground-truth-backed per docs/PLAN.md's testing strategy: synthetic
// facades have known boundary positions, so we assert the detected grid
// matches within a small pixel tolerance rather than just smoke-testing
// shape.
constexpr int kTolerancePx = 3;

void expectBoundariesNear(const std::vector<int>& actual, const std::vector<int>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], kTolerancePx) << "boundary index " << i;
  }
}

}  // namespace

TEST(Grammar, RecoversRegularGridWithinTolerance) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);

  expectBoundariesNear(grid.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  ASSERT_EQ(grid.col_boundaries_px.size(), facade.ground_truth_grid.col_boundaries_px.size());
  for (std::size_t r = 0; r < grid.col_boundaries_px.size(); ++r) {
    expectBoundariesNear(grid.col_boundaries_px[r], facade.ground_truth_grid.col_boundaries_px[r]);
  }
}

TEST(Grammar, RecoversDifferentRowColCountsWithinTolerance) {
  const auto facade = facade_parser::test::makeRegularFacade(3, 4, 50, 70, 25);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);

  expectBoundariesNear(grid.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  ASSERT_EQ(grid.col_boundaries_px.size(), facade.ground_truth_grid.col_boundaries_px.size());
  for (std::size_t r = 0; r < grid.col_boundaries_px.size(); ++r) {
    expectBoundariesNear(grid.col_boundaries_px[r], facade.ground_truth_grid.col_boundaries_px[r]);
  }
}

TEST(Grammar, BuildSplitGrammarCoversFullFacadeBbox) {
  const auto facade = facade_parser::test::makeRegularFacade(3, 4, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);

  ASSERT_GE(grid.row_boundaries_px.size(), 2U);
  EXPECT_EQ(grid.row_boundaries_px.front(), 0);
  EXPECT_EQ(grid.row_boundaries_px.back(), facade.image.rows);
}
