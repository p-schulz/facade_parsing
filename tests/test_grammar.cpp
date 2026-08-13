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

// Regression test for docs/PLAN.md's "Known limitation" writeup: one
// strongly irregular cell (here, a door much narrower than its
// neighboring windows, bottom-aligned in the lowest row) used to make
// that row's column split find spurious extra boundaries around the
// door's own margins — Stage 3 would emit 6 columns instead of 3 for
// that row. The primary fix lives in Stage 2 (periodicity.cpp retries
// lower-ranked autocorrelation candidates when the top-ranked one
// disagrees with direct-peak spacing); Stage 3's median-width merge
// (grammar.cpp) is a secondary safety net. This asserts the column
// *count* is exactly recovered.
//
// Boundary *positions* right next to the irregular cell use a looser
// tolerance than expectBoundariesNear's usual 3px: the "regular grid"
// ground truth assumes every cell has the idealized pitch, but a door
// genuinely narrower than its neighbors shifts where the true
// low-activity wall gap sits by a few pixels — that's the detector
// correctly following the actual pixel content, not an error.
TEST(Grammar, RecoversCorrectColumnCountDespiteOneNarrowCell) {
  constexpr int kLooseTolerancePx = 8;

  const auto facade = facade_parser::test::makeFacadeWithOneNarrowCell(
      /*rows=*/2, /*cols=*/3, /*cell_width_px=*/60, /*cell_height_px=*/80, /*margin_px=*/20,
      /*narrow_row=*/1, /*narrow_col=*/1, /*narrow_width_px=*/40, /*narrow_height_px=*/90);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);

  expectBoundariesNear(grid.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  ASSERT_EQ(grid.col_boundaries_px.size(), facade.ground_truth_grid.col_boundaries_px.size());
  for (std::size_t r = 0; r < grid.col_boundaries_px.size(); ++r) {
    const auto& actual = grid.col_boundaries_px[r];
    const auto& expected = facade.ground_truth_grid.col_boundaries_px[r];
    ASSERT_EQ(actual.size(), expected.size()) << "row " << r << ": wrong column count";
    for (std::size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], kLooseTolerancePx) << "row " << r << " boundary " << i;
    }
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
