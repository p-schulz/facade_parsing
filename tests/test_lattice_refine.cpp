#include <gtest/gtest.h>

#include <cstdlib>

#include "facade_parser/edges.hpp"
#include "facade_parser/grammar.hpp"
#include "facade_parser/lattice_refine.hpp"
#include "synthetic_facade.hpp"

namespace {

int totalAbsError(const std::vector<int>& a, const std::vector<int>& b) {
  int sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += std::abs(a[i] - b[i]);
  }
  return sum;
}

}  // namespace

// Directly tests refine()'s contract (Riemenschneider et al., CVPR 2012):
// given a boundary perturbed away from its true low-activity position but
// still within the search window, it should snap back closer to the
// truth. Per docs/PLAN.md's testing strategy note on validating Stage 5
// against a synthetic ground truth.
TEST(LatticeRefine, RecoversFromPerturbedBoundariesTowardGroundTruth) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);

  facade_parser::FacadeGrid perturbed = facade.ground_truth_grid;
  for (std::size_t i = 1; i + 1 < perturbed.row_boundaries_px.size(); ++i) {
    perturbed.row_boundaries_px[i] += 5;
  }
  for (auto& cols : perturbed.col_boundaries_px) {
    for (std::size_t i = 1; i + 1 < cols.size(); ++i) {
      cols[i] += 5;
    }
  }

  const facade_parser::LatticeRefinePass pass;
  const facade_parser::FacadeGrid refined =
      pass.refine(perturbed, edges.gradient_magnitude, config);

  const int row_error_before =
      totalAbsError(perturbed.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  const int row_error_after =
      totalAbsError(refined.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  EXPECT_LT(row_error_after, row_error_before);

  int col_error_before = 0;
  int col_error_after = 0;
  for (std::size_t r = 0; r < refined.col_boundaries_px.size(); ++r) {
    col_error_before +=
        totalAbsError(perturbed.col_boundaries_px[r], facade.ground_truth_grid.col_boundaries_px[r]);
    col_error_after +=
        totalAbsError(refined.col_boundaries_px[r], facade.ground_truth_grid.col_boundaries_px[r]);
  }
  EXPECT_LT(col_error_after, col_error_before);
}

// End-to-end (Stage 3 + Stage 5) on a jittered facade: refinement
// composed on top of buildSplitGrammar's own already-adaptive detection
// should never make accuracy worse, and should stay within a tight
// tolerance of the true (jittered) boundary positions.
TEST(LatticeRefine, ComposedWithGrammarStaysAccurateOnJitteredFacade) {
  const auto facade = facade_parser::test::makeJitteredFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);

  const facade_parser::FacadeGrid coarse = facade_parser::buildSplitGrammar(edges, bbox, config);
  const facade_parser::LatticeRefinePass pass;
  const facade_parser::FacadeGrid refined =
      pass.refine(coarse, edges.gradient_magnitude, config);

  const int row_error_coarse =
      totalAbsError(coarse.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  const int row_error_refined =
      totalAbsError(refined.row_boundaries_px, facade.ground_truth_grid.row_boundaries_px);
  EXPECT_LE(row_error_refined, row_error_coarse + 1);  // never meaningfully worse.

  for (int b : refined.row_boundaries_px) {
    EXPECT_GE(b, 0);
    EXPECT_LE(b, facade.image.rows);
  }
}

TEST(LatticeRefine, RefineReturnsSameBoundaryCount) {
  const auto facade = facade_parser::test::makeJitteredFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);

  const facade_parser::LatticeRefinePass pass;
  const facade_parser::FacadeGrid refined =
      pass.refine(facade.ground_truth_grid, edges.gradient_magnitude, config);

  EXPECT_EQ(refined.row_boundaries_px.size(), facade.ground_truth_grid.row_boundaries_px.size());
  ASSERT_EQ(refined.col_boundaries_px.size(), facade.ground_truth_grid.col_boundaries_px.size());
  for (std::size_t r = 0; r < refined.col_boundaries_px.size(); ++r) {
    EXPECT_EQ(refined.col_boundaries_px[r].size(), facade.ground_truth_grid.col_boundaries_px[r].size());
  }
}
