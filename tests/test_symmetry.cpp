#include <gtest/gtest.h>

#include <algorithm>

#include "facade_parser/classification.hpp"
#include "facade_parser/edges.hpp"
#include "facade_parser/grammar.hpp"
#include "facade_parser/symmetry.hpp"
#include "synthetic_facade.hpp"

TEST(Symmetry, MirrorAxisIsBboxCenterForSymmetricFacade) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 4, 60, 80, 20);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);
  facade_parser::Config config;

  const auto result =
      facade_parser::checkSymmetry(facade.image, bbox, facade.ground_truth_grid, {}, config);

  EXPECT_NEAR(result.mirror_axis_x_px, bbox.width / 2.0, 1.0);
}

// Simulates an occluded cell (e.g. a tree in front of one window, per
// docs/PLAN.md Stage 6): drop one cell's classification from the input
// element list and confirm checkSymmetry infers it from its mirrored
// counterpart, near the true bbox, at a discounted confidence.
TEST(Symmetry, InfersOccludedCellFromMirroredCounterpart) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 4, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);
  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);
  auto elements = facade_parser::classifyCells(facade.image, edges, grid, config);

  const std::size_t original_count = elements.size();
  elements.erase(std::remove_if(elements.begin(), elements.end(),
                                 [](const facade_parser::Element& e) {
                                   return e.row == 0 && e.col == 0;
                                 }),
                  elements.end());
  ASSERT_EQ(elements.size(), original_count - 1);

  const auto result = facade_parser::checkSymmetry(facade.image, bbox, grid, elements, config);

  ASSERT_EQ(result.inferences.size(), 1U);
  const auto& inference = result.inferences.front();
  EXPECT_EQ(inference.row, 0);
  EXPECT_EQ(inference.col, 0);
  EXPECT_EQ(inference.suggested_type, facade_parser::ElementType::Window);
  EXPECT_NEAR(inference.suggested_bbox_px.x, 20, 3);
  EXPECT_NEAR(inference.suggested_bbox_px.y, 20, 3);
  EXPECT_GT(inference.mirror_confidence, 0.0F);
}

TEST(Symmetry, DisabledCheckStillReportsAxisButNoInferences) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 4, 60, 80, 20);
  facade_parser::Config config;
  config.enable_symmetry_check = false;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);
  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);

  const auto result = facade_parser::checkSymmetry(facade.image, bbox, grid, {}, config);

  EXPECT_NEAR(result.mirror_axis_x_px, bbox.width / 2.0, 1.0);
  EXPECT_TRUE(result.inferences.empty());
}
