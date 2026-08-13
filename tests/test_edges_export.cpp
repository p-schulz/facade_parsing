#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "facade_parser/classification.hpp"
#include "facade_parser/edges.hpp"
#include "facade_parser/edges_export.hpp"
#include "facade_parser/grammar.hpp"
#include "synthetic_facade.hpp"

// A regular window grid has no cornices/ledges of its own — every long
// line segment Stage 1 finds is part of a window frame, which Stage 4
// already reports as a Window element. exportEdgeElements must not
// double-report those frames as separate `edge` elements.
TEST(EdgesExport, WindowFrameSegmentsAreSuppressed) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);
  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);
  const auto cell_elements = facade_parser::classifyCells(facade.image, edges, grid, config);

  const facade_parser::SobelMagnitudeDepthHint depth_hint;
  const auto edge_elements =
      facade_parser::exportEdgeElements(edges, cell_elements, depth_hint, config);

  EXPECT_TRUE(edge_elements.empty());
}

// A horizontal line drawn well outside any window (a cornice-like
// feature) should survive suppression and be tagged accordingly.
TEST(EdgesExport, GenuineCorniceLineIsEmitted) {
  auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  cv::line(facade.image, {0, 5}, {facade.image.cols - 1, 5}, cv::Scalar(30, 30, 30), 2);

  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);
  const cv::Rect bbox(0, 0, facade.image.cols, facade.image.rows);
  const facade_parser::FacadeGrid grid = facade_parser::buildSplitGrammar(edges, bbox, config);
  const auto cell_elements = facade_parser::classifyCells(facade.image, edges, grid, config);

  const facade_parser::SobelMagnitudeDepthHint depth_hint;
  const auto edge_elements =
      facade_parser::exportEdgeElements(edges, cell_elements, depth_hint, config);

  ASSERT_EQ(edge_elements.size(), 1U);
  EXPECT_EQ(edge_elements.front().type, facade_parser::ElementType::Edge);
  EXPECT_EQ(edge_elements.front().edge_kind, facade_parser::EdgeKind::Cornice);
  ASSERT_TRUE(edge_elements.front().depth_hint_value.has_value());
}

TEST(EdgesExport, ShortSegmentsAreDropped) {
  cv::Mat image(200, 300, CV_8UC3, cv::Scalar(210, 205, 200));
  // A short 10px mark — well under the default min_edge_length_px (40).
  cv::line(image, {50, 50}, {60, 50}, cv::Scalar(30, 30, 30), 2);

  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);

  const facade_parser::SobelMagnitudeDepthHint depth_hint;
  const auto edge_elements = facade_parser::exportEdgeElements(edges, {}, depth_hint, config);

  EXPECT_TRUE(edge_elements.empty());
}

TEST(EdgesExport, SobelMagnitudeDepthHintIsAlwaysLowConfidence) {
  const cv::Mat gradient(100, 100, CV_32FC1, cv::Scalar(128.0));
  const facade_parser::SobelMagnitudeDepthHint depth_hint;

  const auto result = depth_hint.estimate(gradient, cv::Rect(10, 10, 20, 20));

  EXPECT_TRUE(result.low_confidence);
  EXPECT_GE(result.value, 0.0F);
  EXPECT_LE(result.value, 1.0F);
}
