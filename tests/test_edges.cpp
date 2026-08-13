#include <gtest/gtest.h>

#include <cmath>

#include "facade_parser/edges.hpp"
#include "synthetic_facade.hpp"

TEST(Edges, ProducesEdgeMapsWithCorrectSize) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;

  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);

  EXPECT_EQ(edges.canny.size(), facade.image.size());
  EXPECT_EQ(edges.gradient_magnitude.size(), facade.image.size());
  EXPECT_GT(cv::countNonZero(edges.canny), 0) << "expected some edges around the drawn windows";
}

// Ground-truth-backed: a regular grid of rectangles has exactly 4 edges
// per window (top/bottom horizontal, left/right vertical); assert the
// detector (FastLineDetector or the HoughLinesP fallback — see
// docs/PLAN.md Stage 1) finds roughly that many, split correctly by
// orientation, and none diagonal.
TEST(Edges, DetectsWindowFrameSegmentsSplitByOrientation) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);
  facade_parser::Config config;

  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(facade.image, config);

  const int num_windows = 2 * 3;
  // Expect roughly 2 horizontal + 2 vertical segments per window; allow
  // slack since a detector may merge or split a rectangle's edge.
  EXPECT_GE(static_cast<int>(edges.horizontal_segments.size()), num_windows);
  EXPECT_GE(static_cast<int>(edges.vertical_segments.size()), num_windows);

  for (const auto& seg : edges.horizontal_segments) {
    EXPECT_LE(std::abs(seg.angle_deg), config.line_angle_tolerance_deg);
  }
  for (const auto& seg : edges.vertical_segments) {
    EXPECT_GE(std::abs(seg.angle_deg), 90.0 - config.line_angle_tolerance_deg);
  }
}

TEST(Edges, NoSegmentsOnBlankWall) {
  const cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(210, 205, 200));
  facade_parser::Config config;

  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(blank, config);

  EXPECT_TRUE(edges.horizontal_segments.empty());
  EXPECT_TRUE(edges.vertical_segments.empty());
}
