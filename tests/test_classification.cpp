#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "facade_parser/classification.hpp"
#include "facade_parser/edges.hpp"

namespace {

// Builds a single-cell FacadeGrid spanning the whole image, so
// classifyCells is exercised in isolation from Stage 3's grid detection
// — per docs/PLAN.md's testing strategy ("synthesize cells with
// controlled contrast/aspect ratio to test the decision thresholds").
facade_parser::FacadeGrid singleCellGrid(cv::Size image_size) {
  facade_parser::FacadeGrid grid;
  grid.row_boundaries_px = {0, image_size.height};
  grid.col_boundaries_px = {{0, image_size.width}};
  return grid;
}

cv::Mat wallImage(cv::Size size) { return cv::Mat(size, CV_8UC3, cv::Scalar(210, 205, 200)); }

}  // namespace

TEST(Classification, EmitsNoElementsWhenWallElementsDisabled) {
  const cv::Mat image = wallImage({100, 100});
  facade_parser::Config config;
  config.emit_wall_elements = false;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);
  const auto grid = singleCellGrid(image.size());

  const auto elements = facade_parser::classifyCells(image, edges, grid, config);

  EXPECT_TRUE(elements.empty());
}

TEST(Classification, PlainWallCellClassifiesAsWall) {
  const cv::Mat image = wallImage({100, 100});
  facade_parser::Config config;
  config.emit_wall_elements = true;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);
  const auto grid = singleCellGrid(image.size());

  const auto elements = facade_parser::classifyCells(image, edges, grid, config);

  ASSERT_EQ(elements.size(), 1U);
  EXPECT_EQ(elements[0].type, facade_parser::ElementType::Wall);
  EXPECT_GT(elements[0].confidence, 0.5F);
}

TEST(Classification, CenteredRectangleWithValidFillAndAspectClassifiesAsWindow) {
  // 200x160 cell, 90x70 dark rectangle centered: fill ratio = 6300/32000
  // ~= 0.20 (inside [0.15, 0.95]), aspect = 90/70 ~= 1.29 (inside
  // [0.3, 3.0]) — should land inside the window decision region.
  cv::Mat image = wallImage({200, 160});
  const cv::Rect window_rect(55, 45, 90, 70);
  cv::rectangle(image, window_rect, cv::Scalar(60, 50, 40), cv::FILLED);

  facade_parser::Config config;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);
  const auto grid = singleCellGrid(image.size());

  const auto elements = facade_parser::classifyCells(image, edges, grid, config);

  ASSERT_EQ(elements.size(), 1U);
  EXPECT_EQ(elements[0].type, facade_parser::ElementType::Window);
  EXPECT_EQ(elements[0].bbox_px, window_rect);
  EXPECT_GT(elements[0].confidence, 0.0F);
}

TEST(Classification, BelowMinFillRatioDoesNotClassifyAsWindow) {
  // Tiny 10x8 rectangle in a 200x160 cell: fill ratio ~= 80/32000 ~=
  // 0.0025, well under window_min_fill_ratio (0.15) — exercises the
  // fill-ratio threshold's lower boundary.
  cv::Mat image = wallImage({200, 160});
  cv::rectangle(image, cv::Rect(95, 76, 10, 8), cv::Scalar(60, 50, 40), cv::FILLED);

  facade_parser::Config config;
  config.emit_wall_elements = true;
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);
  const auto grid = singleCellGrid(image.size());

  const auto elements = facade_parser::classifyCells(image, edges, grid, config);

  ASSERT_EQ(elements.size(), 1U);
  EXPECT_EQ(elements[0].type, facade_parser::ElementType::Wall);
}

TEST(Classification, TallNarrowRectangleInLowestRowClassifiesAsDoor) {
  // 90x220 cell; a 50x190 rectangle nearly filling it (fill ratio ~=
  // 9500/19800 ~= 0.48, aspect = 50/190 ~= 0.26 -- just under
  // window_min_aspect (0.3), but the door heuristic uses height/width,
  // i.e. 190/50 = 3.8 >= door_min_height_width_ratio (1.4)). Use a wider
  // door so aspect also clears the window shape gate first.
  cv::Mat image = wallImage({90, 220});
  const cv::Rect door_rect(15, 20, 60, 200);
  cv::rectangle(image, door_rect, cv::Scalar(60, 50, 40), cv::FILLED);

  facade_parser::Config config;
  facade_parser::FacadeGrid grid;
  grid.row_boundaries_px = {0, 0, image.rows};  // two row bands; cell is in row 1 (lowest).
  grid.col_boundaries_px = {{0, image.cols}, {0, image.cols}};
  const facade_parser::EdgeMaps edges = facade_parser::extractEdges(image, config);

  // Row 0 is degenerate (zero height) purely to make row 1 "the lowest
  // floor band" per the door heuristic; classifyCells must tolerate it
  // without dividing by zero (cellRect for an empty band has area 0).
  const auto elements = facade_parser::classifyCells(image, edges, grid, config);

  ASSERT_EQ(elements.size(), 1U);
  EXPECT_EQ(elements[0].row, 1);
  EXPECT_EQ(elements[0].type, facade_parser::ElementType::Door);
}
