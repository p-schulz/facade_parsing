#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "facade_parser/rectification.hpp"

TEST(Rectification, ChooseTargetSizeMatchesQuadBoundingBox) {
  facade_parser::Quad quad;
  quad.top_left = {0.0F, 0.0F};
  quad.top_right = {200.0F, 0.0F};
  quad.bottom_left = {0.0F, 100.0F};
  quad.bottom_right = {200.0F, 100.0F};

  const cv::Size size = facade_parser::chooseTargetSize(quad);

  EXPECT_EQ(size, cv::Size(200, 100));
}

TEST(Rectification, ChooseTargetSizeUsesTheLongerOfEachOppositeEdgePair) {
  // top_width = 150 (top edge), bottom_width = 200 (bottom edge) ->
  // width should pick 200. left_height = 90 (left edge, vertical),
  // right_height = hypot(200-150, 90-0) = hypot(50, 90) ~= 102.96 (right
  // edge is diagonal) -> height should pick that, not the shorter left.
  facade_parser::Quad quad;
  quad.top_left = {0.0F, 0.0F};
  quad.top_right = {150.0F, 0.0F};
  quad.bottom_left = {0.0F, 90.0F};
  quad.bottom_right = {200.0F, 90.0F};

  const cv::Size size = facade_parser::chooseTargetSize(quad);

  EXPECT_EQ(size.width, 200);
  EXPECT_EQ(size.height, 102);
}

TEST(Rectification, RectifyProducesExactRequestedSize) {
  const cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
  facade_parser::Quad quad;
  quad.top_left = {80.0F, 50.0F};
  quad.top_right = {350.0F, 90.0F};
  quad.bottom_left = {40.0F, 350.0F};
  quad.bottom_right = {380.0F, 320.0F};

  const cv::Mat rectified = facade_parser::rectify(image, quad, cv::Size(300, 250));

  EXPECT_EQ(rectified.size(), cv::Size(300, 250));
}

// Ground-truth-backed per the project's usual testing style: draws a
// solid-filled quad at known (skewed) corners, rectifies it, and
// confirms the output is (almost entirely) filled with that color end
// to end — i.e. the whole quad interior lands inside the axis-aligned
// output without being clipped or offset, which is what "the warped
// output is an axis-aligned rectangle of the expected size" means in
// practice for a homography warp.
TEST(Rectification, RectifyMapsSkewedQuadContentToFillTheFullOutput) {
  cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
  facade_parser::Quad quad;
  quad.top_left = {80.0F, 50.0F};
  quad.top_right = {350.0F, 90.0F};
  quad.bottom_left = {40.0F, 350.0F};
  quad.bottom_right = {380.0F, 320.0F};

  const std::vector<cv::Point> poly = {
      cv::Point(quad.top_left), cv::Point(quad.top_right), cv::Point(quad.bottom_right),
      cv::Point(quad.bottom_left)};
  cv::fillConvexPoly(image, poly, cv::Scalar(255, 255, 255));

  const cv::Size target(300, 300);
  const cv::Mat rectified = facade_parser::rectify(image, quad, target);

  ASSERT_EQ(rectified.size(), target);
  cv::Mat gray;
  cv::cvtColor(rectified, gray, cv::COLOR_BGR2GRAY);
  const int white_pixels = cv::countNonZero(gray > 200);
  const double white_fraction = static_cast<double>(white_pixels) / gray.total();
  EXPECT_GT(white_fraction, 0.95);
}

TEST(Rectification, RectifyConvenienceOverloadMatchesChooseTargetSize) {
  const cv::Mat image(400, 400, CV_8UC3, cv::Scalar(128, 128, 128));
  facade_parser::Quad quad;
  quad.top_left = {0.0F, 0.0F};
  quad.top_right = {200.0F, 0.0F};
  quad.bottom_left = {0.0F, 100.0F};
  quad.bottom_right = {200.0F, 100.0F};

  const cv::Mat rectified = facade_parser::rectify(image, quad);

  EXPECT_EQ(rectified.size(), facade_parser::chooseTargetSize(quad));
}

TEST(Rectification, ProposeCornersFallsBackToInsetRectangleWhenNoLinesFound) {
  const cv::Mat blank(400, 300, CV_8UC3, cv::Scalar(200, 200, 200));

  const facade_parser::Quad quad = facade_parser::proposeCorners(blank, 20.0, 0.10);

  // 10% of 300x400: 30px horizontal margin, 40px vertical margin.
  EXPECT_NEAR(quad.top_left.x, 30.0F, 1.0F);
  EXPECT_NEAR(quad.top_left.y, 40.0F, 1.0F);
  EXPECT_NEAR(quad.top_right.x, 270.0F, 1.0F);
  EXPECT_NEAR(quad.top_right.y, 40.0F, 1.0F);
  EXPECT_NEAR(quad.bottom_left.x, 30.0F, 1.0F);
  EXPECT_NEAR(quad.bottom_left.y, 360.0F, 1.0F);
  EXPECT_NEAR(quad.bottom_right.x, 270.0F, 1.0F);
  EXPECT_NEAR(quad.bottom_right.y, 360.0F, 1.0F);
}

TEST(Rectification, ProposeCornersFindsCornersOfADrawnRectangle) {
  cv::Mat image(500, 700, CV_8UC3, cv::Scalar(220, 220, 220));
  const cv::Rect r(80, 60, 500, 380);  // tl=(80,60) br=(580,440).
  cv::rectangle(image, r, cv::Scalar(20, 20, 20), 4);

  const facade_parser::Quad quad = facade_parser::proposeCorners(image);

  constexpr float kTolerancePx = 5.0F;
  EXPECT_NEAR(quad.top_left.x, 80.0F, kTolerancePx);
  EXPECT_NEAR(quad.top_left.y, 60.0F, kTolerancePx);
  EXPECT_NEAR(quad.top_right.x, 580.0F, kTolerancePx);
  EXPECT_NEAR(quad.top_right.y, 60.0F, kTolerancePx);
  EXPECT_NEAR(quad.bottom_left.x, 80.0F, kTolerancePx);
  EXPECT_NEAR(quad.bottom_left.y, 440.0F, kTolerancePx);
  EXPECT_NEAR(quad.bottom_right.x, 580.0F, kTolerancePx);
  EXPECT_NEAR(quad.bottom_right.y, 440.0F, kTolerancePx);
}
