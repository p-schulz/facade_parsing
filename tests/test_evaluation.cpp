#include <gtest/gtest.h>

#include "facade_parser/evaluation.hpp"

namespace {

facade_parser::Element makeDetected(facade_parser::ElementType type, cv::Rect bbox) {
  facade_parser::Element e;
  e.type = type;
  e.bbox_px = bbox;
  return e;
}

facade_parser::GroundTruthElement makeGt(facade_parser::ElementType type, cv::Rect bbox) {
  return {type, bbox};
}

}  // namespace

TEST(Evaluation, PerfectMatchYieldsF1OfOneAndIouOfOne) {
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(100, 100, 50, 60)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(100, 100, 50, 60)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 1);
  EXPECT_EQ(score.false_positives, 0);
  EXPECT_EQ(score.false_negatives, 0);
  EXPECT_DOUBLE_EQ(score.mean_iou, 1.0);
  EXPECT_DOUBLE_EQ(score.f1, 1.0);
}

TEST(Evaluation, NoOverlapYieldsF1OfZero) {
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(500, 500, 50, 50)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 0);
  EXPECT_EQ(score.false_positives, 1);
  EXPECT_EQ(score.false_negatives, 1);
  EXPECT_DOUBLE_EQ(score.f1, 0.0);
}

TEST(Evaluation, OverlapAboveThresholdMatches) {
  // IoU = 8100 / 11900 ~= 0.6807, above the default 0.5 threshold.
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 100, 100)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(10, 10, 100, 100)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 1);
  EXPECT_EQ(score.false_positives, 0);
  EXPECT_NEAR(score.mean_iou, 0.6807, 1e-3);
}

TEST(Evaluation, OverlapBelowThresholdDoesNotMatch) {
  // IoU = 2500 / 17500 ~= 0.1429, below the default 0.5 threshold.
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 100, 100)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(50, 50, 100, 100)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 0);
  EXPECT_EQ(score.false_positives, 1);
  EXPECT_EQ(score.false_negatives, 1);
}

TEST(Evaluation, ExtraDetectionCountsAsFalsePositive) {
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(500, 500, 50, 50)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 1);
  EXPECT_EQ(score.false_positives, 1);
  EXPECT_EQ(score.false_negatives, 0);
}

TEST(Evaluation, MissingDetectionCountsAsFalseNegative) {
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(500, 500, 50, 50)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 1);
  EXPECT_EQ(score.false_positives, 0);
  EXPECT_EQ(score.false_negatives, 1);
}

TEST(Evaluation, DetectedDoorNeverMatchesGroundTruthWindow) {
  facade_parser::FacadeResult detected;
  detected.elements.push_back(
      makeDetected(facade_parser::ElementType::Door, cv::Rect(0, 0, 50, 50)));
  facade_parser::GroundTruthImage gt;
  gt.elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));

  const auto score = facade_parser::scoreImage(detected, gt);

  EXPECT_EQ(score.true_positives, 0);
  EXPECT_EQ(score.false_positives, 1);  // the door
  EXPECT_EQ(score.false_negatives, 1);  // the window
}

TEST(Evaluation, ScoreDatasetSkipsImagesWithNoGroundTruth) {
  std::vector<facade_parser::FacadeResult> detected(2);
  std::vector<facade_parser::GroundTruthImage> gt(2);
  gt[0].elements.push_back(makeGt(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  detected[0].elements.push_back(
      makeDetected(facade_parser::ElementType::Window, cv::Rect(0, 0, 50, 50)));
  // gt[1] and detected[1] left empty (no ground truth for that image).

  const auto result = facade_parser::scoreDataset(detected, gt);

  ASSERT_EQ(result.per_image.size(), 1U);
  EXPECT_DOUBLE_EQ(result.mean_f1, 1.0);
}

TEST(Evaluation, RenderComparisonOverlayPreservesImageSize) {
  facade_parser::FacadeResult detected;
  facade_parser::GroundTruthImage gt;
  facade_parser::ImageScore score;
  const cv::Mat image(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));

  const cv::Mat overlay = facade_parser::renderComparisonOverlay(image, detected, gt, score);

  EXPECT_EQ(overlay.size(), image.size());
}
