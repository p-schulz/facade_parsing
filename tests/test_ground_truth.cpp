#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include "facade_parser/ground_truth.hpp"

TEST(GroundTruth, SaveAndLoadRoundTrips) {
  facade_parser::GroundTruthImage gt;
  gt.image_path = "facade_01.png";
  gt.image_size_px = cv::Size(1024, 768);
  gt.elements.push_back({facade_parser::ElementType::Window, cv::Rect(100, 200, 80, 120)});
  gt.elements.push_back({facade_parser::ElementType::Door, cv::Rect(300, 500, 60, 150)});

  const std::string path = "/tmp/facade_parser_test_gt.gt.json";
  ASSERT_TRUE(facade_parser::saveGroundTruth(gt, path));

  const auto loaded = facade_parser::loadGroundTruth(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->image_path, "facade_01.png");
  EXPECT_EQ(loaded->image_size_px, cv::Size(1024, 768));
  ASSERT_EQ(loaded->elements.size(), 2U);
  EXPECT_EQ(loaded->elements[0].type, facade_parser::ElementType::Window);
  EXPECT_EQ(loaded->elements[0].bbox_px, cv::Rect(100, 200, 80, 120));
  EXPECT_EQ(loaded->elements[1].type, facade_parser::ElementType::Door);
  EXPECT_EQ(loaded->elements[1].bbox_px, cv::Rect(300, 500, 60, 150));

  std::remove(path.c_str());
}

TEST(GroundTruth, LoadReturnsNulloptForMissingFile) {
  const auto loaded =
      facade_parser::loadGroundTruth("/tmp/facade_parser_test_gt_does_not_exist.gt.json");
  EXPECT_FALSE(loaded.has_value());
}

TEST(GroundTruth, LoadSkipsMalformedElementsWithoutFailingWholeFile) {
  const std::string path = "/tmp/facade_parser_test_gt_malformed.gt.json";
  {
    std::ofstream out(path);
    out << R"({"image_path": "x.png", "image_size_px": [100, 100], "elements": [)"
        << R"({"type": "window", "bbox_px": [1, 2, 3, 4]}, )"
        << R"({"type": "not_a_type", "bbox_px": [1, 2, 3, 4]}, )"
        << R"({"type": "door"})"  // missing bbox_px
        << R"(]})";
  }

  const auto loaded = facade_parser::loadGroundTruth(path);
  ASSERT_TRUE(loaded.has_value());
  ASSERT_EQ(loaded->elements.size(), 1U);
  EXPECT_EQ(loaded->elements[0].type, facade_parser::ElementType::Window);

  std::remove(path.c_str());
}
