#include <gtest/gtest.h>

#include <cstdio>
#include <nlohmann/json.hpp>

#include "facade_parser/io_json.hpp"

TEST(IoJson, ToJsonRoundTripsGridAndWindowElement) {
  facade_parser::FacadeResult result;
  result.source_image = "facade_01.png";
  result.image_size_px = {1024, 768};
  result.grid.row_boundaries_px = {0, 210, 430};
  result.grid.col_boundaries_px = {{0, 180, 360}, {0, 200, 400}};

  facade_parser::Element window;
  window.type = facade_parser::ElementType::Window;
  window.row = 1;
  window.col = 0;
  window.bbox_px = cv::Rect(0, 210, 200, 220);
  window.bbox_norm = cv::Rect2f(0.0F, 0.2734F, 0.1953F, 0.2865F);
  window.confidence = 0.87F;
  result.elements.push_back(window);

  const nlohmann::json j = facade_parser::toJson(result);

  EXPECT_EQ(j.at("source_image").get<std::string>(), "facade_01.png");
  EXPECT_EQ(j.at("image_size_px"), (nlohmann::json{1024, 768}));
  EXPECT_EQ(j.at("grid").at("row_boundaries_px"), (nlohmann::json{0, 210, 430}));
  ASSERT_EQ(j.at("elements").size(), 1U);
  EXPECT_EQ(j.at("elements")[0].at("type").get<std::string>(), "window");
  EXPECT_EQ(j.at("elements")[0].at("row").get<int>(), 1);
}

TEST(IoJson, ToJsonSerializesSymmetryInferences) {
  facade_parser::FacadeResult result;
  result.source_image = "facade_01.png";
  result.image_size_px = {1024, 768};
  result.grid.row_boundaries_px = {0, 768};
  result.grid.col_boundaries_px = {{0, 1024}};

  facade_parser::SymmetryInference inference;
  inference.row = 0;
  inference.col = 0;
  inference.suggested_bbox_px = cv::Rect(100, 200, 50, 60);
  inference.suggested_type = facade_parser::ElementType::Window;
  inference.mirror_confidence = 0.6F;
  result.symmetry_inferences.push_back(inference);

  const nlohmann::json j = facade_parser::toJson(result);

  ASSERT_EQ(j.at("symmetry_inferences").size(), 1U);
  const auto& js = j.at("symmetry_inferences")[0];
  EXPECT_EQ(js.at("row").get<int>(), 0);
  EXPECT_EQ(js.at("col").get<int>(), 0);
  EXPECT_EQ(js.at("suggested_type").get<std::string>(), "window");
  EXPECT_EQ(js.at("suggested_bbox_px"), (nlohmann::json{100, 200, 50, 60}));
  EXPECT_FLOAT_EQ(js.at("mirror_confidence").get<float>(), 0.6F);
}

TEST(IoJson, ConfigRoundTripsThroughJson) {
  facade_parser::Config config;
  config.canny_low = 30.0;
  config.canny_high = 90.0;
  config.hough_threshold_votes = 55;
  config.min_cell_size_px = 12;
  config.window_min_fill_ratio = 0.2;
  config.enable_lattice_refine = false;
  config.emit_wall_elements = true;

  const nlohmann::json j = facade_parser::configToJson(config);
  const facade_parser::Config round_tripped = facade_parser::configFromJson(j);

  EXPECT_DOUBLE_EQ(round_tripped.canny_low, 30.0);
  EXPECT_DOUBLE_EQ(round_tripped.canny_high, 90.0);
  EXPECT_EQ(round_tripped.hough_threshold_votes, 55);
  EXPECT_EQ(round_tripped.min_cell_size_px, 12);
  EXPECT_DOUBLE_EQ(round_tripped.window_min_fill_ratio, 0.2);
  EXPECT_FALSE(round_tripped.enable_lattice_refine);
  EXPECT_TRUE(round_tripped.emit_wall_elements);
}

// A config file missing some fields (e.g. saved by an older build with
// fewer Config members) should keep Config{}'s own default for whatever
// is absent, not zero-initialize it.
TEST(IoJson, ConfigFromJsonKeepsDefaultsForMissingFields) {
  nlohmann::json j;
  j["canny_low"] = 10.0;  // Only one field present.

  const facade_parser::Config config = facade_parser::configFromJson(j);
  const facade_parser::Config defaults;

  EXPECT_DOUBLE_EQ(config.canny_low, 10.0);
  EXPECT_DOUBLE_EQ(config.canny_high, defaults.canny_high);
  EXPECT_EQ(config.min_cell_size_px, defaults.min_cell_size_px);
  EXPECT_EQ(config.enable_symmetry_check, defaults.enable_symmetry_check);
}

TEST(IoJson, WriteAndReadConfigJsonRoundTrips) {
  facade_parser::Config config;
  config.door_min_height_width_ratio = 2.5;

  const std::string path = "/tmp/facade_parser_test_config.json";
  ASSERT_TRUE(facade_parser::writeConfigJson(config, path));

  const auto loaded = facade_parser::readConfigJson(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_DOUBLE_EQ(loaded->door_min_height_width_ratio, 2.5);

  std::remove(path.c_str());
}

TEST(IoJson, ReadConfigJsonReturnsNulloptForMissingFile) {
  const auto loaded = facade_parser::readConfigJson("/tmp/facade_parser_test_does_not_exist.json");
  EXPECT_FALSE(loaded.has_value());
}

TEST(IoJson, RenderDebugOverlayPreservesImageSize) {
  facade_parser::FacadeResult result;
  result.grid.row_boundaries_px = {0, 100};
  result.grid.col_boundaries_px = {{0, 100}};
  const cv::Mat image(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));

  const cv::Mat overlay = facade_parser::renderDebugOverlay(image, result);

  EXPECT_EQ(overlay.size(), image.size());
}
