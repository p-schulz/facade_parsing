#include <gtest/gtest.h>

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

TEST(IoJson, RenderDebugOverlayPreservesImageSize) {
  facade_parser::FacadeResult result;
  result.grid.row_boundaries_px = {0, 100};
  result.grid.col_boundaries_px = {{0, 100}};
  const cv::Mat image(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));

  const cv::Mat overlay = facade_parser::renderDebugOverlay(image, result);

  EXPECT_EQ(overlay.size(), image.size());
}
