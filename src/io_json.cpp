#include "facade_parser/io_json.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace facade_parser {

namespace {

const char* toString(ElementType type) {
  switch (type) {
    case ElementType::Window:
      return "window";
    case ElementType::Door:
      return "door";
    case ElementType::Wall:
      return "wall";
    case ElementType::Edge:
      return "edge";
  }
  return "unknown";
}

const char* toString(EdgeKind kind) {
  switch (kind) {
    case EdgeKind::FloorSeparator:
      return "floor_separator";
    case EdgeKind::Pier:
      return "pier";
    case EdgeKind::Cornice:
      return "cornice";
    case EdgeKind::Unclassified:
      return "unclassified";
  }
  return "unclassified";
}

cv::Scalar colorFor(ElementType type) {
  switch (type) {
    case ElementType::Window:
      return {0, 165, 255};   // orange (BGR: B=0, G=165, R=255)
    case ElementType::Door:
      return {0, 0, 255};     // red
    case ElementType::Wall:
      return {180, 180, 180}; // gray
    case ElementType::Edge:
      return {0, 255, 0};     // green
  }
  return {255, 255, 255};
}

}  // namespace

nlohmann::json toJson(const FacadeResult& result) {
  nlohmann::json j;
  j["source_image"] = result.source_image;
  j["image_size_px"] = {result.image_size_px.width, result.image_size_px.height};

  j["grid"]["row_boundaries_px"] = result.grid.row_boundaries_px;
  j["grid"]["col_boundaries_px"] = result.grid.col_boundaries_px;

  nlohmann::json elements = nlohmann::json::array();
  for (const auto& e : result.elements) {
    nlohmann::json je;
    je["type"] = toString(e.type);
    je["confidence"] = e.confidence;

    if (e.type == ElementType::Edge) {
      nlohmann::json poly_px = nlohmann::json::array();
      nlohmann::json poly_norm = nlohmann::json::array();
      for (const auto& p : e.polyline_px) {
        poly_px.push_back({p.x, p.y});
      }
      for (const auto& p : e.polyline_norm) {
        poly_norm.push_back({p.x, p.y});
      }
      je["polyline_px"] = poly_px;
      je["polyline_norm"] = poly_norm;
      je["kind"] = toString(e.edge_kind);
      if (e.depth_hint_value.has_value()) {
        je["depth_hint"] = {{"value", *e.depth_hint_value}, {"confidence", "low_confidence"}};
      }
    } else {
      je["row"] = e.row;
      je["col"] = e.col;
      je["bbox_px"] = {e.bbox_px.x, e.bbox_px.y, e.bbox_px.width, e.bbox_px.height};
      je["bbox_norm"] = {e.bbox_norm.x, e.bbox_norm.y, e.bbox_norm.width, e.bbox_norm.height};
    }
    elements.push_back(je);
  }
  j["elements"] = elements;

  nlohmann::json symmetry_inferences = nlohmann::json::array();
  for (const auto& s : result.symmetry_inferences) {
    nlohmann::json js;
    js["row"] = s.row;
    js["col"] = s.col;
    js["suggested_type"] = toString(s.suggested_type);
    js["suggested_bbox_px"] = {s.suggested_bbox_px.x, s.suggested_bbox_px.y,
                                s.suggested_bbox_px.width, s.suggested_bbox_px.height};
    js["suggested_bbox_norm"] = {
        s.suggested_bbox_px.x / static_cast<double>(result.image_size_px.width),
        s.suggested_bbox_px.y / static_cast<double>(result.image_size_px.height),
        s.suggested_bbox_px.width / static_cast<double>(result.image_size_px.width),
        s.suggested_bbox_px.height / static_cast<double>(result.image_size_px.height)};
    js["mirror_confidence"] = s.mirror_confidence;
    symmetry_inferences.push_back(js);
  }
  j["symmetry_inferences"] = symmetry_inferences;

  return j;
}

bool writeResultJson(const FacadeResult& result, const std::string& path) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << toJson(result).dump(2);
  return out.good();
}

cv::Mat renderDebugOverlay(const cv::Mat& bgr_image, const FacadeResult& result) {
  cv::Mat overlay = bgr_image.clone();

  // Grid boundaries.
  for (int y : result.grid.row_boundaries_px) {
    cv::line(overlay, {0, y}, {overlay.cols, y}, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
  }
  for (const auto& cols_for_row : result.grid.col_boundaries_px) {
    for (int x : cols_for_row) {
      cv::line(overlay, {x, 0}, {x, overlay.rows}, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  // Classified cells and edges.
  for (const auto& e : result.elements) {
    const cv::Scalar color = colorFor(e.type);
    if (e.type == ElementType::Edge) {
      if (e.polyline_px.size() >= 2) {
        cv::polylines(overlay, e.polyline_px, /*isClosed=*/false, color, 2, cv::LINE_AA);
      }
    } else {
      cv::rectangle(overlay, e.bbox_px, color, 2, cv::LINE_AA);
    }
  }

  // Stage 6 symmetry inferences: thin magenta boxes, distinct from a
  // directly-observed Stage 4 detection's solid color-coded box.
  for (const auto& s : result.symmetry_inferences) {
    cv::rectangle(overlay, s.suggested_bbox_px, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
  }

  return overlay;
}

bool writeDebugOverlay(const cv::Mat& bgr_image, const FacadeResult& result,
                        const std::string& path) {
  return cv::imwrite(path, renderDebugOverlay(bgr_image, result));
}

}  // namespace facade_parser
