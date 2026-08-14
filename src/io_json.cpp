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

// Field-by-field, explicit (not NLOHMANN_DEFINE_TYPE_INTRUSIVE) so
// `types.hpp` stays free of an nlohmann dependency — same separation
// already used for FacadeResult's toJson() above.
nlohmann::json configToJson(const Config& c) {
  nlohmann::json j;
  j["canny_low"] = c.canny_low;
  j["canny_high"] = c.canny_high;
  j["hough_min_line_length_px"] = c.hough_min_line_length_px;
  j["hough_max_line_gap_px"] = c.hough_max_line_gap_px;
  j["hough_threshold_votes"] = c.hough_threshold_votes;
  j["line_angle_tolerance_deg"] = c.line_angle_tolerance_deg;

  j["profile_smoothing_sigma_px"] = c.profile_smoothing_sigma_px;
  j["periodicity_min_score"] = c.periodicity_min_score;
  j["periodicity_agreement_tol_px"] = c.periodicity_agreement_tol_px;
  j["direct_peak_low_activity_frac"] = c.direct_peak_low_activity_frac;

  j["min_cell_size_px"] = c.min_cell_size_px;
  j["min_segment_width_frac_of_median"] = c.min_segment_width_frac_of_median;

  j["otsu_close_kernel_frac"] = c.otsu_close_kernel_frac;
  j["window_min_fill_ratio"] = c.window_min_fill_ratio;
  j["window_max_fill_ratio"] = c.window_max_fill_ratio;
  j["window_min_aspect"] = c.window_min_aspect;
  j["window_max_aspect"] = c.window_max_aspect;
  j["door_min_height_width_ratio"] = c.door_min_height_width_ratio;

  j["enable_lattice_refine"] = c.enable_lattice_refine;
  j["lattice_refine_window_frac"] = c.lattice_refine_window_frac;

  j["enable_symmetry_check"] = c.enable_symmetry_check;

  j["min_edge_length_px"] = c.min_edge_length_px;
  j["edge_claim_margin_px"] = c.edge_claim_margin_px;

  j["emit_wall_elements"] = c.emit_wall_elements;
  return j;
}

namespace {

template <typename T>
void readIfPresent(const nlohmann::json& j, const char* key, T* out) {
  if (j.contains(key)) {
    *out = j.at(key).get<T>();
  }
}

}  // namespace

Config configFromJson(const nlohmann::json& j) {
  Config c;
  readIfPresent(j, "canny_low", &c.canny_low);
  readIfPresent(j, "canny_high", &c.canny_high);
  readIfPresent(j, "hough_min_line_length_px", &c.hough_min_line_length_px);
  readIfPresent(j, "hough_max_line_gap_px", &c.hough_max_line_gap_px);
  readIfPresent(j, "hough_threshold_votes", &c.hough_threshold_votes);
  readIfPresent(j, "line_angle_tolerance_deg", &c.line_angle_tolerance_deg);

  readIfPresent(j, "profile_smoothing_sigma_px", &c.profile_smoothing_sigma_px);
  readIfPresent(j, "periodicity_min_score", &c.periodicity_min_score);
  readIfPresent(j, "periodicity_agreement_tol_px", &c.periodicity_agreement_tol_px);
  readIfPresent(j, "direct_peak_low_activity_frac", &c.direct_peak_low_activity_frac);

  readIfPresent(j, "min_cell_size_px", &c.min_cell_size_px);
  readIfPresent(j, "min_segment_width_frac_of_median", &c.min_segment_width_frac_of_median);

  readIfPresent(j, "otsu_close_kernel_frac", &c.otsu_close_kernel_frac);
  readIfPresent(j, "window_min_fill_ratio", &c.window_min_fill_ratio);
  readIfPresent(j, "window_max_fill_ratio", &c.window_max_fill_ratio);
  readIfPresent(j, "window_min_aspect", &c.window_min_aspect);
  readIfPresent(j, "window_max_aspect", &c.window_max_aspect);
  readIfPresent(j, "door_min_height_width_ratio", &c.door_min_height_width_ratio);

  readIfPresent(j, "enable_lattice_refine", &c.enable_lattice_refine);
  readIfPresent(j, "lattice_refine_window_frac", &c.lattice_refine_window_frac);

  readIfPresent(j, "enable_symmetry_check", &c.enable_symmetry_check);

  readIfPresent(j, "min_edge_length_px", &c.min_edge_length_px);
  readIfPresent(j, "edge_claim_margin_px", &c.edge_claim_margin_px);

  readIfPresent(j, "emit_wall_elements", &c.emit_wall_elements);
  return c;
}

bool writeConfigJson(const Config& config, const std::string& path) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << configToJson(config).dump(2);
  return out.good();
}

std::optional<Config> readConfigJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  nlohmann::json j;
  try {
    in >> j;
  } catch (const nlohmann::json::parse_error&) {
    return std::nullopt;
  }
  return configFromJson(j);
}

}  // namespace facade_parser
