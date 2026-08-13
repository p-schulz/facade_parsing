#include "facade_parser/classification.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

#ifdef FACADE_PARSER_HAS_OPENCV_GEOMETRY
#include <opencv2/geometry.hpp>
#endif

namespace facade_parser {

namespace {

struct CellFeatures {
  bool has_blob = false;
  cv::Rect blob_bbox_local;  // relative to the cell crop.
  double fill_ratio = 0.0;   // contour area / cell area.
  double aspect = 0.0;       // bbox width / bbox height.
  double edge_density = 0.0;
  double contrast_stddev = 0.0;
};

// Otsu threshold (inverted: windows are typically the higher-contrast,
// often darker-toned region against a more uniform wall — see Wenzel &
// Förstner and Recky & Leberl, referenced in docs/PLAN.md Stage 4),
// morphological close to merge mullions/sprossen, largest external
// contour, bounding rect fit.
CellFeatures computeCellFeatures(const cv::Mat& bgr_image, const cv::Mat& canny,
                                  const cv::Rect& cell, const Config& config) {
  CellFeatures features;
  if (cell.area() <= 0) {
    return features;
  }

  cv::Mat gray;
  cv::cvtColor(bgr_image(cell), gray, cv::COLOR_BGR2GRAY);

  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(gray, mean, stddev);
  features.contrast_stddev = stddev[0];

  const cv::Mat canny_crop = canny(cell);
  features.edge_density =
      static_cast<double>(cv::countNonZero(canny_crop)) / static_cast<double>(cell.area());

  cv::Mat binary;
  cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  const int shorter_side = std::min(cell.width, cell.height);
  const int kernel_size =
      std::max(3, static_cast<int>(shorter_side * config.otsu_close_kernel_frac) | 1);
  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
  cv::Mat closed;
  cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return features;
  }

  std::size_t largest_idx = 0;
  double largest_area = 0.0;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area > largest_area) {
      largest_area = area;
      largest_idx = i;
    }
  }
  if (largest_area <= 0.0) {
    return features;
  }

  features.has_blob = true;
  features.blob_bbox_local = cv::boundingRect(contours[largest_idx]);
  features.fill_ratio = largest_area / static_cast<double>(cell.area());
  features.aspect = static_cast<double>(features.blob_bbox_local.width) /
                     std::max(1, features.blob_bbox_local.height);
  return features;
}

// Weighted combination of the three features into a single [0,1] score —
// documented as a heuristic, not a learned classifier, per the design
// prompt (no ML anywhere in this tool).
double windowLikelihoodScore(const CellFeatures& f, const Config& config) {
  const double fill_mid = 0.5 * (config.window_min_fill_ratio + config.window_max_fill_ratio);
  const double fill_half_range = 0.5 * (config.window_max_fill_ratio - config.window_min_fill_ratio);
  const double fill_score =
      fill_half_range > 0.0
          ? std::clamp(1.0 - std::abs(f.fill_ratio - fill_mid) / fill_half_range, 0.0, 1.0)
          : 0.0;

  const double edge_score = std::clamp(f.edge_density / 0.15, 0.0, 1.0);
  const double contrast_score = std::clamp(f.contrast_stddev / 64.0, 0.0, 1.0);

  return std::clamp(0.5 * fill_score + 0.3 * edge_score + 0.2 * contrast_score, 0.0, 1.0);
}

bool isWindowShaped(const CellFeatures& f, const Config& config) {
  return f.has_blob && f.fill_ratio >= config.window_min_fill_ratio &&
         f.fill_ratio <= config.window_max_fill_ratio && f.aspect >= config.window_min_aspect &&
         f.aspect <= config.window_max_aspect;
}

}  // namespace

std::vector<Element> classifyCells(const cv::Mat& bgr_image, const EdgeMaps& edges,
                                    const FacadeGrid& grid, const Config& config) {
  std::vector<Element> elements;

  const int last_row = grid.rows() - 1;
  for (int row = 0; row < grid.rows(); ++row) {
    for (int col = 0; col < grid.cols(row); ++col) {
      const cv::Rect cell = grid.cellRect(row, col);
      const CellFeatures features = computeCellFeatures(bgr_image, edges.canny, cell, config);
      const double score = windowLikelihoodScore(features, config);

      ElementType type = ElementType::Wall;
      double confidence = std::clamp(1.0 - score, 0.0, 1.0);
      cv::Rect bbox = cell;

      if (isWindowShaped(features, config)) {
        type = ElementType::Window;
        confidence = score;
        bbox = features.blob_bbox_local + cell.tl();

        // Door heuristic (docs/PLAN.md Stage 4): lowest floor band (whose
        // cell bottom is, by construction, the facade bbox's bottom
        // edge) and a tall aspect ratio.
        const double height_width_ratio =
            static_cast<double>(bbox.height) / std::max(1, bbox.width);
        if (row == last_row && height_width_ratio >= config.door_min_height_width_ratio) {
          type = ElementType::Door;
        }
      }

      if (type == ElementType::Wall && !config.emit_wall_elements) {
        continue;
      }

      Element element;
      element.type = type;
      element.confidence = static_cast<float>(confidence);
      element.row = row;
      element.col = col;
      element.bbox_px = bbox;
      elements.push_back(element);
    }
  }

  return elements;
}

}  // namespace facade_parser
