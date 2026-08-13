#include "facade_parser/symmetry.hpp"

#include <cmath>
#include <map>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace facade_parser {

namespace {

// Locates the column index c such that col_boundaries[c] <= x <
// col_boundaries[c+1]; -1 if x falls outside every band.
int findColumnContaining(const std::vector<int>& col_boundaries, double x) {
  for (std::size_t c = 0; c + 1 < col_boundaries.size(); ++c) {
    if (x >= col_boundaries[c] && x < col_boundaries[c + 1]) {
      return static_cast<int>(c);
    }
  }
  return -1;
}

// Discount applied to a mirror-inferred detection's confidence: it was
// never directly observed, only inferred from its mirrored counterpart,
// so it should never outrank an actual Stage 4 detection of equal
// underlying confidence.
constexpr float kMirrorInferenceConfidenceDiscount = 0.75F;

}  // namespace

SymmetryCheckResult checkSymmetry(const cv::Mat& bgr_image, const cv::Rect& facade_bbox,
                                   const FacadeGrid& grid, const std::vector<Element>& elements,
                                   const Config& config) {
  SymmetryCheckResult result;
  result.mirror_axis_x_px = facade_bbox.x + facade_bbox.width / 2.0;

  if (facade_bbox.width < 4 || facade_bbox.height < 4) {
    return result;
  }

  // cv::phaseCorrelate (Kuglin & Hines, 1975) between the facade crop and
  // its horizontal flip: for a facade with a mirror axis at A, the
  // flipped image is a translated copy of the original by
  // delta = 2A - (W - 1), so the sub-pixel shift phaseCorrelate reports
  // between them recovers A directly. See docs/PLAN.md, Stage 6.
  cv::Mat gray;
  cv::cvtColor(bgr_image(facade_bbox), gray, cv::COLOR_BGR2GRAY);
  cv::Mat gray32;
  gray.convertTo(gray32, CV_32F);

  cv::Mat flipped32;
  cv::flip(gray32, flipped32, 1);

  cv::Mat hanning;
  cv::createHanningWindow(hanning, gray32.size(), CV_32F);

  const cv::Point2d shift = cv::phaseCorrelate(gray32, flipped32, hanning);
  const double width = facade_bbox.width;
  const double axis_local = (width - 1.0) / 2.0 + shift.x / 2.0;
  result.mirror_axis_x_px = facade_bbox.x + axis_local;

  if (!config.enable_symmetry_check) {
    return result;
  }

  std::map<std::pair<int, int>, const Element*> presence;
  for (const auto& e : elements) {
    if (e.type == ElementType::Window || e.type == ElementType::Door) {
      presence[{e.row, e.col}] = &e;
    }
  }

  for (int row = 0; row < grid.rows(); ++row) {
    const auto& col_boundaries = grid.col_boundaries_px[static_cast<std::size_t>(row)];
    for (int col = 0; col < grid.cols(row); ++col) {
      if (presence.count({row, col}) > 0) {
        continue;  // already detected, nothing to infer.
      }
      const cv::Rect cell = grid.cellRect(row, col);
      const double cell_center_x = cell.x + cell.width / 2.0;
      const double mirrored_center_x = 2.0 * result.mirror_axis_x_px - cell_center_x;

      const int mirror_col = findColumnContaining(col_boundaries, mirrored_center_x);
      if (mirror_col < 0 || mirror_col == col) {
        continue;
      }
      const auto it = presence.find({row, mirror_col});
      if (it == presence.end()) {
        continue;  // mirror partner also undetected; nothing to infer from.
      }

      const Element& partner = *it->second;
      const double mirrored_x = 2.0 * result.mirror_axis_x_px - (partner.bbox_px.x + partner.bbox_px.width);
      SymmetryInference inference;
      inference.row = row;
      inference.col = col;
      inference.suggested_bbox_px =
          cv::Rect(static_cast<int>(std::lround(mirrored_x)), partner.bbox_px.y,
                    partner.bbox_px.width, partner.bbox_px.height);
      inference.suggested_type = partner.type;
      inference.mirror_confidence = partner.confidence * kMirrorInferenceConfidenceDiscount;
      result.inferences.push_back(inference);
    }
  }

  return result;
}

}  // namespace facade_parser
