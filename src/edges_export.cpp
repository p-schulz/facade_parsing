#include "facade_parser/edges_export.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace facade_parser {

DepthHint::Result SobelMagnitudeDepthHint::estimate(const cv::Mat& gradient_magnitude,
                                                      const cv::Rect& region_px) const {
  const cv::Rect clamped = region_px & cv::Rect(0, 0, gradient_magnitude.cols,
                                                 gradient_magnitude.rows);
  DepthHint::Result result;
  if (clamped.area() <= 0) {
    return result;
  }
  const cv::Scalar mean = cv::mean(gradient_magnitude(clamped));
  // Clamp to [0, 1]; 255.0 is an arbitrary reference scale (Sobel output
  // is unbounded) chosen so typical 8-bit-source gradients land near
  // mid-range — this is a coarse proxy, not a calibrated measurement,
  // hence low_confidence stays true. See docs/PLAN.md, Stage 7.
  result.value = static_cast<float>(std::min(1.0, mean[0] / 255.0));
  result.low_confidence = true;
  return result;
}

namespace {

cv::Rect segmentBoundingBox(const LineSegment& seg) {
  const int min_x = static_cast<int>(std::floor(std::min(seg.p1.x, seg.p2.x)));
  const int max_x = static_cast<int>(std::ceil(std::max(seg.p1.x, seg.p2.x)));
  const int min_y = static_cast<int>(std::floor(std::min(seg.p1.y, seg.p2.y)));
  const int max_y = static_cast<int>(std::ceil(std::max(seg.p1.y, seg.p2.y)));
  return cv::Rect(min_x, min_y, std::max(1, max_x - min_x), std::max(1, max_y - min_y));
}

// See docs/PLAN.md Stage 7 for why this is containment-after-dilation
// rather than an IoU test.
bool claimedByWindowOrDoor(const cv::Rect& seg_bbox, const std::vector<Element>& cell_elements,
                            double margin_px) {
  const int margin = static_cast<int>(margin_px);
  for (const auto& el : cell_elements) {
    if (el.type != ElementType::Window && el.type != ElementType::Door) {
      continue;
    }
    cv::Rect dilated = el.bbox_px;
    dilated.x -= margin;
    dilated.y -= margin;
    dilated.width += 2 * margin;
    dilated.height += 2 * margin;
    if ((dilated & seg_bbox) == seg_bbox) {
      return true;
    }
  }
  return false;
}

void appendEdgeElements(const std::vector<LineSegment>& segments, bool is_horizontal,
                         const EdgeMaps& edges, const std::vector<Element>& cell_elements,
                         const DepthHint& depth_hint, const Config& config,
                         std::vector<Element>* out) {
  const double cornice_band_px = 0.05 * edges.gradient_magnitude.rows;

  for (const auto& seg : segments) {
    if (seg.length_px < config.min_edge_length_px) {
      continue;
    }
    const cv::Rect seg_bbox = segmentBoundingBox(seg);
    if (claimedByWindowOrDoor(seg_bbox, cell_elements, config.edge_claim_margin_px)) {
      continue;
    }

    Element element;
    element.type = ElementType::Edge;
    element.polyline_px = {cv::Point(cvRound(seg.p1.x), cvRound(seg.p1.y)),
                            cv::Point(cvRound(seg.p2.x), cvRound(seg.p2.y))};
    // Longer segments are more likely a genuine architectural feature
    // rather than a fragment from noise/texture; saturates at 2x the
    // minimum length.
    element.confidence =
        static_cast<float>(std::clamp(seg.length_px / (2.0 * config.min_edge_length_px), 0.0, 1.0));

    if (is_horizontal) {
      element.edge_kind =
          seg_bbox.y < cornice_band_px ? EdgeKind::Cornice : EdgeKind::FloorSeparator;
    } else {
      element.edge_kind = EdgeKind::Pier;
    }

    const DepthHint::Result depth = depth_hint.estimate(edges.gradient_magnitude, seg_bbox);
    element.depth_hint_value = depth.value;

    out->push_back(std::move(element));
  }
}

}  // namespace

std::vector<Element> exportEdgeElements(const EdgeMaps& edges,
                                         const std::vector<Element>& cell_elements,
                                         const DepthHint& depth_hint, const Config& config) {
  std::vector<Element> result;
  appendEdgeElements(edges.horizontal_segments, /*is_horizontal=*/true, edges, cell_elements,
                      depth_hint, config, &result);
  appendEdgeElements(edges.vertical_segments, /*is_horizontal=*/false, edges, cell_elements,
                      depth_hint, config, &result);
  return result;
}

}  // namespace facade_parser
