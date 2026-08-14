#include "facade_parser/rectification.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <opencv2/imgproc.hpp>

#ifdef FACADE_PARSER_HAS_OPENCV_GEOMETRY
#include <opencv2/geometry.hpp>
#endif

#include "facade_parser/edges.hpp"
#include "facade_parser/types.hpp"

namespace facade_parser {

namespace {

Quad fallbackQuad(cv::Size image_size, double margin_frac) {
  const float mx = static_cast<float>(image_size.width * margin_frac);
  const float my = static_cast<float>(image_size.height * margin_frac);
  Quad q;
  q.top_left = {mx, my};
  q.top_right = {static_cast<float>(image_size.width) - mx, my};
  q.bottom_left = {mx, static_cast<float>(image_size.height) - my};
  q.bottom_right = {static_cast<float>(image_size.width) - mx,
                     static_cast<float>(image_size.height) - my};
  return q;
}

// Longest segment (by LineSegment::length_px) among `segments` whose
// perpendicular-axis midpoint falls inside [region_min, region_max) —
// e.g. for a vertical segment, its horizontal midpoint against the left
// or right third of the image width. This is the "closest to the left/
// right/top/bottom third" heuristic from docs/PLAN.md: segments
// positioned within that outer region, longest (strongest) one wins.
std::optional<LineSegment> longestInRegion(const std::vector<LineSegment>& segments,
                                            bool vertical, float region_min, float region_max) {
  std::optional<LineSegment> best;
  float best_length = -1.0F;
  for (const auto& seg : segments) {
    const float mid = vertical ? (seg.p1.x + seg.p2.x) * 0.5F : (seg.p1.y + seg.p2.y) * 0.5F;
    if (mid < region_min || mid >= region_max) {
      continue;
    }
    if (seg.length_px > best_length) {
      best_length = seg.length_px;
      best = seg;
    }
  }
  return best;
}

// Standard 2D line-line intersection (lines through each segment's own
// two endpoints, extended to infinite lines — the segments themselves
// are typically shorter than the full facade edge, e.g. a partially
// occluded roofline). nullopt for (near-)parallel lines.
std::optional<cv::Point2f> intersectLines(const LineSegment& a, const LineSegment& b) {
  const double x1 = a.p1.x;
  const double y1 = a.p1.y;
  const double x2 = a.p2.x;
  const double y2 = a.p2.y;
  const double x3 = b.p1.x;
  const double y3 = b.p1.y;
  const double x4 = b.p2.x;
  const double y4 = b.p2.y;

  const double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (std::abs(denom) < 1e-6) {
    return std::nullopt;
  }
  const double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  return cv::Point2f(static_cast<float>(x1 + t * (x2 - x1)), static_cast<float>(y1 + t * (y2 - y1)));
}

cv::Point2f clampToImage(cv::Point2f p, cv::Size image_size) {
  p.x = std::clamp(p.x, 0.0F, static_cast<float>(image_size.width));
  p.y = std::clamp(p.y, 0.0F, static_cast<float>(image_size.height));
  return p;
}

}  // namespace

Quad proposeCorners(const cv::Mat& bgr_image, double line_angle_tolerance_deg,
                     double fallback_margin_frac) {
  const cv::Size image_size = bgr_image.size();
  const Quad fallback = fallbackQuad(image_size, fallback_margin_frac);

  // Reuse Stage 1's line detector (edges.hpp) rather than reimplementing
  // Canny + line detection; only the angle tolerance is overridden from
  // its post-rectification default, per the file doc comment.
  Config edge_config;
  edge_config.line_angle_tolerance_deg = line_angle_tolerance_deg;
  const EdgeMaps edges = extractEdges(bgr_image, edge_config);

  const float width = static_cast<float>(image_size.width);
  const float height = static_cast<float>(image_size.height);

  const auto left_edge = longestInRegion(edges.vertical_segments, /*vertical=*/true, 0.0F,
                                          width / 3.0F);
  const auto right_edge = longestInRegion(edges.vertical_segments, /*vertical=*/true,
                                           2.0F * width / 3.0F, width);
  const auto top_edge = longestInRegion(edges.horizontal_segments, /*vertical=*/false, 0.0F,
                                         height / 3.0F);
  const auto bottom_edge = longestInRegion(edges.horizontal_segments, /*vertical=*/false,
                                            2.0F * height / 3.0F, height);

  if (!left_edge || !right_edge || !top_edge || !bottom_edge) {
    return fallback;
  }

  const auto top_left = intersectLines(*left_edge, *top_edge);
  const auto top_right = intersectLines(*right_edge, *top_edge);
  const auto bottom_left = intersectLines(*left_edge, *bottom_edge);
  const auto bottom_right = intersectLines(*right_edge, *bottom_edge);
  if (!top_left || !top_right || !bottom_left || !bottom_right) {
    return fallback;
  }

  Quad quad;
  quad.top_left = clampToImage(*top_left, image_size);
  quad.top_right = clampToImage(*top_right, image_size);
  quad.bottom_left = clampToImage(*bottom_left, image_size);
  quad.bottom_right = clampToImage(*bottom_right, image_size);
  return quad;
}

cv::Size chooseTargetSize(const Quad& quad) {
  const auto dist = [](cv::Point2f a, cv::Point2f b) {
    return std::hypot(static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y));
  };
  const double top_width = dist(quad.top_left, quad.top_right);
  const double bottom_width = dist(quad.bottom_left, quad.bottom_right);
  const double left_height = dist(quad.top_left, quad.bottom_left);
  const double right_height = dist(quad.top_right, quad.bottom_right);

  const int w = static_cast<int>(std::max(top_width, bottom_width));
  const int h = static_cast<int>(std::max(left_height, right_height));
  return cv::Size(std::max(1, w), std::max(1, h));
}

cv::Mat rectify(const cv::Mat& bgr_image, const Quad& quad, cv::Size target_size) {
  // Source/destination correspondence, matching Quad's own field order.
  const cv::Point2f src[4] = {quad.top_left, quad.top_right, quad.bottom_right, quad.bottom_left};
  const cv::Point2f dst[4] = {
      cv::Point2f(0.0F, 0.0F),
      cv::Point2f(static_cast<float>(target_size.width - 1), 0.0F),
      cv::Point2f(static_cast<float>(target_size.width - 1),
                  static_cast<float>(target_size.height - 1)),
      cv::Point2f(0.0F, static_cast<float>(target_size.height - 1)),
  };

  // The whole method: one planar homography (no metric/scale
  // assumptions — see the file doc comment) mapping the quad onto an
  // axis-aligned rectangle, then a perspective warp.
  const cv::Mat homography = cv::getPerspectiveTransform(src, dst);
  cv::Mat rectified;
  cv::warpPerspective(bgr_image, rectified, homography, target_size);
  return rectified;
}

cv::Mat rectify(const cv::Mat& bgr_image, const Quad& quad) {
  return rectify(bgr_image, quad, chooseTargetSize(quad));
}

}  // namespace facade_parser
