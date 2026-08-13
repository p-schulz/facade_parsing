#include "facade_parser/edges.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

#ifdef FACADE_PARSER_HAS_XIMGPROC
#include <opencv2/ximgproc/fast_line_detector.hpp>
#endif

namespace facade_parser {

namespace {

LineSegment makeSegment(cv::Point2f p1, cv::Point2f p2) {
  LineSegment seg;
  seg.p1 = p1;
  seg.p2 = p2;
  const cv::Point2f d = p2 - p1;
  seg.length_px = std::sqrt(d.x * d.x + d.y * d.y);
  double angle = std::atan2(static_cast<double>(d.y), static_cast<double>(d.x)) * 180.0 / CV_PI;
  // Normalize to [-90, 90): a line has no inherent direction, so fold
  // the opposite-pointing representation onto the same angle.
  while (angle < -90.0) angle += 180.0;
  while (angle >= 90.0) angle -= 180.0;
  seg.angle_deg = static_cast<float>(angle);
  return seg;
}

#ifdef FACADE_PARSER_HAS_XIMGPROC
// Primary path: cv::ximgproc::createFastLineDetector(). See docs/PLAN.md,
// Stage 1 decision, for why FLD is preferred over classic LSD when
// ximgproc is available.
std::vector<LineSegment> detectLineSegments(const cv::Mat& gray, const Config& /*config*/) {
  const cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector();
  std::vector<cv::Vec4f> lines;
  fld->detect(gray, lines);

  std::vector<LineSegment> segments;
  segments.reserve(lines.size());
  for (const auto& l : lines) {
    segments.push_back(makeSegment({l[0], l[1]}, {l[2], l[3]}));
  }
  return segments;
}
#else
// Fallback path: cv::HoughLinesP on the Canny edge map, used when the
// OpenCV build has no ximgproc (contrib) module. See docs/PLAN.md,
// Stage 1 decision.
std::vector<LineSegment> detectLineSegments(const cv::Mat& canny, const Config& config) {
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(canny, lines, 1.0, CV_PI / 180.0, config.hough_threshold_votes,
                  config.hough_min_line_length_px, config.hough_max_line_gap_px);

  std::vector<LineSegment> segments;
  segments.reserve(lines.size());
  for (const auto& l : lines) {
    segments.push_back(makeSegment({static_cast<float>(l[0]), static_cast<float>(l[1])},
                                    {static_cast<float>(l[2]), static_cast<float>(l[3])}));
  }
  return segments;
}
#endif

void splitByAngle(const std::vector<LineSegment>& segments, double tolerance_deg,
                   std::vector<LineSegment>* horizontal, std::vector<LineSegment>* vertical) {
  for (const auto& seg : segments) {
    const double angle = std::abs(seg.angle_deg);
    if (angle <= tolerance_deg) {
      horizontal->push_back(seg);
    } else if (std::abs(angle - 90.0) <= tolerance_deg) {
      vertical->push_back(seg);
    }
    // Diagonal segments discarded, see docs/PLAN.md Stage 1.
  }
}

}  // namespace

EdgeMaps extractEdges(const cv::Mat& bgr_image, const Config& config) {
  EdgeMaps result;

  cv::Mat gray;
  cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);

  cv::Canny(gray, result.canny, config.canny_low, config.canny_high);

  cv::Mat dx;
  cv::Mat dy;
  cv::Sobel(gray, dx, CV_32F, 1, 0);
  cv::Sobel(gray, dy, CV_32F, 0, 1);
  cv::magnitude(dx, dy, result.gradient_magnitude);

#ifdef FACADE_PARSER_HAS_XIMGPROC
  const std::vector<LineSegment> segments = detectLineSegments(gray, config);
#else
  const std::vector<LineSegment> segments = detectLineSegments(result.canny, config);
#endif
  splitByAngle(segments, config.line_angle_tolerance_deg, &result.horizontal_segments,
               &result.vertical_segments);

  return result;
}

}  // namespace facade_parser
