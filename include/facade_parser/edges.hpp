/// \file edges.hpp
/// \brief Stage 1 — edge and line extraction.
///
/// Grayscale + Canny for a binary edge map, Sobel gradient magnitude for
/// the continuous-strength signal Stage 2/4 need, and near-horizontal /
/// near-vertical line segment extraction (FastLineDetector when
/// `FACADE_PARSER_HAS_XIMGPROC` is defined, `cv::HoughLinesP` otherwise —
/// see docs/PLAN.md, Stage 1 decision).
#pragma once

#include <opencv2/core.hpp>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Runs Canny + Sobel-magnitude + line segment extraction on `bgr_image`
/// and splits detected segments into near-horizontal / near-vertical sets
/// using `config.line_angle_tolerance_deg`.
EdgeMaps extractEdges(const cv::Mat& bgr_image, const Config& config);

}  // namespace facade_parser
