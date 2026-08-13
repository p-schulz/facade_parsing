/// \file classification.hpp
/// \brief Stage 4 — cell classification.
///
/// Rule-based window/door/wall classification per grid cell, following
/// the feature set (aspect ratio, contrast, edge density) used by Wenzel
/// & Förstner, "Window Detection in Facades"; Recky & Leberl, "Window
/// detection in complex facades"; and Neuhausen & König, "Automatic
/// window detection in facade images" (2018) — see docs/PLAN.md, Stage 4.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Classifies every leaf cell of `grid` as Window / Door / Wall using
/// local Otsu thresholding, contrast/edge-density features, morphological
/// closing, and contour/bounding-rect fitting. Always emits window/door
/// elements; wall elements are only emitted when
/// `config.emit_wall_elements` is set.
std::vector<Element> classifyCells(const cv::Mat& bgr_image, const EdgeMaps& edges,
                                    const FacadeGrid& grid, const Config& config);

}  // namespace facade_parser
