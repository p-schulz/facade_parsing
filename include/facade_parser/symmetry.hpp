/// \file symmetry.hpp
/// \brief Stage 6 — symmetry check (optional pass).
///
/// Estimates a vertical mirror axis via `cv::phaseCorrelate` (Kuglin &
/// Hines, 1975) between the facade crop and its horizontal flip, then
/// uses the axis to flag grid cells whose mirrored counterpart was
/// classified in Stage 4 but the cell itself wasn't (e.g. occluded by a
/// tree or shadow). See docs/PLAN.md, Stage 6, for why flagging — not
/// auto-filling — is the default.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "facade_parser/types.hpp"

namespace facade_parser {

struct SymmetryCheckResult {
  double mirror_axis_x_px = 0.0;
  std::vector<SymmetryInference> inferences;
};

/// Runs the phase-correlation mirror-axis estimate on `bgr_image`
/// restricted to `facade_bbox`, then cross-references `grid`/`elements`
/// to produce `inferences`.
SymmetryCheckResult checkSymmetry(const cv::Mat& bgr_image, const cv::Rect& facade_bbox,
                                   const FacadeGrid& grid, const std::vector<Element>& elements,
                                   const Config& config);

}  // namespace facade_parser
