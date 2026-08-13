/// \file edges_export.hpp
/// \brief Stage 7 — edge/ledge/projection extraction.
///
/// Filters Stage 1's long line segments down to those not already
/// claimed by a Stage-4 window/door bbox and emits them as `edge`
/// elements (cornices, ledges, string courses, floor separators, piers).
/// Also defines the default `DepthHint` implementation: explicitly a
/// low-confidence gradient-magnitude proxy, not a physical depth
/// estimate — see the `DepthHint` doc comment in types.hpp and
/// docs/PLAN.md, Stage 7, for why relief-depth-from-shading is out of
/// scope for this module.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Default `DepthHint`: returns the normalized mean Sobel gradient
/// magnitude within `region_px`, always tagged `low_confidence = true`.
/// A future implementation backed by a real depth/displacement map
/// (e.g. photogrammetry) can be swapped in without changing Stage 8.
class SobelMagnitudeDepthHint final : public DepthHint {
 public:
  Result estimate(const cv::Mat& gradient_magnitude, const cv::Rect& region_px) const override;
};

/// Emits `edge` elements from `edges`' long segments
/// (`config.min_edge_length_px`) that aren't fully contained in any
/// window/door bbox in `cell_elements` dilated by
/// `config.edge_claim_margin_px` (see docs/PLAN.md Stage 7 for why this
/// isn't a plain IoU test), tagging each with a coarse `EdgeKind` and a
/// `depth_hint` value from `depth_hint`.
std::vector<Element> exportEdgeElements(const EdgeMaps& edges,
                                         const std::vector<Element>& cell_elements,
                                         const DepthHint& depth_hint, const Config& config);

}  // namespace facade_parser
