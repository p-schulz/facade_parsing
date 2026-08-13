/// \file grammar.hpp
/// \brief Stage 3 — split-grammar segmentation.
///
/// Implements Müller et al. (SIGGRAPH 2007) §4's recursive split grammar:
/// a floor-split on the whole-facade horizontal profile, then a
/// per-floor-band tile-split on that band's vertical profile. The
/// recursion is internal; the result is flattened into a `FacadeGrid`
/// (see docs/PLAN.md, "Data structure decision").
#pragma once

#include <opencv2/core.hpp>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Builds the split-grammar grid for `facade_bbox` within `edges`'
/// gradient magnitude image. Recursion stops per-cell when a band is
/// smaller than `config.min_cell_size_px` or no statistically
/// significant periodicity is found in its local profile.
FacadeGrid buildSplitGrammar(const EdgeMaps& edges, const cv::Rect& facade_bbox,
                              const Config& config);

}  // namespace facade_parser
