/// \file lattice_refine.hpp
/// \brief Stage 5 — irregular lattice refinement (optional pass).
///
/// Implements Riemenschneider, Krispel, Thaller, Donoser, Havemann,
/// Fellner, Bischof, "Irregular Lattices for Complex Shape Grammar
/// Facade Parsing" (CVPR 2012): after the global grid from Stages 2–3,
/// each row/column boundary is refined independently by searching a
/// small window around its globally predicted position for the local
/// profile extremum. Implemented as a strategy object so it can be
/// disabled (`Config::enable_lattice_refine`) for debugging or for
/// near-perfectly-regular synthetic test images.
#pragma once

#include <opencv2/core.hpp>

#include "facade_parser/types.hpp"

namespace facade_parser {

class LatticeRefinePass {
 public:
  /// Refines every boundary in `grid` within
  /// `config.lattice_refine_window_frac` of the local cell size, snapping
  /// to the nearest profile extremum in `gradient_magnitude`. Returns a
  /// new grid; `grid` is left unmodified.
  FacadeGrid refine(const FacadeGrid& grid, const cv::Mat& gradient_magnitude,
                     const Config& config) const;
};

}  // namespace facade_parser
