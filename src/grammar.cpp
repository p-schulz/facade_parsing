#include "facade_parser/grammar.hpp"

#include <algorithm>

#include "facade_parser/periodicity.hpp"

namespace facade_parser {

namespace {

// Greedily drops interior boundaries that would leave an undersized
// segment on either side, until none remain (or only the two endpoints
// are left). Keeps the outer two boundaries fixed — they're the
// facade/band's own extent, not a detected split.
std::vector<int> enforceMinSize(std::vector<int> boundaries, int min_size) {
  bool changed = true;
  while (changed && boundaries.size() > 2) {
    changed = false;
    for (std::size_t i = 1; i + 1 < boundaries.size(); ++i) {
      const int seg_before = boundaries[i] - boundaries[i - 1];
      const int seg_after = boundaries[i + 1] - boundaries[i];
      if (seg_before < min_size || seg_after < min_size) {
        boundaries.erase(boundaries.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }
  return boundaries;
}

// Guards against the "known limitation" documented in docs/PLAN.md: one
// strongly irregular cell (e.g. a door much narrower than its
// neighboring windows) can make directPeaks find extra low-activity
// runs around its own margins, over-splitting its row/band beyond the
// actual number of bays. Repeatedly finds the segment furthest below
// `min_width_frac` of the row's own median segment width and merges it
// into whichever neighbor is smaller (so a spurious narrow split
// doesn't get folded into an already-correct, larger neighboring cell),
// until none remain undersized or too few segments are left to compute
// a meaningful median.
//
// This is a local, cheap heuristic, not a full fix: it assumes the
// row/band is *mostly* regular and one (or a few) cells are the outlier
// — a facade whose bays are irregular throughout would need something
// closer to Riemenschneider et al.'s full irregular-lattice search (see
// docs/PLAN.md).
std::vector<int> enforceMedianWidthConsistency(std::vector<int> boundaries,
                                                double min_width_frac) {
  // Fewer than 3 segments: no statistically meaningful median to compare
  // against (with 2 segments there's no way to tell which one, if
  // either, is the "outlier").
  while (boundaries.size() >= 4) {
    std::vector<int> widths;
    widths.reserve(boundaries.size() - 1);
    for (std::size_t i = 0; i + 1 < boundaries.size(); ++i) {
      widths.push_back(boundaries[i + 1] - boundaries[i]);
    }

    std::vector<int> sorted_widths = widths;
    std::sort(sorted_widths.begin(), sorted_widths.end());
    const int median_width = sorted_widths[sorted_widths.size() / 2];
    const int min_width = static_cast<int>(median_width * min_width_frac);

    int narrowest_idx = -1;
    int narrowest_width = 0;
    for (std::size_t i = 0; i < widths.size(); ++i) {
      if (widths[i] < min_width && (narrowest_idx < 0 || widths[i] < narrowest_width)) {
        narrowest_idx = static_cast<int>(i);
        narrowest_width = widths[i];
      }
    }
    if (narrowest_idx < 0) {
      break;  // Nothing undersized relative to the median; done.
    }

    const std::size_t k = static_cast<std::size_t>(narrowest_idx);
    std::size_t remove_index;
    if (k == 0) {
      remove_index = 1;  // No left neighbor; merge into the right one.
    } else if (k + 1 == widths.size()) {
      remove_index = k;  // No right neighbor; merge into the left one.
    } else {
      // Merge with whichever neighbor is smaller, so the narrow segment
      // doesn't get folded into an already-large (probably correct)
      // neighboring cell.
      remove_index = (widths[k - 1] <= widths[k + 1]) ? k : (k + 1);
    }
    boundaries.erase(boundaries.begin() + static_cast<std::ptrdiff_t>(remove_index));
  }
  return boundaries;
}

}  // namespace

// Two-level split grammar (Müller et al. §4): a floor-split on the
// whole-facade horizontal profile, then a tile-split per floor band on
// that band's own vertical profile. See docs/PLAN.md, "Data structure
// decision", for why this doesn't recurse further into a sub-cell tree.
FacadeGrid buildSplitGrammar(const EdgeMaps& edges, const cv::Rect& facade_bbox,
                              const Config& config) {
  FacadeGrid grid;

  const std::vector<float> row_profile =
      computeRowProfile(edges.gradient_magnitude, facade_bbox);
  const PeriodicityResult row_result = analyzePeriodicity(row_profile, config);
  std::vector<int> row_boundaries =
      enforceMinSize(row_result.boundary_positions_px, config.min_cell_size_px);
  // Deliberately NOT applying enforceMedianWidthConsistency to row
  // (floor) boundaries: unlike bay widths within a single floor, floor
  // *heights* legitimately vary a lot on real facades (a taller ground
  // floor, a short attic/roof band, etc. — see docs/PLAN.md's
  // "Mitigated: a single strongly irregular cell..." writeup, where
  // applying this filter to rows on a real photo merged away a real
  // shorter band and lost most of the detected windows). The irregular-
  // cell failure mode this guards against was only ever observed at the
  // column level, within one floor's own bays.
  for (int& b : row_boundaries) {
    b += facade_bbox.y;
  }
  grid.row_boundaries_px = row_boundaries;

  for (std::size_t i = 0; i + 1 < row_boundaries.size(); ++i) {
    const cv::Rect band(facade_bbox.x, row_boundaries[i], facade_bbox.width,
                         row_boundaries[i + 1] - row_boundaries[i]);
    const std::vector<float> col_profile = computeColumnProfile(edges.gradient_magnitude, band);
    const PeriodicityResult col_result = analyzePeriodicity(col_profile, config);
    std::vector<int> col_boundaries =
        enforceMinSize(col_result.boundary_positions_px, config.min_cell_size_px);
    col_boundaries =
        enforceMedianWidthConsistency(col_boundaries, config.min_segment_width_frac_of_median);
    for (int& b : col_boundaries) {
      b += facade_bbox.x;
    }
    grid.col_boundaries_px.push_back(std::move(col_boundaries));
  }

  return grid;
}

}  // namespace facade_parser
