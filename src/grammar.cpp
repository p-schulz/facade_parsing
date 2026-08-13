#include "facade_parser/grammar.hpp"

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
    for (int& b : col_boundaries) {
      b += facade_bbox.x;
    }
    grid.col_boundaries_px.push_back(std::move(col_boundaries));
  }

  return grid;
}

}  // namespace facade_parser
