#include "facade_parser/lattice_refine.hpp"

#include <algorithm>
#include <cstdlib>

#include "facade_parser/periodicity.hpp"

namespace facade_parser {

namespace {

// Snaps each interior boundary to the position of minimum activity
// within +/- window_frac * local_cell_size of its globally predicted
// position (Riemenschneider et al., CVPR 2012; see docs/PLAN.md Stage
// 5). Search windows are computed from the *original* boundary
// positions (not chained through previously-refined neighbors), so
// refinement of one boundary can't drift subsequent ones.
std::vector<int> refineBoundaries(const std::vector<int>& boundaries,
                                   const std::vector<float>& profile, double window_frac) {
  if (boundaries.size() < 3 || profile.empty()) {
    return boundaries;
  }

  std::vector<int> result = boundaries;
  const int n = static_cast<int>(profile.size());

  for (std::size_t i = 1; i + 1 < boundaries.size(); ++i) {
    const int prev = boundaries[i - 1];
    const int cur = boundaries[i];
    const int next = boundaries[i + 1];
    const int local_size = std::min(cur - prev, next - cur);
    const int half_window = std::max(1, static_cast<int>(local_size * window_frac));

    const int lo = std::clamp(cur - half_window, prev + 1, n - 1);
    const int hi = std::clamp(cur + half_window, 0, std::min(next - 1, n - 1));
    if (lo > hi) {
      continue;
    }

    // A genuine wall/pier gap is usually a wide near-zero plateau, not a
    // sharp point minimum, so a plain argmin would deterministically
    // snap to whichever tied position is scanned first (the window's own
    // edge), and a "nearest tied position to cur" rule would just leave
    // an already-inside-the-plateau `cur` wherever it started instead of
    // centering it. Two passes: find the window's value range, then
    // snap to the centroid of every position within `epsilon` of the
    // minimum — the middle of the low-activity region.
    float min_val = profile[static_cast<std::size_t>(lo)];
    float max_val = min_val;
    for (int x = lo + 1; x <= hi; ++x) {
      const float v = profile[static_cast<std::size_t>(x)];
      min_val = std::min(min_val, v);
      max_val = std::max(max_val, v);
    }
    const float epsilon = std::max(1e-3F, 0.05F * (max_val - min_val));

    long long sum = 0;
    int count = 0;
    for (int x = lo; x <= hi; ++x) {
      if (profile[static_cast<std::size_t>(x)] <= min_val + epsilon) {
        sum += x;
        ++count;
      }
    }
    const int best = count > 0 ? static_cast<int>(sum / count) : cur;
    result[i] = best;
  }

  return result;
}

}  // namespace

FacadeGrid LatticeRefinePass::refine(const FacadeGrid& grid, const cv::Mat& gradient_magnitude,
                                      const Config& config) const {
  FacadeGrid refined = grid;
  if (grid.row_boundaries_px.size() < 2) {
    return refined;
  }

  const cv::Rect full(0, 0, gradient_magnitude.cols, gradient_magnitude.rows);
  const std::vector<float> row_profile = computeRowProfile(gradient_magnitude, full);
  refined.row_boundaries_px =
      refineBoundaries(grid.row_boundaries_px, row_profile, config.lattice_refine_window_frac);

  for (std::size_t r = 0; r < grid.col_boundaries_px.size(); ++r) {
    const cv::Rect band(0, refined.row_boundaries_px[r], gradient_magnitude.cols,
                         refined.row_boundaries_px[r + 1] - refined.row_boundaries_px[r]);
    const std::vector<float> col_profile = computeColumnProfile(gradient_magnitude, band);
    refined.col_boundaries_px[r] =
        refineBoundaries(grid.col_boundaries_px[r], col_profile, config.lattice_refine_window_frac);
  }

  return refined;
}

}  // namespace facade_parser
