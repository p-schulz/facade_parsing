#include "synthetic_facade.hpp"

#include <opencv2/imgproc.hpp>

namespace facade_parser::test {

namespace {

int jitterAt(const std::vector<int>& jitter, std::size_t index) {
  return (index < jitter.size()) ? jitter[index] : 0;
}

SyntheticFacade makeFacadeImpl(int rows, int cols, int cell_width_px, int cell_height_px,
                                int margin_px, const std::vector<int>& row_jitter_px,
                                const std::vector<int>& col_jitter_px) {
  SyntheticFacade out;

  const int width = margin_px + cols * (cell_width_px + margin_px);
  const int height = margin_px + rows * (cell_height_px + margin_px);
  out.image = cv::Mat(height, width, CV_8UC3, cv::Scalar(210, 205, 200));  // plain wall.

  std::vector<int> row_top(static_cast<std::size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    row_top[static_cast<std::size_t>(r)] =
        margin_px + r * (cell_height_px + margin_px) + jitterAt(row_jitter_px, static_cast<std::size_t>(r));
  }
  std::vector<int> col_left(static_cast<std::size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    col_left[static_cast<std::size_t>(c)] =
        margin_px + c * (cell_width_px + margin_px) + jitterAt(col_jitter_px, static_cast<std::size_t>(c));
  }

  // Ground-truth grid boundaries are the midpoints of the wall/pier gaps
  // between consecutive windows — the same low-activity locations
  // buildSplitGrammar (via analyzePeriodicity) is designed to find, NOT
  // the windows' own edges. See docs/PLAN.md, Stage 2's directPeaks
  // decision, for why boundaries live in the gaps rather than on window
  // edges.
  out.ground_truth_grid.row_boundaries_px.resize(static_cast<std::size_t>(rows) + 1);
  out.ground_truth_grid.row_boundaries_px.front() = 0;
  out.ground_truth_grid.row_boundaries_px.back() = height;
  for (int r = 1; r < rows; ++r) {
    const int prev_bottom = row_top[static_cast<std::size_t>(r - 1)] + cell_height_px;
    const int this_top = row_top[static_cast<std::size_t>(r)];
    out.ground_truth_grid.row_boundaries_px[static_cast<std::size_t>(r)] =
        (prev_bottom + this_top) / 2;
  }

  std::vector<int> col_boundaries(static_cast<std::size_t>(cols) + 1);
  col_boundaries.front() = 0;
  col_boundaries.back() = width;
  for (int c = 1; c < cols; ++c) {
    const int prev_right = col_left[static_cast<std::size_t>(c - 1)] + cell_width_px;
    const int this_left = col_left[static_cast<std::size_t>(c)];
    col_boundaries[static_cast<std::size_t>(c)] = (prev_right + this_left) / 2;
  }
  out.ground_truth_grid.col_boundaries_px.assign(static_cast<std::size_t>(rows), col_boundaries);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      cv::rectangle(out.image,
                    cv::Rect(col_left[static_cast<std::size_t>(c)], row_top[static_cast<std::size_t>(r)],
                              cell_width_px, cell_height_px),
                    cv::Scalar(60, 50, 40), cv::FILLED);
    }
  }

  return out;
}

}  // namespace

SyntheticFacade makeRegularFacade(int rows, int cols, int cell_width_px, int cell_height_px,
                                   int margin_px) {
  return makeFacadeImpl(rows, cols, cell_width_px, cell_height_px, margin_px, {}, {});
}

SyntheticFacade makeJitteredFacade(int rows, int cols, int cell_width_px, int cell_height_px,
                                    int margin_px) {
  // Fixed, hard-coded offsets (not runtime RNG) per docs/PLAN.md.
  static const std::vector<int> kRowJitter = {0, 3, -2, 4, -3, 2, -4, 1};
  static const std::vector<int> kColJitter = {0, -2, 3, -1, 2, -3, 4, -2};
  return makeFacadeImpl(rows, cols, cell_width_px, cell_height_px, margin_px, kRowJitter,
                        kColJitter);
}

}  // namespace facade_parser::test
