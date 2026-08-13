/// \file types.hpp
/// \brief Core value types shared by every pipeline stage.
///
/// See docs/PLAN.md ("Data structure decision: grid, not tree") for the
/// rationale behind representing the split-grammar result as a flattened
/// grid instead of a tree.
#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace facade_parser {

/// Classification assigned to a grid cell, or "edge" for Stage 7 polylines.
enum class ElementType { Window, Door, Wall, Edge };

/// Sub-classification for `Edge` elements (Stage 7).
enum class EdgeKind { FloorSeparator, Pier, Cornice, Unclassified };

/// A single near-horizontal or near-vertical line segment from Stage 1,
/// in pixel coordinates.
struct LineSegment {
  cv::Point2f p1;
  cv::Point2f p2;
  float length_px = 0.0F;
  float angle_deg = 0.0F;  ///< In [-90, 90), 0 = horizontal.
};

/// Result of Stage 1 (edges.hpp).
struct EdgeMaps {
  cv::Mat canny;               ///< CV_8UC1 binary edge map.
  cv::Mat gradient_magnitude;  ///< CV_32FC1, same size as input.
  std::vector<LineSegment> horizontal_segments;
  std::vector<LineSegment> vertical_segments;
};

/// A single detected periodicity peak on a 1D projection profile
/// (Stage 2). `lag_px` is the candidate period; `score` is normalized
/// to the profile's own scale for cross-stage comparability.
struct PeriodicityPeak {
  int lag_px = 0;
  float score = 0.0F;
};

/// Stage 2 output for one profile (either the whole-facade P_h, or a
/// per-floor-band P_v).
struct PeriodicityResult {
  std::vector<float> smoothed_profile;
  std::vector<int> boundary_positions_px;  ///< Ascending, inclusive of both ends.
  bool low_confidence = false;             ///< Set when autocorrelation and
                                            ///< direct peak-picking disagreed
                                            ///< beyond tolerance (see PLAN.md).
};

/// Flattened split-grammar result (Stage 3). See docs/PLAN.md for the
/// grid-vs-tree rationale.
struct FacadeGrid {
  std::vector<int> row_boundaries_px;               ///< size R+1.
  std::vector<std::vector<int>> col_boundaries_px;   ///< size R, each row's own boundaries.

  int rows() const { return static_cast<int>(row_boundaries_px.size()) - 1; }
  int cols(int row) const {
    return static_cast<int>(col_boundaries_px.at(static_cast<std::size_t>(row)).size()) - 1;
  }

  /// Axis-aligned pixel rect for cell (row, col); kept inline since
  /// `types.hpp` is header-only (see docs/PLAN.md module table).
  cv::Rect cellRect(int row, int col) const {
    const int y0 = row_boundaries_px.at(static_cast<std::size_t>(row));
    const int y1 = row_boundaries_px.at(static_cast<std::size_t>(row) + 1);
    const auto& cols_for_row = col_boundaries_px.at(static_cast<std::size_t>(row));
    const int x0 = cols_for_row.at(static_cast<std::size_t>(col));
    const int x1 = cols_for_row.at(static_cast<std::size_t>(col) + 1);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
  }
};

/// A classified grid cell (Stage 4) or an emitted edge (Stage 7),
/// normalized coordinates included per docs/OUTPUT_FORMAT.md.
struct Element {
  ElementType type = ElementType::Wall;
  float confidence = 0.0F;

  // Populated for Window / Door / Wall:
  int row = -1;
  int col = -1;
  cv::Rect bbox_px;
  cv::Rect2f bbox_norm;

  // Populated for Edge:
  std::vector<cv::Point> polyline_px;
  std::vector<cv::Point2f> polyline_norm;
  EdgeKind edge_kind = EdgeKind::Unclassified;
  std::optional<float> depth_hint_value;  ///< See DepthHint below; always
                                           ///< `low_confidence` for the
                                           ///< default implementation.
};

/// Result of a symmetry check pass (Stage 6) for a single grid cell that
/// had no Stage-4 detection but whose mirrored counterpart did.
struct SymmetryInference {
  int row = 0;
  int col = 0;
  cv::Rect suggested_bbox_px;
  ElementType suggested_type = ElementType::Window;
  float mirror_confidence = 0.0F;
};

/// Full pipeline result, mirrors docs/OUTPUT_FORMAT.md.
struct FacadeResult {
  std::string source_image;
  cv::Size image_size_px;
  FacadeGrid grid;
  std::vector<Element> elements;
  std::vector<SymmetryInference> symmetry_inferences;
};

/// Strategy interface for Stage 7's per-edge depth/relief proxy. The
/// default implementation is a coarse, explicitly low-confidence Sobel-
/// magnitude proxy (see edges_export.hpp); a future implementation
/// backed by a real depth/displacement map can be swapped in without
/// touching Stage 8's export code. See docs/PLAN.md Stage 7.
class DepthHint {
 public:
  virtual ~DepthHint() = default;

  /// Returns a value in [0, 1] and whether the estimate should be
  /// treated as low confidence by consumers. This module only ever
  /// returns low_confidence = true; a future depth-map-backed
  /// implementation is expected to return false.
  struct Result {
    float value = 0.0F;
    bool low_confidence = true;
  };

  virtual Result estimate(const cv::Mat& gradient_magnitude, const cv::Rect& region_px) const = 0;
};

/// Aggregate configuration for every stage. Grouped in one struct (rather
/// than per-stage config structs) so the CLI has a single flat set of
/// flags to expose and the pipeline has a single object to thread through
/// all eight stages.
struct Config {
  // Stage 1
  double canny_low = 50.0;
  double canny_high = 150.0;
  double hough_min_line_length_px = 30.0;
  double hough_max_line_gap_px = 5.0;
  int hough_threshold_votes = 40;  ///< HoughLinesP accumulator threshold (fallback path only).
  double line_angle_tolerance_deg = 10.0;  ///< Distance from 0/90 deg to
                                            ///< still count as h/v.

  // Stage 2
  double profile_smoothing_sigma_px = 2.0;
  double periodicity_min_score = 0.15;     ///< Fraction of zero-lag autocorrelation.
  double periodicity_agreement_tol_px = 8.0;
  double direct_peak_low_activity_frac = 0.25;  ///< Fraction of profile max below
                                                 ///< which a run is "low activity"
                                                 ///< (candidate separator); see
                                                 ///< docs/PLAN.md Stage 2.

  // Stage 3
  int min_cell_size_px = 24;
  double min_segment_width_frac_of_median = 0.5;  ///< A column (bay) split
                                                    ///< whose resulting
                                                    ///< segment is narrower
                                                    ///< than this fraction
                                                    ///< of that floor band's
                                                    ///< own median bay
                                                    ///< width gets merged
                                                    ///< into a neighbor.
                                                    ///< Guards against one
                                                    ///< strongly irregular
                                                    ///< cell (e.g. a narrow
                                                    ///< door) producing
                                                    ///< spurious extra
                                                    ///< low-activity runs
                                                    ///< that over-split its
                                                    ///< row. Deliberately
                                                    ///< NOT applied to row
                                                    ///< (floor) boundaries
                                                    ///< — floor heights
                                                    ///< legitimately vary
                                                    ///< a lot on real
                                                    ///< facades. See
                                                    ///< docs/PLAN.md's
                                                    ///< "Mitigated: a
                                                    ///< single strongly
                                                    ///< irregular cell...".

  // Stage 4
  double otsu_close_kernel_frac = 0.08;   ///< Fraction of cell's shorter side.
  double window_min_fill_ratio = 0.15;
  double window_max_fill_ratio = 0.95;
  double window_min_aspect = 0.3;
  double window_max_aspect = 3.0;
  double door_min_height_width_ratio = 1.8;  ///< Ordinary portrait windows
                                              ///< commonly run up to ~1.5-1.6;
                                              ///< tuned above that band so a
                                              ///< tall bottom-row window
                                              ///< isn't misread as a door.

  // Stage 5
  bool enable_lattice_refine = true;
  double lattice_refine_window_frac = 0.10;  ///< +/- fraction of local cell size.

  // Stage 6
  bool enable_symmetry_check = true;

  // Stage 7
  double min_edge_length_px = 40.0;
  double edge_claim_margin_px = 6.0;  ///< A line segment is considered
                                       ///< "already part of a window/door
                                       ///< frame" (and dropped) when it
                                       ///< lies fully within that
                                       ///< element's bbox dilated by this
                                       ///< margin. A plain IoU test
                                       ///< doesn't work here: a straight
                                       ///< segment's own bounding box is a
                                       ///< near-zero-area sliver, so its
                                       ///< IoU against a solid window
                                       ///< rect is tiny even when the
                                       ///< segment traces that window's
                                       ///< own edge exactly. See
                                       ///< docs/PLAN.md Stage 7.

  // Stage 8
  bool emit_wall_elements = false;
};

}  // namespace facade_parser
