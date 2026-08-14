/// \file rectification.hpp
/// \brief Stage 0 — perspective (keystone) correction.
///
/// A raw facade photo has camera perspective distortion; Stages 1-4
/// (periodicity/autocorrelation, split-grammar, contour fitting) assume
/// a fronto-parallel, axis-aligned facade plane and work much better
/// once that's true. This module proposes the four facade corners in a
/// raw photo, then warps the facade plane onto an axis-aligned rectangle
/// given those corners (adjusted by the user, typically, via the GUI).
///
/// Deliberately **not** a metric/scale-accurate rectification: only
/// right angles matter here. Real-world facade width/height come from
/// an external GIS pipeline (OpenStreetMap / LGL-BW) and the rectified
/// texture is later resized to a power-of-two resolution regardless, so
/// preserving exact proportions isn't a goal. No camera calibration,
/// vanishing-point estimation, or lens undistortion — a single planar
/// homography (`cv::getPerspectiveTransform` + `cv::warpPerspective`) is
/// the whole method.
///
/// This is a GUI-triggered pre-process, not part of
/// `facade_parser::run()` (pipeline.hpp) — see docs/PLAN.md, "Stage 0:
/// Rectification", for why it's a separate module rather than a new
/// `Config` surface.
#pragma once

#include <opencv2/core.hpp>

namespace facade_parser {

/// Four corners of a (planar) facade in image pixel space.
struct Quad {
  cv::Point2f top_left;
  cv::Point2f top_right;
  cv::Point2f bottom_left;
  cv::Point2f bottom_right;
};

/// Proposes a starting `Quad` for a raw (unrectified) facade photo, by
/// reusing Stage 1's line detector (`extractEdges`, edges.hpp) rather
/// than reimplementing Canny + line detection: classifies segments as
/// near-vertical / near-horizontal within `line_angle_tolerance_deg` of
/// the respective axis (wider than Stage 1's own post-rectification
/// default of 10 deg — a raw photo's facade edges aren't yet close to
/// axis-aligned by definition, that's the whole reason this pass
/// exists), then for each outer third of the image picks that region's
/// longest candidate segment as the corresponding facade edge (left/
/// right thirds among vertical segments -> the facade's side edges;
/// top/bottom thirds among horizontal segments -> roofline/eave and
/// ground/sockel line) and pairwise-intersects the four chosen lines.
///
/// Falls back to a centered inset rectangle (`fallback_margin_frac`
/// from each border) when fewer than all four edges are found, or a
/// pairwise intersection is degenerate (near-parallel lines) — so the
/// caller (the GUI) always gets a valid, image-bounded starting quad to
/// display and let the user drag-correct, rather than a failure.
Quad proposeCorners(const cv::Mat& bgr_image, double line_angle_tolerance_deg = 20.0,
                     double fallback_margin_frac = 0.10);

/// The target rectangle size for `rectify()`: the quad's own bounding
/// box (max of the top/bottom edge lengths for width, max of the left/
/// right edge lengths for height). No metric or fixed-aspect-ratio goal
/// — see the file doc comment — this just avoids obviously
/// upsampling/downsampling relative to the quad's own on-screen size.
cv::Size chooseTargetSize(const Quad& quad);

/// Computes the planar homography mapping `quad`'s four corners onto an
/// axis-aligned `target_size` rectangle (`cv::getPerspectiveTransform`)
/// and warps `bgr_image` with it (`cv::warpPerspective`). Corner
/// ordering matches `Quad`'s named fields directly — no metric
/// assumptions are made about `quad` itself (it need not be a "nice"
/// shape; a heavily skewed quad still warps to a plain rectangle by
/// construction).
cv::Mat rectify(const cv::Mat& bgr_image, const Quad& quad, cv::Size target_size);

/// Convenience overload: target size from `chooseTargetSize(quad)`.
cv::Mat rectify(const cv::Mat& bgr_image, const Quad& quad);

}  // namespace facade_parser
