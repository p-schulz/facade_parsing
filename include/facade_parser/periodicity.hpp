/// \file periodicity.hpp
/// \brief Stage 2 — periodicity / grid detection.
///
/// Implements Müller, Zeng, Wonka, Van Gool, "Image-based Procedural
/// Modeling of Facades" (SIGGRAPH 2007), §4.1: projection profiles of
/// gradient magnitude, smoothed and analyzed for periodicity two ways —
/// DFT-based autocorrelation (Wiener–Khinchin theorem) and direct
/// NMS peak-picking — cross-validated against each other (see
/// docs/PLAN.md).
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// P_v[x] = sum over y in `roi` of `gradient_magnitude` at column x.
std::vector<float> computeColumnProfile(const cv::Mat& gradient_magnitude, const cv::Rect& roi);

/// P_h[y] = sum over x in `roi` of `gradient_magnitude` at row y.
std::vector<float> computeRowProfile(const cv::Mat& gradient_magnitude, const cv::Rect& roi);

/// Gaussian-smooths `profile` in place-equivalent fashion (returns a copy)
/// using `config.profile_smoothing_sigma_px`.
std::vector<float> smoothProfile(const std::vector<float>& profile, const Config& config);

/// Wiener–Khinchin autocorrelation: zero-pads `profile` to the next
/// `cv::getOptimalDFTSize`, computes the power spectrum via `cv::dft`,
/// and inverse-transforms back to the autocorrelation function. Returns
/// candidate periods ordered by descending score.
std::vector<PeriodicityPeak> autocorrelationPeaks(const std::vector<float>& profile,
                                                    const Config& config);

/// Direct boundary localization on `profile`: thresholds at
/// `config.direct_peak_low_activity_frac` of the profile max to find
/// contiguous low-activity runs (candidate wall/pier gaps between
/// periodic features), takes each run's center, then applies
/// minimum-distance non-maximum suppression using `min_distance_px`
/// (normally `autocorrelationPeaks`'s top candidate; a threshold-only
/// pass is used when it's `<= 0`). Judgment call, documented in
/// docs/PLAN.md Stage 2: boundaries are placed at low-*activity*
/// (valley) regions rather than at profile maxima, since a window/pier
/// grid's separators are the plain wall gaps, not the window edges
/// themselves — a raw maxima scan on a filled-rectangle profile finds
/// two peaks per bay (the window's left/right edges) with no reliable
/// way to disambiguate which one to treat as "the" boundary.
std::vector<PeriodicityPeak> directPeaks(const std::vector<float>& profile, int min_distance_px,
                                          const Config& config);

/// Runs both peak-finding strategies on `profile`, cross-validates them,
/// and returns the resulting boundary positions (see docs/PLAN.md for the
/// disagreement-handling rule that sets `PeriodicityResult::low_confidence`).
PeriodicityResult analyzePeriodicity(const std::vector<float>& profile, const Config& config);

}  // namespace facade_parser
