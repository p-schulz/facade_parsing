/// \file evaluation.hpp
/// \brief Detection-vs-ground-truth matching and scoring for the
/// auto-tuning pipeline (see docs/PLAN.md, "Auto-tuning pipeline").
///
/// Scores a `FacadeResult` (Stage 4's output) against a hand-drawn
/// `GroundTruthImage` (ground_truth.hpp) using greedy IoU matching,
/// independently per element type (Window/Door — v1 scope; see
/// ground_truth.hpp). This is the objective function `autotune.hpp`'s
/// optimizer maximizes.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "facade_parser/ground_truth.hpp"
#include "facade_parser/types.hpp"

namespace facade_parser {

/// One matched (or unmatched) pair. Exactly one of `detected_index` /
/// `ground_truth_index` is -1 for an unmatched entry (false
/// positive/negative respectively); both are set (and `iou > 0`) for a
/// match.
struct ElementMatch {
  int detected_index = -1;
  int ground_truth_index = -1;
  double iou = 0.0;
};

struct ScoringOptions {
  double iou_match_threshold = 0.5;  ///< Minimum IoU to count a
                                      ///< detection/ground-truth pair as
                                      ///< a match rather than a false
                                      ///< positive + false negative pair.
};

/// Per-image score. `f1` is the standard 2*P*R/(P+R) over
/// (true_positives, false_positives, false_negatives); `mean_iou` is
/// averaged over matched (true-positive) pairs only — 0 if there are none.
struct ImageScore {
  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;
  double mean_iou = 0.0;
  double f1 = 0.0;
  std::vector<ElementMatch> matches;
};

/// Matches `detected`'s Window/Door elements against `ground_truth`'s,
/// independently per type (a detected door is never matched against a
/// ground-truth window, even if their boxes overlap): computes all
/// pairwise IoU within a type bucket, then greedily assigns matches
/// highest-IoU-first while IoU >= `options.iou_match_threshold`, each
/// detection/ground-truth element used at most once. `Wall`/`Edge`
/// detections are ignored (see ground_truth.hpp — v1 annotation scope is
/// Window/Door only).
ImageScore scoreImage(const FacadeResult& detected, const GroundTruthImage& ground_truth,
                       const ScoringOptions& options = {});

/// Aggregate score over a whole dataset: `mean_f1`/`mean_iou` are the
/// unweighted mean of each image's own `f1`/`mean_iou` (an image with no
/// ground truth at all is skipped, not counted as a 0).
struct DatasetScore {
  double mean_f1 = 0.0;
  double mean_iou = 0.0;
  std::vector<ImageScore> per_image;
};

DatasetScore scoreDataset(const std::vector<FacadeResult>& detected,
                           const std::vector<GroundTruthImage>& ground_truth,
                           const ScoringOptions& options = {});

/// Draws `score.matches` directly over a copy of `image`: true positives
/// green, false positives (unmatched detections) red, false negatives
/// (unmatched ground truth) yellow — so tuning isn't "blind": the user
/// can see exactly where a config's detections diverge from their
/// annotations before (and after) running the auto-tuner.
cv::Mat renderComparisonOverlay(const cv::Mat& image, const FacadeResult& detected,
                                 const GroundTruthImage& ground_truth, const ImageScore& score);

}  // namespace facade_parser
