#include "facade_parser/evaluation.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace facade_parser {

namespace {

double computeIoU(const cv::Rect& a, const cv::Rect& b) {
  const cv::Rect inter = a & b;
  if (inter.area() <= 0) {
    return 0.0;
  }
  const double union_area = a.area() + b.area() - inter.area();
  return union_area > 0.0 ? inter.area() / union_area : 0.0;
}

struct Candidate {
  int detected_index;
  int ground_truth_index;
  double iou;
};

// Greedy IoU matching (see evaluation.hpp doc comment) restricted to one
// element type at a time, so a detected door is never matched against a
// ground-truth window. Appends matches/FP/FN to `score->matches` and
// accumulates the match count/IoU sum into `score` and `*iou_sum`
// respectively (divided into a mean by the caller once, after both
// element types have been processed).
void matchType(ElementType type, const FacadeResult& detected, const GroundTruthImage& ground_truth,
               double iou_threshold, ImageScore* score, double* iou_sum) {
  std::vector<int> detected_indices;
  for (std::size_t i = 0; i < detected.elements.size(); ++i) {
    if (detected.elements[i].type == type) {
      detected_indices.push_back(static_cast<int>(i));
    }
  }
  std::vector<int> gt_indices;
  for (std::size_t i = 0; i < ground_truth.elements.size(); ++i) {
    if (ground_truth.elements[i].type == type) {
      gt_indices.push_back(static_cast<int>(i));
    }
  }

  std::vector<Candidate> candidates;
  for (int di : detected_indices) {
    for (int gi : gt_indices) {
      const double iou = computeIoU(detected.elements[static_cast<std::size_t>(di)].bbox_px,
                                     ground_truth.elements[static_cast<std::size_t>(gi)].bbox_px);
      if (iou >= iou_threshold) {
        candidates.push_back({di, gi, iou});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.iou > b.iou; });

  std::vector<bool> detected_used(detected.elements.size(), false);
  std::vector<bool> gt_used(ground_truth.elements.size(), false);

  for (const auto& c : candidates) {
    if (detected_used[static_cast<std::size_t>(c.detected_index)] ||
        gt_used[static_cast<std::size_t>(c.ground_truth_index)]) {
      continue;
    }
    detected_used[static_cast<std::size_t>(c.detected_index)] = true;
    gt_used[static_cast<std::size_t>(c.ground_truth_index)] = true;
    score->matches.push_back({c.detected_index, c.ground_truth_index, c.iou});
    ++score->true_positives;
    *iou_sum += c.iou;
  }

  for (int di : detected_indices) {
    if (!detected_used[static_cast<std::size_t>(di)]) {
      score->matches.push_back({di, -1, 0.0});
      ++score->false_positives;
    }
  }
  for (int gi : gt_indices) {
    if (!gt_used[static_cast<std::size_t>(gi)]) {
      score->matches.push_back({-1, gi, 0.0});
      ++score->false_negatives;
    }
  }
}

}  // namespace

ImageScore scoreImage(const FacadeResult& detected, const GroundTruthImage& ground_truth,
                       const ScoringOptions& options) {
  ImageScore score;
  double iou_sum = 0.0;
  matchType(ElementType::Window, detected, ground_truth, options.iou_match_threshold, &score,
            &iou_sum);
  matchType(ElementType::Door, detected, ground_truth, options.iou_match_threshold, &score,
            &iou_sum);

  if (score.true_positives > 0) {
    score.mean_iou = iou_sum / score.true_positives;
  }

  const double precision_denom = score.true_positives + score.false_positives;
  const double recall_denom = score.true_positives + score.false_negatives;
  const double precision = precision_denom > 0.0 ? score.true_positives / precision_denom : 0.0;
  const double recall = recall_denom > 0.0 ? score.true_positives / recall_denom : 0.0;
  score.f1 = (precision + recall) > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

  return score;
}

DatasetScore scoreDataset(const std::vector<FacadeResult>& detected,
                           const std::vector<GroundTruthImage>& ground_truth,
                           const ScoringOptions& options) {
  DatasetScore result;
  double sum_f1 = 0.0;
  double sum_iou = 0.0;
  int counted = 0;

  const std::size_t n = std::min(detected.size(), ground_truth.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (ground_truth[i].elements.empty()) {
      continue;  // No ground truth to score against: excluded, not counted as 0.
    }
    const ImageScore image_score = scoreImage(detected[i], ground_truth[i], options);
    result.per_image.push_back(image_score);
    sum_f1 += image_score.f1;
    sum_iou += image_score.mean_iou;
    ++counted;
  }

  if (counted > 0) {
    result.mean_f1 = sum_f1 / counted;
    result.mean_iou = sum_iou / counted;
  }
  return result;
}

cv::Mat renderComparisonOverlay(const cv::Mat& image, const FacadeResult& detected,
                                 const GroundTruthImage& ground_truth, const ImageScore& score) {
  cv::Mat overlay = image.clone();
  for (const auto& m : score.matches) {
    if (m.detected_index >= 0 && m.ground_truth_index >= 0) {
      cv::rectangle(overlay, detected.elements[static_cast<std::size_t>(m.detected_index)].bbox_px,
                    cv::Scalar(0, 200, 0), 2, cv::LINE_AA);  // true positive: green
    } else if (m.detected_index >= 0) {
      cv::rectangle(overlay, detected.elements[static_cast<std::size_t>(m.detected_index)].bbox_px,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);  // false positive: red
    } else if (m.ground_truth_index >= 0) {
      cv::rectangle(overlay,
                    ground_truth.elements[static_cast<std::size_t>(m.ground_truth_index)].bbox_px,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);  // false negative: yellow
    }
  }
  return overlay;
}

}  // namespace facade_parser
