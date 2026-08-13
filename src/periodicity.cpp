#include "facade_parser/periodicity.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace facade_parser {

std::vector<float> computeColumnProfile(const cv::Mat& gradient_magnitude, const cv::Rect& roi) {
  const cv::Mat crop = gradient_magnitude(roi);
  cv::Mat column_sums;
  cv::reduce(crop, column_sums, 0, cv::REDUCE_SUM, CV_32F);
  std::vector<float> profile(static_cast<std::size_t>(column_sums.cols));
  for (int x = 0; x < column_sums.cols; ++x) {
    profile[static_cast<std::size_t>(x)] = column_sums.at<float>(0, x);
  }
  return profile;
}

std::vector<float> computeRowProfile(const cv::Mat& gradient_magnitude, const cv::Rect& roi) {
  const cv::Mat crop = gradient_magnitude(roi);
  cv::Mat row_sums;
  cv::reduce(crop, row_sums, 1, cv::REDUCE_SUM, CV_32F);
  std::vector<float> profile(static_cast<std::size_t>(row_sums.rows));
  for (int y = 0; y < row_sums.rows; ++y) {
    profile[static_cast<std::size_t>(y)] = row_sums.at<float>(y, 0);
  }
  return profile;
}

std::vector<float> smoothProfile(const std::vector<float>& profile, const Config& config) {
  if (profile.empty()) {
    return profile;
  }
  cv::Mat row(1, static_cast<int>(profile.size()), CV_32F, const_cast<float*>(profile.data()));
  cv::Mat smoothed;
  cv::GaussianBlur(row, smoothed, cv::Size(0, 1), config.profile_smoothing_sigma_px);
  std::vector<float> out(profile.size());
  for (std::size_t i = 0; i < profile.size(); ++i) {
    out[i] = smoothed.at<float>(0, static_cast<int>(i));
  }
  return out;
}

// Wiener-Khinchin: autocorrelation is the inverse DFT of the power
// spectrum. Zero-padding to >= 2n avoids circular wraparound
// contaminating lags in [0, n) (Müller et al. §4.1; see docs/PLAN.md
// Stage 2).
std::vector<PeriodicityPeak> autocorrelationPeaks(const std::vector<float>& profile,
                                                    const Config& config) {
  const int n = static_cast<int>(profile.size());
  if (n < 8) {
    return {};
  }

  double mean = 0.0;
  for (float v : profile) {
    mean += v;
  }
  mean /= n;

  const int padded_n = cv::getOptimalDFTSize(2 * n);
  cv::Mat padded = cv::Mat::zeros(1, padded_n, CV_32F);
  for (int i = 0; i < n; ++i) {
    padded.at<float>(0, i) = static_cast<float>(profile[static_cast<std::size_t>(i)] - mean);
  }

  cv::Mat spectrum;
  cv::dft(padded, spectrum, cv::DFT_COMPLEX_OUTPUT);

  cv::Mat power;
  cv::mulSpectrums(spectrum, spectrum, power, 0, /*conjB=*/true);

  cv::Mat autocorr;
  cv::idft(power, autocorr, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

  const float zero_lag = autocorr.at<float>(0, 0);
  if (zero_lag <= 0.0F) {
    return {};
  }

  const int min_lag = std::max(4, static_cast<int>(0.03 * n));
  const int max_lag = std::min(n / 2, autocorr.cols - 1);

  std::vector<PeriodicityPeak> peaks;
  for (int lag = min_lag; lag < max_lag; ++lag) {
    const float v = autocorr.at<float>(0, lag);
    const float prev = autocorr.at<float>(0, lag - 1);
    const float next = autocorr.at<float>(0, lag + 1);
    if (v >= prev && v >= next) {
      const float score = v / zero_lag;
      if (score >= static_cast<float>(config.periodicity_min_score)) {
        peaks.push_back({lag, score});
      }
    }
  }

  std::sort(peaks.begin(), peaks.end(),
            [](const PeriodicityPeak& a, const PeriodicityPeak& b) { return a.score > b.score; });
  return peaks;
}

// See periodicity.hpp doc comment: this scans low-activity runs (not
// literal maxima) since separators are wall/pier gaps, not window edges.
std::vector<PeriodicityPeak> directPeaks(const std::vector<float>& profile, int min_distance_px,
                                          const Config& config) {
  const int n = static_cast<int>(profile.size());
  if (n < 3) {
    return {};
  }

  const float max_val = *std::max_element(profile.begin(), profile.end());
  if (max_val <= 0.0F) {
    return {};
  }
  const float threshold = max_val * static_cast<float>(config.direct_peak_low_activity_frac);

  struct Run {
    int start;
    int end;  // inclusive
  };
  std::vector<Run> runs;
  int run_start = -1;
  for (int i = 0; i < n; ++i) {
    const bool below = profile[static_cast<std::size_t>(i)] <= threshold;
    if (below && run_start < 0) {
      run_start = i;
    }
    if ((!below || i == n - 1) && run_start >= 0) {
      const int run_end = below ? i : i - 1;
      runs.push_back({run_start, run_end});
      run_start = -1;
    }
  }

  std::vector<PeriodicityPeak> candidates;
  candidates.reserve(runs.size());
  for (const auto& run : runs) {
    // Drop runs touching either end of the profile: those are the
    // facade's own outer margins (already implied by the FacadeGrid's
    // forced 0/n endpoints), not interior separators. Checked on the
    // run's extent, not its center, since Gaussian smoothing can shrink
    // an edge-touching run's low-activity extent without moving its
    // start away from index 0 / n-1.
    if (run.start <= 0 || run.end >= n - 1) {
      continue;
    }
    const int center = (run.start + run.end) / 2;
    double sum = 0.0;
    for (int i = run.start; i <= run.end; ++i) {
      sum += profile[static_cast<std::size_t>(i)];
    }
    const double mean_val = sum / (run.end - run.start + 1);
    candidates.push_back({center, static_cast<float>(1.0 - mean_val / max_val)});
  }

  if (min_distance_px > 0) {
    std::sort(candidates.begin(), candidates.end(),
              [](const PeriodicityPeak& a, const PeriodicityPeak& b) {
                return a.score > b.score;
              });
    std::vector<PeriodicityPeak> accepted;
    for (const auto& c : candidates) {
      const bool too_close =
          std::any_of(accepted.begin(), accepted.end(), [&](const PeriodicityPeak& a) {
            return std::abs(a.lag_px - c.lag_px) < min_distance_px;
          });
      if (!too_close) {
        accepted.push_back(c);
      }
    }
    candidates = std::move(accepted);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const PeriodicityPeak& a, const PeriodicityPeak& b) { return a.lag_px < b.lag_px; });
  return candidates;
}

namespace {

// One attempt at building a boundary list from a single candidate
// period: runs directPeaks with that period as the NMS distance, then
// checks whether the resulting spacing actually agrees with it.
struct BoundaryAttempt {
  std::vector<int> boundaries;
  bool low_confidence = true;
  bool agrees = false;  ///< True once `boundaries` is non-empty and confident.
};

BoundaryAttempt tryBoundariesForPeriod(const std::vector<float>& smoothed, int n, int period,
                                        const Config& config) {
  BoundaryAttempt attempt;
  const auto direct = directPeaks(smoothed, period, config);
  if (direct.empty()) {
    return attempt;
  }

  attempt.boundaries.push_back(0);
  for (const auto& p : direct) {
    attempt.boundaries.push_back(p.lag_px);
  }
  attempt.boundaries.push_back(n);

  if (direct.size() >= 2 && period > 0) {
    double total_spacing = 0.0;
    for (std::size_t i = 0; i + 1 < direct.size(); ++i) {
      total_spacing += (direct[i + 1].lag_px - direct[i].lag_px);
    }
    const double avg_spacing = total_spacing / static_cast<double>(direct.size() - 1);
    attempt.low_confidence = std::abs(avg_spacing - period) > config.periodicity_agreement_tol_px;
  }
  attempt.agrees = !attempt.low_confidence;
  return attempt;
}

}  // namespace

PeriodicityResult analyzePeriodicity(const std::vector<float>& profile, const Config& config) {
  PeriodicityResult result;
  result.smoothed_profile = smoothProfile(profile, config);
  const int n = static_cast<int>(result.smoothed_profile.size());

  if (n < 2) {
    result.boundary_positions_px = {0, n};
    result.low_confidence = true;
    return result;
  }

  const auto autocorr_peaks = autocorrelationPeaks(result.smoothed_profile, config);

  // Try each autocorrelation candidate period, best-scored first,
  // accepting the first whose direct-peak spacing actually agrees with
  // it — rather than committing to the top-scored candidate no matter
  // what. One strongly irregular cell in an otherwise-regular row (e.g.
  // a door much narrower than its neighboring windows) can make a
  // sub-harmonic of the true period score marginally higher than the
  // true period itself, which used to make this row over-split; the
  // true period is typically still in the candidate list, just not
  // ranked first, so a disagreeing top candidate is worth a second try
  // rather than an immediate low_confidence result. See docs/PLAN.md's
  // "Known limitation" writeup.
  BoundaryAttempt best;
  bool have_best = false;
  for (const auto& peak : autocorr_peaks) {
    BoundaryAttempt attempt = tryBoundariesForPeriod(result.smoothed_profile, n, peak.lag_px, config);
    if (attempt.boundaries.empty()) {
      continue;
    }
    if (!have_best) {
      best = attempt;
      have_best = true;
    }
    if (attempt.agrees) {
      best = attempt;
      break;
    }
  }

  if (!have_best) {
    // No autocorrelation candidate produced a usable split (or there
    // were no candidates at all): fall back to a threshold-only direct
    // pass, matching the original no-period behavior.
    BoundaryAttempt attempt = tryBoundariesForPeriod(result.smoothed_profile, n, 0, config);
    if (!attempt.boundaries.empty()) {
      best = attempt;
      have_best = true;
    }
  }

  if (!have_best) {
    result.boundary_positions_px = {0, n};
    result.low_confidence = true;
    return result;
  }

  result.boundary_positions_px = best.boundaries;
  result.low_confidence = best.low_confidence;
  return result;
}

}  // namespace facade_parser
