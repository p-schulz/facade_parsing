/// \file autotune.hpp
/// \brief Deterministic parameter search for the auto-tuning pipeline
/// (see docs/PLAN.md, "Auto-tuning pipeline"): coordinate descent /
/// pattern search from a few fixed starting configs, no RNG anywhere
/// (consistent with the rest of this codebase's "no reliance on RNG"
/// rule).
///
/// Deliberately pipeline-agnostic: `autoTune()` optimizes an injected
/// `ScoreFn`, not `facade_parser::run()` directly, so the optimizer
/// itself is testable in milliseconds against a synthetic scoring
/// function (see tests/test_autotune.cpp) without needing real images.
/// `makeDatasetScoreFn()` is the only place this module touches the real
/// pipeline/OpenCV — it builds the `ScoreFn` production code actually
/// uses.
#pragma once

#include <functional>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

#include "facade_parser/evaluation.hpp"
#include "facade_parser/ground_truth.hpp"
#include "facade_parser/types.hpp"

namespace facade_parser {

/// Higher is better. Typically `scoreDataset(...).mean_f1`, but any
/// deterministic function of `Config` works (see tests/test_autotune.cpp
/// for a non-pipeline example).
using ScoreFn = std::function<double(const Config&)>;

/// One tunable `Config` field, accessed through get/set closures so
/// `double` and `int` members share one representation (int fields round
/// on `set`). `name` matches the CLI flag / GUI label for cross-reference
/// (see apps/facade_parser_cli.cpp, apps/facade_parser_gui/main.cpp).
struct TunableParam {
  std::string name;
  double min = 0.0;
  double max = 1.0;
  double initial_step = 0.0;
  std::function<double(const Config&)> get;
  std::function<void(Config&, double)> set;
};

/// The default search space: Stage 1's Canny thresholds and angle
/// tolerance, all of Stage 2, both of Stage 3, all of Stage 4, and
/// Stage 5's refine window fraction — box-detection-affecting params
/// only. Excludes: `hough_*` (only used in the HoughLinesP fallback path,
/// no effect when ximgproc/FastLineDetector is available), Stage 7's
/// `min_edge_length_px`/`edge_claim_margin_px` (edges are out of scope
/// for v1's box-only annotations), and all three `bool` fields (don't
/// fit a shrinking-step numeric search — see `defaultStartingConfigs()`
/// for how `enable_lattice_refine` is varied instead; `enable_symmetry_check`
/// is excluded entirely since Stage 6 output never reaches
/// `FacadeResult::elements`, so it cannot affect a box-detection score;
/// `emit_wall_elements` stays off since Wall isn't a v1 annotation type).
std::vector<TunableParam> defaultTunableParams();

/// 2-3 hand-picked, hard-coded starting configs (not randomly sampled —
/// see file doc comment) giving the multi-start search real diversity:
/// the current defaults, a recall-biased "loose thresholds" variant, and
/// a precision-biased "strict thresholds" variant. `enable_lattice_refine`
/// differs across these instead of being a `TunableParam`.
std::vector<Config> defaultStartingConfigs();

struct TuneOptions {
  std::vector<TunableParam> params = defaultTunableParams();
  std::vector<Config> starting_configs = defaultStartingConfigs();
  int max_iterations_per_start = 6;
  double step_shrink_factor = 0.5;
  double min_step_frac = 0.01;  ///< Stop refining a param once its step
                                 ///< shrinks below this fraction of
                                 ///< (max - min).
  std::optional<double> max_duration_seconds;  ///< Optional wall-clock
                                                ///< budget, checked
                                                ///< between trials.
};

/// One single-parameter trial, for progress reporting (a GUI progress
/// bar / log) and for `tests/test_autotune.cpp`'s convergence assertions.
struct TuneStepResult {
  int start_index = 0;
  int iteration = 0;
  std::string param_name;
  double old_value = 0.0;
  double new_value = 0.0;
  double old_score = 0.0;
  double new_score = 0.0;
  bool improved = false;
};

struct TuneResult {
  Config best_config;
  double best_score = 0.0;
  std::vector<TuneStepResult> history;
};

/// Return false to request cooperative early stop (e.g. a GUI Cancel
/// button) — checked after every trial, same as `max_duration_seconds`.
using TuneProgressCallback = std::function<bool(const TuneStepResult&)>;

/// Deterministic coordinate descent from each of `options.starting_configs`
/// in turn: repeatedly sweeps `options.params`, for each trying +/-step
/// from the current value and taking the first improvement (greedy),
/// shrinking the step when neither direction improves; a start converges
/// when a full sweep yields no improvement or every param's step has
/// shrunk below `options.min_step_frac`. Returns the best result found
/// across all starts.
TuneResult autoTune(const ScoreFn& evaluate, const TuneOptions& options,
                     const TuneProgressCallback& progress_callback = nullptr);

/// Production convenience wrapper: builds a `ScoreFn` that runs
/// `facade_parser::run()` over `images` with the trial `Config` and
/// reduces the result via `scoreDataset(...).mean_f1` against
/// `ground_truth`. `images.size()` must equal `ground_truth.size()`
/// (paired by index).
ScoreFn makeDatasetScoreFn(const std::vector<cv::Mat>& images,
                            const std::vector<GroundTruthImage>& ground_truth,
                            const ScoringOptions& scoring_options = {});

}  // namespace facade_parser
