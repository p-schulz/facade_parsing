#include "facade_parser/autotune.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "facade_parser/pipeline.hpp"

namespace facade_parser {

namespace {

// int fields round on set; get() promotes to double for the shared
// TunableParam representation.
TunableParam intParam(std::string name, int Config::*field, double min, double max,
                       double initial_step) {
  TunableParam param;
  param.name = std::move(name);
  param.min = min;
  param.max = max;
  param.initial_step = initial_step;
  param.get = [field](const Config& c) { return static_cast<double>(c.*field); };
  param.set = [field](Config& c, double v) { c.*field = static_cast<int>(std::lround(v)); };
  return param;
}

TunableParam doubleParam(std::string name, double Config::*field, double min, double max,
                          double initial_step) {
  TunableParam param;
  param.name = std::move(name);
  param.min = min;
  param.max = max;
  param.initial_step = initial_step;
  param.get = [field](const Config& c) { return c.*field; };
  param.set = [field](Config& c, double v) { c.*field = v; };
  return param;
}

}  // namespace

std::vector<TunableParam> defaultTunableParams() {
  std::vector<TunableParam> params;

  // Stage 1 (Canny + angle tolerance only; hough_* excluded, see header).
  params.push_back(doubleParam("canny_low", &Config::canny_low, 0.0, 255.0, 30.0));
  params.push_back(doubleParam("canny_high", &Config::canny_high, 0.0, 255.0, 30.0));
  params.push_back(doubleParam("line_angle_tolerance_deg", &Config::line_angle_tolerance_deg, 0.0,
                                45.0, 5.0));

  // Stage 2.
  params.push_back(doubleParam("profile_smoothing_sigma_px", &Config::profile_smoothing_sigma_px,
                                0.1, 15.0, 2.0));
  params.push_back(
      doubleParam("periodicity_min_score", &Config::periodicity_min_score, 0.0, 1.0, 0.1));
  params.push_back(doubleParam("periodicity_agreement_tol_px",
                                &Config::periodicity_agreement_tol_px, 0.0, 50.0, 5.0));
  params.push_back(doubleParam("direct_peak_low_activity_frac",
                                &Config::direct_peak_low_activity_frac, 0.0, 1.0, 0.1));

  // Stage 3.
  params.push_back(intParam("min_cell_size_px", &Config::min_cell_size_px, 2.0, 300.0, 30.0));
  params.push_back(doubleParam("min_segment_width_frac_of_median",
                                &Config::min_segment_width_frac_of_median, 0.0, 1.0, 0.1));

  // Stage 4.
  params.push_back(
      doubleParam("otsu_close_kernel_frac", &Config::otsu_close_kernel_frac, 0.0, 0.5, 0.05));
  params.push_back(
      doubleParam("window_min_fill_ratio", &Config::window_min_fill_ratio, 0.0, 1.0, 0.1));
  params.push_back(
      doubleParam("window_max_fill_ratio", &Config::window_max_fill_ratio, 0.0, 1.0, 0.1));
  params.push_back(
      doubleParam("window_min_aspect", &Config::window_min_aspect, 0.05, 5.0, 0.5));
  params.push_back(
      doubleParam("window_max_aspect", &Config::window_max_aspect, 0.05, 5.0, 0.5));
  params.push_back(doubleParam("door_min_height_width_ratio",
                                &Config::door_min_height_width_ratio, 1.0, 6.0, 0.5));

  // Stage 5 (the refine window itself; enable_lattice_refine is a bool —
  // varied across defaultStartingConfigs() instead, see header).
  params.push_back(doubleParam("lattice_refine_window_frac", &Config::lattice_refine_window_frac,
                                0.0, 0.5, 0.05));

  return params;
}

std::vector<Config> defaultStartingConfigs() {
  std::vector<Config> configs;

  configs.push_back(Config{});  // Current hand-tuned defaults.

  Config loose;  // Recall-biased: catch more candidates, let scoring sort it out.
  loose.canny_low = 30.0;
  loose.canny_high = 90.0;
  loose.window_min_fill_ratio = 0.08;
  loose.window_max_fill_ratio = 0.98;
  loose.window_min_aspect = 0.15;
  loose.window_max_aspect = 4.0;
  loose.door_min_height_width_ratio = 1.5;
  loose.enable_lattice_refine = false;
  configs.push_back(loose);

  Config strict;  // Precision-biased: only confident, well-formed detections.
  strict.canny_low = 70.0;
  strict.canny_high = 180.0;
  strict.window_min_fill_ratio = 0.25;
  strict.window_max_fill_ratio = 0.85;
  strict.window_min_aspect = 0.5;
  strict.window_max_aspect = 2.0;
  strict.door_min_height_width_ratio = 2.2;
  strict.enable_lattice_refine = true;
  configs.push_back(strict);

  return configs;
}

TuneResult autoTune(const ScoreFn& evaluate, const TuneOptions& options,
                     const TuneProgressCallback& progress_callback) {
  TuneResult result;
  bool have_best = false;
  bool cancelled = false;

  const auto start_time = std::chrono::steady_clock::now();
  const auto timeExceeded = [&]() {
    if (!options.max_duration_seconds.has_value()) {
      return false;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    return elapsed >= *options.max_duration_seconds;
  };

  for (std::size_t start_index = 0;
       start_index < options.starting_configs.size() && !cancelled; ++start_index) {
    Config current = options.starting_configs[start_index];
    double current_score = evaluate(current);

    // Per-param step size, persisted *across* sweeps — only ever shrinks
    // (never resets to initial_step on the next sweep). This is what
    // makes convergence actually reach fine precision within a bounded
    // iteration budget: a param that's already been shrunk down close to
    // `min_step` on a previous sweep resumes from there, rather than
    // re-discovering the same shrink level by halving from scratch every
    // single sweep (the latter was tried first and wasted most of the
    // iteration budget re-shrinking instead of converging).
    std::vector<double> steps;
    steps.reserve(options.params.size());
    for (const auto& param : options.params) {
      steps.push_back(param.initial_step);
    }

    for (int iteration = 0; iteration < options.max_iterations_per_start && !cancelled;
         ++iteration) {
      bool improved_this_sweep = false;

      for (std::size_t param_index = 0; param_index < options.params.size(); ++param_index) {
        if (cancelled || timeExceeded()) {
          cancelled = true;
          break;
        }
        const auto& param = options.params[param_index];
        double& step = steps[param_index];
        const double min_step = options.min_step_frac * (param.max - param.min);
        bool improved_this_param = false;

        while (step >= min_step && !improved_this_param) {
          for (double direction : {1.0, -1.0}) {
            if (cancelled || timeExceeded()) {
              cancelled = true;
              break;
            }

            Config trial = current;
            const double trial_value =
                std::clamp(param.get(current) + direction * step, param.min, param.max);
            param.set(trial, trial_value);
            const double trial_score = evaluate(trial);

            TuneStepResult step_result;
            step_result.start_index = static_cast<int>(start_index);
            step_result.iteration = iteration;
            step_result.param_name = param.name;
            step_result.old_value = param.get(current);
            step_result.new_value = trial_value;
            step_result.old_score = current_score;
            step_result.new_score = trial_score;
            step_result.improved = trial_score > current_score;
            result.history.push_back(step_result);

            if (progress_callback && !progress_callback(step_result)) {
              cancelled = true;
            }

            if (step_result.improved) {
              current = trial;
              current_score = trial_score;
              improved_this_sweep = true;
              improved_this_param = true;
              break;  // Take the improvement; keep this step size for next sweep.
            }
          }

          if (!improved_this_param) {
            step *= options.step_shrink_factor;  // Both directions failed at this size.
          }
        }
      }

      if (!improved_this_sweep) {
        break;  // Converged for this starting config.
      }
    }

    if (!have_best || current_score > result.best_score) {
      result.best_config = current;
      result.best_score = current_score;
      have_best = true;
    }
  }

  return result;
}

ScoreFn makeDatasetScoreFn(const std::vector<cv::Mat>& images,
                            const std::vector<GroundTruthImage>& ground_truth,
                            const ScoringOptions& scoring_options) {
  return [images, ground_truth, scoring_options](const Config& config) {
    std::vector<FacadeResult> detected;
    detected.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
      detected.push_back(run(images[i], ground_truth[i].image_path, config));
    }
    return scoreDataset(detected, ground_truth, scoring_options).mean_f1;
  };
}

}  // namespace facade_parser
