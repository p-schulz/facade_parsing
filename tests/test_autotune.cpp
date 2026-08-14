#include <gtest/gtest.h>

#include "facade_parser/autotune.hpp"

namespace {

// A synthetic ScoreFn with a known optimum, used to test the optimizer
// itself in isolation from the real pipeline/OpenCV (see autotune.hpp's
// file doc comment for why autoTune() is pipeline-agnostic). Reuses two
// arbitrary Config double fields as stand-in 2D coordinates — their
// actual pipeline meaning is irrelevant here.
constexpr double kTargetX = 120.0;  // canny_low
constexpr double kTargetY = 0.6;    // periodicity_min_score

double syntheticScore(const facade_parser::Config& c) {
  const double dx = c.canny_low - kTargetX;
  const double dy = c.periodicity_min_score - kTargetY;
  return -(dx * dx + dy * dy);  // Maximized (== 0) exactly at the target.
}

facade_parser::TuneOptions makeSyntheticOptions() {
  facade_parser::TuneOptions options;
  options.params = {
      {"canny_low", 0.0, 255.0, 30.0,
       [](const facade_parser::Config& c) { return c.canny_low; },
       [](facade_parser::Config& c, double v) { c.canny_low = v; }},
      {"periodicity_min_score", 0.0, 1.0, 0.1,
       [](const facade_parser::Config& c) { return c.periodicity_min_score; },
       [](facade_parser::Config& c, double v) { c.periodicity_min_score = v; }},
  };
  facade_parser::Config start;
  start.canny_low = 10.0;
  start.periodicity_min_score = 0.05;
  options.starting_configs = {start};
  options.max_iterations_per_start = 60;
  // Precision floor is min_step_frac * (max - min) per param (search
  // stops refining a param once its step shrinks below that) — 0.0002
  // here means canny_low (range 255) can get within ~0.0255 of the
  // target and periodicity_min_score (range 1.0) within ~0.0001,
  // comfortably inside this test's tolerances below.
  options.min_step_frac = 0.0002;
  return options;
}

}  // namespace

TEST(AutoTune, ConvergesNearKnownOptimum) {
  const auto options = makeSyntheticOptions();

  const auto result = facade_parser::autoTune(syntheticScore, options);

  EXPECT_NEAR(result.best_config.canny_low, kTargetX, 0.1);
  EXPECT_NEAR(result.best_config.periodicity_min_score, kTargetY, 0.001);
  EXPECT_NEAR(result.best_score, 0.0, 0.001);
  EXPECT_FALSE(result.history.empty());
}

TEST(AutoTune, NeverProducesAWorseResultThanTheStartingConfig) {
  const auto options = makeSyntheticOptions();
  const double starting_score = syntheticScore(options.starting_configs.front());

  const auto result = facade_parser::autoTune(syntheticScore, options);

  EXPECT_GE(result.best_score, starting_score);
}

TEST(AutoTune, MultiStartKeepsTheBestAcrossStarts) {
  auto options = makeSyntheticOptions();
  // A second start already at the optimum with a step size too small to
  // move it (min_step_frac stops it immediately) — should still win over
  // a first start that has to travel further and may not fully converge
  // within the iteration budget.
  facade_parser::Config already_optimal;
  already_optimal.canny_low = kTargetX;
  already_optimal.periodicity_min_score = kTargetY;
  options.starting_configs.push_back(already_optimal);
  options.max_iterations_per_start = 1;  // Too few for the far start to converge.

  const auto result = facade_parser::autoTune(syntheticScore, options);

  EXPECT_NEAR(result.best_score, 0.0, 1e-9);
}

TEST(AutoTune, ProgressCallbackReturningFalseStopsEarly) {
  const auto options = makeSyntheticOptions();
  int call_count = 0;
  const facade_parser::TuneProgressCallback stop_after_three = [&](const facade_parser::TuneStepResult&) {
    ++call_count;
    return call_count < 3;
  };

  facade_parser::autoTune(syntheticScore, options, stop_after_three);

  EXPECT_EQ(call_count, 3);
}

TEST(AutoTune, DefaultTunableParamsIsNonEmptyAndWithinBounds) {
  const auto params = facade_parser::defaultTunableParams();

  ASSERT_FALSE(params.empty());
  const facade_parser::Config defaults;
  for (const auto& param : params) {
    const double value = param.get(defaults);
    EXPECT_GE(value, param.min) << param.name;
    EXPECT_LE(value, param.max) << param.name;
    EXPECT_GT(param.initial_step, 0.0) << param.name;
  }
}

TEST(AutoTune, DefaultStartingConfigsIsNonEmpty) {
  EXPECT_GE(facade_parser::defaultStartingConfigs().size(), 2U);
}
