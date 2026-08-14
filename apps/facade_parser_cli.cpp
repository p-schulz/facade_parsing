#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>
#include <opencv2/imgcodecs.hpp>

#include "facade_parser/autotune.hpp"
#include "facade_parser/evaluation.hpp"
#include "facade_parser/ground_truth.hpp"
#include "facade_parser/io_json.hpp"
#include "facade_parser/pipeline.hpp"

namespace {

std::filesystem::path withExtension(const std::filesystem::path& dir, const std::string& stem,
                                     const std::string& ext) {
  return dir / (stem + ext);
}

// Registers a CLI11 option per `Config` field (see docs/PLAN.md's
// "Module breakdown" table and types.hpp for what each one does),
// bound directly to `config`'s members so an unspecified flag keeps the
// struct's own default. Grouped by pipeline stage to mirror
// types.hpp / docs/PLAN.md, so `--help` output and the GUI's tuning
// panel stay easy to cross-reference against each other.
void addConfigOptions(CLI::App& app, facade_parser::Config& config) {
  auto* s1 = app.add_option_group("Stage 1: Edges");
  s1->add_option("--canny-low", config.canny_low, "Canny hysteresis low threshold")
      ->capture_default_str();
  s1->add_option("--canny-high", config.canny_high, "Canny hysteresis high threshold")
      ->capture_default_str();
  s1->add_option("--hough-min-line-length", config.hough_min_line_length_px,
                "HoughLinesP fallback: minimum line length (px)")
      ->capture_default_str();
  s1->add_option("--hough-max-line-gap", config.hough_max_line_gap_px,
                "HoughLinesP fallback: maximum gap to bridge (px)")
      ->capture_default_str();
  s1->add_option("--hough-threshold-votes", config.hough_threshold_votes,
                "HoughLinesP fallback: accumulator vote threshold")
      ->capture_default_str();
  s1->add_option("--line-angle-tolerance-deg", config.line_angle_tolerance_deg,
                "Max angle from 0/90 deg to still count as horizontal/vertical")
      ->capture_default_str();

  auto* s2 = app.add_option_group("Stage 2: Periodicity");
  s2->add_option("--profile-smoothing-sigma", config.profile_smoothing_sigma_px,
                "Gaussian sigma (px) for smoothing projection profiles")
      ->capture_default_str();
  s2->add_option("--periodicity-min-score", config.periodicity_min_score,
                "Min autocorrelation score (fraction of zero-lag) to accept a period")
      ->capture_default_str();
  s2->add_option("--periodicity-agreement-tol", config.periodicity_agreement_tol_px,
                "Max px disagreement between autocorrelation and direct peaks before "
                "flagging low_confidence")
      ->capture_default_str();
  s2->add_option("--direct-peak-low-activity-frac", config.direct_peak_low_activity_frac,
                "Fraction of profile max below which a run counts as a low-activity gap")
      ->capture_default_str();

  auto* s3 = app.add_option_group("Stage 3: Split grammar");
  s3->add_option("--min-cell-size", config.min_cell_size_px,
                "Minimum row/column band size (px) before recursion stops")
      ->capture_default_str();
  s3->add_option("--min-segment-width-frac", config.min_segment_width_frac_of_median,
                "A column (bay) split narrower than this fraction of its floor band's median "
                "bay width gets merged into a neighbor (guards against one irregular cell, "
                "e.g. a narrow door, over-splitting its row; NOT applied to row/floor "
                "boundaries, since floor heights legitimately vary)")
      ->capture_default_str();

  auto* s4 = app.add_option_group("Stage 4: Classification");
  s4->add_option("--otsu-close-kernel-frac", config.otsu_close_kernel_frac,
                "Morphological close kernel size, as a fraction of the cell's shorter side")
      ->capture_default_str();
  s4->add_option("--window-min-fill-ratio", config.window_min_fill_ratio,
                "Min contour-area/cell-area ratio to count as a window")
      ->capture_default_str();
  s4->add_option("--window-max-fill-ratio", config.window_max_fill_ratio,
                "Max contour-area/cell-area ratio to count as a window")
      ->capture_default_str();
  s4->add_option("--window-min-aspect", config.window_min_aspect,
                "Min bounding-box width/height ratio to count as a window")
      ->capture_default_str();
  s4->add_option("--window-max-aspect", config.window_max_aspect,
                "Max bounding-box width/height ratio to count as a window")
      ->capture_default_str();
  s4->add_option("--door-min-height-width-ratio", config.door_min_height_width_ratio,
                "Min height/width ratio (lowest floor band only) to reclassify a window as a door")
      ->capture_default_str();

  auto* s5 = app.add_option_group("Stage 5: Lattice refinement");
  s5->add_flag("--no-lattice-refine{false}", config.enable_lattice_refine,
              "Disable Stage 5 lattice refinement")
      ->default_val(true);
  s5->add_option("--lattice-refine-window-frac", config.lattice_refine_window_frac,
                "+/- fraction of local cell size to search when snapping a boundary")
      ->capture_default_str();

  auto* s6 = app.add_option_group("Stage 6: Symmetry");
  s6->add_flag("--no-symmetry-check{false}", config.enable_symmetry_check,
              "Disable Stage 6 symmetry check")
      ->default_val(true);

  auto* s7 = app.add_option_group("Stage 7: Edges/ledges");
  s7->add_option("--min-edge-length", config.min_edge_length_px,
                "Minimum line segment length (px) to emit as an edge element")
      ->capture_default_str();
  s7->add_option("--edge-claim-margin", config.edge_claim_margin_px,
                "Dilation margin (px) used to suppress segments already inside a window/door bbox")
      ->capture_default_str();

  auto* s8 = app.add_option_group("Stage 8: Export");
  s8->add_flag("--emit-wall-elements", config.emit_wall_elements,
              "Also emit `wall` elements for unclassified cells (default: implicit only)");
}

// Scans `dataset_dir` for `*.gt.json` sidecar annotation files (written
// by the GUI's annotation tool — see ground_truth.hpp), resolving each
// one's image relative to its own directory. Skips (with a warning, not
// a hard failure) any annotation file that fails to load or whose image
// can't be read, so one bad entry doesn't block tuning against the rest
// of the dataset.
bool loadDataset(const std::string& dataset_dir, std::vector<cv::Mat>* images,
                  std::vector<facade_parser::GroundTruthImage>* ground_truth) {
  for (const auto& entry : std::filesystem::directory_iterator(dataset_dir)) {
    const std::string filename = entry.path().filename().string();
    constexpr std::string_view kSuffix = ".gt.json";
    if (filename.size() <= kSuffix.size() ||
        filename.compare(filename.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
      continue;
    }

    const auto gt = facade_parser::loadGroundTruth(entry.path().string());
    if (!gt.has_value()) {
      std::cerr << "facade_parser: skipping malformed annotation file: " << entry.path() << "\n";
      continue;
    }

    const std::filesystem::path image_path = entry.path().parent_path() / gt->image_path;
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "facade_parser: skipping " << entry.path() << ": couldn't read image "
                << image_path << "\n";
      continue;
    }

    images->push_back(image);
    ground_truth->push_back(*gt);
  }
  return !images->empty();
}

int runTune(const std::string& dataset_dir, const std::string& output_path, int iterations,
            double iou_threshold) {
  std::vector<cv::Mat> images;
  std::vector<facade_parser::GroundTruthImage> ground_truth;
  if (!loadDataset(dataset_dir, &images, &ground_truth)) {
    std::cerr << "facade_parser: no usable *.gt.json annotations found in " << dataset_dir
              << "\n";
    return 1;
  }
  std::cout << "facade_parser: tuning against " << images.size() << " annotated image(s)\n";

  facade_parser::ScoringOptions scoring_options;
  scoring_options.iou_match_threshold = iou_threshold;
  const facade_parser::ScoreFn score_fn =
      facade_parser::makeDatasetScoreFn(images, ground_truth, scoring_options);

  facade_parser::TuneOptions tune_options;
  tune_options.max_iterations_per_start = iterations;

  const facade_parser::TuneProgressCallback progress = [](const facade_parser::TuneStepResult& s) {
    std::cout << "  start=" << s.start_index << " iter=" << s.iteration << " param=" << s.param_name
               << " " << s.old_value << " -> " << s.new_value << "  score " << s.old_score << " -> "
               << s.new_score << (s.improved ? "  (improved)" : "") << "\n";
    return true;  // The CLI never cancels; use the GUI for interactive control.
  };

  const facade_parser::TuneResult result = facade_parser::autoTune(score_fn, tune_options, progress);
  std::cout << "facade_parser: best score (mean F1) = " << result.best_score << "\n";

  if (!facade_parser::writeConfigJson(result.best_config, output_path)) {
    std::cerr << "facade_parser: failed to write " << output_path << "\n";
    return 1;
  }
  std::cout << "facade_parser: wrote tuned config to " << output_path << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Pass 1: peek at --config-file only (ignoring everything else,
  // including the `tune` subcommand's own tokens) so it can be loaded
  // *before* the real app captures per-flag CLI11 defaults from
  // `config`'s current values below. This is what makes the ordering
  // rule work: individually-passed flags always override the file, but
  // unspecified flags keep the file's values instead of Config{}'s
  // hardcoded defaults. See docs/PLAN.md, "Auto-tuning pipeline".
  std::string config_file_path;
  {
    CLI::App pre_app;
    pre_app.allow_extras();
    pre_app.add_option("--config-file", config_file_path);
    try {
      pre_app.parse(argc, argv);
    } catch (const CLI::ParseError&) {
      // Ignore here; the real parse below reports actual errors properly.
    }
  }

  facade_parser::Config config;
  if (!config_file_path.empty()) {
    if (const auto loaded = facade_parser::readConfigJson(config_file_path)) {
      config = *loaded;
    }
    // If the file doesn't exist or fails to parse, fall through silently
    // here — the real app's --config-file option below has
    // ->check(CLI::ExistingFile), so CLI11_PARSE will report a proper
    // user-facing error and exit before any of `config` is used.
  }

  CLI::App app{"facade_parser: classical-CV facade grid/window/door/edge extraction"};

  std::string input_path;
  std::string output_dir = ".";

  app.add_option("-i,--input", input_path, "Path to a rectified facade photo")
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", output_dir, "Output directory for JSON + debug overlay")
      ->default_val(".");
  app.add_option("--config-file", config_file_path,
                 "Load Config defaults from a JSON file (see `tune`'s --output); "
                 "individual flags below still override")
      ->check(CLI::ExistingFile);
  addConfigOptions(app, config);

  CLI::App* tune_cmd =
      app.add_subcommand("tune", "Search Config's parameter space against ground-truth "
                                  "annotations (see docs/PLAN.md, \"Auto-tuning pipeline\")");
  std::string dataset_dir;
  std::string tune_output = "tuned_config.json";
  int iterations = 6;
  double iou_threshold = 0.5;
  tune_cmd
      ->add_option("--dataset", dataset_dir,
                   "Directory containing images with sidecar *.gt.json annotation files "
                   "(see ground_truth.hpp / the GUI's annotation tool)")
      ->required()
      ->check(CLI::ExistingDirectory);
  tune_cmd->add_option("--output", tune_output, "Where to write the best Config found, as JSON")
      ->default_val("tuned_config.json");
  tune_cmd
      ->add_option("--iterations", iterations,
                   "Max coordinate-descent sweeps per starting config")
      ->default_val(6);
  tune_cmd
      ->add_option("--iou-threshold", iou_threshold,
                   "Min IoU to count a detection as matching a ground-truth box")
      ->default_val(0.5);

  CLI11_PARSE(app, argc, argv);

  if (tune_cmd->parsed()) {
    return runTune(dataset_dir, tune_output, iterations, iou_threshold);
  }

  if (input_path.empty()) {
    std::cerr << "facade_parser: --input is required (unless using the `tune` subcommand)\n";
    return 1;
  }

  const cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "facade_parser: failed to read image: " << input_path << "\n";
    return 1;
  }

  const std::filesystem::path in(input_path);
  const facade_parser::FacadeResult result = facade_parser::run(image, in.filename().string(), config);

  std::filesystem::create_directories(output_dir);
  const std::filesystem::path json_path = withExtension(output_dir, in.stem().string(), ".json");
  const std::filesystem::path overlay_path =
      withExtension(output_dir, in.stem().string() + "_overlay", ".png");

  if (!facade_parser::writeResultJson(result, json_path.string())) {
    std::cerr << "facade_parser: failed to write " << json_path << "\n";
    return 1;
  }
  if (!facade_parser::writeDebugOverlay(image, result, overlay_path.string())) {
    std::cerr << "facade_parser: failed to write " << overlay_path << "\n";
    return 1;
  }

  std::cout << "facade_parser: wrote " << json_path << " and " << overlay_path << "\n";
  return 0;
}
