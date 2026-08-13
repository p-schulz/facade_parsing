#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

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

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"facade_parser: classical-CV facade grid/window/door/edge extraction"};

  std::string input_path;
  std::string output_dir = ".";
  facade_parser::Config config;

  app.add_option("-i,--input", input_path, "Path to a rectified facade photo")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", output_dir, "Output directory for JSON + debug overlay")
      ->default_val(".");
  addConfigOptions(app, config);

  CLI11_PARSE(app, argc, argv);

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
