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

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"facade_parser: classical-CV facade grid/window/door/edge extraction"};

  std::string input_path;
  std::string output_dir = ".";
  bool no_lattice_refine = false;
  bool no_symmetry_check = false;

  app.add_option("-i,--input", input_path, "Path to a rectified facade photo")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", output_dir, "Output directory for JSON + debug overlay")
      ->default_val(".");
  app.add_flag("--no-lattice-refine", no_lattice_refine, "Disable Stage 5 lattice refinement");
  app.add_flag("--no-symmetry-check", no_symmetry_check, "Disable Stage 6 symmetry check");

  CLI11_PARSE(app, argc, argv);

  const cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "facade_parser: failed to read image: " << input_path << "\n";
    return 1;
  }

  facade_parser::Config config;
  config.enable_lattice_refine = !no_lattice_refine;
  config.enable_symmetry_check = !no_symmetry_check;

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
