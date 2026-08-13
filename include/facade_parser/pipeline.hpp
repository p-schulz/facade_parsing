/// \file pipeline.hpp
/// \brief Orchestrates Stages 1-8 behind a single entry point.
///
/// Not part of the original per-stage header list in the design prompt;
/// added so the CLI and tests don't hand-assemble all eight stages
/// themselves. See docs/PLAN.md, "Module breakdown" note.
#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Runs the full pipeline (Stages 1-8) on `bgr_image`. `source_image_name`
/// is stored verbatim as `FacadeResult::source_image` (basename, not a
/// path — the caller decides what to pass). `facade_bbox` defaults to the
/// whole image when not given.
FacadeResult run(const cv::Mat& bgr_image, const std::string& source_image_name,
                  const Config& config = {});

}  // namespace facade_parser
