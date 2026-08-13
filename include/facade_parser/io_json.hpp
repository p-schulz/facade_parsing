/// \file io_json.hpp
/// \brief Stage 8 — export (JSON serialization + debug overlay).
///
/// Serializes a `FacadeResult` to the schema documented in
/// docs/OUTPUT_FORMAT.md, and renders a debug overlay image (grid lines,
/// color-coded classified cells, edge polylines) for visual QA.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core.hpp>
#include <string>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// Converts `result` to its JSON representation (docs/OUTPUT_FORMAT.md).
nlohmann::json toJson(const FacadeResult& result);

/// Writes `toJson(result)` to `path`. Returns false on I/O failure.
bool writeResultJson(const FacadeResult& result, const std::string& path);

/// Draws `result`'s grid boundaries, classified cell rectangles
/// (color-coded by `ElementType`), and edge polylines over a copy of
/// `bgr_image`.
cv::Mat renderDebugOverlay(const cv::Mat& bgr_image, const FacadeResult& result);

/// Convenience: `cv::imwrite`s `renderDebugOverlay(bgr_image, result)` to
/// `path`. Returns false on I/O failure.
bool writeDebugOverlay(const cv::Mat& bgr_image, const FacadeResult& result,
                        const std::string& path);

}  // namespace facade_parser
