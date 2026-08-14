/// \file io_json.hpp
/// \brief Stage 8 — export (JSON serialization + debug overlay).
///
/// Serializes a `FacadeResult` to the schema documented in
/// docs/OUTPUT_FORMAT.md, and renders a debug overlay image (grid lines,
/// color-coded classified cells, edge polylines) for visual QA.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core.hpp>
#include <optional>
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

/// Serializes every `Config` field to JSON (flat object, field names
/// match the struct member names). Used by the CLI's `tune` subcommand
/// and the GUI's auto-tuner to persist/reload a tuned config — see
/// docs/PLAN.md, "Auto-tuning pipeline".
nlohmann::json configToJson(const Config& config);

/// Reads back a `Config` produced by `configToJson`. Fields absent from
/// `j` keep `Config{}`'s own default (so an older/partial config file
/// still loads cleanly against a newer `Config` with more fields).
Config configFromJson(const nlohmann::json& j);

/// Writes `configToJson(config)` to `path`. Returns false on I/O failure.
bool writeConfigJson(const Config& config, const std::string& path);

/// Reads a `Config` from `path`. Returns nullopt on I/O failure or
/// malformed JSON.
std::optional<Config> readConfigJson(const std::string& path);

}  // namespace facade_parser
