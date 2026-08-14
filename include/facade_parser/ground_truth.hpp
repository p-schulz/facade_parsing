/// \file ground_truth.hpp
/// \brief Ground-truth annotation data model and JSON I/O for the
/// auto-tuning pipeline (see docs/PLAN.md, "Auto-tuning pipeline").
///
/// A `GroundTruthImage` is a user-drawn set of window/door boxes for one
/// facade photo, persisted as a sidecar `<stem>.gt.json` next to the
/// image (see docs/OUTPUT_FORMAT.md for the exact schema). This is
/// deliberately a separate, simpler schema from `FacadeResult`
/// (io_json.hpp) — no confidence, no edges/polylines, no grid — since
/// it's authored by a person, not produced by the pipeline. v1 scope is
/// boxes only (`ElementType::Window` / `ElementType::Door`); edges are a
/// documented future extension, not implemented here.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "facade_parser/types.hpp"

namespace facade_parser {

/// One hand-drawn ground-truth box. `type` is restricted to Window/Door
/// in this module's own API (loadGroundTruth/saveGroundTruth do not
/// reject other values on read, so a hand-edited file with e.g. `"wall"`
/// still round-trips, but nothing in the GUI or evaluation.hpp currently
/// produces or consumes anything but Window/Door — see docs/PLAN.md).
struct GroundTruthElement {
  ElementType type = ElementType::Window;
  cv::Rect bbox_px;
};

/// One image's worth of ground truth.
struct GroundTruthImage {
  std::string image_path;  ///< Relative to the annotation file's own
                            ///< directory, so a dataset stays portable as
                            ///< long as the image and its `.gt.json`
                            ///< move together.
  cv::Size image_size_px;
  std::vector<GroundTruthElement> elements;
};

/// Loads a `GroundTruthImage` from a `.gt.json` file. Returns nullopt on
/// I/O failure or malformed JSON (never throws for those cases —
/// exceptions are reserved for genuine programmer errors elsewhere in
/// this codebase, and "file doesn't parse" is an expected condition
/// here, not one).
std::optional<GroundTruthImage> loadGroundTruth(const std::string& path);

/// Writes `gt` to `path`. Returns false on I/O failure.
bool saveGroundTruth(const GroundTruthImage& gt, const std::string& path);

}  // namespace facade_parser
