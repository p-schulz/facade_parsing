/// \file main.cpp
/// \brief imgui-based GUI for facade_parser:
///  - a "File" menu to load a single image (Preview mode) or a whole
///    dataset of images (Dataset mode) and save results;
///  - in Preview mode, a rectification toolbar (see docs/PLAN.md,
///    "Stage 0: Rectification") — draggable corner handles over the raw
///    photo, a "Rectify" button, and a "Detect Features" button that
///    runs the same detection entry point either mode uses;
///  - a left-hand panel exposing every `Config` threshold (see
///    types.hpp) for interactive real-world tuning;
///  - a Dataset panel (dataset mode only) for drawing window/door
///    ground-truth boxes on each loaded photo and running the
///    auto-tuner against them (see docs/PLAN.md, "Auto-tuning
///    pipeline");
///  - a main pane previewing either the raw photo with corner handles,
///    the plain debug overlay, or (Dataset mode) a ground-truth/
///    detection comparison overlay.
/// Wraps the same library entry points (`facade_parser::run`,
/// `scoreImage`, `autoTune`, `rectify`, ...) the CLI uses — no pipeline,
/// tuning, or rectification logic lives here, only UI and glue.
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Drags in the system OpenGL headers.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "facade_parser/autotune.hpp"
#include "facade_parser/evaluation.hpp"
#include "facade_parser/ground_truth.hpp"
#include "facade_parser/io_json.hpp"
#include "facade_parser/pipeline.hpp"
#include "facade_parser/rectification.hpp"
#include "file_dialog_mac.h"

namespace {

// --- Dataset / annotation state --------------------------------------

// One loaded dataset photo plus its (possibly still-empty) ground truth.
// `source_bgr` is decoded once when added to the dataset and kept in
// memory for the rest of the session — datasets are expected to be a
// handful of photos, not thousands.
struct DatasetEntry {
  std::string image_path;  // Absolute path, as chosen in the file dialog.
  cv::Mat source_bgr;
  facade_parser::GroundTruthImage ground_truth;
  bool ground_truth_dirty = false;  // Unsaved annotation edits.
  std::optional<facade_parser::ImageScore> last_score;
};

// --- Background auto-tune worker --------------------------------------

// Snapshot of the background tune thread's progress, read by the main
// thread under `TuneWorker::mutex` each frame. Kept deliberately small
// (no per-trial history) — the GUI only needs a live trial count and
// best-score-so-far, not the full trace the CLI prints to stdout.
struct TuneSnapshot {
  bool running = false;
  bool done = false;
  int trials_done = 0;
  double best_score = 0.0;
  facade_parser::TuneResult result;  // Valid once `done` is true.
};

// Owns the background thread. The worker only ever touches its own
// captured copies of the dataset (passed by value into the thread
// lambda) and this snapshot under `mutex` — it never reaches back into
// AppState, since imgui/AppState aren't thread-safe. Destroying a
// TuneWorker requests cancellation and joins, so replacing or dropping
// `AppState::tune_worker` is always safe.
struct TuneWorker {
  std::thread thread;
  std::mutex mutex;
  TuneSnapshot snapshot;
  std::atomic<bool> cancel_requested{false};

  ~TuneWorker() {
    if (thread.joinable()) {
      cancel_requested = true;
      thread.join();
    }
  }
};

// Which image the PreviewPanel shows in Preview mode — see
// docs/PLAN.md, "Stage 0: Rectification". Not used in Dataset mode
// (that has its own annotate/comparison toggles, unaffected by this).
enum class PreviewView { Original, Rectified };

struct AppState {
  // Currently displayed image (Preview mode: the opened file; Dataset
  // mode: whichever entry is selected) and its rendered overlay.
  bool has_image = false;
  std::string image_path;
  cv::Mat source_bgr;
  facade_parser::Config config;
  facade_parser::FacadeResult result;
  cv::Mat overlay_bgr;
  GLuint texture = 0;
  int texture_width = 0;
  int texture_height = 0;
  std::string status;
  bool status_is_error = false;

  // Rectification (Preview mode only — see docs/PLAN.md, "Stage 0:
  // Rectification"). `corners` starts equal to `detected_corners` on
  // load and is what the user drags; "Reset to detected corners"
  // restores it from `detected_corners`.
  facade_parser::Quad detected_corners;
  facade_parser::Quad corners;
  bool has_rectified = false;
  cv::Mat rectified_bgr;
  GLuint original_texture = 0;  // Plain source_bgr, corner handles drawn as a live overlay.
  int original_texture_width = 0;
  int original_texture_height = 0;
  int dragging_corner_index = -1;  // -1 = none, else 0=TL 1=TR 2=BR 3=BL.
  PreviewView preview_view = PreviewView::Original;

  // Dataset mode.
  std::vector<DatasetEntry> dataset;
  int viewing_dataset_index = -1;  // -1 == Preview mode (single image).
  facade_parser::ElementType annotate_type = facade_parser::ElementType::Window;
  bool annotate_mode = false;
  bool show_comparison = false;
  bool dragging = false;
  ImVec2 drag_start_screen{0, 0};
  int right_clicked_annotation_index = -1;

  std::unique_ptr<TuneWorker> tune_worker;
};

// --- Rendering / pipeline glue -----------------------------------------

void uploadTexture(const cv::Mat& bgr, GLuint* texture, int* width_out, int* height_out) {
  cv::Mat rgba;
  cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

  if (*texture == 0) {
    glGenTextures(1, texture);
  }
  glBindTexture(GL_TEXTURE_2D, *texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               rgba.data);
  *width_out = rgba.cols;
  *height_out = rgba.rows;
}

void uploadOverlayTexture(AppState& state) {
  uploadTexture(state.overlay_bgr, &state.texture, &state.texture_width, &state.texture_height);
}

// Preview-mode (single image) render: re-runs the pipeline and shows
// the plain debug overlay. Uses `state.rectified_bgr` once one exists
// (see docs/PLAN.md, "Stage 0: Rectification" — "make the rectified
// image the default input, keep both paths"), else falls back to the
// raw `state.source_bgr` exactly as before rectification existed.
void runPipeline(AppState& state) {
  const cv::Mat& input = state.has_rectified ? state.rectified_bgr : state.source_bgr;
  const std::filesystem::path p(state.image_path);
  state.result = facade_parser::run(input, p.filename().string(), state.config);
  state.overlay_bgr = facade_parser::renderDebugOverlay(input, state.result);
  uploadOverlayTexture(state);

  int windows = 0;
  int doors = 0;
  int edges = 0;
  for (const auto& e : state.result.elements) {
    if (e.type == facade_parser::ElementType::Window) {
      ++windows;
    } else if (e.type == facade_parser::ElementType::Door) {
      ++doors;
    } else if (e.type == facade_parser::ElementType::Edge) {
      ++edges;
    }
  }
  char buf[320];
  std::snprintf(buf, sizeof(buf), "%s %s (%dx%d) - %d window(s), %d door(s), %d edge(s)",
                state.has_rectified ? "Detected on rectified" : "Loaded",
                p.filename().string().c_str(), input.cols, input.rows, windows, doors, edges);
  state.status = buf;
  state.status_is_error = false;
}

// Dataset-mode render: shows ground-truth boxes (yellow, "false
// negative" styling — see renderComparisonOverlay) alone, or overlaid
// against live detections (green/red/yellow TP/FP/FN) when
// `state.show_comparison` is on. Reuses evaluation.hpp's
// renderComparisonOverlay for both cases via an empty FacadeResult when
// not comparing, rather than a separate GT-only drawing routine.
void refreshDatasetPreview(AppState& state) {
  DatasetEntry& entry = state.dataset[static_cast<std::size_t>(state.viewing_dataset_index)];

  facade_parser::FacadeResult result;
  if (state.show_comparison) {
    const std::filesystem::path p(entry.image_path);
    result = facade_parser::run(entry.source_bgr, p.filename().string(), state.config);
  }
  const facade_parser::ImageScore score = facade_parser::scoreImage(result, entry.ground_truth);
  entry.last_score = score;

  state.source_bgr = entry.source_bgr;
  state.image_path = entry.image_path;
  state.result = result;
  state.overlay_bgr =
      facade_parser::renderComparisonOverlay(entry.source_bgr, result, entry.ground_truth, score);
  uploadOverlayTexture(state);

  char buf[320];
  std::snprintf(buf, sizeof(buf), "%s - %zu annotation(s): TP=%d FP=%d FN=%d F1=%.3f",
                std::filesystem::path(entry.image_path).filename().string().c_str(),
                entry.ground_truth.elements.size(), score.true_positives, score.false_positives,
                score.false_negatives, score.f1);
  state.status = buf;
  state.status_is_error = false;
}

// Re-renders whichever mode is active — the one function config edits /
// dataset-selection changes / annotation edits all call.
void refreshPreview(AppState& state) {
  if (state.viewing_dataset_index >= 0) {
    refreshDatasetPreview(state);
  } else if (state.has_image) {
    runPipeline(state);
  }
}

void loadImage(AppState& state, const std::string& path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    state.status = "Failed to read image: " + path;
    state.status_is_error = true;
    return;
  }

  state.source_bgr = image;
  state.image_path = path;
  state.has_image = true;
  state.viewing_dataset_index = -1;  // Leave dataset mode, if it was active.

  // Rectification state, per docs/PLAN.md "State management": a fresh
  // image clears any previous corners/rectified result and regenerates
  // the automatic proposal.
  state.detected_corners = facade_parser::proposeCorners(image);
  state.corners = state.detected_corners;
  state.has_rectified = false;
  state.rectified_bgr = cv::Mat();
  state.preview_view = PreviewView::Original;
  uploadTexture(state.source_bgr, &state.original_texture, &state.original_texture_width,
                &state.original_texture_height);

  runPipeline(state);  // Raw-image detection, same as before rectification existed.
}

void saveJson(AppState& state, const std::string& path) {
  if (!facade_parser::writeResultJson(state.result, path)) {
    state.status = "Failed to write JSON: " + path;
    state.status_is_error = true;
    return;
  }
  state.status = "Saved JSON to " + path;
  state.status_is_error = false;
}

// Loads each of `paths` into a new DatasetEntry, auto-loading its
// sidecar `<stem>.gt.json` if one already exists next to the image
// (same convention the CLI's `tune --dataset` scan uses — see
// apps/facade_parser_cli.cpp's loadDataset). Unreadable files are
// skipped, not fatal to the rest of the batch.
void loadDataset(AppState& state, const std::vector<std::string>& paths) {
  for (const auto& path : paths) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
      continue;
    }

    DatasetEntry entry;
    entry.image_path = path;
    entry.source_bgr = image;

    const std::filesystem::path p(path);
    const std::filesystem::path gt_path = p.parent_path() / (p.stem().string() + ".gt.json");
    if (const auto loaded = facade_parser::loadGroundTruth(gt_path.string())) {
      entry.ground_truth = *loaded;
    } else {
      entry.ground_truth.image_path = p.filename().string();
      entry.ground_truth.image_size_px = image.size();
    }
    state.dataset.push_back(std::move(entry));
  }

  if (!state.dataset.empty() && state.viewing_dataset_index < 0) {
    state.viewing_dataset_index = 0;
    state.has_image = true;
    refreshDatasetPreview(state);
  }
}

void saveCurrentAnnotations(AppState& state) {
  if (state.viewing_dataset_index < 0) {
    return;
  }
  DatasetEntry& entry = state.dataset[static_cast<std::size_t>(state.viewing_dataset_index)];
  const std::filesystem::path p(entry.image_path);
  const std::filesystem::path gt_path = p.parent_path() / (p.stem().string() + ".gt.json");
  entry.ground_truth.image_path = p.filename().string();
  entry.ground_truth.image_size_px = entry.source_bgr.size();

  if (facade_parser::saveGroundTruth(entry.ground_truth, gt_path.string())) {
    entry.ground_truth_dirty = false;
    state.status = "Saved annotations to " + gt_path.string();
    state.status_is_error = false;
  } else {
    state.status = "Failed to save annotations to " + gt_path.string();
    state.status_is_error = true;
  }
}

// --- Screen <-> image pixel coordinate transform ------------------------
// Only needed at the moment a drag finishes / a click is hit-tested (see
// the preview-pane annotation handling in main()) — everything else
// (existing boxes, comparison colors) is baked into the overlay cv::Mat
// via OpenCV drawing and displayed like any other texture, so there's no
// need for a parallel per-frame imgui-draw-list transform for static
// content.

cv::Rect screenRectToImageRect(ImVec2 p0, ImVec2 p1, ImVec2 image_min, ImVec2 image_max,
                                cv::Size image_pixel_size) {
  const float x0 = std::clamp(std::min(p0.x, p1.x), image_min.x, image_max.x);
  const float x1 = std::clamp(std::max(p0.x, p1.x), image_min.x, image_max.x);
  const float y0 = std::clamp(std::min(p0.y, p1.y), image_min.y, image_max.y);
  const float y1 = std::clamp(std::max(p0.y, p1.y), image_min.y, image_max.y);
  const float scale_x =
      static_cast<float>(image_pixel_size.width) / std::max(1.0F, image_max.x - image_min.x);
  const float scale_y =
      static_cast<float>(image_pixel_size.height) / std::max(1.0F, image_max.y - image_min.y);
  const int px0 = static_cast<int>((x0 - image_min.x) * scale_x);
  const int py0 = static_cast<int>((y0 - image_min.y) * scale_y);
  const int px1 = static_cast<int>((x1 - image_min.x) * scale_x);
  const int py1 = static_cast<int>((y1 - image_min.y) * scale_y);
  return cv::Rect(px0, py0, std::max(1, px1 - px0), std::max(1, py1 - py0));
}

cv::Point2f screenPointToImagePoint(ImVec2 p, ImVec2 image_min, ImVec2 image_max,
                                     cv::Size image_pixel_size) {
  const float scale_x =
      static_cast<float>(image_pixel_size.width) / std::max(1.0F, image_max.x - image_min.x);
  const float scale_y =
      static_cast<float>(image_pixel_size.height) / std::max(1.0F, image_max.y - image_min.y);
  return {(p.x - image_min.x) * scale_x, (p.y - image_min.y) * scale_y};
}

// Forward transform (image pixel -> screen), needed for the
// rectification corner handles: unlike the dataset annotation boxes
// (baked into the overlay cv::Mat and displayed like any other
// texture), the handles are live `ImDrawList` overlay drawn fresh every
// frame, so dragging one doesn't require re-uploading a texture.
ImVec2 imagePointToScreen(cv::Point2f p, ImVec2 image_min, ImVec2 image_max,
                           cv::Size image_pixel_size) {
  const float scale_x =
      (image_max.x - image_min.x) / static_cast<float>(std::max(1, image_pixel_size.width));
  const float scale_y =
      (image_max.y - image_min.y) / static_cast<float>(std::max(1, image_pixel_size.height));
  return {image_min.x + p.x * scale_x, image_min.y + p.y * scale_y};
}

// Indexes a Quad's four named corners 0=TL 1=TR 2=BR 3=BL (matching the
// order the rectification toolbar draws/hit-tests them in) — keeps
// Quad's own public API named-field, matching this codebase's style
// elsewhere (bbox_px, polyline_px, ...), while still letting the drag
// interaction below iterate over "corner index" conveniently.
cv::Point2f* cornerByIndex(facade_parser::Quad& quad, int index) {
  switch (index) {
    case 0:
      return &quad.top_left;
    case 1:
      return &quad.top_right;
    case 2:
      return &quad.bottom_right;
    default:
      return &quad.bottom_left;
  }
}

// --- Auto-tune background thread ----------------------------------------

// Starts a fresh tune run against every dataset entry that has at least
// one annotation. Replacing `state.tune_worker` (if a previous run is
// still active) safely cancels and joins it first via TuneWorker's
// destructor.
void startAutoTune(AppState& state) {
  std::vector<cv::Mat> images;
  std::vector<facade_parser::GroundTruthImage> ground_truth;
  for (const auto& entry : state.dataset) {
    if (!entry.ground_truth.elements.empty()) {
      images.push_back(entry.source_bgr);
      ground_truth.push_back(entry.ground_truth);
    }
  }
  if (images.empty()) {
    state.status = "No annotated images in the dataset to tune against.";
    state.status_is_error = true;
    return;
  }

  state.tune_worker = std::make_unique<TuneWorker>();
  TuneWorker* worker = state.tune_worker.get();
  worker->snapshot.running = true;

  // `images`/`ground_truth` are captured by value: cv::Mat's copy is a
  // cheap shallow (ref-counted) copy, and the worker must never share
  // mutable state with the main thread's AppState.
  worker->thread = std::thread([worker, images, ground_truth]() {
    const facade_parser::ScoreFn score_fn =
        facade_parser::makeDatasetScoreFn(images, ground_truth);
    const facade_parser::TuneOptions options;  // Defaults from autotune.hpp.

    const facade_parser::TuneProgressCallback progress =
        [worker](const facade_parser::TuneStepResult& step) {
          {
            const std::lock_guard<std::mutex> lock(worker->mutex);
            worker->snapshot.trials_done += 1;
            worker->snapshot.best_score =
                std::max({worker->snapshot.best_score, step.old_score, step.new_score});
          }
          return !worker->cancel_requested.load();
        };

    const facade_parser::TuneResult result = facade_parser::autoTune(score_fn, options, progress);

    const std::lock_guard<std::mutex> lock(worker->mutex);
    worker->snapshot.result = result;
    worker->snapshot.best_score = result.best_score;
    worker->snapshot.running = false;
    worker->snapshot.done = true;
  });
}

void glfwErrorCallback(int error, const char* description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

// "(?)" marker that shows `desc` in a tooltip on hover — used to surface
// the same per-parameter explanations that live as comments on `Config`
// in types.hpp, without cluttering the panel with wrapped body text.
void helpMarker(const char* desc) {
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip()) {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0F);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

// Slider bound directly to a `double` field (ImGui has no SliderDouble;
// SliderScalar with ImGuiDataType_Double avoids a lossy float round trip
// through Config's double members). Returns true once the edit is
// committed (mouse released / field defocused), not on every drag frame
// — re-running the full pipeline on every intermediate drag value would
// make dragging feel laggy for no benefit.
bool tuneDouble(const char* label, double* value, double min, double max,
                 const char* fmt = "%.3f") {
  ImGui::SliderScalar(label, ImGuiDataType_Double, value, &min, &max, fmt);
  return ImGui::IsItemDeactivatedAfterEdit();
}

bool tuneInt(const char* label, int* value, int min, int max) {
  ImGui::SliderInt(label, value, min, max);
  return ImGui::IsItemDeactivatedAfterEdit();
}

// Draws every `Config` field, grouped by pipeline stage to match
// types.hpp / docs/PLAN.md / the CLI's --help groups. Returns true if
// any value was committed this frame, so the caller knows to re-run.
bool drawTuningPanel(facade_parser::Config& c) {
  bool changed = false;

  if (ImGui::Button("Reset to defaults")) {
    c = facade_parser::Config{};
    changed = true;
  }
  ImGui::Spacing();

  if (ImGui::CollapsingHeader("Stage 1: Edges", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= tuneDouble("Canny low", &c.canny_low, 0.0, 255.0, "%.0f");
    changed |= tuneDouble("Canny high", &c.canny_high, 0.0, 255.0, "%.0f");
    changed |= tuneDouble("Hough min line length", &c.hough_min_line_length_px, 1.0, 300.0, "%.0f");
    helpMarker("HoughLinesP fallback only (used when OpenCV ximgproc/FastLineDetector isn't available).");
    changed |= tuneDouble("Hough max line gap", &c.hough_max_line_gap_px, 0.0, 100.0, "%.0f");
    changed |= tuneInt("Hough threshold votes", &c.hough_threshold_votes, 1, 300);
    changed |= tuneDouble("Line angle tolerance (deg)", &c.line_angle_tolerance_deg, 0.0, 45.0, "%.1f");
    helpMarker("Max angle from 0/90 deg to still count as horizontal/vertical.");
  }

  if (ImGui::CollapsingHeader("Stage 2: Periodicity", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= tuneDouble("Profile smoothing sigma", &c.profile_smoothing_sigma_px, 0.1, 15.0, "%.2f");
    changed |= tuneDouble("Periodicity min score", &c.periodicity_min_score, 0.0, 1.0, "%.3f");
    helpMarker("Min autocorrelation score (fraction of zero-lag) to accept a candidate period.");
    changed |= tuneDouble("Periodicity agreement tol (px)", &c.periodicity_agreement_tol_px, 0.0, 50.0, "%.1f");
    helpMarker("Max px disagreement between autocorrelation and direct-peak spacing before the "
               "result is flagged low_confidence.");
    changed |= tuneDouble("Direct-peak low-activity frac", &c.direct_peak_low_activity_frac, 0.0, 1.0, "%.3f");
    helpMarker("Fraction of profile max below which a run counts as a low-activity wall/pier gap "
               "(candidate separator position).");
  }

  if (ImGui::CollapsingHeader("Stage 3: Split grammar", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= tuneInt("Min cell size (px)", &c.min_cell_size_px, 2, 300);
    changed |= tuneDouble("Min segment width frac", &c.min_segment_width_frac_of_median, 0.0, 1.0, "%.3f");
    helpMarker("A column (bay) split narrower than this fraction of its floor band's median bay "
               "width gets merged into a neighbor. Guards against one irregular cell (e.g. a "
               "narrow door) over-splitting its row. NOT applied to row/floor boundaries, since "
               "floor heights legitimately vary a lot on real facades.");
  }

  if (ImGui::CollapsingHeader("Stage 4: Classification", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= tuneDouble("Otsu close kernel frac", &c.otsu_close_kernel_frac, 0.0, 0.5, "%.3f");
    changed |= tuneDouble("Window min fill ratio", &c.window_min_fill_ratio, 0.0, 1.0, "%.3f");
    changed |= tuneDouble("Window max fill ratio", &c.window_max_fill_ratio, 0.0, 1.0, "%.3f");
    changed |= tuneDouble("Window min aspect", &c.window_min_aspect, 0.05, 5.0, "%.3f");
    changed |= tuneDouble("Window max aspect", &c.window_max_aspect, 0.05, 5.0, "%.3f");
    changed |= tuneDouble("Door min height/width ratio", &c.door_min_height_width_ratio, 1.0, 6.0, "%.2f");
    helpMarker("Lowest floor band only. Ordinary portrait windows commonly run up to ~1.5-1.6; "
               "keep this above that band or tall bottom-row windows get misread as doors.");
  }

  if (ImGui::CollapsingHeader("Stage 5: Lattice refinement", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Checkbox("Enable lattice refinement", &c.enable_lattice_refine)) {
      changed = true;
    }
    changed |= tuneDouble("Refine window frac", &c.lattice_refine_window_frac, 0.0, 0.5, "%.3f");
    helpMarker("+/- fraction of local cell size searched when snapping a boundary to the nearest "
               "low-activity position.");
  }

  if (ImGui::CollapsingHeader("Stage 6: Symmetry", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Checkbox("Enable symmetry check", &c.enable_symmetry_check)) {
      changed = true;
    }
  }

  if (ImGui::CollapsingHeader("Stage 7: Edges/ledges", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= tuneDouble("Min edge length (px)", &c.min_edge_length_px, 0.0, 400.0, "%.0f");
    changed |= tuneDouble("Edge claim margin (px)", &c.edge_claim_margin_px, 0.0, 50.0, "%.1f");
    helpMarker("Dilation margin used to decide a line segment is already part of a window/door "
               "frame (and should be dropped, not emitted as its own edge).");
  }

  if (ImGui::CollapsingHeader("Stage 8: Export", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Checkbox("Emit wall elements", &c.emit_wall_elements)) {
      changed = true;
    }
    helpMarker("Also emit `wall` elements for unclassified cells (default: implicit only, i.e. "
               "\"no window/door here\").");
  }

  return changed;
}

// Progress/result controls for the background tune started by
// startAutoTune(). Reads TuneWorker's snapshot under its mutex each
// frame; never touches the worker thread's own captured data.
void drawAutoTunePanel(AppState& state) {
  ImGui::SeparatorText("Auto-Tune");

  if (!state.tune_worker) {
    if (ImGui::Button("Run Auto-Tune")) {
      startAutoTune(state);
    }
    return;
  }

  bool running = false;
  bool done = false;
  int trials_done = 0;
  double best_score = 0.0;
  facade_parser::Config best_config;
  {
    const std::lock_guard<std::mutex> lock(state.tune_worker->mutex);
    running = state.tune_worker->snapshot.running;
    done = state.tune_worker->snapshot.done;
    trials_done = state.tune_worker->snapshot.trials_done;
    best_score = state.tune_worker->snapshot.best_score;
    if (done) {
      best_config = state.tune_worker->snapshot.result.best_config;
    }
  }

  if (running) {
    ImGui::Text("Tuning... %d trial(s) so far, best mean F1: %.3f", trials_done, best_score);
    if (ImGui::Button("Cancel")) {
      state.tune_worker->cancel_requested = true;
    }
  } else if (done) {
    ImGui::Text("Done (%d trials). Best mean F1: %.3f", trials_done, best_score);
    if (ImGui::Button("Apply Tuned Config")) {
      state.config = best_config;
      state.tune_worker.reset();
      refreshPreview(state);
    }
    ImGui::SameLine();
    if (ImGui::Button("Discard")) {
      state.tune_worker.reset();
    }
  }
}

void drawDatasetPanel(AppState& state) {
  if (state.dataset.empty()) {
    ImGui::TextDisabled("File > Open Dataset... to load photos for annotation and tuning.");
    return;
  }

  ImGui::Text("%zu image(s) loaded", state.dataset.size());
  ImGui::BeginChild("DatasetList", ImVec2(0, 220), ImGuiChildFlags_Borders);
  for (int i = 0; i < static_cast<int>(state.dataset.size()); ++i) {
    const auto& entry = state.dataset[static_cast<std::size_t>(i)];
    std::string label = std::filesystem::path(entry.image_path).filename().string();
    label += " (" + std::to_string(entry.ground_truth.elements.size()) + ")";
    if (entry.last_score.has_value()) {
      char score_buf[32];
      std::snprintf(score_buf, sizeof(score_buf), "  F1=%.2f", entry.last_score->f1);
      label += score_buf;
    }
    if (ImGui::Selectable(label.c_str(), state.viewing_dataset_index == i)) {
      state.viewing_dataset_index = i;
      state.has_image = true;
      refreshDatasetPreview(state);
    }
  }
  ImGui::EndChild();

  if (state.viewing_dataset_index < 0) {
    return;
  }
  DatasetEntry& entry = state.dataset[static_cast<std::size_t>(state.viewing_dataset_index)];

  ImGui::Spacing();
  ImGui::SeparatorText("Annotate");

  int type_int = static_cast<int>(state.annotate_type);
  ImGui::RadioButton("Window", &type_int, static_cast<int>(facade_parser::ElementType::Window));
  ImGui::SameLine();
  ImGui::RadioButton("Door", &type_int, static_cast<int>(facade_parser::ElementType::Door));
  state.annotate_type = static_cast<facade_parser::ElementType>(type_int);

  ImGui::Checkbox("Annotate mode", &state.annotate_mode);
  helpMarker("Drag on the image to draw a box of the selected type. Right-click an existing box "
             "to delete it or change its type.");
  if (ImGui::Checkbox("Show live detections (comparison)", &state.show_comparison)) {
    refreshDatasetPreview(state);
  }
  helpMarker("Green = matched (true positive), red = unmatched detection (false positive), "
             "yellow = unmatched annotation (false negative).");

  if (ImGui::Button("Save Annotations")) {
    saveCurrentAnnotations(state);
  }
  if (entry.ground_truth_dirty) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0F, 0.8F, 0.3F, 1.0F), "(unsaved)");
  }

  ImGui::Spacing();
  drawAutoTunePanel(state);
}

}  // namespace

int main() {
  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit()) {
    return 1;
  }

  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

  GLFWwindow* window = glfwCreateWindow(1600, 900, "facade_parser", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // The main layout (menu bar + fixed panes) doesn't use any
  // user-movable/resizable imgui windows, so there's no layout worth
  // persisting across runs — disable the imgui.ini it would otherwise
  // write to the working directory.
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  AppState state;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool open_image_requested = false;
    bool open_dataset_requested = false;
    bool save_json_requested = false;

    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open Image...", "Cmd+O")) {
          open_image_requested = true;
        }
        if (ImGui::MenuItem("Open Dataset...")) {
          open_dataset_requested = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Save JSON...", "Cmd+S", false, state.has_image)) {
          save_json_requested = true;
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }

    if (open_image_requested) {
      if (auto path = facade_parser::gui::openImageDialog()) {
        loadImage(state, *path);
      }
    }
    if (open_dataset_requested) {
      const std::vector<std::string> paths = facade_parser::gui::openMultipleImagesDialog();
      if (!paths.empty()) {
        loadDataset(state, paths);
      }
    }
    if (save_json_requested) {
      const std::string stem = state.image_path.empty()
                                    ? "facade"
                                    : std::filesystem::path(state.image_path).stem().string();
      if (auto path = facade_parser::gui::saveJsonDialog(stem + ".json")) {
        saveJson(state, *path);
      }
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float menu_bar_height = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + menu_bar_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y - menu_bar_height));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoNavFocus;
    ImGui::Begin("##main", nullptr, flags);

    if (!state.status.empty()) {
      ImGui::TextColored(state.status_is_error ? ImVec4(1.0F, 0.45F, 0.45F, 1.0F)
                                                : ImVec4(0.6F, 0.9F, 0.6F, 1.0F),
                          "%s", state.status.c_str());
      ImGui::Separator();
    }

    constexpr float kDatasetPanelWidth = 260.0F;
    constexpr float kTuningPanelWidth = 380.0F;

    ImGui::BeginChild("DatasetPanel", ImVec2(kDatasetPanelWidth, 0), ImGuiChildFlags_Borders);
    drawDatasetPanel(state);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("TuningPanel", ImVec2(kTuningPanelWidth, 0), ImGuiChildFlags_Borders);
    const bool config_changed = drawTuningPanel(state.config);
    if (config_changed) {
      refreshPreview(state);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("PreviewPanel", ImVec2(0, 0), ImGuiChildFlags_Borders);
    if (state.has_image) {
      const bool in_dataset_mode = state.viewing_dataset_index >= 0;

      if (!in_dataset_mode) {
        // Rectification toolbar (Preview mode only) — see docs/PLAN.md,
        // "Stage 0: Rectification". Labeled "Corners"/"Result" rather
        // than "Original"/"Rectified": the result view shows whatever
        // facade_parser::run() was last given (raw image if the user
        // hasn't rectified yet — "Detect Features" works either way,
        // per the request's own "keep both paths" clause), so
        // "Rectified" wouldn't always be accurate as a label.
        int view_int = static_cast<int>(state.preview_view);
        ImGui::RadioButton("Corners", &view_int, static_cast<int>(PreviewView::Original));
        ImGui::SameLine();
        ImGui::RadioButton("Result", &view_int, static_cast<int>(PreviewView::Rectified));
        state.preview_view = static_cast<PreviewView>(view_int);

        ImGui::SameLine();
        if (ImGui::Button("Reset to detected corners")) {
          state.corners = state.detected_corners;
        }
        ImGui::SameLine();
        if (ImGui::Button("Rectify")) {
          state.rectified_bgr = facade_parser::rectify(state.source_bgr, state.corners);
          state.has_rectified = true;
          state.preview_view = PreviewView::Rectified;
          // Show the plain warp immediately — Rectify and Detect
          // Features are deliberately separate actions (per the
          // request); detection re-runs explicitly via that button, or
          // implicitly next time a tuning-panel slider is edited.
          state.overlay_bgr = state.rectified_bgr.clone();
          uploadOverlayTexture(state);
        }
        ImGui::SameLine();
        if (ImGui::Button("Detect Features")) {
          state.preview_view = PreviewView::Rectified;
          runPipeline(state);
        }
        ImGui::Spacing();
      }

      const bool show_corners = !in_dataset_mode && state.preview_view == PreviewView::Original;
      const GLuint active_texture = show_corners ? state.original_texture : state.texture;
      const int active_width = show_corners ? state.original_texture_width : state.texture_width;
      const int active_height =
          show_corners ? state.original_texture_height : state.texture_height;

      const ImVec2 avail = ImGui::GetContentRegionAvail();
      const float aspect = static_cast<float>(active_width) / static_cast<float>(active_height);
      float draw_w = avail.x;
      float draw_h = draw_w / aspect;
      if (draw_h > avail.y) {
        draw_h = avail.y;
        draw_w = draw_h * aspect;
      }
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail.x - draw_w) * 0.5F);
      ImGui::Image(static_cast<ImTextureID>(active_texture), ImVec2(draw_w, draw_h));

      const ImVec2 image_min = ImGui::GetItemRectMin();
      const ImVec2 image_max = ImGui::GetItemRectMax();
      const bool image_hovered = ImGui::IsItemHovered();
      const cv::Size image_pixel_size(active_width, active_height);

      if (show_corners) {
        // Draggable corner-handle overlay for the current `state.corners`
        // — pure ImDrawList overlay (not baked into the texture), so
        // dragging never needs a re-upload. See docs/PLAN.md, "Stage 0:
        // Rectification".
        constexpr float kHandleRadius = 7.0F;
        constexpr float kHitRadius = kHandleRadius * 2.5F;
        const ImVec2 mouse = ImGui::GetMousePos();
        const auto screenPosOf = [&](int i) {
          return imagePointToScreen(*cornerByIndex(state.corners, i), image_min, image_max,
                                     image_pixel_size);
        };

        if (state.dragging_corner_index < 0 && image_hovered &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          for (int i = 0; i < 4; ++i) {
            const ImVec2 sp = screenPosOf(i);
            const float dx = mouse.x - sp.x;
            const float dy = mouse.y - sp.y;
            if (dx * dx + dy * dy <= kHitRadius * kHitRadius) {
              state.dragging_corner_index = i;
              break;
            }
          }
        }
        if (state.dragging_corner_index >= 0) {
          if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            cv::Point2f new_pos =
                screenPointToImagePoint(mouse, image_min, image_max, image_pixel_size);
            new_pos.x = std::clamp(new_pos.x, 0.0F, static_cast<float>(image_pixel_size.width));
            new_pos.y = std::clamp(new_pos.y, 0.0F, static_cast<float>(image_pixel_size.height));
            *cornerByIndex(state.corners, state.dragging_corner_index) = new_pos;
          }
          if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            state.dragging_corner_index = -1;
          }
        }

        // Draw connecting lines + handles using the now-current
        // positions (reflecting any update just applied above, so
        // there's no one-frame lag while dragging).
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 screen_pts[4];
        for (int i = 0; i < 4; ++i) {
          screen_pts[i] = screenPosOf(i);
        }
        for (int i = 0; i < 4; ++i) {
          draw_list->AddLine(screen_pts[i], screen_pts[(i + 1) % 4], IM_COL32(0, 255, 255, 255),
                             2.0F);
        }
        for (int i = 0; i < 4; ++i) {
          const float dx = mouse.x - screen_pts[i].x;
          const float dy = mouse.y - screen_pts[i].y;
          const bool active =
              (dx * dx + dy * dy <= kHitRadius * kHitRadius) || state.dragging_corner_index == i;
          draw_list->AddCircleFilled(screen_pts[i], kHandleRadius,
                                     active ? IM_COL32(255, 255, 0, 255) : IM_COL32(0, 255, 255, 255));
        }
      }

      if (in_dataset_mode && state.annotate_mode) {
        DatasetEntry& entry = state.dataset[static_cast<std::size_t>(state.viewing_dataset_index)];

        if (image_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          state.dragging = true;
          state.drag_start_screen = ImGui::GetMousePos();
        }
        if (state.dragging) {
          const ImVec2 current = ImGui::GetMousePos();
          ImGui::GetWindowDrawList()->AddRect(state.drag_start_screen, current,
                                              IM_COL32(255, 255, 0, 255), 0.0F, 0, 2.0F);
          if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            state.dragging = false;
            const cv::Rect box = screenRectToImageRect(state.drag_start_screen, current, image_min,
                                                        image_max, image_pixel_size);
            if (box.width >= 3 && box.height >= 3) {
              entry.ground_truth.elements.push_back({state.annotate_type, box});
              entry.ground_truth_dirty = true;
              refreshDatasetPreview(state);
            }
          }
        }

        if (image_hovered && !state.dragging && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
          const cv::Point2f click =
              screenPointToImagePoint(ImGui::GetMousePos(), image_min, image_max, image_pixel_size);
          int hit = -1;
          for (int i = static_cast<int>(entry.ground_truth.elements.size()) - 1; i >= 0; --i) {
            if (entry.ground_truth.elements[static_cast<std::size_t>(i)].bbox_px.contains(
                    cv::Point(static_cast<int>(click.x), static_cast<int>(click.y)))) {
              hit = i;
              break;
            }
          }
          state.right_clicked_annotation_index = hit;
          if (hit >= 0) {
            ImGui::OpenPopup("AnnotationContextMenu");
          }
        }
        if (ImGui::BeginPopup("AnnotationContextMenu")) {
          const int idx = state.right_clicked_annotation_index;
          if (idx >= 0 && idx < static_cast<int>(entry.ground_truth.elements.size())) {
            auto& el = entry.ground_truth.elements[static_cast<std::size_t>(idx)];
            const char* other_label = el.type == facade_parser::ElementType::Window
                                           ? "Change to Door"
                                           : "Change to Window";
            if (ImGui::MenuItem(other_label)) {
              el.type = el.type == facade_parser::ElementType::Window
                            ? facade_parser::ElementType::Door
                            : facade_parser::ElementType::Window;
              entry.ground_truth_dirty = true;
              refreshDatasetPreview(state);
            }
            if (ImGui::MenuItem("Delete")) {
              entry.ground_truth.elements.erase(entry.ground_truth.elements.begin() + idx);
              entry.ground_truth_dirty = true;
              refreshDatasetPreview(state);
            }
          }
          ImGui::EndPopup();
        }
      }
    } else {
      ImGui::TextDisabled(
          "File > Open Image... for a single preview, or File > Open Dataset... to annotate and "
          "auto-tune against several photos.");
    }
    ImGui::EndChild();

    ImGui::End();

    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.12F, 0.12F, 0.13F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  state.tune_worker.reset();  // Cancel + join any still-running tune before teardown.

  if (state.texture != 0) {
    glDeleteTextures(1, &state.texture);
  }
  if (state.original_texture != 0) {
    glDeleteTextures(1, &state.original_texture);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
