/// \file main.cpp
/// \brief imgui-based GUI for facade_parser: a "File" menu to load an
/// image and save the resulting JSON, a left-hand panel exposing every
/// `Config` threshold (see types.hpp) for interactive real-world tuning,
/// and a main pane previewing the pipeline's debug overlay (grid,
/// classified cell boxes, edge polylines) over the loaded image. Wraps
/// the same library entry point (`facade_parser::run`) the CLI uses —
/// no pipeline logic lives here.
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Drags in the system OpenGL headers.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "facade_parser/io_json.hpp"
#include "facade_parser/pipeline.hpp"
#include "file_dialog_mac.h"

namespace {

struct AppState {
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
};

void uploadOverlayTexture(AppState& state) {
  cv::Mat rgba;
  cv::cvtColor(state.overlay_bgr, rgba, cv::COLOR_BGR2RGBA);

  if (state.texture == 0) {
    glGenTextures(1, &state.texture);
  }
  glBindTexture(GL_TEXTURE_2D, state.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               rgba.data);
  state.texture_width = rgba.cols;
  state.texture_height = rgba.rows;
}

// Re-runs the pipeline on the already-loaded `state.source_bgr` using
// `state.config` (no disk re-read) — used both after loading a new
// image and after every tuning-panel edit.
void runPipeline(AppState& state) {
  const std::filesystem::path p(state.image_path);
  state.result = facade_parser::run(state.source_bgr, p.filename().string(), state.config);
  state.overlay_bgr = facade_parser::renderDebugOverlay(state.source_bgr, state.result);
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
  std::snprintf(buf, sizeof(buf), "Loaded %s (%dx%d) - %d window(s), %d door(s), %d edge(s)",
                p.filename().string().c_str(), state.source_bgr.cols, state.source_bgr.rows,
                windows, doors, edges);
  state.status = buf;
  state.status_is_error = false;
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
  runPipeline(state);
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

  GLFWwindow* window = glfwCreateWindow(1440, 860, "facade_parser", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // The main layout (menu bar + fixed tuning/preview panes) doesn't use
  // any user-movable/resizable imgui windows, so there's no layout worth
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
    bool save_json_requested = false;

    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open Image...", "Cmd+O")) {
          open_image_requested = true;
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

    constexpr float kTuningPanelWidth = 380.0F;
    ImGui::BeginChild("TuningPanel", ImVec2(kTuningPanelWidth, 0), ImGuiChildFlags_Borders);
    const bool config_changed = drawTuningPanel(state.config);
    if (config_changed && state.has_image) {
      runPipeline(state);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("PreviewPanel", ImVec2(0, 0), ImGuiChildFlags_Borders);
    if (state.has_image) {
      const ImVec2 avail = ImGui::GetContentRegionAvail();
      const float aspect =
          static_cast<float>(state.texture_width) / static_cast<float>(state.texture_height);
      float draw_w = avail.x;
      float draw_h = draw_w / aspect;
      if (draw_h > avail.y) {
        draw_h = avail.y;
        draw_w = draw_h * aspect;
      }
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail.x - draw_w) * 0.5F);
      ImGui::Image(static_cast<ImTextureID>(state.texture), ImVec2(draw_w, draw_h));
    } else {
      ImGui::TextDisabled("File > Open Image... to load a rectified facade photo.");
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

  if (state.texture != 0) {
    glDeleteTextures(1, &state.texture);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
