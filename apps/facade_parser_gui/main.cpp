/// \file main.cpp
/// \brief Minimal imgui-based GUI for facade_parser: a "File" menu to
/// load an image and save the resulting JSON, and a main pane previewing
/// the pipeline's debug overlay (grid, classified cell boxes, edge
/// polylines) over the loaded image. Wraps the same library entry point
/// (`facade_parser::run`) the CLI uses — no pipeline logic lives here.
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

void loadImage(AppState& state, const std::string& path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    state.status = "Failed to read image: " + path;
    state.status_is_error = true;
    return;
  }

  state.source_bgr = image;
  state.image_path = path;
  const std::filesystem::path p(path);
  state.result = facade_parser::run(state.source_bgr, p.filename().string());
  state.overlay_bgr = facade_parser::renderDebugOverlay(state.source_bgr, state.result);
  uploadOverlayTexture(state);
  state.has_image = true;

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

  GLFWwindow* window = glfwCreateWindow(1280, 800, "facade_parser", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // The single main pane is fixed (no user-movable/resizable imgui
  // windows), so there's no layout worth persisting across runs —
  // disable the imgui.ini it would otherwise write to the working
  // directory.
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
