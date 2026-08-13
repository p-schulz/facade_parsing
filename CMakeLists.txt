cmake_minimum_required(VERSION 3.16)
project(facade_parser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FACADE_PARSER_BUILD_TESTS "Build GoogleTest-based unit tests" ON)

# --- OpenCV -------------------------------------------------------------
# System-provided, not FetchContent'd (see docs/PLAN.md: "Dependencies").
# Only core/imgproc/imgcodecs are actually called unconditionally by this
# codebase; no stage currently uses cv::Feature2D-family APIs (OpenCV 5
# also renamed the module from `features2d` to `features`), so it's
# intentionally not a required component here.
find_package(OpenCV REQUIRED CONFIG COMPONENTS core imgproc imgcodecs)

# ximgproc (contrib) is optional: Stage 1 falls back to cv::HoughLinesP
# when it isn't present. See docs/PLAN.md Stage 1 decision.
find_package(OpenCV QUIET CONFIG COMPONENTS ximgproc)
if(OpenCV_ximgproc_FOUND)
  message(STATUS "facade_parser: OpenCV ximgproc found — using FastLineDetector")
else()
  message(STATUS "facade_parser: OpenCV ximgproc NOT found — falling back to HoughLinesP")
endif()

# geometry is optional: OpenCV 5 moved cv::contourArea/cv::boundingRect
# (used by Stage 4) out of imgproc into a new `geometry` module; earlier
# OpenCV versions (and possibly future ones) keep them in imgproc, so
# link this module only when it actually exists as a separate target.
find_package(OpenCV QUIET CONFIG COMPONENTS geometry)
if(OpenCV_geometry_FOUND)
  message(STATUS "facade_parser: OpenCV geometry module found (OpenCV 5+ contourArea/boundingRect home)")
else()
  message(STATUS "facade_parser: OpenCV geometry module not found — assuming contourArea/boundingRect live in imgproc")
endif()

# --- Header-only third-party deps: system package preferred, ------------
# --- FetchContent (network) only as a fallback ---------------------------
# A pure FetchContent setup means every fresh build directory needs to
# `git clone` these from GitHub, which hangs or crawls on a slow/flaky
# connection (observed in practice — see the "configure never finishes"
# issue this was added to fix). `find_package(... CONFIG QUIET)` first
# picks up a Homebrew-installed copy (`brew install nlohmann-json cli11
# googletest`) with no network access needed at all; FetchContent only
# runs for whichever of the three isn't already installed.
include(FetchContent)

find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
  message(STATUS "facade_parser: using system nlohmann_json ${nlohmann_json_VERSION}")
else()
  message(STATUS "facade_parser: system nlohmann_json not found, fetching via git")
  FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(json)
endif()

find_package(CLI11 CONFIG QUIET)
if(CLI11_FOUND)
  message(STATUS "facade_parser: using system CLI11 ${CLI11_VERSION}")
else()
  message(STATUS "facade_parser: system CLI11 not found, fetching via git")
  FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
  )
  FetchContent_MakeAvailable(cli11)
endif()

# --- facade_parser library ----------------------------------------------
add_library(facade_parser
  src/edges.cpp
  src/periodicity.cpp
  src/grammar.cpp
  src/classification.cpp
  src/lattice_refine.cpp
  src/symmetry.cpp
  src/edges_export.cpp
  src/io_json.cpp
  src/pipeline.cpp
)
add_library(facade_parser::facade_parser ALIAS facade_parser)

target_include_directories(facade_parser
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(facade_parser
  PUBLIC
    opencv_core opencv_imgproc opencv_imgcodecs
    nlohmann_json::nlohmann_json
)

if(OpenCV_ximgproc_FOUND)
  target_link_libraries(facade_parser PUBLIC opencv_ximgproc)
  target_compile_definitions(facade_parser PUBLIC FACADE_PARSER_HAS_XIMGPROC=1)
endif()

if(OpenCV_geometry_FOUND)
  target_link_libraries(facade_parser PUBLIC opencv_geometry)
  target_compile_definitions(facade_parser PUBLIC FACADE_PARSER_HAS_OPENCV_GEOMETRY=1)
endif()

target_compile_options(facade_parser PRIVATE -Wall -Wextra)

# --- CLI ------------------------------------------------------------------
add_executable(facade_parser_cli apps/facade_parser_cli.cpp)
target_link_libraries(facade_parser_cli PRIVATE facade_parser::facade_parser CLI11::CLI11)

# --- GUI (macOS only) -----------------------------------------------------
# imgui itself is vendored (not fetched) under external/imgui, per the
# task's instruction to use it from there. GLFW (window/GL-context
# creation) follows the same system-package-preferred pattern as the
# other dependencies above; the native Open/Save panels are a small
# Objective-C++ file (file_dialog_mac.mm), which is why this target is
# gated to APPLE — porting to another OS would need a different file
# dialog backend, out of scope here.
option(FACADE_PARSER_BUILD_GUI "Build the imgui-based GUI application (macOS only)" ON)

if(FACADE_PARSER_BUILD_GUI AND APPLE)
  enable_language(OBJCXX)

  find_package(glfw3 CONFIG QUIET)
  if(glfw3_FOUND)
    message(STATUS "facade_parser: using system GLFW ${glfw3_VERSION}")
  else()
    message(STATUS "facade_parser: system GLFW not found, fetching via git")
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.4
    )
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
  endif()

  find_package(OpenGL REQUIRED)

  set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
  add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

  add_executable(facade_parser_gui
    apps/facade_parser_gui/main.cpp
    apps/facade_parser_gui/file_dialog_mac.mm
  )
  target_link_libraries(facade_parser_gui PRIVATE
    facade_parser::facade_parser
    imgui
    "-framework Cocoa"
    "-framework UniformTypeIdentifiers"
  )
elseif(FACADE_PARSER_BUILD_GUI)
  message(STATUS "facade_parser: skipping GUI target — only implemented for macOS")
endif()

# --- Tests ----------------------------------------------------------------
if(FACADE_PARSER_BUILD_TESTS)
  enable_testing()

  find_package(GTest CONFIG QUIET)
  if(GTest_FOUND)
    message(STATUS "facade_parser: using system GoogleTest ${GTest_VERSION}")
  else()
    message(STATUS "facade_parser: system GoogleTest not found, fetching via git")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_subdirectory(tests)
endif()
