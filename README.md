# facade_parser

Standalone, offline C++ library + CLI that analyzes a single rectified
building-facade photo and extracts a structured grid of floors,
window/pier bays, windows, doors, and prominent horizontal/vertical edges
(cornices, ledges, string courses) using classical (non-deep-learning)
computer vision. See `docs/PLAN.md` for the full design rationale and
`docs/OUTPUT_FORMAT.md` for the JSON schema.

**Status: all 8 stages implemented**, each with unit tests backed by
procedurally-generated synthetic facades with known ground truth (see
`tests/synthetic_facade.hpp` and `docs/PLAN.md`'s testing strategy). See
"Known limitations" below for what's still rough at the edges.

## Build

Requires OpenCV (with `core`, `imgproc`, `imgcodecs`; `ximgproc` and
`geometry` optional — see below) discoverable via
`find_package(OpenCV CONFIG)`, and CMake ≥ 3.16. `nlohmann/json`,
`CLI11`, and `GoogleTest` are picked up from a system install via
`find_package(... CONFIG QUIET)` when present — e.g.
`brew install nlohmann-json cli11 googletest` — and only fetched via
`FetchContent` (i.e. `git clone` from GitHub) for whichever of the three
isn't already installed. Installing all three via Homebrew first is
strongly recommended: a fresh `FetchContent` clone of all three over a
slow or restricted connection can make the first configure take a very
long time or effectively hang.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

If OpenCV isn't found automatically (e.g. a Homebrew install not on the
default CMake search path), pass `-DOpenCV_DIR=/path/to/lib/cmake/opencvN`.

`ximgproc` (OpenCV contrib) is optional: when present, Stage 1 uses
`cv::ximgproc::createFastLineDetector`; otherwise it falls back to
`cv::HoughLinesP`. `geometry` is also optional — OpenCV 5 moved
`cv::contourArea`/`cv::boundingRect` (used by Stage 4) into that module;
older OpenCV keeps them in `imgproc`. The CMake configure step logs which
variant of each was selected.

The GUI (see below) additionally needs GLFW, also resolved
system-package-first (`brew install glfw`) with a `FetchContent`
fallback, plus the system OpenGL framework (no install needed on
macOS). It's built automatically alongside the CLI; pass
`-DFACADE_PARSER_BUILD_GUI=OFF` to skip it. The GUI target only exists on
macOS — see "GUI usage" below for why.

## CLI usage

```
./build/facade_parser_cli --input facade.png --output ./out
```

Writes `out/facade.json` (schema: `docs/OUTPUT_FORMAT.md`) and
`out/facade_overlay.png` (debug visualization: grid lines, color-coded
classified cells, edge polylines). Flags:

- `--no-lattice-refine` — disable Stage 5 (irregular lattice refinement).
- `--no-symmetry-check` — disable Stage 6 (mirror-symmetry inference).

## GUI usage (macOS)

```
./build/facade_parser_gui
```

An [imgui](https://github.com/ocornut/imgui) desktop app (vendored under
`external/imgui`, per the design choice to use it directly rather than
fetch it) wrapping the same `facade_parser::run()` entry point the CLI
uses — no pipeline logic is duplicated in `apps/facade_parser_gui/`.

- **File > Open Image...** — native `NSOpenPanel`, filtered to
  PNG/JPEG/BMP/TIFF. Loading an image immediately runs the full
  pipeline and shows the debug overlay (grid lines, color-coded
  window/door/wall boxes, edge polylines) in the main pane, scaled to
  fit the window while preserving aspect ratio.
- **File > Save JSON...** — native `NSSavePanel`, writes the current
  result via the same `writeResultJson()` the CLI uses; disabled until
  an image is loaded.
- The status line under the menu bar reports the loaded file's size and
  a window/door/edge count, or the last error (e.g. an unreadable
  file), color-coded green/red.

Windowing is GLFW + OpenGL3 (the standard imgui desktop backend
pairing); the Open/Save panels are a small Objective-C++ helper
(`apps/facade_parser_gui/file_dialog_mac.mm`), which is the reason this
target is macOS-only for now — porting it would mean swapping in a
different native file-dialog backend, not a pipeline change.

## Pipeline stages

Each stage is documented in detail, with its OpenCV API choices and the
literature it implements, in `docs/PLAN.md`:

1. **Edge and line extraction** (`edges.hpp`) — Canny, Sobel gradient
   magnitude, near-horizontal/vertical line segments.
2. **Periodicity / grid detection** (`periodicity.hpp`) — projection
   profiles, DFT-based autocorrelation (Wiener–Khinchin), NMS peak
   picking.
3. **Split-grammar segmentation** (`grammar.hpp`) — recursive
   floor-split then tile-split (Müller et al., SIGGRAPH 2007),
   flattened into a `FacadeGrid`.
4. **Cell classification** (`classification.hpp`) — Otsu threshold,
   contrast/edge-density features, contour/rect fit, window/door/wall
   decision with confidence.
5. **Irregular lattice refinement** (`lattice_refine.hpp`, optional) —
   per-boundary local snapping (Riemenschneider et al., CVPR 2012).
6. **Symmetry check** (`symmetry.hpp`, optional) — mirror-axis estimate
   via `cv::phaseCorrelate`, flags occluded cells whose mirrored
   counterpart was detected.
7. **Edge/ledge/projection extraction** (`edges_export.hpp`) — emits
   unclaimed long line segments as `edge` elements; ships a
   `DepthHint` interface with a default low-confidence gradient-magnitude
   proxy (see Known limitations).
8. **Export** (`io_json.hpp`) — JSON serialization + debug overlay PNG.

## Known limitations

- **No physical relief/depth estimation.** Stage 7 explicitly does not
  attempt to recover protrusion depth from shading — that's ill-posed
  from a single rectified RGB image without a light-direction model. The
  default `DepthHint` implementation (`SobelMagnitudeDepthHint`) is a
  coarse gradient-magnitude proxy, always tagged `low_confidence`. A
  future implementation backed by a real depth/displacement map (e.g.
  from photogrammetry) can be swapped in via the `DepthHint` interface
  without touching Stage 8.
- **A single strongly irregular cell used to be able to break a row's
  column split** — now mitigated. Stage 2 (`analyzePeriodicity`) retries
  lower-ranked autocorrelation candidates when the top-ranked one
  disagrees with direct-peak spacing (a door much narrower than its
  neighboring windows can otherwise make a sub-harmonic of the true
  period score marginally higher), and Stage 3 has a secondary
  median-width consistency pass (`Config::min_segment_width_frac_of_median`)
  as a safety net. This is still a local heuristic, not a general fix
  for facades with irregular rhythm throughout a row — see docs/PLAN.md's
  "Mitigated: a single strongly irregular cell..." writeup for the full
  story and why a complete fix would need something closer to
  Riemenschneider et al.'s full irregular-lattice search.
- No deep learning, pretrained models, or cloud APIs anywhere in this
  tool, by design.
- No rectification step — input images are assumed already
  orthorectified/perspective-corrected.
- Automated tests validate against **synthetic** ground-truth facades
  only (`tests/synthetic_facade.hpp`); no numeric accuracy claim is made
  against real photos anywhere in this repo. Use the CLI directly on a
  real photo for qualitative visual QA via the debug overlay.
