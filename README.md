# facade_parser

Standalone, offline C++ library + CLI that analyzes a single rectified
building-facade photo and extracts a structured grid of floors,
window/pier bays, windows, doors, and prominent horizontal/vertical edges
(cornices, ledges, string courses) using classical (non-deep-learning)
computer vision. See `docs/PLAN.md` for the full design rationale and
`docs/OUTPUT_FORMAT.md` for the JSON schema.

**Status: all 8 stages implemented**, each with unit tests backed by
procedurally-generated synthetic facades with known ground truth (see
`tests/synthetic_facade.hpp` and `docs/PLAN.md`'s testing strategy).
There's also an auto-tuning pipeline (`ground_truth.hpp`,
`evaluation.hpp`, `autotune.hpp`; the CLI's `tune` subcommand; the GUI's
Dataset panel) for searching `Config`'s ~20 tunable thresholds against
hand-drawn annotations on real photos instead of guessing — see "Dataset
mode" under GUI usage, and `docs/PLAN.md`'s "Auto-tuning pipeline"
section. See "Known limitations" below for what's still rough at the
edges.

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
classified cells, edge polylines). Every `Config` field (see
`include/facade_parser/types.hpp`) is exposed as a flag, grouped by
pipeline stage — run `--help` for the full list, e.g.
`--canny-low`, `--window-min-fill-ratio`, `--no-lattice-refine`
(disables Stage 5), `--no-symmetry-check` (disables Stage 6).

`--config-file <path>` loads a `Config` saved as JSON (e.g. by `tune`,
below) as the baseline *before* any other flags are applied — an
individually-passed flag always overrides the file, but an unspecified
one keeps the file's value instead of the compiled-in default:

```
./build/facade_parser_cli --input facade.png --config-file tuned_config.json --output ./out
```

### `tune` subcommand (auto-tuning pipeline)

```
./build/facade_parser_cli tune --dataset ./my_dataset --output tuned_config.json
```

Searches `Config`'s parameter space (deterministic coordinate descent —
see `docs/PLAN.md`, "Auto-tuning pipeline") for the values that best
match hand-drawn ground-truth annotations, scanning `--dataset` for
`*.gt.json` sidecar files (schema: `docs/OUTPUT_FORMAT.md`) — the same
files the GUI's annotation tool writes (below); each resolves its own
image relative to its own directory. Prints live progress to stdout
(this can take minutes on a non-trivial dataset — see the performance
note in `docs/PLAN.md`) and writes the best `Config` found as JSON.
`--iterations N` (default 6) caps coordinate-descent sweeps per starting
config; `--iou-threshold` (default 0.5) sets the minimum IoU to count a
detection as matching an annotation.

## GUI usage (macOS)

```
./build/facade_parser_gui
```

An [imgui](https://github.com/ocornut/imgui) desktop app (vendored under
`external/imgui`, per the design choice to use it directly rather than
fetch it) wrapping the same `facade_parser::run()` entry point the CLI
uses — no pipeline logic is duplicated in `apps/facade_parser_gui/`.

- **File > Open Image...** — native `NSOpenPanel`, filtered to
  PNG/JPEG/BMP/TIFF. Loading an image proposes four facade corners
  automatically (see "Rectification" below), then immediately runs the
  full pipeline on the raw image and shows the debug overlay (grid
  lines, color-coded window/door/wall boxes, edge polylines) in the main
  pane, scaled to fit the window while preserving aspect ratio.
- **File > Save JSON...** — native `NSSavePanel`, writes the current
  result via the same `writeResultJson()` the CLI uses; disabled until
  an image is loaded.
- **File > Save Rectified Image...** — native `NSSavePanel` (PNG),
  writes the current `rectified_bgr` plain warp to disk; disabled until
  you've rectified an image in Preview mode (see "Rectification"
  below) — it's a raw pixel dump, not the JSON result, useful for
  feeding the warped photo into another tool.
- **Tuning panel** (right column, full window height) — every `Config`
  field, grouped by stage, as sliders/checkboxes with a "Reset to
  defaults" button. Edits re-run the pipeline on whatever's currently
  shown (single image or the selected dataset photo) once you release
  the slider.
- **Layout is resizable** — every panel boundary (Preview/Dataset split
  at the bottom of the main area, the three Dataset sub-panels, and the
  Tuning panel's left edge) is a drag handle: hover it for a resize
  cursor, drag to resize. Sizes reset to their defaults on relaunch —
  imgui's own `imgui.ini` layout persistence is deliberately disabled
  (see `apps/facade_parser_gui/main.cpp`'s `main()`), so nothing is
  written to disk.
- The status line under the menu bar reports the loaded file's size and
  a window/door/edge count (or dataset image's annotation/score
  summary), or the last error (e.g. an unreadable file), color-coded
  green/red.

Windowing is GLFW + OpenGL3 (the standard imgui desktop backend
pairing); the Open/Save panels are a small Objective-C++ helper
(`apps/facade_parser_gui/file_dialog_mac.mm`), which is the reason this
target is macOS-only for now — porting it would mean swapping in a
different native file-dialog backend, not a pipeline change.

### Rectification (perspective correction)

Real facade photos have keystone distortion; Stage 1-4 detection works
much better once the facade plane is fronto-parallel and axis-aligned.
This is Preview-mode only (not Dataset mode) — see `docs/PLAN.md`,
"Stage 0: Rectification", for the full design writeup, including why
it's deliberately **not** metric-accurate (only right angles matter;
real-world scale comes from an external GIS pipeline applied
downstream).

- A **"Corners" / "Result" toggle** above the preview switches between
  the raw photo (with the four draggable corner handles) and whatever
  `facade_parser::run()` last produced (raw-image detection until you
  rectify, then the rectified one).
- On load, four corners are proposed automatically by reusing Stage 1's
  own line detector (wider angle tolerance, since an unrectified
  photo's edges aren't axis-aligned yet) and picking each side's
  longest candidate edge; falls back to a centered inset rectangle if
  fewer than four are found. Drag a handle to correct it (clamped to
  the image bounds); **Reset to detected corners** restores the
  automatic proposal.
- **Rectify** warps the facade quad onto an axis-aligned rectangle
  (`cv::getPerspectiveTransform` + `cv::warpPerspective`) and switches
  to the "Result" view showing the plain warp. It's cached until you
  click Rectify again (e.g. after adjusting a corner) or load a new
  image. Once rectified, **File > Save Rectified Image...** writes that
  warp to a PNG.
- **Detect Features** calls the same, unmodified
  `facade_parser::run()` the CLI uses, on the rectified image once one
  exists (raw image otherwise — detection works either way, rectifying
  first just tends to improve it). Tuning-panel slider edits do the
  same automatically once a rectified image exists, so you don't have
  to keep re-clicking it while tuning.

### Dataset mode: annotation + auto-tuning

The Dataset panel spans the bottom of the main area (below the Preview
panel, full width up to the Tuning panel), split into three resizable
sub-columns: **image list** (left), **annotate options** (middle), and
**auto-tune** (right) — drag either divider between them to resize.

- **File > Open Dataset...** — native multi-select `NSOpenPanel`. Each
  chosen photo is added to the image list sub-column, auto-loading its
  sidecar `<stem>.gt.json` if one already exists next to it (schema:
  `docs/OUTPUT_FORMAT.md`). Click an entry in the list to view it.
- **Annotate mode** — drag on the image to draw a box of the selected
  type (Window/Door radio buttons); right-click an existing box for a
  "Delete" / "Change type" context menu. **Save Annotations** writes the
  sidecar `.gt.json`; an "(unsaved)" marker shows when there are pending
  edits.
- **Show live detections (comparison)** — overlays the current
  Config's detections against the annotations: green = matched, red =
  unmatched detection, yellow = unmatched annotation. Off, it just shows
  the annotations themselves (still yellow — no live pipeline run
  needed).
- **Run Auto-Tune** — searches for the `Config` that best matches every
  annotated dataset photo (deterministic coordinate descent, no RNG —
  see `docs/PLAN.md`, "Auto-tuning pipeline"), on a background thread so
  the UI stays responsive; shows a live trial count and best-score
  readout with a **Cancel** button. When done, **Apply Tuned Config**
  loads the result into the tuning panel for review (not applied
  automatically — you decide whether it's actually better) or
  **Discard** it.

This can take a while on a real dataset (a full sweep is easily
thousands of pipeline evaluations) — that's why progress and
cancellation are front and center rather than a blocking spinner.

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
