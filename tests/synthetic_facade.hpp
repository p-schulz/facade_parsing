/// \file synthetic_facade.hpp
/// \brief Test-only helper: procedurally draws a regular-grid facade
/// image and returns it alongside its ground-truth FacadeGrid, per
/// docs/PLAN.md's testing strategy. Not part of the public library.
#pragma once

#include <opencv2/core.hpp>

#include "facade_parser/types.hpp"

namespace facade_parser::test {

struct SyntheticFacade {
  cv::Mat image;                 // CV_8UC3
  facade_parser::FacadeGrid ground_truth_grid;
};

/// Draws `rows` floor bands x `cols` bays of windows on a plain wall
/// background, at exactly regular spacing (no jitter), sized
/// `cell_width_px` x `cell_height_px` per bay with `margin_px` between
/// cells. Returns the image plus the exact boundary positions used to
/// draw it.
SyntheticFacade makeRegularFacade(int rows, int cols, int cell_width_px, int cell_height_px,
                                   int margin_px);

/// Same as makeRegularFacade, but each interior boundary is offset by a
/// fixed (not runtime-random — see docs/PLAN.md "no RNG" constraint),
/// hard-coded jitter pattern, to exercise Stage 5's lattice refinement.
SyntheticFacade makeJitteredFacade(int rows, int cols, int cell_width_px, int cell_height_px,
                                    int margin_px);

}  // namespace facade_parser::test
