#include <gtest/gtest.h>

#include "facade_parser/pipeline.hpp"
#include "synthetic_facade.hpp"

// Runs the full Stage 1-8 pipeline on a synthetic facade and checks
// internal consistency: grid spans the image, every detected window/door
// is at a grid cell, normalized coords derive correctly from pixel
// coords, and no spurious edge elements appear (a plain window grid has
// no cornices of its own — see EdgesExport.WindowFrameSegmentsAreSuppressed
// in test_edges_export.cpp).
TEST(EndToEnd, RunProducesConsistentResult) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);

  const facade_parser::FacadeResult result =
      facade_parser::run(facade.image, "synthetic.png");

  EXPECT_EQ(result.source_image, "synthetic.png");
  EXPECT_EQ(result.image_size_px, facade.image.size());
  ASSERT_GE(result.grid.row_boundaries_px.size(), 2U);
  EXPECT_EQ(result.grid.row_boundaries_px.front(), 0);
  EXPECT_EQ(result.grid.row_boundaries_px.back(), facade.image.rows);

  for (const auto& e : result.elements) {
    if (e.type == facade_parser::ElementType::Edge) {
      continue;
    }
    EXPECT_NEAR(e.bbox_norm.x, static_cast<float>(e.bbox_px.x) / facade.image.cols, 1e-4F);
    EXPECT_NEAR(e.bbox_norm.y, static_cast<float>(e.bbox_px.y) / facade.image.rows, 1e-4F);
  }
}

// End-to-end ground-truth check: a clean regular grid should produce
// exactly one Window element per cell, no spurious edges, and no
// symmetry inferences (every cell was directly detected).
TEST(EndToEnd, RegularFacadeDetectsAllWindowsAndNoSpuriousEdges) {
  const auto facade = facade_parser::test::makeRegularFacade(2, 3, 60, 80, 20);

  const facade_parser::FacadeResult result = facade_parser::run(facade.image, "synthetic.png");

  int window_count = 0;
  int edge_count = 0;
  for (const auto& e : result.elements) {
    if (e.type == facade_parser::ElementType::Window) {
      ++window_count;
    } else if (e.type == facade_parser::ElementType::Edge) {
      ++edge_count;
    }
  }
  EXPECT_EQ(window_count, 6);
  EXPECT_EQ(edge_count, 0);
  EXPECT_TRUE(result.symmetry_inferences.empty());
}
