#include "facade_parser/pipeline.hpp"

#include "facade_parser/classification.hpp"
#include "facade_parser/edges.hpp"
#include "facade_parser/edges_export.hpp"
#include "facade_parser/grammar.hpp"
#include "facade_parser/lattice_refine.hpp"
#include "facade_parser/symmetry.hpp"

namespace facade_parser {

namespace {

Element normalized(Element e, const cv::Size& image_size) {
  if (e.type == ElementType::Edge) {
    e.polyline_norm.reserve(e.polyline_px.size());
    for (const auto& p : e.polyline_px) {
      e.polyline_norm.emplace_back(static_cast<float>(p.x) / static_cast<float>(image_size.width),
                                    static_cast<float>(p.y) /
                                        static_cast<float>(image_size.height));
    }
  } else {
    e.bbox_norm = cv::Rect2f(static_cast<float>(e.bbox_px.x) / static_cast<float>(image_size.width),
                              static_cast<float>(e.bbox_px.y) / static_cast<float>(image_size.height),
                              static_cast<float>(e.bbox_px.width) /
                                  static_cast<float>(image_size.width),
                              static_cast<float>(e.bbox_px.height) /
                                  static_cast<float>(image_size.height));
  }
  return e;
}

}  // namespace

FacadeResult run(const cv::Mat& bgr_image, const std::string& source_image_name,
                  const Config& config) {
  FacadeResult result;
  result.source_image = source_image_name;
  result.image_size_px = bgr_image.size();

  const cv::Rect facade_bbox(0, 0, bgr_image.cols, bgr_image.rows);

  // Stage 1
  const EdgeMaps edges = extractEdges(bgr_image, config);

  // Stage 2 + 3
  result.grid = buildSplitGrammar(edges, facade_bbox, config);

  // Stage 5 (toggleable)
  if (config.enable_lattice_refine) {
    result.grid = LatticeRefinePass{}.refine(result.grid, edges.gradient_magnitude, config);
  }

  // Stage 4
  std::vector<Element> cell_elements = classifyCells(bgr_image, edges, result.grid, config);

  // Stage 6 (toggleable)
  if (config.enable_symmetry_check) {
    const SymmetryCheckResult symmetry =
        checkSymmetry(bgr_image, facade_bbox, result.grid, cell_elements, config);
    result.symmetry_inferences = symmetry.inferences;
  }

  // Stage 7
  const SobelMagnitudeDepthHint depth_hint;
  std::vector<Element> edge_elements =
      exportEdgeElements(edges, cell_elements, depth_hint, config);

  result.elements.reserve(cell_elements.size() + edge_elements.size());
  for (auto& e : cell_elements) {
    result.elements.push_back(normalized(std::move(e), result.image_size_px));
  }
  for (auto& e : edge_elements) {
    result.elements.push_back(normalized(std::move(e), result.image_size_px));
  }

  return result;
}

}  // namespace facade_parser
