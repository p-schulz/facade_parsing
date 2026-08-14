#include "facade_parser/ground_truth.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

namespace facade_parser {

namespace {

const char* toString(ElementType type) {
  switch (type) {
    case ElementType::Window:
      return "window";
    case ElementType::Door:
      return "door";
    case ElementType::Wall:
      return "wall";
    case ElementType::Edge:
      return "edge";
  }
  return "unknown";
}

std::optional<ElementType> parseElementType(const std::string& s) {
  if (s == "window") return ElementType::Window;
  if (s == "door") return ElementType::Door;
  if (s == "wall") return ElementType::Wall;
  if (s == "edge") return ElementType::Edge;
  return std::nullopt;
}

}  // namespace

std::optional<GroundTruthImage> loadGroundTruth(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }

  nlohmann::json j;
  try {
    in >> j;
  } catch (const nlohmann::json::parse_error&) {
    return std::nullopt;
  }

  GroundTruthImage gt;
  if (!j.contains("image_path") || !j.contains("elements")) {
    return std::nullopt;
  }
  gt.image_path = j.at("image_path").get<std::string>();

  if (j.contains("image_size_px")) {
    const auto& size = j.at("image_size_px");
    if (size.is_array() && size.size() == 2) {
      gt.image_size_px = cv::Size(size[0].get<int>(), size[1].get<int>());
    }
  }

  for (const auto& je : j.at("elements")) {
    if (!je.contains("type") || !je.contains("bbox_px")) {
      continue;  // Skip malformed entries rather than failing the whole file.
    }
    const auto type = parseElementType(je.at("type").get<std::string>());
    const auto& bbox = je.at("bbox_px");
    if (!type.has_value() || !bbox.is_array() || bbox.size() != 4) {
      continue;
    }
    GroundTruthElement element;
    element.type = *type;
    element.bbox_px =
        cv::Rect(bbox[0].get<int>(), bbox[1].get<int>(), bbox[2].get<int>(), bbox[3].get<int>());
    gt.elements.push_back(element);
  }

  return gt;
}

bool saveGroundTruth(const GroundTruthImage& gt, const std::string& path) {
  nlohmann::json j;
  j["image_path"] = gt.image_path;
  j["image_size_px"] = {gt.image_size_px.width, gt.image_size_px.height};

  nlohmann::json elements = nlohmann::json::array();
  for (const auto& element : gt.elements) {
    nlohmann::json je;
    je["type"] = toString(element.type);
    je["bbox_px"] = {element.bbox_px.x, element.bbox_px.y, element.bbox_px.width,
                      element.bbox_px.height};
    elements.push_back(je);
  }
  j["elements"] = elements;

  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << j.dump(2);
  return out.good();
}

}  // namespace facade_parser
