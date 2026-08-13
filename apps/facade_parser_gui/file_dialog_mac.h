/// \file file_dialog_mac.h
/// \brief Native macOS Open/Save panels for the GUI's File menu.
///
/// Implemented in file_dialog_mac.mm (Objective-C++, NSOpenPanel /
/// NSSavePanel). Kept behind a plain C++ interface so main.cpp doesn't
/// need to be Objective-C++ itself.
#pragma once

#include <optional>
#include <string>

namespace facade_parser::gui {

/// Shows a native "Open" panel filtered to common raster image types.
/// Returns the chosen absolute path, or nullopt if the user cancelled.
std::optional<std::string> openImageDialog();

/// Shows a native "Save" panel for a .json file, defaulting the file
/// name field to `suggested_name`. Returns the chosen absolute path, or
/// nullopt if the user cancelled.
std::optional<std::string> saveJsonDialog(const std::string& suggested_name);

}  // namespace facade_parser::gui
