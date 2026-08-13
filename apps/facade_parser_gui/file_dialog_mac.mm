#include "file_dialog_mac.h"

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

namespace facade_parser::gui {

std::optional<std::string> openImageDialog() {
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    panel.allowedContentTypes = @[
      UTTypePNG, UTTypeJPEG, UTTypeBMP, UTTypeTIFF
    ];

    // The GLFW window isn't a full NSApp-activated app by default; make
    // sure the panel actually comes to the front.
    [NSApp activateIgnoringOtherApps:YES];

    if ([panel runModal] == NSModalResponseOK && panel.URL != nil) {
      return std::string(panel.URL.path.UTF8String);
    }
    return std::nullopt;
  }
}

std::optional<std::string> saveJsonDialog(const std::string& suggested_name) {
  @autoreleasepool {
    NSSavePanel* panel = [NSSavePanel savePanel];
    panel.allowedContentTypes = @[ UTTypeJSON ];
    panel.nameFieldStringValue = [NSString stringWithUTF8String:suggested_name.c_str()];

    [NSApp activateIgnoringOtherApps:YES];

    if ([panel runModal] == NSModalResponseOK && panel.URL != nil) {
      return std::string(panel.URL.path.UTF8String);
    }
    return std::nullopt;
  }
}

}  // namespace facade_parser::gui
