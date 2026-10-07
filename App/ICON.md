# Embed ANE icon

`App/AppIcon.icon` is an Icon Composer document: a rounded Neural Engine package with a pin-1 mark around a die whose 3×3 cores glow at the brightness of embedding values. The package has no pins drawn, so the outline stays soft. The menu bar glyph (`App/Sources/StatusGlyph.swift`) uses the same package-and-die shape.

Layers in `AppIcon.icon/Assets/` are 1024-pt SVGs with no background or mask; the system applies the icon shape.

| Group (front to back) | Layer | Liquid Glass |
|---|---|---|
| Cores | `3-cores.svg` | off, so the core colors stay readable |
| Die | `2-die.svg` | on, 20% translucency |
| Package | `1-package.svg` | on, 40% translucency |

The background is an automatic gradient from `#1B3440`. In `icon.json` the first group is the front-most.

The App target compiles the document directly (`ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon`), which requires Xcode 26 or later. On macOS 26 and later the icon renders with Liquid Glass; for the macOS 15 deployment target Xcode also generates fallback images at build time, so no separate asset catalog is kept. Edit the icon in Xcode → Open Developer Tool → Icon Composer, or edit the SVGs and `icon.json` directly.
