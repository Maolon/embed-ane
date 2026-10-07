#!/bin/bash
# Build Embed ANE.app (Release, arm64), sign it ad hoc, and package:
#   dist/Embed-ANE-<version>.dmg            app + /Applications link
#   dist/embed-ane-<version>-arm64.tar.gz   the `embed-ane` CLI
#   dist/SHA256SUMS
# Ad-hoc signing needs no Apple Developer account. Gatekeeper still flags the
# download, so the README explains how to open it the first time.
# Usage: scripts/package-dmg.sh [version]   (default: 0.0.0-dev)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-0.0.0-dev}"
DIST="$ROOT/dist"
DERIVED="$ROOT/.build/ReleaseDerivedData"
cd "$ROOT"
rm -rf "$DIST"; mkdir -p "$DIST"

EMBED_ANE_CONFIGURATION=Release EMBED_ANE_MARKETING_VERSION="$VERSION" EMBED_ANE_DERIVED_DATA="$DERIVED" \
  bash scripts/build-app.sh
APP="$DERIVED/Build/Products/Release/Embed ANE.app"
[[ -d "$APP" ]] || { echo "missing $APP" >&2; exit 1; }

# Ad-hoc signature with the hardened runtime; --deep also signs the worker
# entry point and embedded bundles.
codesign --force --deep --options runtime --timestamp=none --sign - "$APP"
codesign --verify --deep --strict "$APP"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Embed ANE" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov \
  "$DIST/Embed-ANE-$VERSION.dmg" >/dev/null

swift build -c release --product embed-ane
CLI_STAGE="$STAGE/cli"; mkdir -p "$CLI_STAGE"
cp "$(swift build -c release --show-bin-path)/embed-ane" "$CLI_STAGE/"
codesign --force --options runtime --timestamp=none --sign - "$CLI_STAGE/embed-ane"
cp LICENSE "$CLI_STAGE/"
tar -C "$CLI_STAGE" -czf "$DIST/embed-ane-$VERSION-arm64.tar.gz" embed-ane LICENSE

(cd "$DIST" && shasum -a 256 ./*.dmg ./*.tar.gz > SHA256SUMS)
cat "$DIST/SHA256SUMS"
