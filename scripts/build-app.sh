#!/bin/bash
# Unsigned compile only. No model load, application launch or login registration.
# EMBED_ANE_CONFIGURATION (Debug|Release) and EMBED_ANE_MARKETING_VERSION are optional.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOCK="App/App.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
# Xcode's workspace has a different origin graph than the root Swift package.
# Seed a version-2 lock with the exact root pins (no stale version-3 originHash).
python3 - "$LOCK" <<'PYCODE'
import json, pathlib, sys
root = json.loads(pathlib.Path('Package.resolved').read_text())
path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({'pins': root['pins'], 'version': 2}, indent=2) + '\n')
print('Dependency pins:')
for pin in root['pins']:
    print(pin['identity'], pin['state'].get('version', ''), pin['state']['revision'])
PYCODE
xcodebuild -version
xcodebuild \
  -project App/App.xcodeproj \
  -scheme App \
  -configuration "${EMBED_ANE_CONFIGURATION:-Debug}" \
  -destination 'generic/platform=macOS' \
  -derivedDataPath "${EMBED_ANE_DERIVED_DATA:-$ROOT/.build/AppDerivedData}" \
  -clonedSourcePackagesDirPath "$ROOT/.build/AppSourcePackages" \
  -onlyUsePackageVersionsFromResolvedFile \
  build CODE_SIGNING_ALLOWED=NO ARCHS=arm64 ONLY_ACTIVE_ARCH=NO ${EMBED_ANE_MARKETING_VERSION:+MARKETING_VERSION=$EMBED_ANE_MARKETING_VERSION}
python3 - "$LOCK" <<'PYCODE'
import json, pathlib, sys
def pins(path):
    value = json.loads(pathlib.Path(path).read_text())
    return {p['identity']: (p['kind'], p['location'], p['state']) for p in value['pins']}
if pins('Package.resolved') != pins(sys.argv[1]):
    raise SystemExit('ERROR: Xcode dependency pins diverged from root Package.resolved')
print('App compiled unsigned; all dependency pins match the root lock.')
PYCODE
