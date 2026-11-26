#!/usr/bin/env bash
# Builds "Planetary Studio.app" and packs it into a drag-to-Applications DMG.
#
#   packaging/make_dmg.sh                      # binary for this Mac's architecture
#   packaging/make_dmg.sh --apple-silicon      # Apple Silicon (arm64) only
#   packaging/make_dmg.sh --intel              # Intel (x86_64) only
#   packaging/make_dmg.sh --universal          # Intel + Apple Silicon in one binary
#
# Output: target/dmg/PlanetaryStudio-<version>-macos-<apple-silicon|intel|universal>.dmg
set -euo pipefail

APP_NAME="Planetary Studio"
BUNDLE_ID="dev.planetarystudio.app"
BIN="ps-app"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$ROOT/packaging/private_paths.sh"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
OUT="$ROOT/target/dmg"
STAGE="$OUT/stage"
APP="$STAGE/$APP_NAME.app"

# ---- build ---------------------------------------------------------------------------------
build_for() {
    rustup target add "$1" >/dev/null
    echo "==> cargo build --release --target $1"
    cargo build --release -p "$BIN" --target "$1"
}

mkdir -p "$OUT"
case "${1:-}" in
    --universal)
        build_for x86_64-apple-darwin
        build_for aarch64-apple-darwin
        BINARY="$OUT/$BIN"
        lipo -create -output "$BINARY" target/x86_64-apple-darwin/release/"$BIN" target/aarch64-apple-darwin/release/"$BIN"
        SUFFIX="macos-universal"
        ;;
    --apple-silicon)
        build_for aarch64-apple-darwin
        BINARY="target/aarch64-apple-darwin/release/$BIN"
        SUFFIX="macos-apple-silicon"
        ;;
    --intel)
        build_for x86_64-apple-darwin
        BINARY="target/x86_64-apple-darwin/release/$BIN"
        SUFFIX="macos-intel"
        ;;
    "")
        echo "==> cargo build --release"
        cargo build --release -p "$BIN"
        BINARY="target/release/$BIN"
        if [ "$(uname -m)" = "arm64" ]; then SUFFIX="macos-apple-silicon"; else SUFFIX="macos-intel"; fi
        ;;
    *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
esac

# ---- .app bundle ---------------------------------------------------------------------------
echo "==> bundle $APP_NAME.app"
rm -rf "$STAGE"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY" "$APP/Contents/MacOS/$BIN"

ICONSET="$OUT/AppIcon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z $size $size packaging/icon_1024.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) packaging/icon_1024.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                 <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>          <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>           <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>              <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>   <string>$VERSION</string>
    <key>CFBundleExecutable</key>           <string>$BIN</string>
    <key>CFBundleIconFile</key>             <string>AppIcon</string>
    <key>CFBundlePackageType</key>          <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key>       <string>11.0</string>
    <key>LSApplicationCategoryType</key>    <string>public.app-category.photography</string>
    <key>NSHighResolutionCapable</key>      <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict>
</plist>
PLIST

# Ad-hoc signature: required for Apple Silicon to launch the binary at all. It is not a
# Developer ID signature, so Gatekeeper on other Macs still asks for confirmation.
codesign --force --deep --sign - "$APP"

# ---- DMG -----------------------------------------------------------------------------------
echo "==> dmg"
ln -s /Applications "$STAGE/Applications"
DMG="$OUT/PlanetaryStudio-$VERSION-$SUFFIX.dmg"
rm -f "$DMG"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null

echo
echo "Creato: $DMG ($(du -h "$DMG" | cut -f1))"
