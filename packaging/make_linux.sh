#!/usr/bin/env bash
# Builds the Linux x86_64 packages: a tar.gz (binary, .desktop, icon) and an AppImage.
# Runs on Linux (GitHub Actions, ubuntu-22.04 for an old enough glibc); from macOS there is no
# simple cross-compiler for it.
#
# Output: target/linux/PlanetaryStudio-<version>-linux-x86_64.tar.gz and .AppImage
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$ROOT/packaging/private_paths.sh"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
OUT="$ROOT/target/linux"
NAME="PlanetaryStudio-$VERSION-linux-x86_64"

echo "==> cargo build --release"
cargo build --release -p ps-app

rm -rf "$OUT" && mkdir -p "$OUT/$NAME"
cp target/release/ps-app "$OUT/$NAME/planetary-studio"
cp packaging/linux/planetary-studio.desktop "$OUT/$NAME/"
cp packaging/icon_256.png "$OUT/$NAME/planetary-studio.png"
cp LICENSE.md "$OUT/$NAME/"
(cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME")

# AppImage: one file that runs on most distributions.
APPDIR="$OUT/AppDir"
mkdir -p "$APPDIR/usr/bin"
cp target/release/ps-app "$APPDIR/usr/bin/planetary-studio"
cp packaging/linux/planetary-studio.desktop "$APPDIR/"
cp packaging/icon_256.png "$APPDIR/planetary-studio.png"
ln -sf usr/bin/planetary-studio "$APPDIR/AppRun"
TOOL="$OUT/appimagetool"
curl -fsSL -o "$TOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x "$TOOL"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$TOOL" "$APPDIR" "$OUT/$NAME.AppImage"

echo
echo "Creati: $OUT/$NAME.tar.gz e $OUT/$NAME.AppImage"
