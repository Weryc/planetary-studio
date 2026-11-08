#!/usr/bin/env bash
# Cross-compiles the Windows build from macOS and packs it into a zip.
# Needs: rustup target add x86_64-pc-windows-gnu && brew install mingw-w64
#
# Output: target/windows/PlanetaryStudio-<version>-windows-x64.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

TARGET="x86_64-pc-windows-gnu"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
OUT="$ROOT/target/windows"
STAGE="$OUT/Planetary Studio"

echo "==> cargo build --release --target $TARGET"
cargo build --release -p ps-app --target "$TARGET"

rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "target/$TARGET/release/ps-app.exe" "$STAGE/Planetary Studio.exe"

ZIP="$OUT/PlanetaryStudio-$VERSION-windows-x64.zip"
rm -f "$ZIP"
(cd "$OUT" && zip -qr "$ZIP" "Planetary Studio")

echo
echo "Creato: $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "Eseguibile: $STAGE/Planetary Studio.exe"
