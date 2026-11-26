#!/usr/bin/env bash
# Builds the Windows x64 package: a zip with the portable executable.
#
# - On a Windows machine (GitHub Actions, Git Bash) it builds natively with the MSVC toolchain.
# - From macOS or Linux it cross-compiles with MinGW:
#     rustup target add x86_64-pc-windows-gnu && brew install mingw-w64
#
# Output: target/windows/PlanetaryStudio-<version>-windows-x64.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$ROOT/packaging/private_paths.sh"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) NATIVE=1; TARGET="x86_64-pc-windows-msvc" ;;
    *)                    NATIVE=0; TARGET="x86_64-pc-windows-gnu" ;;
esac
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
OUT="$ROOT/target/windows"
STAGE="$OUT/Planetary Studio"

echo "==> cargo build --release --target $TARGET"
cargo build --release -p ps-app --target "$TARGET"

rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "target/$TARGET/release/ps-app.exe" "$STAGE/Planetary Studio.exe"
if [ "$NATIVE" = 0 ]; then
    # The symbol table names the build folder (import libraries): not needed to run, left out.
    x86_64-w64-mingw32-strip "$STAGE/Planetary Studio.exe"
fi

ZIP="$OUT/PlanetaryStudio-$VERSION-windows-x64.zip"
rm -f "$ZIP"
if command -v zip >/dev/null; then
    (cd "$OUT" && zip -qr "$ZIP" "Planetary Studio")
else
    # Git Bash on the Windows runners has no zip: PowerShell does the same.
    powershell.exe -NoProfile -Command "Compress-Archive -Path '$(cygpath -w "$STAGE")' -DestinationPath '$(cygpath -w "$ZIP")'"
fi

echo
echo "Creato: $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "Eseguibile: $STAGE/Planetary Studio.exe"
