# Sourced by the packaging scripts: keeps the paths of the machine that builds the release out
# of the binaries. Rust records source paths for panic messages and debug info; without this
# they would carry the builder's home folder and project location.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# The last matching prefix wins: the general home first, the specific folders after.
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/home/builder --remap-path-prefix=$HOME/.cargo/registry/src=/cargo/registry/src --remap-path-prefix=$HOME/.cargo/git/checkouts=/cargo/git --remap-path-prefix=$HOME/.rustup=/rustup --remap-path-prefix=$ROOT=/planetary-studio"
