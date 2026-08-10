#!/usr/bin/env bash
# Fetch and verify the pinned HRX amdxdna *public package* release artifact.
#
# After XADX removal, FLM consumes HRX via find_package(hrx CONFIG REQUIRED) from
# the public package (CMake package config + libhrx.so + public headers), not the
# former HRX_DIR/HRX_BUILD source+build tree. This script downloads + verifies +
# extracts the package and prints its CMake package prefix (feed it to the FLM
# build via -DCMAKE_PREFIX_PATH). The Linux agent owns finalizing this path.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: fetch-hrx-release.sh [out-dir]

Downloads the HRX public package pinned in ./hrx-release.env, verifies its
checksum, extracts it, locates the HRX CMake package config, and prints the
package prefix for find_package(hrx).

When running in GitHub Actions, the script also writes:
  prefix=<package-prefix>
to $GITHUB_OUTPUT.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here"
release_env="$here/hrx-release.env"
[[ -f "$release_env" ]] || die "missing $release_env"

# shellcheck disable=SC1090
source "$release_env"

: "${HRX_RELEASE_REPO:?missing HRX_RELEASE_REPO in $release_env}"
: "${HRX_RELEASE_TAG:?missing HRX_RELEASE_TAG in $release_env}"
: "${HRX_RELEASE_ASSET:?missing HRX_RELEASE_ASSET in $release_env}"
: "${HRX_RELEASE_SHA256:?missing HRX_RELEASE_SHA256 in $release_env}"

case "$HRX_RELEASE_TAG$HRX_RELEASE_ASSET$HRX_RELEASE_SHA256" in
  *TODO*)
    die "hrx-release.env still has TODO placeholders. The Linux agent must publish the HRX public package and fill in REPO/TAG/ASSET/SHA256 first."
    ;;
esac

out_dir="${1:-$root/.hrx-release}"
mkdir -p "$out_dir"

base_url="https://github.com/${HRX_RELEASE_REPO}/releases/download/${HRX_RELEASE_TAG}"
tarball="$out_dir/$HRX_RELEASE_ASSET"
sha_file="$tarball.sha256"

download() {
  local url="$1"
  local path="$2"
  echo "download: $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$path" "$url"
}

download "$base_url/$HRX_RELEASE_ASSET" "$tarball"
download "$base_url/$HRX_RELEASE_ASSET.sha256" "$sha_file"

remote_sha="$(awk '{print $1; exit}' "$sha_file")"
[[ -n "$remote_sha" ]] || die "empty remote sha256 file: $sha_file"
[[ "$remote_sha" == "$HRX_RELEASE_SHA256" ]] ||
  die "remote sha256 mismatch: expected $HRX_RELEASE_SHA256, got $remote_sha"

actual_sha="$(sha256sum "$tarball" | awk '{print $1}')"
[[ "$actual_sha" == "$HRX_RELEASE_SHA256" ]] ||
  die "tarball sha256 mismatch: expected $HRX_RELEASE_SHA256, got $actual_sha"

# Public package ships as .tar.zst (fall back to gzip for legacy assets).
# Decompress zstd via the standalone binary piped into tar, so extraction does
# not depend on tar being built with its zstd plugin, and fail with a clear
# message when the zstd package is not installed.
case "$HRX_RELEASE_ASSET" in
  *.tar.zst)
    command -v zstd >/dev/null 2>&1 ||
      die "zstd is required to extract $HRX_RELEASE_ASSET but was not found on PATH (install the 'zstd' package)"
    zstd -dc "$tarball" | tar -C "$out_dir" -xf -
    ;;
  *.tar.gz|*.tgz) tar -C "$out_dir" -xzf "$tarball" ;;
  *) tar -C "$out_dir" -xf "$tarball" ;;
esac

# Locate the HRX CMake package config; its dir feeds find_package(hrx CONFIG).
config_file="$(find "$out_dir" -type f \( -name 'hrx-config.cmake' -o -name 'hrxConfig.cmake' \) 2>/dev/null | head -n1)"
[[ -n "$config_file" ]] ||
  die "no HRX CMake package config found under $out_dir -- is this a public HRX package built with the CMake packaging flow and IREE_HAL_DRIVER_AMDXDNA=ON?"

config_dir="$(dirname "$config_file")"
# Prefer the install prefix (.../<prefix>/lib/cmake/hrx -> <prefix>); else the config dir.
prefix="$config_dir"
if [[ "$(basename "$config_dir")" == "hrx" ]]; then
  cmake_dir="$(dirname "$config_dir")"
  if [[ "$(basename "$cmake_dir")" == "cmake" ]]; then
    prefix="$(dirname "$(dirname "$cmake_dir")")"
  fi
fi

echo "HRX_CMAKE_CONFIG=$config_file"
echo "HRX_CMAKE_PREFIX=$prefix"
echo ""
echo "Configure FLM with: -DCMAKE_PREFIX_PATH=\"$prefix\""
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "prefix=$prefix" >> "$GITHUB_OUTPUT"
fi
