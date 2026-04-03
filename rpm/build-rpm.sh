#!/bin/bash
set -euo pipefail

# Build an RPM package for FastFlowLM
# Usage: ./rpm/build-rpm.sh [version]
#
# Prerequisites (Fedora/RHEL):
#   sudo dnf install rpm-build rpmdevtools
#   sudo dnf builddep rpm/fastflowlm.spec

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION="${1:-0.9.38}"
PKG_NAME="fastflowlm"

echo "==> Building ${PKG_NAME}-${VERSION} RPM"

# Set up rpmbuild tree
RPMBUILD_DIR="${REPO_ROOT}/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create source tarball from the repo
echo "==> Creating source tarball..."
TARBALL_DIR="${PKG_NAME}-${VERSION}"
TAR_PATH="${RPMBUILD_DIR}/SOURCES/${PKG_NAME}-${VERSION}.tar.gz"

git -C "$REPO_ROOT" archive \
    --format=tar.gz \
    --prefix="${TARBALL_DIR}/" \
    --output="$TAR_PATH" \
    HEAD

# If there are submodules, we need to include them
if [ -f "$REPO_ROOT/.gitmodules" ]; then
    echo "==> Including submodules in tarball..."
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT

    # Extract the archive
    tar -xzf "$TAR_PATH" -C "$TMPDIR"

    # Add submodule contents
    git -C "$REPO_ROOT" submodule foreach --quiet --recursive \
        'SUBPATH="${displaypath}"; \
         mkdir -p "'"$TMPDIR/${TARBALL_DIR}"'/${SUBPATH}"; \
         git archive --format=tar HEAD | tar -x -C "'"$TMPDIR/${TARBALL_DIR}"'/${SUBPATH}"'

    # Recreate the tarball
    tar -czf "$TAR_PATH" -C "$TMPDIR" "${TARBALL_DIR}"
fi

# Copy spec file
cp "$SCRIPT_DIR/fastflowlm.spec" "$RPMBUILD_DIR/SPECS/"

# Build the RPM
echo "==> Running rpmbuild..."
rpmbuild \
    --define "_topdir ${RPMBUILD_DIR}" \
    --define "flm_version ${VERSION}" \
    -ba "$RPMBUILD_DIR/SPECS/fastflowlm.spec"

echo ""
echo "==> Build complete!"
echo "    RPMs:  $(find "$RPMBUILD_DIR/RPMS" -name '*.rpm')"
echo "    SRPMs: $(find "$RPMBUILD_DIR/SRPMS" -name '*.rpm')"
