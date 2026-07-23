#!/bin/bash
set -euo pipefail

# Build psana wheel
# Usage: ./build_wheel.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_DIR}/builddir"
INSTALL_DIR="${REPO_DIR}/install"
DIST_DIR="${REPO_DIR}/dist"

echo "======================================"
echo "Building psana wheel"
echo "======================================"
echo ""

# Check Python version
PYTHON="${PYTHON:-python3}"
PYVER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: ${PYVER}"
echo "Python executable: ${PYTHON}"
echo ""

# Check requirements
if ! command -v meson >/dev/null; then
    echo "ERROR: meson not found in PATH"
    exit 1
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "WARNING: CONDA_PREFIX not set, using /usr as fallback"
    CONDA_PREFIX="/usr"
fi

if [ -z "${EPICS_BASE:-}" ]; then
    echo "WARNING: EPICS_BASE not set, you may need to set this"
    EPICS_BASE="${CONDA_PREFIX}"
fi

if [ -z "${EPICS_HOST_ARCH:-}" ]; then
    echo "WARNING: EPICS_HOST_ARCH not set, using rhel7-x86_64 as fallback"
    EPICS_HOST_ARCH="rhel7-x86_64"
fi

echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "EPICS_BASE: ${EPICS_BASE}"
echo "EPICS_HOST_ARCH: ${EPICS_HOST_ARCH}"
echo ""

# Clean previous build
if [ -d "${BUILD_DIR}" ] || [ -d "${INSTALL_DIR}" ]; then
    echo "=== Cleaning previous build ==="
    rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
    echo ""
fi

# Add conda include path to compiler flags
export CXXFLAGS="${CXXFLAGS:-} -I${CONDA_PREFIX}/include"
export CFLAGS="${CFLAGS:-} -I${CONDA_PREFIX}/include"

# Set PKG_CONFIG_PATH to find conda's Python
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Run Meson build
echo "=== Running Meson setup ==="
meson setup "${BUILD_DIR}" \
  --prefix="${INSTALL_DIR}" \
  -Dconda_prefix="${CONDA_PREFIX}" \
  -Depics_base="${EPICS_BASE}" \
  -Depics_host_arch="${EPICS_HOST_ARCH}" \
  -Dbuild_daq=false

echo ""
echo "=== Running Meson compile ==="
meson compile -C "${BUILD_DIR}"

echo ""
echo "=== Running Meson install ==="
meson install -C "${BUILD_DIR}"

echo ""
echo "=== Verifying Meson install ==="
echo "Checking for Python packages in install tree..."
if [ -d "${INSTALL_DIR}/lib/python${PYVER}/site-packages" ]; then
    echo "Found: ${INSTALL_DIR}/lib/python${PYVER}/site-packages/"
    ls -la "${INSTALL_DIR}/lib/python${PYVER}/site-packages/" || true
else
    echo "ERROR: Python site-packages directory not found!"
    echo "Looking for: ${INSTALL_DIR}/lib/python${PYVER}/site-packages/"
    echo ""
    echo "Available directories in ${INSTALL_DIR}/lib/:"
    ls -la "${INSTALL_DIR}/lib/" || true
    exit 1
fi

echo ""
echo "Checking for shared libraries..."
ls -la "${INSTALL_DIR}/lib/" | grep "\.so" || echo "No .so files found in lib/"

echo ""
echo "Checking for executables..."
if [ -d "${INSTALL_DIR}/bin" ]; then
    ls -la "${INSTALL_DIR}/bin/" || true
else
    echo "No bin/ directory found"
fi

# Update pyproject.toml source mapping for correct Python version
echo ""
echo "=== Updating pyproject.toml for Python ${PYVER} ==="
# Create a temporary pyproject.toml with correct Python version paths
cp pyproject.toml pyproject.toml.bak

# Set packages to empty list (disable auto-detection, use sources mapping only)
# This handles both cases: packages = ["psana", "psalg"] or packages = []
sed -i 's|^packages = .*|packages = []|' pyproject.toml

# Update sources mapping to use the current Python version
sed -i "s|\"install/lib\" = \"lib\"|\"install/lib/python${PYVER}/site-packages/psana\" = \"psana\"\n\"install/lib/python${PYVER}/site-packages/psalg\" = \"psalg\"|" pyproject.toml

# Build wheel
echo ""
echo "=== Building wheel with Hatchling ==="
"${PYTHON}" -m pip install --quiet build hatchling

# Set Python tag for the wheel (cpXYZ for CPython X.Y.Z)
PYTAG="cp${PYVER//./}"  # e.g., "3.13" -> "cp313"

# Build with correct platform tags
"${PYTHON}" -m build --wheel --outdir "${DIST_DIR}"

# Rename wheel to have correct platform-specific tags
# Hatchling may produce a generic wheel, but we have compiled extensions
# so we need platform-specific tags
OLD_WHEEL="${DIST_DIR}/psana-4.3-py3-none-any.whl"
NEW_WHEEL="${DIST_DIR}/psana-4.3-${PYTAG}-${PYTAG}-linux_x86_64.whl"

if [ -f "${OLD_WHEEL}" ]; then
    mv "${OLD_WHEEL}" "${NEW_WHEEL}"
    echo "Renamed wheel to: $(basename ${NEW_WHEEL})"
fi

# Restore original pyproject.toml
mv pyproject.toml.bak pyproject.toml

# Show results
echo ""
echo "======================================"
echo "Build complete!"
echo "======================================"
if [ -f "${DIST_DIR}"/*.whl ]; then
    ls -lh "${DIST_DIR}"/*.whl
    echo ""
    echo "Wheel contents summary:"
    unzip -l "${DIST_DIR}"/*.whl | grep -E "psana/|psalg/|\.so" | head -30
else
    echo "ERROR: No wheel file found in ${DIST_DIR}/"
    exit 1
fi
