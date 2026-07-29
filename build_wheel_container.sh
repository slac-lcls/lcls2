#!/bin/bash
set -euo pipefail

# Container-only prepare phase, intended to run as cibuildwheel's before-build
# step inside a manylinux container (no Conda, no EPICS).
#
# Unlike build_wheel.sh (the manual/Conda workflow, left untouched), this
# script does NOT set a Conda fallback, does NOT inject conda include/pkgconfig
# paths, and does NOT call the final wheel build or restore pyproject.toml
# afterward. It only runs Meson configure/compile/install and patches
# pyproject.toml's source mapping for the active Python version; cibuildwheel's
# own build step (hatchling) runs immediately after this against the patched
# pyproject.toml. The container checkout is ephemeral, so no restore step is
# needed.
#
# See docs/topics/psana-wheel-packaging.md (as-of 2026-07-29 container-build
# sections) for why build_wheel.sh's Conda-oriented defaults don't apply here.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_DIR}/builddir"
INSTALL_DIR="${REPO_DIR}/install"

echo "======================================"
echo "Preparing psana build (container)"
echo "======================================"
echo ""

PYTHON="${PYTHON:-python3}"
PYVER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: ${PYVER}"
echo "Python executable: ${PYTHON}"
echo ""

if ! command -v meson >/dev/null; then
    echo "ERROR: meson not found in PATH"
    exit 1
fi

# Clean previous build
if [ -d "${BUILD_DIR}" ] || [ -d "${INSTALL_DIR}" ]; then
    echo "=== Cleaning previous build ==="
    rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
    echo ""
fi

# Meson's Cython-language sanity check (runs automatically at project()
# declaration, before any of our code) needs Python.h on the compiler's
# DEFAULT search path. That's not guaranteed in a Conda env (headers under
# $CONDA_PREFIX/include) or in a manylinux container (headers under
# /opt/pythonX.Y/include, also not a default path). sysconfig resolves this
# correctly in both cases -- unlike build_wheel.sh's Conda-only
# -I${CONDA_PREFIX}/include, this works for any active Python, Conda or not.
# Confirmed locally 2026-07-29: without this, meson setup fails immediately
# with "ERROR: Compiler cython cannot compile programs." See PODMAN_TEST_LOG.md.
PY_INCLUDE="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export CFLAGS="${CFLAGS:-} -I${PY_INCLUDE}"
export CXXFLAGS="${CXXFLAGS:-} -I${PY_INCLUDE}"
echo "Python include path (sysconfig): ${PY_INCLUDE}"
echo ""

# No -Dconda_prefix, no -Depics_base/-Depics_host_arch: conda_prefix defaults
# to '' (meson.build now skips the conda linker flags when empty), and
# epics_base/epics_host_arch default to '' as well (harmless now that
# libdir_install_rpath only uses them when build_daq=true).
echo "=== Running Meson setup (container defaults, build_daq=false) ==="
meson setup "${BUILD_DIR}" \
  --prefix="${INSTALL_DIR}" \
  -Dbuild_daq=false

echo ""
echo "=== Running Meson compile ==="
meson compile -C "${BUILD_DIR}"

echo ""
echo "=== Running Meson install ==="
meson install -C "${BUILD_DIR}"

echo ""
echo "=== Verifying Meson install ==="
if [ ! -d "${INSTALL_DIR}/lib/python${PYVER}/site-packages" ]; then
    echo "ERROR: Python site-packages directory not found!"
    echo "Looking for: ${INSTALL_DIR}/lib/python${PYVER}/site-packages/"
    echo ""
    echo "Available directories in ${INSTALL_DIR}/lib/:"
    ls -la "${INSTALL_DIR}/lib/" || true
    exit 1
fi
echo "Found: ${INSTALL_DIR}/lib/python${PYVER}/site-packages/"

# Patch pyproject.toml source mapping for this Python version.
# Deliberately NOT restored here (unlike build_wheel.sh) -- cibuildwheel's
# own build step runs against this patched file immediately after.
echo ""
echo "=== Patching pyproject.toml for Python ${PYVER} ==="
sed -i 's|^packages = .*|packages = []|' "${REPO_DIR}/pyproject.toml"
sed -i "s|\"install/lib\" = \"lib\"|\"install/lib/python${PYVER}/site-packages/psana\" = \"psana\"\n\"install/lib/python${PYVER}/site-packages/psalg\" = \"psalg\"|" "${REPO_DIR}/pyproject.toml"

echo ""
echo "Prepare phase complete. cibuildwheel's build step will now package install/ via hatchling."
