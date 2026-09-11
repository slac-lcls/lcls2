#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build_psana.sh [options]

Build and install the analysis-only psana stack (xtcdata, psalg, and psana)
with the repository's root Meson build.  psdaq is not built; CUDA subprojects
are disabled unless --with-cuda is requested.

Run this from an activated development environment containing the psana build
dependencies.  The completed install includes an activate.sh helper.

Options:
  -p, --prefix DIR        Installation prefix (default: <repo>/install_psana)
  -t, --build-type TYPE   Meson build type: debug, debugoptimized, release,
                          minsize, or plain (default: debugoptimized)
  -j, --jobs N            Parallel build jobs (default: nproc or 4)
      --build-dir DIR     Meson build directory (default: <repo>/builddir_psana)
      --clean             Remove the selected install and build directories
                          before building
      --python-only       Refresh installed Python files and entry points from
                          an existing native build without compiling
      --with-cuda         Allow nvcc detection and CUDA subprojects
      --build-list LIST   Accepted for compatibility; ignored by Meson
      --with-psalg        Accepted for compatibility; psalg is always built
  -h, --help              Show this message

Examples:
  ./build_psana.sh --clean -j 8
  ./build_psana.sh --python-only
  source ./install_psana/activate.sh
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install_prefix="$repo_dir/install_psana"
build_dir="$repo_dir/builddir_psana"
build_type="debugoptimized"
jobs=""
clean_first=0
python_only=0
with_cuda=0
legacy_build_list=""
legacy_with_psalg=0

log() {
  printf '[build_psana] %s\n' "$*"
}

die() {
  printf '[build_psana] Error: %s\n' "$*" >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "Option $1 requires a value."
}

normalize_build_type() {
  case "$1" in
    Debug|debug)
      printf 'debug'
      ;;
    Release|release)
      printf 'release'
      ;;
    RelWithDebInfo|relwithdebinfo|debugoptimized)
      printf 'debugoptimized'
      ;;
    MinSizeRel|minsizerel|minsize)
      printf 'minsize'
      ;;
    plain|Plain)
      printf 'plain'
      ;;
    *)
      die "Unsupported build type: $1"
      ;;
  esac
}

remove_path_entry() {
  local remove_dir="$1"
  local current_path="$2"
  local updated_path=""
  local entry=""
  IFS=':' read -r -a path_entries <<< "$current_path"
  for entry in "${path_entries[@]}"; do
    if [[ -n "$entry" && "$entry" != "$remove_dir" ]]; then
      if [[ -z "$updated_path" ]]; then
        updated_path="$entry"
      else
        updated_path="${updated_path}:$entry"
      fi
    fi
  done
  printf '%s' "$updated_path"
}

validate_clean_target() {
  local target="$1"
  local label="$2"
  case "$target" in
    ""|/|"$repo_dir"|"${HOME:-__unset_home__}")
      die "Refusing to clean unsafe $label path: $target"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)
      require_value "$@"
      install_prefix="$2"
      shift 2
      ;;
    -t|--build-type)
      require_value "$@"
      build_type="$(normalize_build_type "$2")"
      shift 2
      ;;
    -j|--jobs)
      require_value "$@"
      jobs="$2"
      shift 2
      ;;
    --build-dir)
      require_value "$@"
      build_dir="$2"
      shift 2
      ;;
    --clean)
      clean_first=1
      shift
      ;;
    --python-only)
      python_only=1
      shift
      ;;
    --with-cuda)
      with_cuda=1
      shift
      ;;
    -b|--build-list)
      require_value "$@"
      legacy_build_list="$2"
      shift 2
      ;;
    --with-psalg)
      legacy_with_psalg=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1 (use --help for usage)"
      ;;
  esac
done

if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  elif [[ "${OSTYPE:-}" == darwin* ]] && command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
  else
    jobs=4
  fi
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer: $jobs"

if [[ -n "$legacy_build_list" ]]; then
  log "Warning: --build-list is ignored by the Meson build: $legacy_build_list"
fi
if [[ "$legacy_with_psalg" -eq 1 ]]; then
  log "Warning: --with-psalg is unnecessary; psalg is always built."
fi
if [[ "$clean_first" -eq 1 && "$python_only" -eq 1 ]]; then
  die "--python-only cannot be combined with --clean."
fi

python_bin="${PYTHON:-python3}"
command -v "$python_bin" >/dev/null 2>&1 || \
  die "Python interpreter '$python_bin' was not found. Set PYTHON to override."
command -v meson >/dev/null 2>&1 || die "meson was not found in PATH."
command -v ninja >/dev/null 2>&1 || die "ninja was not found in PATH."
"$python_bin" -m pip --version >/dev/null 2>&1 || \
  die "pip was not found for $python_bin."

install_prefix="$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$install_prefix")"
build_dir="$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$build_dir")"

conda_prefix="${CONDA_PREFIX:-}"
[[ -n "$conda_prefix" ]] || \
  die "CONDA_PREFIX is not set. Activate the psana development environment first."
[[ -f "$conda_prefix/include/rapidjson/document.h" ]] || \
  die "RapidJSON headers were not found under $conda_prefix/include."

if [[ "$clean_first" -eq 1 ]]; then
  validate_clean_target "$install_prefix" "install prefix"
  validate_clean_target "$build_dir" "build directory"
  log "Cleaning previous build outputs"
  rm -rf "$install_prefix" "$build_dir"
fi

if [[ "$python_only" -eq 1 ]]; then
  [[ -d "$build_dir/meson-info" ]] || \
    die "--python-only requires an existing Meson build: $build_dir"
  [[ -d "$install_prefix/lib" ]] || \
    die "--python-only requires an existing install: $install_prefix"
  configured_prefix="$("$python_bin" - "$build_dir/meson-info/intro-buildoptions.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    options = json.load(stream)
print(next(option["value"] for option in options if option["name"] == "prefix"))
PY
)"
  [[ "$configured_prefix" == "$install_prefix" ]] || \
    die "Meson build prefix is $configured_prefix, not $install_prefix; rerun with the matching --prefix or use --clean."
fi

mkdir -p "$install_prefix"

pyver="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
site_packages_dir="$install_prefix/lib/python${pyver}/site-packages"

meson_options=(
  "-Dconda_prefix=$conda_prefix"
  "-Dprefix=$install_prefix"
  "-Dbuild_daq=false"
  "-Dpython.bytecompile=-1"
  "--buildtype=$build_type"
)

if [[ -n "${EPICS_BASE:-}" ]]; then
  meson_options+=("-Depics_base=$EPICS_BASE")
fi
if [[ -n "${EPICS_HOST_ARCH:-}" ]]; then
  meson_options+=("-Depics_host_arch=$EPICS_HOST_ARCH")
fi

# Root Meson builds psalg sources that include headers supplied by the active
# environment (RapidJSON on the NERSC build environment).
build_cpath="$conda_prefix/include${CPATH:+:$CPATH}"
build_path="$PATH"
restore_linker_env=0

if command -v nvcc >/dev/null 2>&1 && [[ "$with_cuda" -eq 1 ]]; then
  restore_linker_env=1
  export BUILD_PSANA_OLD_LDFLAGS="${LDFLAGS:-}"
  export BUILD_PSANA_OLD_CXXFLAGS="${CXXFLAGS:-}"
  export LDFLAGS=""
  export CXXFLAGS=""
  if [[ -n "${CUDA_ROOT:-}" && -e "$CUDA_ROOT" ]]; then
    meson_options+=("-Dcustom_cuda_path=$CUDA_ROOT")
  fi
elif command -v nvcc >/dev/null 2>&1; then
  nvcc_dir="$(dirname "$(command -v nvcc)")"
  build_path="$(remove_path_entry "$nvcc_dir" "$PATH")"
fi

cleanup_linker_env() {
  if [[ "$restore_linker_env" -eq 1 ]]; then
    export LDFLAGS="$BUILD_PSANA_OLD_LDFLAGS"
    export CXXFLAGS="$BUILD_PSANA_OLD_CXXFLAGS"
    unset BUILD_PSANA_OLD_LDFLAGS BUILD_PSANA_OLD_CXXFLAGS
  fi
}
trap cleanup_linker_env EXIT

log "Source directory : $repo_dir"
log "Install prefix   : $install_prefix"
log "Build directory  : $build_dir"
log "Build type       : $build_type"
log "Parallel jobs    : $jobs"
log "Python           : $(command -v "$python_bin")"
log "Conda prefix     : $conda_prefix"
log "Python only      : $python_only"
log "CUDA enabled     : $with_cuda"

if [[ "$python_only" -eq 0 ]]; then
  if [[ -d "$build_dir/meson-info" ]]; then
    log "Reconfiguring Meson build"
    PATH="$build_path" CPATH="$build_cpath" \
      meson setup --reconfigure "$build_dir" "${meson_options[@]}"
  else
    log "Configuring Meson build"
    PATH="$build_path" CPATH="$build_cpath" \
      meson setup "$build_dir" "${meson_options[@]}"
  fi

  log "Compiling Meson targets"
  PATH="$build_path" CPATH="$build_cpath" \
    meson compile -C "$build_dir" -j "$jobs"
else
  log "Skipping Meson configure and compile (--python-only)"
fi

log "Installing Meson targets"
PATH="$build_path" CPATH="$build_cpath" \
  meson install --only-changed --no-rebuild --quiet -C "$build_dir"

# pyproject.toml's wheel metadata expects native libraries below ./install.
# Use a private staging project so a custom --prefix works and an unrelated,
# stale ./install tree can never leak into this installation.
package_dir="$build_dir/python-package"
rm -rf "$package_dir"
mkdir -p "$package_dir/install/lib"
cp "$repo_dir/pyproject.toml" \
   "$repo_dir/hatch_build.py" \
   "$repo_dir/README.md" \
   "$repo_dir/LICENSE.md" \
   "$package_dir/"

shopt -s nullglob
installed_libraries=("$install_prefix"/lib/*.so*)
shopt -u nullglob
[[ ${#installed_libraries[@]} -gt 0 ]] || \
  die "No shared libraries were installed under $install_prefix/lib."
for library in "${installed_libraries[@]}"; do
  ln -s "$library" "$package_dir/install/lib/$(basename "$library")"
done

log "Installing Python metadata and command-line entry points"
PATH="$build_path" CPATH="$build_cpath" \
  "$python_bin" -m pip install "$package_dir" \
    --no-compile \
    --no-deps \
    --no-build-isolation \
    --prefix="$install_prefix"

activation_file="$install_prefix/activate.sh"
log "Writing runtime activation helper: $activation_file"
cat >"$activation_file" <<EOF
# Source this file after activating the Python environment used to build psana.
export PATH="$install_prefix/bin":\${PATH:-}
export PYTHONPATH="$site_packages_dir":\${PYTHONPATH:-}
export LD_LIBRARY_PATH="$install_prefix/lib":\${LD_LIBRARY_PATH:-}
EOF

log "Verifying the installed package"
{
  cd "$install_prefix"
  PATH="$install_prefix/bin:$PATH" \
  PYTHONPATH="$site_packages_dir${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH="$install_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  BUILD_PSANA_EXPECTED_SITE="$site_packages_dir" \
    "$python_bin" - <<'PY'
import os
from pathlib import Path

import psana
import psana.dgram

location = Path(psana.__file__).resolve()
expected_site = Path(os.environ["BUILD_PSANA_EXPECTED_SITE"]).resolve()
if expected_site not in location.parents:
    raise SystemExit(f"Imported psana from an unexpected path: {location}")
print(f"Verified psana: {location}")
PY
}

log "Build completed and verified."
cat <<EOF

Activate this install in the current shell with:

  source "$activation_file"

Then, for example:

  python -c "import psana; print(psana.__file__)"
  detnames <xtc2-file>
EOF
