#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build_psana.sh [options]

Build the minimum set of components (xtcdata + psana) required to run
psana/psana/tests/user_loops.py without sourcing setup_env.sh.

Options:
  -p, --prefix DIR        Installation prefix (default: <repo>/install_psana)
  -b, --build-list LIST   BUILD_LIST passed to psana (default: PSANA:DGRAM)
  -t, --build-type TYPE   CMAKE_BUILD_TYPE (default: RelWithDebInfo)
  -j, --jobs N            Parallel build jobs (default: nproc or 4)
      --cmake-prefix DIR  Extra CMAKE_PREFIX_PATH entry (optional)
      --with-psalg        Force psalg build even if BUILD_LIST does not need it
      --clean             Remove previous install and build directories first
  -h, --help              Show this message

Examples:
  ./build_psana.sh --clean
  ./build_psana.sh -b "PSANA:DGRAM:NDARRAY" --with-psalg
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install_prefix="$repo_dir/install_psana"
build_type="RelWithDebInfo"
build_list="PSANA:DGRAM"
jobs=""
cmake_prefix=""
force_psalg=0
clean_first=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)
      install_prefix="$2"
      shift 2
      ;;
    -b|--build-list)
      build_list="$2"
      shift 2
      ;;
    -t|--build-type)
      build_type="$2"
      shift 2
      ;;
    -j|--jobs)
      jobs="$2"
      shift 2
      ;;
    --cmake-prefix)
      cmake_prefix="$2"
      shift 2
      ;;
    --with-psalg)
      force_psalg=1
      shift
      ;;
    --clean)
      clean_first=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  elif [[ "$OSTYPE" == darwin* ]] && command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
  else
    jobs=4
  fi
fi

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter '$python_bin' not found. Set PYTHON to override." >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required but was not found in PATH." >&2
  exit 1
fi
if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
  echo "pip is required but was not found for $python_bin." >&2
  exit 1
fi

xtc_build_dir="$repo_dir/xtcdata/build_psana"
psalg_build_dir="$repo_dir/psalg/build_psana"
site_packages_dir=""

log() {
  printf '[build_psana] %s\n' "$*"
}

if [[ "$clean_first" -eq 1 ]]; then
  log "Cleaning previous build outputs"
  rm -rf "$install_prefix" "$xtc_build_dir" "$psalg_build_dir"
fi

mkdir -p "$install_prefix"

pyver="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
site_packages_dir="$install_prefix/lib/python${pyver}/site-packages"
mkdir -p "$site_packages_dir"

cmake_prefix_args=()
if [[ -n "${cmake_prefix:-}" ]]; then
  cmake_prefix_args+=("-DCMAKE_PREFIX_PATH=$cmake_prefix")
fi

requires_psalg=0
if [[ "$build_list" =~ (SHMEM|PEAKFINDER|CFD|NDARRAY|PYCALGOS) ]]; then
  requires_psalg=1
fi
if [[ "$force_psalg" -eq 1 ]]; then
  requires_psalg=1
fi

build_shmem="OFF"
if [[ "$build_list" =~ SHMEM ]]; then
  build_shmem="ON"
fi

log "Install prefix : $install_prefix"
log "BUILD_LIST     : $build_list"
log "CMake type     : $build_type"
log "Parallel jobs  : $jobs"
log "Python         : $python_bin"

# Build xtcdata (always required)
log "Configuring xtcdata"
cmake -S "$repo_dir/xtcdata" \
      -B "$xtc_build_dir" \
      -DCMAKE_INSTALL_PREFIX="$install_prefix" \
      -DCMAKE_BUILD_TYPE="$build_type" \
      "${cmake_prefix_args[@]}"
log "Building xtcdata"
cmake --build "$xtc_build_dir" --target install -j "$jobs"

psalg_env=()
if [[ "$requires_psalg" -eq 1 ]]; then
  log "Configuring psalg (BUILD_SHMEM=$build_shmem)"
  cmake -S "$repo_dir/psalg" \
        -B "$psalg_build_dir" \
        -DCMAKE_INSTALL_PREFIX="$install_prefix" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_SHMEM="$build_shmem" \
        -DCMAKE_PREFIX_PATH="$install_prefix${cmake_prefix:+;$cmake_prefix}"
  log "Building psalg"
  cmake --build "$psalg_build_dir" --target install -j "$jobs"
  log "Installing psalg python bindings"
  (cd "$repo_dir/psalg" && \
    "$python_bin" -m pip install --no-deps --no-build-isolation --prefix="$install_prefix" --editable .)
  psalg_env=(PSALGDIR="$install_prefix")
fi

log "Building psana extensions"
(
  cd "$repo_dir/psana"
  env \
    INSTDIR="$install_prefix" \
    XTCDATADIR="$install_prefix" \
    BUILD_LIST="$build_list" \
    "${psalg_env[@]}" \
    "$python_bin" setup.py build_ext -f --inplace
)

log "Installing psana python package"
(
  cd "$repo_dir/psana"
  env \
    INSTDIR="$install_prefix" \
    XTCDATADIR="$install_prefix" \
    BUILD_LIST="$build_list" \
    "${psalg_env[@]}" \
    "$python_bin" -m pip install --no-deps --no-build-isolation --prefix="$install_prefix" --editable .
)

sitecustomize="$site_packages_dir/sitecustomize.py"
if [[ ! -f "$sitecustomize" ]]; then
  log "Creating sitecustomize.py at $sitecustomize"
  cat >"$sitecustomize" <<EOF
import site
site.addsitedir(r"$site_packages_dir")
EOF
fi

log "Build completed."
cat <<EOF
To run psana with this install prefix, add the following to your environment:

  export PYTHONPATH="$site_packages_dir":\${PYTHONPATH:-}
  export PATH="$install_prefix/bin":\${PATH:-}

Then execute:

  TEST_XTC_DIR="$repo_dir/psana/psana/tests" pytest psana/psana/tests/user_loops.py
EOF
