#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build_psana.sh [options]

Build a minimal local psana install using the current Meson + root package flow.
This helper is intended for DataSource, detnames, and related psana workflows
without building psdaq.

Options:
  -p, --prefix DIR        Installation prefix (default: <repo>/install_psana)
  -t, --build-type TYPE   Meson build type: debug, debugoptimized, release,
                          minsize, plain (default: debugoptimized)
  -j, --jobs N            Parallel build jobs (default: nproc or 4)
      --build-dir DIR     Meson build directory (default: <repo>/builddir_psana)
      --clean             Remove previous install and build directories first
      --python-only       Skip Meson configure/compile/install and only refresh
                          the installed Python package
      --with-cuda         Allow nvcc detection and CUDA subprojects
      --build-list LIST   Legacy option from setup.py flow; ignored now
      --with-psalg        Legacy option from setup.py flow; ignored now
  -h, --help              Show this message

Examples:
  ./build_psana.sh --clean
  ./build_psana.sh --build-type release
  ./build_psana.sh -p /tmp/psana-min
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

warn_legacy_flags() {
  if [[ -n "$legacy_build_list" ]]; then
    printf '[build_psana] Warning: --build-list is ignored by the Meson build flow: %s\n' "$legacy_build_list" >&2
  fi
  if [[ "$legacy_with_psalg" -eq 1 ]]; then
    printf '[build_psana] Warning: --with-psalg is ignored; psalg is built by the Meson project.\n' >&2
  fi
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
      printf 'Unsupported build type: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)
      install_prefix="$2"
      shift 2
      ;;
    -t|--build-type)
      build_type="$(normalize_build_type "$2")"
      shift 2
      ;;
    -j|--jobs)
      jobs="$2"
      shift 2
      ;;
    --build-dir)
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
      printf 'Unknown option: %s\n' "$1" >&2
      usage
      exit 1
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

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf "Python interpreter '%s' not found. Set PYTHON to override.\n" "$python_bin" >&2
  exit 1
fi
if ! command -v meson >/dev/null 2>&1; then
  printf "meson is required but was not found in PATH.\n" >&2
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  printf "ninja is required but was not found in PATH.\n" >&2
  exit 1
fi
if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
  printf "pip is required but was not found for %s.\n" "$python_bin" >&2
  exit 1
fi

log() {
  printf '[build_psana] %s\n' "$*"
}

remove_path_entry() {
  local remove_dir="$1"
  local current_path="$2"
  local updated_path=""
  local entry=""
  IFS=':' read -r -a _path_entries <<< "$current_path"
  for entry in "${_path_entries[@]}"; do
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

warn_legacy_flags

if [[ "$clean_first" -eq 1 && "$python_only" -eq 1 ]]; then
  printf -- "--python-only cannot be combined with --clean.\n" >&2
  exit 1
fi

if [[ "$clean_first" -eq 1 ]]; then
  log "Cleaning previous build outputs"
  rm -rf "$install_prefix" "$build_dir"
fi

if [[ "$python_only" -eq 1 ]]; then
  if [[ ! -d "$build_dir" ]]; then
    printf -- "--python-only requires an existing build directory: %s\n" "$build_dir" >&2
    printf -- "Run a normal ./build_psana.sh first.\n" >&2
    exit 1
  fi
  if [[ ! -d "$install_prefix/lib" ]]; then
    printf -- "--python-only requires an existing install tree under: %s\n" "$install_prefix" >&2
    printf -- "Run a normal ./build_psana.sh first.\n" >&2
    exit 1
  fi
fi

mkdir -p "$install_prefix"

pyver="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
site_packages_dir="$install_prefix/lib/python${pyver}/site-packages"
mkdir -p "$site_packages_dir"

conda_prefix="${CONDA_PREFIX:-}"
if [[ -z "$conda_prefix" ]]; then
  printf "CONDA_PREFIX is not set. Activate the psana build environment first.\n" >&2
  exit 1
fi

epics_base="${EPICS_BASE:-}"
epics_host_arch="${EPICS_HOST_ARCH:-}"

meson_options=(
  "-Dconda_prefix=$conda_prefix"
  "-Dprefix=$install_prefix"
  "-Dbuild_daq=false"
  "-Dpython.bytecompile=-1"
  "--buildtype=$build_type"
)

if [[ -n "$epics_base" ]]; then
  meson_options+=("-Depics_base=$epics_base")
fi
if [[ -n "$epics_host_arch" ]]; then
  meson_options+=("-Depics_host_arch=$epics_host_arch")
fi

build_path="${PATH}"
restore_linker_env=0
if command -v nvcc >/dev/null 2>&1 && [[ "$with_cuda" -eq 1 ]]; then
  restore_linker_env=1
  export LDFLAGS_OLD="${LDFLAGS:-}"
  export CXXFLAGS_OLD="${CXXFLAGS:-}"
  export LDFLAGS=""
  export CXXFLAGS=""
  if [[ -n "${CUDA_ROOT:-}" && -e "${CUDA_ROOT:-}" ]]; then
    meson_options+=("-Dcustom_cuda_path=$CUDA_ROOT")
  fi
fi

if command -v nvcc >/dev/null 2>&1 && [[ "$with_cuda" -eq 0 ]]; then
  nvcc_path="$(command -v nvcc)"
  nvcc_dir="$(dirname "$nvcc_path")"
  build_path="$(remove_path_entry "$nvcc_dir" "$PATH")"
fi

pip_cmd=("$python_bin" -m pip)
if command -v uv >/dev/null 2>&1; then
  pip_cmd=(uv pip)
fi

cleanup_linker_env() {
  if [[ "$restore_linker_env" -eq 1 ]]; then
    export LDFLAGS="${LDFLAGS_OLD}"
    export CXXFLAGS="${CXXFLAGS_OLD}"
    unset LDFLAGS_OLD
    unset CXXFLAGS_OLD
  fi
}
trap cleanup_linker_env EXIT

log "Install prefix : $install_prefix"
log "Build directory : $build_dir"
log "Build type     : $build_type"
log "Parallel jobs  : $jobs"
log "Python         : $python_bin"
log "Conda prefix   : $conda_prefix"
log "Pip frontend   : ${pip_cmd[*]}"
log "Python only    : $python_only"
log "CUDA enabled   : $with_cuda"

if [[ "$python_only" -eq 0 && ! -d "$build_dir" ]]; then
  log "Configuring Meson build"
  (
    cd "$repo_dir"
    PATH="$build_path" meson setup "$build_dir" "${meson_options[@]}"
  )
fi

if [[ "$python_only" -eq 0 ]]; then
  log "Compiling Meson targets"
  PATH="$build_path" meson compile -C "$build_dir" -j "$jobs"

  log "Installing Meson targets"
  PATH="$build_path" meson install --only-changed --no-rebuild --quiet -C "$build_dir"
else
  log "Skipping Meson configure/compile/install (--python-only)"
fi

log "Installing Python package"
(
  cd "$repo_dir"
  PATH="$build_path" "${pip_cmd[@]}" install . \
    --no-compile \
    --no-deps \
    --no-build-isolation \
    --prefix="$install_prefix" \
    --config-settings setup-args="${meson_options[*]}" \
    --config-settings compile-args="-j$jobs" \
    --config-settings install-args="--only-changed --no-rebuild"
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

Then verify with:

  python -c "import psana; from psana import DataSource; print('psana ok')"
  detnames <xtc2-file>
EOF
