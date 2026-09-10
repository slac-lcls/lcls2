#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build_psana.sh [options]

Build and install the analysis-only psana stack (xtcdata, psalg, and psana)
with the repository's root Meson build.  psdaq is not built; CUDA compiler
discovery is suppressed unless --with-cuda is requested.

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
      --with-cuda         Allow the root Meson project to discover nvcc
      --perlmutter-setup FILE
                          After a successful build, write a sourceable
                          Perlmutter runtime setup script to FILE
      --build-list LIST   Accepted for compatibility; ignored by Meson
      --with-psalg        Accepted for compatibility; psalg is always built
  -h, --help              Show this message

Examples:
  ./build_psana.sh --clean -j 8
  ./build_psana.sh --python-only
  ./build_psana.sh --perlmutter-setup /path/to/setup_env_perlmutter.sh
  source ./install_psana/activate.sh
EOF
}

repo_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_install_prefix="$repo_dir/install_psana"
default_build_dir="$repo_dir/builddir_psana"
install_prefix="$default_install_prefix"
build_dir="$default_build_dir"
build_type="debugoptimized"
jobs=""
clean_first=0
python_only=0
with_cuda=0
perlmutter_setup=""
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

validate_clean_targets() {
  "$python_bin" - \
    "$install_prefix" \
    "$build_dir" \
    "$repo_dir" \
    "$default_install_prefix" \
    "$default_build_dir" \
    "${HOME:-}" \
    "$conda_prefix" \
    "$install_marker" \
    "$build_marker" <<'PY'
from pathlib import Path
import sys


def resolved(value):
    return Path(value).expanduser().resolve()


def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


install, build, repo, default_install, default_build = map(resolved, sys.argv[1:6])
home = resolved(sys.argv[6]) if sys.argv[6] else None
conda = resolved(sys.argv[7]) if sys.argv[7] else None
targets = (
    ("install prefix", install, default_install, Path(sys.argv[8]), "install"),
    ("build directory", build, default_build, Path(sys.argv[9]), "build"),
)
protected = [("source repository", repo)]
if home is not None and home != Path(home.anchor):
    protected.append(("home directory", home))

def has_managed_marker(marker, kind, target):
    try:
        contents = marker.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    return contents == [
        "lcls2 build_psana managed root v1",
        kind,
        str(target),
    ]


for label, target, allowed_repo_target, marker, marker_kind in targets:
    if target == Path(target.anchor):
        raise SystemExit(f"Refusing to clean unsafe {label}: {target}")
    for protected_label, protected_path in protected:
        if within(protected_path, target):
            raise SystemExit(
                f"Refusing to clean {label} {target}; it contains the {protected_label}"
            )
    if within(target, repo) and target != allowed_repo_target:
        raise SystemExit(
            f"Refusing to clean {label} inside the source repository: {target}"
        )
    if conda is not None and (within(target, conda) or within(conda, target)):
        raise SystemExit(
            f"Refusing to clean {label} overlapping the active environment: {target}"
        )
    if (
        target.exists()
        and not has_managed_marker(marker, marker_kind, target)
    ):
        raise SystemExit(
            f"Refusing to clean existing unmanaged {label}: {target}. "
            "Choose a new path or remove it explicitly after verifying its contents."
        )

if within(install, build) or within(build, install):
    raise SystemExit(
        f"Refusing overlapping clean targets: install={install}, build={build}"
    )
PY
}

write_managed_marker() {
  local marker="$1"
  local kind="$2"
  local target="$3"
  local resolved_target marker_tmp

  resolved_target="$("$python_bin" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$target")"
  mkdir -p "$(dirname "$marker")"
  marker_tmp="$(mktemp "${marker}.tmp.XXXXXX")"
  printf 'lcls2 build_psana managed root v1\n%s\n%s\n' \
    "$kind" "$resolved_target" >"$marker_tmp"
  chmod 0644 "$marker_tmp"
  mv "$marker_tmp" "$marker"
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
    --perlmutter-setup)
      require_value "$@"
      perlmutter_setup="$2"
      shift 2
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

python_request="${PYTHON:-python3}"
command -v "$python_request" >/dev/null 2>&1 || \
  die "Python interpreter '$python_request' was not found. Set PYTHON to override."
python_bin="$("$python_request" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
command -v meson >/dev/null 2>&1 || die "meson was not found in PATH."
command -v ninja >/dev/null 2>&1 || die "ninja was not found in PATH."
"$python_bin" -m pip --version >/dev/null 2>&1 || \
  die "pip was not found for $python_bin."

install_prefix="$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$install_prefix")"
build_dir="$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$build_dir")"
install_marker="${install_prefix}.build_psana-install-root"
build_marker="${build_dir}.build_psana-build-root"
install_needs_marker=0
build_needs_marker=0
[[ -e "$install_prefix" || -L "$install_prefix" ]] || install_needs_marker=1
[[ -e "$build_dir" || -L "$build_dir" ]] || build_needs_marker=1
if [[ -n "$perlmutter_setup" ]]; then
  perlmutter_setup="$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$perlmutter_setup")"
  [[ ! -d "$perlmutter_setup" ]] || \
    die "Perlmutter setup path is a directory: $perlmutter_setup"
fi

conda_prefix="${CONDA_PREFIX:-}"
if [[ -n "$perlmutter_setup" && -z "$conda_prefix" ]]; then
  die "--perlmutter-setup requires an active Conda environment."
fi

if [[ "$clean_first" -eq 1 ]]; then
  validate_clean_targets
  log "Cleaning previous build outputs"
  rm -rf "$install_prefix" "$build_dir"
  install_needs_marker=1
  build_needs_marker=1
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
if [[ "$install_needs_marker" -eq 1 ]]; then
  write_managed_marker "$install_marker" install "$install_prefix"
fi
if [[ "$build_needs_marker" -eq 1 ]]; then
  write_managed_marker "$build_marker" build "$build_dir"
fi

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
# environment (RapidJSON in the active build environment).
python_include="$("$python_bin" -c \
  'import sysconfig; print(sysconfig.get_path("include") or "")')"
[[ -n "$python_include" && -f "$python_include/Python.h" ]] || \
  die "Python headers were not found for $python_bin."
build_cpath="$python_include${CPATH:+:$CPATH}"
if [[ -n "$conda_prefix" ]]; then
  build_cpath="$conda_prefix/include${build_cpath:+:$build_cpath}"
fi
python_dir="$(dirname "$python_bin")"
build_path="$python_dir${PATH:+:$PATH}"
restore_linker_env=0
masked_path_root=""

mask_nvcc_from_path() {
  local entry entry_dir item item_name masked_entry path_index=0
  local filtered_path=""

  masked_path_root="$(mktemp -d "${TMPDIR:-/tmp}/build-psana-path.XXXXXX")"
  while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue
    entry_dir="$(cd -P "$entry" 2>/dev/null && pwd)" || entry_dir="$entry"
    masked_entry="$entry_dir"

    if [[ -x "$entry_dir/nvcc" ]]; then
      masked_entry="$masked_path_root/$path_index"
      mkdir -p "$masked_entry"
      for item in "$entry_dir"/*; do
        [[ -e "$item" || -L "$item" ]] || continue
        item_name="$(basename "$item")"
        [[ "$item_name" == nvcc ]] || ln -s "$item" "$masked_entry/$item_name"
      done
    fi

    if [[ -z "$filtered_path" ]]; then
      filtered_path="$masked_entry"
    else
      filtered_path="$filtered_path:$masked_entry"
    fi
    path_index=$((path_index + 1))
  done < <(printf '%s' "$build_path" | tr ':' '\n')

  build_path="$filtered_path"
}

run_with_build_env() {
  if [[ -n "$build_cpath" ]]; then
    PATH="$build_path" CPATH="$build_cpath" "$@"
  else
    PATH="$build_path" env -u CPATH "$@"
  fi
}

meson_python="$(PATH="$build_path" command -v python3 || true)"
[[ -n "$meson_python" ]] || die "python3 was not found in the build PATH."
meson_python="$("$meson_python" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
[[ "$meson_python" == "$python_bin" ]] || \
  die "PYTHON resolves to $python_bin, but Meson would use $meson_python."

cxx_bin="${CXX:-c++}"
command -v "$cxx_bin" >/dev/null 2>&1 || \
  die "C++ compiler '$cxx_bin' was not found. Set CXX to override."
printf '#include <rapidjson/document.h>\n' | \
  run_with_build_env "$cxx_bin" -E -x c++ - >/dev/null 2>&1 || \
  die "RapidJSON headers were not found by $cxx_bin. Add them to CPATH or the active environment."

cuda_compiler=""
if [[ "$with_cuda" -eq 1 ]]; then
  if [[ -n "${CUDA_ROOT:-}" ]]; then
    [[ -d "$CUDA_ROOT" ]] || die "CUDA_ROOT is not a directory: $CUDA_ROOT"
    cuda_root="$(cd -P "$CUDA_ROOT" && pwd)"
    [[ -x "$cuda_root/bin/nvcc" ]] || \
      die "CUDA_ROOT does not contain an executable bin/nvcc: $cuda_root"
    build_path="$cuda_root/bin:$build_path"
    meson_options+=("-Dcustom_cuda_path=$cuda_root")
  fi
  cuda_compiler="$(PATH="$build_path" command -v nvcc || true)"
  [[ -n "$cuda_compiler" ]] || \
    die "--with-cuda requires nvcc in PATH or under CUDA_ROOT/bin."
  restore_linker_env=1
  export BUILD_PSANA_OLD_LDFLAGS="${LDFLAGS:-}"
  export BUILD_PSANA_OLD_CXXFLAGS="${CXXFLAGS:-}"
  export LDFLAGS=""
  export CXXFLAGS=""
elif PATH="$build_path" command -v nvcc >/dev/null 2>&1; then
  mask_nvcc_from_path
fi

cleanup_linker_env() {
  if [[ "$restore_linker_env" -eq 1 ]]; then
    export LDFLAGS="$BUILD_PSANA_OLD_LDFLAGS"
    export CXXFLAGS="$BUILD_PSANA_OLD_CXXFLAGS"
    unset BUILD_PSANA_OLD_LDFLAGS BUILD_PSANA_OLD_CXXFLAGS
  fi
  if [[ -n "$masked_path_root" && -d "$masked_path_root" ]]; then
    rm -rf -- "$masked_path_root"
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
log "CUDA discovery   : $with_cuda${cuda_compiler:+ ($cuda_compiler)}"
if [[ -n "$perlmutter_setup" ]]; then
  log "Perlmutter setup: $perlmutter_setup"
fi

if [[ "$python_only" -eq 0 ]]; then
  if [[ -d "$build_dir/meson-info" ]]; then
    log "Reconfiguring Meson build"
    run_with_build_env \
      meson setup --reconfigure "$build_dir" "$repo_dir" "${meson_options[@]}"
  else
    log "Configuring Meson build"
    run_with_build_env meson setup "$build_dir" "$repo_dir" "${meson_options[@]}"
  fi

  log "Compiling Meson targets"
  run_with_build_env meson compile -C "$build_dir" -j "$jobs"
else
  log "Skipping Meson configure and compile (--python-only)"
fi

log "Installing Meson targets"
run_with_build_env \
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

# Meson has already installed the psana and psalg package trees. This private
# wheel is intentionally limited to metadata, entry points, and native library
# artifacts, so disable Hatchling's source-tree package collection explicitly.
"$python_bin" - "$package_dir/pyproject.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
declaration = 'packages = ["psana", "psalg"]'
if text.count(declaration) != 1:
    raise SystemExit(f"Expected exactly one Hatch package declaration in {path}")
path.write_text(text.replace(declaration, "packages = []"), encoding="utf-8")
PY

shopt -s nullglob
installed_libraries=("$install_prefix"/lib/*.so*)
shopt -u nullglob
[[ ${#installed_libraries[@]} -gt 0 ]] || \
  die "No shared libraries were installed under $install_prefix/lib."
for library in "${installed_libraries[@]}"; do
  ln -s "$library" "$package_dir/install/lib/$(basename "$library")"
done

log "Installing Python metadata and command-line entry points"
run_with_build_env "$python_bin" -m pip install "$package_dir" \
    --no-compile \
    --no-deps \
    --no-build-isolation \
    --prefix="$install_prefix"

activation_file="$install_prefix/activate.sh"
log "Writing runtime activation helper: $activation_file"
printf -v install_bin_quoted '%q' "$install_prefix/bin"
printf -v site_packages_quoted '%q' "$site_packages_dir"
printf -v install_lib_quoted '%q' "$install_prefix/lib"
cat >"$activation_file" <<EOF
# Source this file after activating the Python environment used to build psana.
export PATH=$install_bin_quoted\${PATH:+:\$PATH}
export PYTHONPATH=$site_packages_quoted\${PYTHONPATH:+:\$PYTHONPATH}
export LD_LIBRARY_PATH=$install_lib_quoted\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
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

if [[ -n "$perlmutter_setup" ]]; then
  [[ "$perlmutter_setup" != "$activation_file" ]] || \
    die "Perlmutter setup path must differ from $activation_file"

  perlmutter_setup_dir="$(dirname "$perlmutter_setup")"
  mkdir -p "$perlmutter_setup_dir"
  perlmutter_setup_tmp="$(mktemp "${perlmutter_setup}.tmp.XXXXXX")"
  printf -v conda_prefix_quoted '%q' "$conda_prefix"
  printf -v activation_file_quoted '%q' "$activation_file"
  printf -v install_prefix_quoted '%q' "$install_prefix"

  log "Writing Perlmutter runtime setup: $perlmutter_setup"
  cat >"$perlmutter_setup_tmp" <<EOF
#!/usr/bin/env bash
# Generated by build_psana.sh. Source this file from a Perlmutter bash shell.
if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source \${BASH_SOURCE[0]}" >&2
  exit 1
fi

module load conda/Miniforge3-24.11.3-0 gcc-native/13.2 || return 1
conda activate $conda_prefix_quoted || return 1
source $activation_file_quoted || return 1

export PSANA_INSTALL=$install_prefix_quoted
unset SIT_PSDM_OFFSITE
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
export MPICH_GPU_SUPPORT_ENABLED=0
EOF
  chmod 0755 "$perlmutter_setup_tmp"
  mv "$perlmutter_setup_tmp" "$perlmutter_setup"
fi

log "Build completed and verified."
cat <<EOF

Activate this install in the current shell with:

  source "$activation_file"

Then, for example:

  python -c "import psana; print(psana.__file__)"
  detnames <xtc2-file>
EOF

if [[ -n "$perlmutter_setup" ]]; then
  cat <<EOF

On Perlmutter, activate the build and its Conda environment with:

  source "$perlmutter_setup"
EOF
fi
