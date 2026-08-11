# Podman/cibuildwheel Container Build — Working Log

**Purpose of this file:** self-contained record of what's been tried, what the results were, and what's left, so that if this session is interrupted, a new session (or a person) can pick up entirely from this file plus the working tree — no other context required. Repo: `/sdf/home/m/mavaylon/mavaylon/pip_lcls2/lcls2`, branch `wheel_test`. All changes described below are **uncommitted working-tree edits**.

Fuller narrative/evidence trail (meson.build line numbers, conda-list cross-checks, etc.) also lives in the separate docs KB at `/sdf/home/m/mavaylon/stanford-opencode-docs/.../stanford-knowledge-base/docs/decisions/2026-07-29-cibuildwheel-container-fixes.md` and the corresponding `topics/psana-wheel-packaging.md` sections — but this file should be readable on its own.

---

## Current status (as of 2026-07-29, updated)

**Major update:** a real `cibuildwheel --platform linux` run has now succeeded all the way through image pull/extraction, `before-all`, `before-build`, and Meson configure, and reached actual C++/Cython compilation for the first time. The storage/`lchown` problem (see "Podman rootless-cgroup workaround" section below) appears to be resolved, not just worked around — see "BREAKTHROUGH" section near the end of this file for exactly what changed and why. Two new, real, previously-unseen problems were found during actual compilation (numpy header coverage, a RapidJSON/compiler version incompatibility) — see the same section. Neither has been fixed yet. `build_wheel.sh` (the original, working, Conda-based manual build script) remains untouched throughout all of this.

---

## Files changed, and exactly why

### 1. `meson.build` (root) — two guards

```diff
 # Have to set the linker flags manually, as nvcc's linker conflicts with
 # the gcc linker args that conda sets.
+if conda_base != ''
 add_project_link_arguments(
   [ ... cpp block, unchanged ... ],
   language: 'cpp'
 )
 add_project_link_arguments(
   [ ... cuda block, unchanged ... ],
   language: 'cuda'
 )
+endif

 epics_lib = join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))
-libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir')) + ':' + epics_lib
+if get_option('build_daq')
+  libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir')) + ':' + epics_lib
+else
+  libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
+endif
 bindir_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))
```

- **Guard 1** (`if conda_base != ''`): previously the Conda `-L`/`-Wl,-rpath` linker flags were added unconditionally, even when `conda_prefix` is empty (the container case) — producing meaningless relative-path flags instead of failing loudly. Now skipped entirely when there's no Conda env.
- **Guard 2** (`if get_option('build_daq')`): previously every shared library in the project (150+ `install_rpath:` call sites across xtcdata/psalg/psana, not just DAQ code) got an EPICS path baked into its RPATH even when `build_daq=false` (the wheel-build default). Now only DAQ builds get it.
- **No application source code was touched** — both changes are build-configuration only.
- **Locally verified**: regression-tested against the existing Conda invocation (still works, unaffected) and against an empty-`conda_prefix` invocation (clean `$ORIGIN`-relative RPATHs, no Conda/EPICS path leaked in).

### 2. New file `build_wheel_container.sh` — container-only prepare script

`build_wheel.sh` (existing, untouched) only works inside a Conda env — it exports `CXXFLAGS="-I${CONDA_PREFIX}/include"`, sets `-Dconda_prefix=...`, etc. A real manylinux container has no Conda, so a new script was needed rather than modifying the existing one. Current full contents:

```bash
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
```

### 3. `pyproject.toml` — new `[tool.cibuildwheel]` block

(Note: `git diff pyproject.toml` also shows an unrelated, pre-existing, uncommitted `[project.dependencies]`/`[project.optional-dependencies]` block from before this work started — confirmed via `git show HEAD:pyproject.toml` that it predates this session, last real commit is 2026-05-18. Not part of this effort, flagged here so it isn't misattributed.)

```toml
[tool.cibuildwheel]
build = "cp311-*"
skip = "pp* *-musllinux*"
container-engine = "podman"

[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux_2_28"
before-all = [
    "dnf install -y libcurl-devel rapidjson-devel",
]
before-build = [
    "pip install meson ninja cython hatchling",
    "bash {project}/build_wheel_container.sh",
]
```

`cp311` only for this first run. `openmpi-devel` deliberately **not** installed — confirmed via a repo-wide grep of every `meson.build` file (including `psdaq/`) that psana's C++/Cython has zero direct MPI linkage; `mpi4py` is a pure Python-level runtime dependency. `container-engine = "podman"` added so `cibuildwheel` uses Podman instead of defaulting to Docker. **Verified against the official docs** (`cibuildwheel.pypa.io/en/stable/options/`, fetched 2026-07-29): `container-engine` is confirmed to be a top-level `[tool.cibuildwheel]` option (global, Linux-only, default `docker`) — placement in this file is correct as written.

---

## Everything tried locally, in order, with results

All local testing used real Conda environments as an imperfect stand-in for the container, since Podman itself wasn't being exercised yet. Two environments were used: `psana-wheel-py311` (Python 3.11.15, Meson 1.11.1) and `psana-wheel-build` (Python 3.13.13, Meson 1.11.1) — both at `/sdf/group/lcls/ds/ana/sw/conda_bld/mavaylon/.conda/envs/`. No local Python 3.12 environment exists.

1. **Regression check** — ran `meson setup` with `-Dconda_prefix=$CONDA_PREFIX` etc., matching `build_wheel.sh`'s existing invocation, against the *modified* `meson.build`. Result: succeeded exactly as before (curl found, OpenMP found, 42 targets). Confirms Guard 1/2 don't break the existing working Conda path.
2. **Container-path-analog `meson setup`** — same, but with `-Dconda_prefix` omitted. Result: succeeded; `build.ninja` inspection confirmed every embedded RPATH was `$ORIGIN`-relative to sibling library dirs only, no Conda/EPICS path present. (Manually supplied `CFLAGS`/`CXXFLAGS` pointing at Conda's include dir for this run only, to keep Python.h discoverable — see next item for why that mattered.)
3. **First real run of `build_wheel_container.sh` as originally written** (no header-path help at all) — **failed immediately**: `meson.build:1:0: ERROR: Compiler cython cannot compile programs.` Root cause: Meson's Cython-language sanity check (runs at `project()` declaration, before any of our code) needs `Python.h` on the compiler's *default* search path, and the script sets no `CFLAGS`. (Note: a wrapper-script bug of mine — ending the test command with `echo "EXIT CODE: $?"` — briefly made this look like it exited 0; always read the actual log, not a trailing echo.)
4. **Validated the `sysconfig`-based fix in isolation** — `meson setup` with `CFLAGS="-I$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"`: succeeded cleanly for `psana-wheel-py311`.
5. **Full `meson setup` + `meson compile`, cp311, with the `sysconfig` fix** — got much further: **107 of 174 build targets compiled successfully** (including curl-dependent code, OpenMP code, Cython extensions), then failed on every RapidJSON-dependent file: `fatal error: rapidjson/document.h: No such file or directory` (from `psalg/psalg/calib/src/*.cc` and downstream `libdetector.so` files that link against `libcalib`).
6. **Confirmed RapidJSON headers do exist** in `psana-wheel-py311` (`rapidjson-1.1.0` conda package, header at `$CONDA_PREFIX/include/rapidjson/document.h`) — they're just not on gcc's default path either, same category of problem as `Python.h`, but **deliberately left unfixed** — see "RapidJSON: known unresolved risk" below for why.
7. **Same full test repeated under `psana-wheel-build` (Python 3.13.13)** — stopped at the **exact same point** (same ~107/174 targets succeeded, same RapidJSON error, nothing else different). This is good evidence the C++/Cython source itself compiles fine under 3.13 — the only obstacle on either version is the one shared, environment-level RapidJSON gap, not a version-specific incompatibility. (This directly serves task #4 in the TODO list below.)
8. **Applied the validated fix to `build_wheel_container.sh` for real** (see file contents above) and to `pyproject.toml` (`container-engine = "podman"`).
9. Cleaned up all scratch build directories used for the above (`builddir_cp311_full`, `builddir_py313_full`, etc. — none of these are part of the working tree; `git status` only shows the three intended files).

## RapidJSON: known unresolved risk, deliberately not fixed locally

`build_wheel_container.sh` still has no explicit handling for RapidJSON's header path. This is intentional, not an oversight: the failure we saw locally is specifically a "does this count as a default search path" question, and Conda's prefix layout structurally cannot answer that the same way a real system does. `dnf install rapidjson-devel` on a real `manylinux_2_28` (AlmaLinux 8) image conventionally drops headers under `/usr/include`, which **is** a default gcc search path — unlike Conda's isolated prefix. There's a real chance this "fails" locally but works fine in the actual container. Continuing to fight this locally (e.g., faking a `/usr/include`-like layout) has diminishing returns compared to just running the real container, which is the only environment that can answer this question honestly. **Watch specifically for a RapidJSON header error in the first Podman build log** — if it recurs there, the fix is either an explicit `-I` flag in `build_wheel_container.sh` or (better, longer-term) an explicit `dependency('rapidjson')` in `psalg/psalg/calib/meson.build`, which is currently undeclared there regardless.

## Other open items carried forward (not addressed by any of the above)

- `libsasl2`/`liblber`/`libldap` (three of the 30 bundled libs) have no conda-list entry — system-layer origin, unverified against the actual `manylinux_2_28` base image.
- No local Python 3.12 environment exists — 3.12 compile-compatibility is completely untested (3.11 and 3.13 both are, see above).
- If/when the build target widens past `cp311`, `meson.build`'s numpy-include lookup uses a hardcoded `'python3'` string in `run_command()` in one place, separate from the `py` object Meson auto-detects elsewhere — an inconsistency that's never actually been proven to cause a problem, but hasn't been tested across a multi-version matrix either.

---

## TODO (mirrors the task tracker — task IDs are session-local, listed here for continuity)

1. **[blocking, not yet done] Run the actual `cibuildwheel` + Podman build for `cp311`.** Command below. This is the next thing that needs to happen.
2. **[blocked on 1] Widen `build = "cp311-*"` to `"cp311-* cp312-* cp313-*"`** in `pyproject.toml`, once cp311 is proven, and verify cibuildwheel's real per-version matrix (not just local compile compatibility).
3. Done — remaining KB updates for the 2026-07-29 fixes have already been written (see decision file path at top of this file).
4. Local 3.13 compile-compatibility check — done, see item 7 above. Strong positive signal; no local 3.12 environment exists to run the equivalent check.

---

## Podman rootless-cgroup workaround (2026-07-29) — infra issue, revert when fixed

**Symptom:** `podman run --rm hello-world` failed with:
```
Error: runc: runc create failed: unable to start container process: error during container init: error mounting "cgroup" to rootfs at "/sys/fs/cgroup": stat /opt/weka/data/agent/tmpfss/cgroup/weka: permission denied: OCI permission denied
```
This is a rootless-cgroup-delegation problem specific to `sdfiana023`'s Weka-backed cgroup mount — not something introduced by any of the psana build changes above. **This should be reported to whoever owns Podman/cgroup config on these nodes** as a real infra issue; the below is a user-level workaround, not a fix.

**Manually confirmed working:** downloading a standalone `crun` binary and disabling cgroups:
```bash
curl -L -o ~/bin/crun https://github.com/containers/crun/releases/download/1.21/crun-1.21-linux-amd64
chmod +x ~/bin/crun
podman run --runtime ~/bin/crun --cgroups=disabled --cgroupns=host hello-world   # succeeded
```

**Made persistent with two changes** (both clearly marked `WORKAROUND` inline, both trivially revertible):

1. **New file `~/.config/containers/containers.conf`** (did not exist before — safe to delete entirely when reverting):
   ```toml
   [engine]
   runtime = "crun"

   [engine.runtimes]
   crun = ["/sdf/home/m/mavaylon/bin/crun"]

   [containers]
   cgroups = "disabled"
   ```
   This makes every Podman invocation (not just the one `cibuildwheel` exposes via `create-args`) use the working `crun` binary and skip cgroup creation by default.

2. **`pyproject.toml`**, `[tool.cibuildwheel]` block — changed from:
   ```toml
   container-engine = "podman"
   ```
   to:
   ```toml
   container-engine = { name = "podman", create-args = ["--cgroups=disabled"] }
   ```
   Belt-and-suspenders: explicitly passes `--cgroups=disabled` to the specific `podman create` call `cibuildwheel` makes, in case something isn't covered by `containers.conf` alone.

**To revert once the infra team fixes proper rootless cgroup delegation:**
- Delete `~/.config/containers/containers.conf` (or the whole `~/.config/containers/` directory, if nothing else has since added files there).
- In `pyproject.toml`, change `container-engine = { name = "podman", create-args = ["--cgroups=disabled"] }` back to the plain `container-engine = "podman"`.
- Optionally remove `~/bin/crun` if not needed for anything else.

**Update (reviewed by Thorsten, via boss):** confirmed the diagnosis independently — `runc` (Podman's default runtime) walks the full cgroup hierarchy when setting up a container, and that walk hits the same Weka-exposed, unreadable cgroup path; `crun` doesn't need to walk that same path, which is why swapping runtimes sidesteps it. Two refinements applied as a result:

1. Added `cgroupns = "host"` to `~/.config/containers/containers.conf` under `[containers]` (previously only `cgroups = "disabled"` was set) — matches the original manually-tested working command exactly (`podman run --runtime ~/bin/crun --cgroups=disabled --cgroupns=host hello-world`).
2. Updated `pyproject.toml`'s `container-engine` config to make the workaround explicit in `create-args` rather than relying solely on the global `containers.conf`, and added `disable-host-mount = true`:
   ```toml
   container-engine = { name = "podman", create-args = ["--cgroups=disabled", "--cgroupns=host"], disable-host-mount = true }
   ```
   `disable-host-mount` turns off cibuildwheel's default behavior of bind-mounting the entire host filesystem into the container at `/host` (visible in the earlier failed command as `--volume=/:/host`) — since most of the host filesystem is Weka-backed, this removes one more place Weka could interfere. `build_wheel_container.sh` doesn't need `/host`; it only needs the project directory, which cibuildwheel mounts separately regardless of this setting.

**Important caveat, set expectations accordingly:** neither of these changes touches the second problem (the storage/`lchown` ownership-remap error during image extraction). That failure comes from Podman's persistent image storage defaulting to living under `~/.local/share/containers/storage`, on Weka — a completely separate mechanism from the `/host` bind mount `disable-host-mount` turns off. **The `lchown` error is still expected to occur** on the next run unless the `/lscratch` storage relocation (proposed earlier, never applied) is also done.

**Confirmed by an actual `cibuildwheel --platform linux` run after these changes:** the generated `podman create` command no longer includes `--volume=/:/host` (confirms `disable-host-mount` took effect) and now includes both `--cgroups=disabled` and `--cgroupns=host` (confirms `create-args` took effect). No cgroup-related error occurred this run. **The run still failed at the exact same storage step as before** — same error text, same blob hash, same file (`/usr/bin/write`):
```
Error: writing blob: adding layer with blob "sha256:a0ca16fa6f59daa991d1f4bac1474eee6feecd5a9d67c587c38ec0b4bd0f853f": Error processing tar file(exit status 1): potentially insufficient UIDs or GIDs available in user namespace (requested 0:5 for /usr/bin/write): Check /etc/subuid and /etc/subgid if configured locally and run podman-system-migrate: lchown /usr/bin/write: invalid argument
```
This is strong confirmation that the two problems are genuinely independent: fixing cgroups/host-mount had zero effect on the storage issue, exactly as predicted, since they're different Podman subsystems (OCI runtime vs. image storage). **The `/lscratch` storage relocation is now the only remaining blocker to getting past image extraction entirely.**

**Update:** `create-args` also now explicitly includes `--runtime /sdf/home/m/mavaylon/bin/crun`, redundant with `containers.conf` (already confirmed working) but makes the config self-contained. Flagged inline in `pyproject.toml` as account-specific — the absolute path only exists under this user's home directory, so anyone else running this needs their own `crun` and either their own `containers.conf` or an adjusted path here.

## Exact command to run (Podman, on this node)

Run this from the repo root (`/sdf/home/m/mavaylon/mavaylon/pip_lcls2/lcls2`). No Conda environment needs to be active for this — `cibuildwheel` manages its own containers.

```bash
cd /sdf/home/m/mavaylon/mavaylon/pip_lcls2/lcls2

# 1. Confirm Podman itself works on this node
podman --version
podman run --rm hello-world   # optional smoke test that Podman can actually pull/run an image

# 2. Install cibuildwheel (needs some Python + pip on the host; any recent
#    Python works here, it's just orchestrating containers, not compiling)
pip install --user cibuildwheel

# 3. Run it. pyproject.toml's [tool.cibuildwheel] already specifies:
#    build = "cp311-*", container-engine = "podman", the manylinux_2_28 image,
#    and the before-all/before-build steps described above.
cibuildwheel --platform linux 2>&1 | tee cibuildwheel_run_$(date +%Y%m%d_%H%M%S).log
```

**What to watch for in the output, in order:**
1. Does Podman successfully pull `manylinux_2_28_x86_64` and does `before-all` (`dnf install libcurl-devel rapidjson-devel`) succeed?
2. Does `before-build` succeed — specifically, does `build_wheel_container.sh` get through `meson setup`/`compile`/`install` without the Cython/Python.h error (should be fixed now) and **does it hit the RapidJSON error** (open question, see above)?
3. Does the hatchling build step succeed against the patched `pyproject.toml`?
4. Does `auditwheel repair` run automatically and produce a `manylinux_2_28` tagged wheel in `./wheelhouse/`?

The resulting log file (`cibuildwheel_run_*.log`) plus this file is everything a next session needs to diagnose whatever happens. Paste the tail of that log back, or point a new session at both files, either way.

---

## BREAKTHROUGH (2026-07-29): storage problem appears resolved, real compile errors reached for the first time

### What happened

After the `containers.conf`/`create-args`/`disable-host-mount` refinements above (still failed at the same storage/`lchown` step, confirmed), the user ran:

```bash
podman system migrate --new-runtime=crun
```

then reran `cibuildwheel --platform linux`. This time, **the entire image pull and extraction succeeded** — every blob copied cleanly, ending in "Writing manifest to image destination" / "Storing signatures," with **no `lchown` error at all**. This is the exact step that failed identically in every previous attempt. `podman system migrate` appears to reset/reinitialize internal storage state in a way that fixed the underlying extraction problem, not just worked around it — this may mean the `/lscratch` storage relocation (proposed earlier, never applied) is no longer necessary, though this hasn't been stress-tested (only one successful run so far).

### Everything that succeeded after that (all first-time confirmations in a real container)

- `before-all` (`dnf install -y libcurl-devel rapidjson-devel`) — clean install, no errors.
- `before-build`'s `pip install meson ninja cython hatchling` — succeeded.
- `build_wheel_container.sh`'s `sysconfig`-based Python.h fix — **confirmed working in a real manylinux container**, not just the local Conda approximation: `Python include path (sysconfig): /opt/python/cp311-cp311/include/python3.11`.
- Meson configure — fully succeeded: `Library curl found: YES`, `Run-time dependency OpenMP found: YES 4.5`, `Build targets in project: 42`. Matches local predictions exactly.
- RapidJSON header discovery — **confirmed found automatically** at `/usr/include/rapidjson/document.h`, with no explicit `-I` flag needed, validating the earlier "test as-is, don't pre-emptively fix" decision from the original dependency audit.
- Meson compile started and got to **107+ of 174 targets** before `ninja` stopped due to failures — this is real compilation in a real container, further than any test this session has reached.

### New problem A: numpy headers not reaching most targets

Errors seen:
```
/project/psana/psana/peakFinder/peakFinder_ext.pyx:18:8: 'numpy.pxd' not found
```
```
../psana/src/container.cc:5:10: fatal error: numpy/arrayobject.h: No such file or directory
```

**Root cause, diagnosed:** the old Conda-based `build_wheel.sh` sets a blanket `CXXFLAGS="-I${CONDA_PREFIX}/include"`, and Conda's numpy package happens to install its headers into that same shared, system-style include directory — so every compilation unit got numpy's headers *by accident*, regardless of whether its `meson.build` target actually declared `numpy_dep`. `build_wheel_container.sh` deliberately does not do that blanket injection (that's what fixed the Python.h and RapidJSON discovery concerns). The side effect: any target that doesn't explicitly list `numpy_dep` in its `dependencies:` no longer gets numpy's include path at all. Confirmed by inspecting the actual failing compile commands — several have no `-I` for numpy anywhere in them.

**Proposed fix (not yet applied):** audit `psana/meson.build` and the `psalg/*/meson.build` files for every target using `#include <numpy/arrayobject.h>` or `cimport numpy`, and ensure `numpy_dep` is listed in that target's `dependencies:`. This is a build-configuration-only fix (no `.cc`/`.pyx` source changes) consistent with everything else done in this effort.

### New problem B: RapidJSON version/compiler incompatibility (new, not the discovery gap we expected)

Error seen, repeated across every file that includes `MDBWebUtils.hh` (which pulls in RapidJSON):
```
/usr/include/rapidjson/document.h: In member function 'rapidjson::GenericStringRef<CharType>& rapidjson::GenericStringRef<CharType>::operator=(const rapidjson::GenericStringRef<CharType>&)':
/usr/include/rapidjson/document.h:319:82: error: assignment of read-only member 'rapidjson::GenericStringRef<CharType>::length'
```

This is **not** the header-discovery risk that was flagged earlier (the header was found fine, at the expected default path). This is a genuine compile error inside RapidJSON's own header: `length` is declared `const`, and this old version's assignment operator tries to assign to it anyway. AlmaLinux 8's `rapidjson-devel` ships RapidJSON 1.1.0 (tagged 2016) — a version old enough to predate stricter const-correctness enforcement in the GCC 14.2.1 used by this manylinux image. **Not yet researched or fixed** — needs investigation into whether a newer RapidJSON build/package is available, or a minimal patch is the accepted community fix, before touching anything.

### Status / next steps

Neither new problem has been fixed yet, by explicit instruction (log first, fix later). Task list updated: problem A (numpy_dep) and problem B (RapidJSON/compiler incompatibility) tracked as new, separate tasks. Original task #1 ("run first cibuildwheel/Podman test for cp311") is substantially achieved, a real, far-reaching container build run happened, even though the wheel isn't complete yet — remaining work is now these two new, real, previously-unseen problems.
