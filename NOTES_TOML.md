# pyproject.toml — Notes

Extended rationale/history for `pyproject.toml`, moved out of inline comments
for readability. Each section below corresponds to a section in the toml.

## `[project].dependencies`

**Core dependency list correction (2026-08-27):** Core dependencies were
identified via AST-based import analysis (2026-07-18), corrected 2026-08-27:
a fresh-env install test found six of these (amitypes/amityping, krtc,
prometheus_client, pymongo, pyzmq, kafka) are imported unconditionally in the
core DataSource chain, not gated behind any optional feature -- five were
misfiled as optional extras and two (krtc, kafka) weren't declared anywhere.
See the stanford-knowledge-base
`decisions/2026-08-27-wheel-dependency-classification-gaps.md`.

The original analysis excluded `psdaq/` (not packaged in the wheel, only
built with `build_daq=true`).

**`amityping` pin:** `amitypes` is the import name; `amityping` is the
distribution name, only published on the lcls-ii conda channel, not PyPI --
pulled directly from git instead. Pinned to the commit the `1.3.0` tag
resolved to (verified 2026-08-27) rather than the tag itself, matching the
RapidJSON pin's reproducibility rationale (see `[tool.cibuildwheel.linux]`
`before-all` below) -- tags are mutable and can move or be deleted upstream.

## `[tool.hatch.build.sources]` / `[tool.hatch.build.targets.wheel]`

`build_wheel_container.sh` (run by cibuildwheel's `before-build` step)
replaces the `"install/lib" = "lib"` line with the correct Python-version-
specific mapping (`"install/lib/pythonX.Y/site-packages/psana" = "psana"`,
same for `psalg`), and sets `packages = []` so hatchling uses the `sources`
mapping exclusively instead of auto-discovering the raw (uncompiled)
`psana`/`psalg` source trees at the project root.

## `[tool.cibuildwheel]`

**Build target:** cp311 only for the first Podman run (widen to cp312/cp313
once confirmed working). See `build_wheel_container.sh` and
`docs/topics/psana-wheel-packaging.md` (2026-07-29 sections, in the
stanford-knowledge-base repo) for the reasoning behind the before-all/
before-build split -- most notably that MPI is deliberately NOT installed
there, because psana's C++/Cython code has no direct MPI linkage (mpi4py is
a pure Python-level runtime dependency), confirmed by a repo-wide grep of
every `meson.build` file.

**`container-engine` workaround (2026-07-29):** `create-args` unblocks a
rootless-cgroup permission failure specific to `sdfiana023`'s Weka-backed
cgroup mount -- see `PODMAN_TEST_LOG.md`. `cgroupns=host` and
`disable-host-mount` were added per review (Thorsten, via boss), matching
the manually-confirmed working command:
```
podman run --runtime ~/bin/crun --cgroups=disabled --cgroupns=host
```
and avoiding cibuildwheel's default `--volume=/:/host` bind mount of the
entire host filesystem (which includes Weka-backed paths) into the
container. Once the node/infra team fixes proper cgroup delegation, this can
revert to the plain form: `container-engine = "podman"`.

The explicit `--runtime` path is account-specific
(`/sdf/home/m/mavaylon/bin/crun` only exists under this user's home
directory) and is redundant with `~/.config/containers/containers.conf`,
which already sets `crun` as the default runtime and was confirmed working
in the prior run. Added anyway for an explicit, self-contained config.
Anyone else running this needs their own `crun` at their own path, either
via their own `containers.conf` or by changing this path to match.

## `[tool.cibuildwheel.linux].before-all`

**RapidJSON fix:** AlmaLinux 8's `rapidjson-devel` is the stale upstream
v1.1.0 tag from 2016 -- its `GenericStringRef` assignment operator assigns
to a `const` member, which GCC 14 rejects as a hard error under C++20. No
RPM repo (including EPEL) has ever packaged anything newer, because
RapidJSON itself hasn't tagged a release since 2016. Upstream's master
branch has the fix (`operator=` declared deleted); pulled directly from
GitHub, pinned to a specific commit for reproducibility, rather than relying
on the broken dnf package.

## `[tool.cibuildwheel.linux].before-build`

**numpy install ordering:** numpy must be installed before
`build_wheel_container.sh` runs -- `meson.build`'s `numpy_inc` lookup shells
out to `python3 -c "import numpy; ..."` to find its include path, which
silently fails/returns nothing if numpy isn't already installed in this
container.
