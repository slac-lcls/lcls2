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

**Build target:** cp311/cp312/cp313, widened 2026-09-01 from the original
cp311-only first Podman run. `numpy_dep`'s include-path lookup in
`meson.build` was hardened at the same time (resolve via the already-detected
`py` installation object, with `check: true`, instead of a bare `'python3'`
string with no failure check) specifically because this widening means the
build now runs under three different interpreters per container invocation
for the first time -- not because a version mismatch was ever observed.
**Not yet built or verified under cp312/cp313** -- only locally reasoned
about; run `cibuildwheel --platform linux` on a Podman-configured node and
fresh-env-install-test each resulting wheel before trusting this. See
`build_wheel_container.sh` and
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

**2026-09-01: dropped the explicit `--runtime` path.** It was
account-specific (`/sdf/home/m/mavaylon/bin/crun` only exists under this
user's home directory) -- not portable in a file that's checked into git
and meant to work for anyone building this project. It was also redundant:
`~/.config/containers/containers.conf` (a separate, per-account Podman
config, not checked in) already sets `crun` as the default runtime, and
that alone was confirmed working. Anyone building locally on SDF needs
their own `containers.conf` doing that -- that's real per-account setup,
but it now lives where account-specific config belongs (each person's own
environment), not hardcoded into the shared `pyproject.toml`.

This whole `container-engine` block is for building locally on SDF via
Podman only. `.github/workflows/release.yml` overrides it back to plain
Docker via `CIBW_CONTAINER_ENGINE`, since GitHub-hosted runners have none
of SDF's Weka/rootless-Podman issues to work around.

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
