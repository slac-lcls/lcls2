# How to run the Podman/cibuildwheel container build

Quick-reference for actually running this. Full background, diagnosis, and everything tried is in `PODMAN_TEST_LOG.md` — this file is just the "what to type" version.

## What actually runs this

It is **not** just running `build_wheel_container.sh` directly. That script is one internal step that gets called automatically, inside the container, by `cibuildwheel`. `cibuildwheel` is the actual command you run — it pulls the manylinux container image, runs `before-all` (installs `libcurl-devel`/`rapidjson-devel` via `dnf`), runs `before-build` (which calls `build_wheel_container.sh`), builds the wheel, and repairs it with `auditwheel`. All of that is driven by the `[tool.cibuildwheel]` config already in `pyproject.toml`.

## Reproducing the errors (intentional — do not apply the storage fix)

This file deliberately does **not** include the `/lscratch` storage fix. If the goal is to reproduce what was actually seen, leave it that way. Which error(s) show up depends entirely on which account runs this:

- **Running as `mavaylon`** (the account this was diagnosed on): the cgroups workaround is already in place for this account, so `podman run --rm hello-world` will succeed, and `cibuildwheel` will get further before failing on the **second** error (image extraction / ownership remap on Weka).
- **Running as a different account with no workaround applied**: `podman run --rm hello-world` should fail immediately with the **first** error (cgroup mount permission denied) — no setup needed to see this one, it's the default state for any account on this node right now.
- **To see both errors in sequence as a different user**: apply just the `containers.conf`/`crun` workaround for that account (see below) to get past error 1, then run `cibuildwheel` to hit error 2. Do not add the storage relocation — that's the one piece intentionally left broken for reproduction.

## Prerequisite: the Podman workaround (account-specific, not yet global)

This node (`sdfiana023`) has a known infrastructure issue: rootless Podman fails against Weka-backed storage in two separate ways (cgroups, and container image extraction). Full explanation in `PODMAN_TEST_LOG.md`. The fix so far is **two user-account-level config files**, currently only set up for the `mavaylon` account:

1. `~/.config/containers/containers.conf` — switches the default container runtime to `crun` and disables cgroups.
2. A `crun` binary at `~/bin/crun` (downloaded from the official `containers/crun` GitHub releases, v1.21).

**If your boss runs this under a different user account, these need to be replicated for that account first**, or the run will fail at the exact same cgroup error as the very first attempt. If running as `mavaylon` on `sdfiana023`, this is already in place.

**Update, reviewed by Thorsten (via boss):** two refinements applied to the cgroups-side workaround — `cgroupns = "host"` added to `containers.conf`, and `pyproject.toml`'s `container-engine` now explicitly sets `create-args = ["--cgroups=disabled", "--cgroupns=host"]` plus `disable-host-mount = true` (stops cibuildwheel from bind-mounting the entire Weka-backed host filesystem into the container at `/host`, which isn't needed for this build). See `PODMAN_TEST_LOG.md` for the full writeup.

**Status of the second workaround (storage relocation): STILL NOT DONE.** We diagnosed that Podman's default storage location (`~/.local/share/containers/storage`) sits on Weka and causes a second failure during image extraction, and identified `/lscratch` as the fix, but never actually created that config. **Running this right now will very likely fail again at that same storage/ownership step**, not because anything is wrong with the run itself, but because that fix is still outstanding.

## Commands to run

```bash
cd /sdf/home/m/mavaylon/mavaylon/pip_lcls2/lcls2

# Confirm Podman itself is working first
podman run --rm hello-world

# Confirm which container runtime Podman is actually using (should print "crun")
podman info --format '{{.Host.OCIRuntime.Name}}'

# Install cibuildwheel if not already installed
pip install --user cibuildwheel

# Run the actual build
cibuildwheel --platform linux 2>&1 | tee cibuildwheel_run_$(date +%Y%m%d_%H%M%S).log
```

## What to expect right now

1. `podman run --rm hello-world` should succeed (cgroups workaround is in place).
2. `cibuildwheel` will start, pull the manylinux image, and most likely fail during image extraction with an `lchown ... invalid argument` error — the known, not-yet-fixed storage issue.
3. If it gets past that (meaning the storage fix was applied before this run), the next things to watch for are whether `before-build` clears the RapidJSON header discovery step, which was never actually confirmed working in a real container, only in local Conda-based approximations.

See `PODMAN_TEST_LOG.md` for the full diagnostic trail, exact error text, and everything already ruled in or out.
