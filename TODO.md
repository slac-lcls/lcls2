# TODO

## Build system

- Investigate swapping the wheel build backend from `hatchling` + `hatch_build.py`
  + the `build_wheel_container.sh`/`build_wheel.sh` meson-orchestration + toml
  packages/sources patching to `meson-python` instead. meson-python is a full
  PEP 517 backend built specifically for Meson projects (driven by SciPy's own
  needs), used with cibuildwheel by most of the scientific Python stack
  (SciPy, scikit-image). It would likely eliminate hatch_build.py entirely
  (native platform-tag inference) and the pyproject.toml packages/sources
  sed-patching dance in build_wheel.sh. Not yet evaluated against this
  project's specific layout (xtcdata C++-only, psalg standalone .so libs,
  psana as the actual package, force-include-based library bundling) --
  real migration effort, not a drop-in swap. Raised 2026-08-27.
