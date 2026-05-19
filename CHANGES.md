# psana Pip Wheel - Changes Summary

## What Was Built

Two pip-installable wheels for psana:
```
dist/psana-4.3-cp311-cp311-linux_x86_64.whl  (13 MB) - Python 3.11
dist/psana-4.3-cp313-cp313-linux_x86_64.whl  (13 MB) - Python 3.13
```

## Why This Was Done

**Goal:** Enable analysis users to install psana with a simple `pip install`, without needing:
- Source checkout
- Meson build
- `setup_env.sh` 
- Manual `PYTHONPATH` configuration

**Target users:** Analysis users on Rocky 9 with Python 3.11+

**Not affected:** DAQ/dev workflows (continue using Meson/source installs)

---

## What Changed in the Repository

### 1. Modified: `pyproject.toml`

**Before:**
```toml
[project]
name = "lcls2"                    # Wrong - package doesn't exist
requires-python = ">=3.8"

[tool.hatch.build.targets.wheel]
packages = ["lcls2"]              # Wrong - package doesn't exist
```

**After:**
```toml
[project]
name = "psana"                    # Correct package name
requires-python = ">=3.11"        # Source code minimum version

[tool.hatch.build.sources]
# Dynamic: build script fills in pythonX.Y for each build
"install/lib" = "lib"
"install/bin" = "bin"

[tool.hatch.build.targets.wheel]
packages = ["psana", "psalg"]     # Actual packages (disabled during build)

[tool.hatch.build.targets.wheel.force-include]
# Bundle all shared libraries in the wheel
"install/lib/libxtc.so" = "psana/.libs/libxtc.so"
"install/lib/libpsalg.so" = "psana/.libs/libpsalg.so"
# ... 7 more libraries
```

**Why:**
- Fixed package name from non-existent "lcls2" to actual "psana"
- Added configuration to bundle shared libraries (libxtc.so, libpsalg.so, etc.)
- Set minimum Python version to match source code compatibility (3.11+)
- Made source paths dynamic so same config works for any Python version

### 2. New: `build_wheel.sh` (159 lines)

**What it does:**
1. Detects Python version from active conda environment
2. Runs Meson build (compiles C++/Cython extensions)
3. Temporarily updates `pyproject.toml` with correct Python version paths
4. Builds wheel with Hatchling
5. Renames wheel with correct platform tags (cp311/cp313)
6. Restores original `pyproject.toml`

**Why:**
- Automates the entire build process
- Keeps repository version-agnostic (no hardcoded Python version)
- Makes building wheels for new Python versions trivial (just activate different env)

**Usage:**
```bash
conda activate psana-wheel-py311
./build_wheel.sh
# Output: psana-4.3-cp311-cp311-linux_x86_64.whl
```

---

## Key Technical Decisions

### 1. Why "psana" not "lcls2"?

**Problem:** `pyproject.toml` had `name = "lcls2"` but no such package exists in the repo.

**Reality:** The actual packages are:
- `psana` (in `psana/psana/` directory)
- `psalg` (in `psalg/psalg/` directory)

**Decision:** Name the wheel "psana" since that's the primary package analysis users import.

### 2. Why include `xtcdata` libraries but not `xtcdata` package?

**Finding:** `xtcdata` is C++ only:
- Builds `libxtc.so` (shared library)
- Has NO Python package (no `__init__.py`, no `.py` files)
- Only provides C++ headers and executables

**Decision:** 
- Bundle `libxtc.so` in wheel (needed by psana extensions)
- Include xtc executables (`xtcreader`, `xtcwriter`, etc.)
- Do NOT list xtcdata as a Python package (it isn't one)

### 3. Why not include headers?

**Policy:** Treat the pip wheel as a **runtime distribution**, not a development SDK.

**Reasoning:**
- Analysis users don't compile custom C++ code against psana
- Including headers would bloat the wheel
- Developers who need headers use source/Meson installs

**Result:** Headers are already excluded in all `meson.build` files (commented out `install_headers()`)

### 4. Why Python version-specific wheels?

**Problem:** Compiled extensions are Python version-specific:
- Python 3.11 extensions: `.cpython-311-*.so`
- Python 3.13 extensions: `.cpython-313-*.so`
- These are NOT compatible with each other

**Decision:** Build separate wheels for each Python version:
- `psana-4.3-cp311-*.whl` for Python 3.11
- `psana-4.3-cp313-*.whl` for Python 3.13

**Benefit:** pip automatically installs the correct wheel for the user's Python version

### 5. Why bundle shared libraries in `psana/.libs/`?

**Problem:** Extensions like `dgram.so` depend on:
- `libxtc.so`
- `libpsalg.so`
- `libgeometry.so`
- etc.

**Options considered:**
- System libraries: Won't work (not installed on user systems)
- Separate package: Too complex for users
- Bundle in wheel: Simple, self-contained

**Decision:** Bundle all 9 lcls2-built libraries in `psana/.libs/` directory

**Future:** Could use `auditwheel` to fix RPATHs for better portability

---

## Python Version Strategy

### Repository (version-agnostic):
```toml
requires-python = ">=3.11"  # Source supports 3.11, 3.12, 3.13, etc.
```

### Build Process (version-specific):
- Build script detects active Python version
- Meson compiles with that Python
- Wheel is tagged for that specific version

### Result:
- Repository has NO hardcoded Python version
- Can build wheels for any Python ≥ 3.11
- Each wheel only works with its specific Python version

### Building for Python 3.12 (example):
```bash
conda create -n psana-wheel-py312 python=3.12 numpy meson ninja epics-base rapidjson cython
conda activate psana-wheel-py312
./build_wheel.sh
# Output: psana-4.3-cp312-cp312-linux_x86_64.whl
```

**No code changes needed!**

---

## How `src/` vs `psana/` Works

**Confusion:** Why both `psana/src/` and `psana/psana/` directories?

**Explanation:**

```
psana/                      # Build directory (NOT a package)
├── src/                    # Build-time C++ sources (3 files)
│   ├── dgram.cc           → builds dgram.so
│   ├── container.cc       → builds container.so
│   └── dgramchunk.pyx     → builds dgramchunk.so
│
└── psana/                  # ACTUAL Python package
    ├── __init__.py         # Makes this a package
    ├── datasource.py       # Pure Python
    ├── dgramedit.pyx       # Cython source (builds dgramedit.so)
    └── app/, detector/, etc.
```

**After Meson install, everything merges:**

```
site-packages/psana/
├── dgram.so              # from src/dgram.cc
├── container.so          # from src/container.cc
├── dgramchunk.so         # from src/dgramchunk.pyx
├── dgramedit.so          # from psana/dgramedit.pyx
├── __init__.py           # from psana/psana/__init__.py
├── datasource.py         # from psana/psana/datasource.py
└── app/, detector/, etc.
```

**Key insight:** The `src/` directory is just developer organization. It doesn't appear in the wheel - only the compiled `.so` files do.

---

## What's in Each Wheel

### Python Packages:
- `psana/` - Complete package with all submodules
- `psalg/` - Utilities (daqPipes, syslog)

### Compiled Extensions (18):
- dgram, container, shmem, peakFinder_ext, psalg_ext, peakfinder8
- dgramCreate, dgramchunk, dgramedit, dgramlite, eventbuilder
- parallelreader, smdreader, quadanode, hsd, ndarray
- utilsdetector_ext, constFracDiscrim

### Bundled Shared Libraries (9):
- libxtc.so, libpsalg.so, libutils.so, libgeometry.so
- libcalib.so, libdigitizer.so, libdetector.so
- libshmemcli.so, libshmemsrv.so

### Executables (8):
- xtcreader, xtcwriter, amiwriter, smdwriter, xtcupdate
- shmemClient, shmemServer, shmemWriter

### Console Scripts (65+):
- detnames, calibman, config_dump, geometry_convert
- All from `[project.scripts]` in pyproject.toml

---

## What Users Get

### Before (Meson install):
```bash
source setup_env.sh
export PYTHONPATH=/path/to/lcls2/install/lib/python3.11/site-packages
python -c "import psana"
```

### After (pip install):
```bash
pip install psana-4.3-cp311-cp311-linux_x86_64.whl
python -c "import psana"  # Just works!
```

**Benefits:**
- ✅ No `setup_env.sh`
- ✅ No `PYTHONPATH` manipulation
- ✅ No Python version conflicts
- ✅ Standard pip workflow
- ✅ Works in virtual environments

---

## Limitations & Constraints

1. **Platform:** Linux x86_64 only (no macOS, no Windows)
2. **Python version:** Each wheel requires specific Python version
3. **No test data:** Wheels don't include XTC test files
4. **Runtime only:** No headers for custom C++ development
5. **Not on PyPI yet:** Users must install from local .whl file

---

## Build Requirements

To rebuild wheels, you need:
- conda environment with Python 3.11 or 3.13
- Dependencies: numpy, meson, ninja, epics-base, rapidjson, cython ≥3.2

Build time: ~5 minutes per wheel

---

## Summary

**Changed:** 2 files (pyproject.toml + new build_wheel.sh)
**Result:** Can build pip-installable wheels for any Python ≥3.11
**Testing:** Ready to distribute to analysis users
**Future:** Build more Python versions as needed (same script works)
