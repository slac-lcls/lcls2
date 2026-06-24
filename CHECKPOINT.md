# Checkpoint: Psana Wheel Build Status & Open Questions

**Date:** 2026-05-19  
**Status:** Manylinux wheels successfully created, but dependency list needs validation

---

## What We Accomplished

### ✅ Created Portable Manylinux Wheels
- **Wheel:** `dist/manylinux/psana-4.3-cp313-cp313-manylinux_2_28_x86_64.whl` (20 MB)
- **Method:** Manual build on S3DF using `auditwheel repair` (NOT cibuildwheel - no Docker)
- **Environment:** `/sdf/group/lcls/ds/ana/sw/conda_bld/mavaylon/.conda/envs/psana-wheel-build`
- **Portability:** Works on any Linux with glibc ≥ 2.28 (Rocky 9+, Ubuntu 20.04+)

### ✅ What Auditwheel Did
1. Bundled 30 external libraries (libcurl, libssl, libkrb5, etc.)
2. Fixed RPATH to use `$ORIGIN/../psana.libs` (relative paths)
3. Tagged wheel as `manylinux_2_28_x86_64`
4. Verified compliance with manylinux standards

### ✅ Files Modified
- `build_wheel.sh` - Added automatic auditwheel repair step
- `pyproject.toml` - Added Python runtime dependencies
- `WHEEL_BUILD_SUMMARY.md` - Full documentation

---

## 🚨 OPEN QUESTIONS FOR VALERIO

### Question 1: Dependency List Validation

**Current dependencies in `pyproject.toml`:**
```python
dependencies = [
    "numpy",
    "h5py",
    "matplotlib",
    "psutil",
    "requests",
    "pymongo",
    "mpi4py",
]
```

**What I found by analyzing the code:**
- ✅ **Used extensively:** numpy (248 imports), mpi4py (27), h5py (16), matplotlib (14)
- ✅ **Used in specific modules:** psutil (3), requests (4), pymongo (7)
- ❌ **MISSING from pyproject.toml but used in code:**
  - `scipy` - Used in psana/xtcav, psana/peakFinder (scipy.signal, scipy.linalg, scipy.interpolate)
  - `pyyaml` - Used for YAML config files
  - `PyQt5` - Used heavily in psana/graphqt (70+ imports of PyQt5.QtCore, QtWidgets, QtGui)
  - `amitypes` - Used in 21 files (but only available from lcls-ii conda channel, not PyPI)

**QUESTION:** What should be the official dependency list for the wheel?

**Options:**
1. **Add everything as required:**
   ```python
   dependencies = ["numpy", "h5py", "matplotlib", "mpi4py", "psutil", 
                   "requests", "pymongo", "scipy", "pyyaml", "PyQt5"]
   ```
   - Pro: Users get everything, no import errors
   - Con: Heavyweight install even for users who don't need GUI/analysis tools

2. **Split core vs optional:**
   ```python
   dependencies = ["numpy", "h5py", "mpi4py"]  # Core only
   
   [project.optional-dependencies]
   gui = ["PyQt5", "matplotlib"]
   analysis = ["scipy", "matplotlib"]
   database = ["pymongo", "requests", "pyyaml", "psutil"]
   all = ["PyQt5", "scipy", "matplotlib", "pymongo", "requests", "pyyaml", "psutil"]
   ```
   - Pro: Users choose what they need
   - Con: Import errors if they use features without installing extras

3. **Use existing environment file as source of truth:**
   - Extract runtime deps from `.daq_20250402.txt`
   - Question: Does that file include build-only deps we shouldn't include?

**ACTION NEEDED:** Which approach should I use? Is there an existing requirements.txt that defines runtime-only dependencies?

---

### Question 2: MPI Dependency

**Current situation:**
- mpi4py is listed as a dependency
- But MPI itself (OpenMPI/MPICH) must be installed separately by the user
- This is standard for HPC packages

**QUESTION:** Should we document this clearly? Add a note about MPI requirements?

**Current behavior when testing:**
```bash
# Without MPI installed:
pip install psana-4.3-*.whl
python -c "import psana"
# Error: RuntimeError: cannot load MPI library

# With MPI installed:
conda install openmpi  # or apt-get install libopenmpi-dev
pip install psana-4.3-*.whl
python -c "import psana"  # Works!
```

---

### Question 3: cibuildwheel vs Manual Approach

**What Valerio mentioned:** "cibuildwheel" for automated building

**What we did:** Manual `auditwheel repair` approach

**QUESTION:** Are the wheels we produced acceptable, or do you require cibuildwheel for production?

**Comparison:**
| Aspect | Our Manual Approach | cibuildwheel |
|--------|---------------------|--------------|
| Output Quality | ✅ Same manylinux wheels | ✅ Same manylinux wheels |
| Python Versions | One at a time (3.13 only) | All versions (3.9-3.13) automatically |
| Setup | ✅ Working now | Requires Docker config |
| Use Case | Testing, development | Production, PyPI release |

**My understanding:** Both produce identical quality wheels. cibuildwheel just automates multi-version builds.

**Is this correct?** Should I proceed with cibuildwheel setup, or are manual wheels sufficient for now?

---

### Question 4: amityping Dependency

**Issue:** `amityping` is imported in 21 files but:
- Only available from `lcls-ii` conda channel
- NOT available on PyPI
- Currently marked as optional: `pip install psana[ami]`

**QUESTION:** 
- Is amityping required for core functionality?
- Should we vendor it (include in the wheel)?
- Or keep it as optional and document it?

---

## Technical Details for Reference

### Build Process Summary
1. **CMake build** → Compiles C/C++ extensions, installs to `install/lib/python3.13/`
2. **Hatchling** → Packages Python + extensions into initial wheel (13 MB)
3. **Auditwheel repair** → Bundles external libs, fixes RPATH, creates manylinux wheel (20 MB)

### What's Bundled (30 libraries)
```
libbrotlicommon, libbrotlidec, libcalib, libcom_err, libcrypt, libcrypto, 
libcurl, libgeometry, libgomp, libgssapi_krb5, libidn2, libk5crypto, 
libkeyutils, libkrb5, libkrb5support, liblber, libldap, libnghttp2, 
libpcre2, libpsl, libsasl2, libselinux, libshmemcli, libshmemsrv, libssh, 
libssl, libunistring, libutils, libxtc
```

### System Requirements
- Linux with glibc ≥ 2.28 (Rocky 9+, Ubuntu 20.04+, Debian 11+)
- Python 3.13
- **MPI library** (OpenMPI or MPICH) - user must install separately

---

## Code Analysis Results

### Import Frequency in psana/
```
269 sys
248 numpy
210 logging
200 os
 70 PyQt5.QtCore
 66 PyQt5.QtWidgets
 47 PyQt5.QtGui
 27 mpi4py
 21 amitypes
 16 h5py
 14 matplotlib.pyplot
  7 pymongo
  4 scipy (various submodules)
  3 psutil
  3 requests
  2 yaml
```

### Dependencies Found in Official Environment (`.daq_20250402.txt`)
```
numpy       1.26.4
h5py        3.12.1
matplotlib  3.9.4
mpi4py      4.0.1
psutil      6.1.0
requests    2.32.3
scipy       1.13.1
pyyaml      6.0.2
amityping   1.2.0
```

**Note:** Official environment is Python 3.9, we built for Python 3.13

---

## Next Steps (Pending Your Answers)

1. **Clarify dependency list** → Update pyproject.toml with correct/complete dependencies
2. **Rebuild wheel** → With validated dependency list
3. **Test wheel** → In clean environment to verify all imports work
4. **Document MPI requirements** → Clear instructions for users
5. **Decide on cibuildwheel** → If needed for production, set up GitHub Actions workflow

---

## Testing Instructions (For Future Reference)

```bash
# Create fresh test environment
conda create -n test-psana python=3.13 -y
conda activate test-psana

# Install MPI (required)
conda install openmpi -y

# Install wheel
pip install /sdf/scratch/users/m/mavaylon/gitrepos/new_lcls2/lcls2/dist/manylinux/psana-4.3-cp313-cp313-manylinux_2_28_x86_64.whl

# Test basic import
python -c "import psana; from psana import DataSource; print('✓ Core works')"

# Test features that require scipy
python -c "from psana.xtcav import xtcavDisp; print('✓ xtcav works')"  # Requires scipy

# Test GUI tools
python -c "from psana.graphqt.ColorTable import ColorTable; print('✓ GUI works')"  # Requires PyQt5
```

---

## Questions Summary

1. **What's the correct/complete dependency list for pyproject.toml?**
2. **Should dependencies be split into core vs optional extras?**
3. **Are manual auditwheel-repaired wheels acceptable, or is cibuildwheel required?**
4. **How should we handle amityping (not on PyPI)?**
5. **Should we build for Python 3.9-3.12 as well, or just 3.13?**

---

## Files to Review

- `pyproject.toml` - Current dependency list (incomplete)
- `build_wheel.sh` - Build script with auditwheel
- `WHEEL_BUILD_SUMMARY.md` - Full technical documentation
- `.daq_20250402.txt` - Official conda environment (Python 3.9)
- `dist/manylinux/psana-4.3-cp313-cp313-manylinux_2_28_x86_64.whl` - Current wheel
