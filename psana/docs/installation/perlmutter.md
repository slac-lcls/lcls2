# Psana on NERSC Perlmutter

Use the shared installation for analysis, or build a private checkout with the
site's Cray MPI. This guide covers the analysis stack: xtcdata, psalg and psana.

Status: **implemented-reference**, with verification scope recorded below.
Reviewed source: `410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` on
`features/psana2-gpu`. Verified on **2026-10-15**. Relevant source:
[`build_psana.sh`](../../../build_psana.sh),
[`pyproject.toml`](../../../pyproject.toml), and
[`CalibConstants.py`](../../psana/pscalib/calib/CalibConstants.py).
An installed shared release can differ from this source revision.

## Use the shared installation

Users with access to the LCLS project directory can activate the maintained
installation from a Perlmutter Bash shell:

```bash
source /global/cfs/cdirs/lcls/users/psana2_build/setup_env_perlmutter.sh
PS_PARALLEL=none python -c 'import psana, psana.dgram; print(psana.__file__); print(psana.dgram.__file__)'
```

At verification, this selected the environment
`/global/cfs/cdirs/lcls/users/psana2_build/envs/psana-build-cray` and release
`/global/cfs/cdirs/lcls/users/psana2_build/releases/20260921-89179807e807-cray`,
using Python 3.9.20 and NumPy 1.26.4. Both imports succeeded. The setup script
selects a release, so inspect the printed paths when reproducing an analysis.
This shared release was not upgraded as part of the newer GPU environment tests.

Use a project account you are authorized to charge when requesting compute
resources. For LCLS members, the usual CPU/GPU account names are `lcls` and
`lcls_g`; verify your own associations rather than assuming access:

```bash
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
```

## Choose a development environment

Create environments, install packages and compile on a **login node**. Package
indexes may be unreachable from compute nodes. Run analysis, MPI/GPU checks and
benchmarks inside a Slurm allocation.

Start in a fresh shell before changing between installations. The dependency
environment supplies Python and build tools; the psana install prefix supplies
the code compiled from your checkout. Activate both at runtime.

### Reuse the shared build environment

For a private build using the shared Python 3.9 dependency stack:

```bash
module load conda/Miniforge3-24.11.3-0 gcc-native/13.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/cfs/cdirs/lcls/users/psana2_build/envs/psana-build-cray
```

Then continue with [Build your checkout](#build-your-checkout). Install additional
dependencies in a private environment rather than modifying this shared one.
The KvikIO 26.10 recipe below requires the newer Python/NumPy stack.

### Create a private Python 3.11 environment

The following package choices were used for the verified private builds. Choose
a new prefix on storage available to both login and compute nodes. Home or
project storage is appropriate for a maintained environment; scratch is useful
for disposable builds and experiments.

```bash
module load conda/Miniforge3-24.11.3-0 gcc-native/13.2
source "$(conda info --base)/etc/profile.d/conda.sh"

env_prefix="$HOME/.conda-envs/psana-perlmutter-311"
export CONDA_PKGS_DIRS="$HOME/.cache/psana-conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
export PYTHONNOUSERSITE=1
unset PYTHONPATH LD_PRELOAD

mamba create -y -p "$env_prefix" -c conda-forge \
    python=3.11.14 numpy=2.3.5 pip setuptools wheel \
    meson=1.9.2 ninja 'hatchling<1.32' cmake cython=3.2.2 \
    rapidjson libcurl krb5
conda activate "$env_prefix"

# Confirm the active wrapper belongs to the site's Cray MPI installation.
module -t list
command -v mpicc
mpicc -show
export MPICH_GPU_SUPPORT_ENABLED=0
MPICC="$(command -v mpicc)" python -m pip install \
    --no-cache-dir --no-binary=mpi4py mpi4py==4.1.1

python -m pip install \
    h5py matplotlib scipy requests psutil amityping lcls-krtc \
    prometheus_client pymongo pyzmq kafka-python pytest
python -m pip check
```

On the verified system, `mpicc` resolved under
`/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/bin`. Compile mpi4py with the loaded site
wrapper; a generic MPI wheel or a SLAC/Open MPI environment does not establish
compatibility with Perlmutter's Cray launcher.

### Optional GPU dependencies

Add these to the private Python 3.11 environment on a login node:

```bash
module load cudatoolkit/12.9
python -m pip install \
    cupy-cuda12x==14.0.1 \
    kvikio-cu12==26.10.0 libkvikio-cu12==26.10.0
python -m pip check
```

This installs one CuPy flavor. The tested resolution also installed CUDA 12.9
runtime/NVRTC wheels; it used no private CUDA shim or patched KvikIO library.
Verify the runtime on a GPU node using the commands below before relying on a
new resolution of the dependencies.

Save the resolved environment after installing the packages you need:

```bash
conda list --explicit > conda-explicit.txt
python -m pip freeze > requirements.lock.txt
module -t list 2> modules.txt
```

These record the Conda and pip parts separately. Retain both, plus the Cray MPI
wrapper and module versions; recreating a prefix does not replace the site's MPI.

## Build your checkout

With the chosen dependency environment active, create a checkout of the reviewed
revision. The example uses a private development branch so later edits have a
clear starting point:

```bash
git clone https://github.com/slac-lcls/lcls2.git "$HOME/lcls2-perlmutter"
cd "$HOME/lcls2-perlmutter"
git switch -c my-psana-work 410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8

unset PYTHONPATH
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export MPICH_GPU_SUPPORT_ENABLED=0

./build_psana.sh --build-type release -j 8 \
    --perlmutter-setup "$PWD/setup_env_perlmutter.sh"
source ./setup_env_perlmutter.sh
PS_PARALLEL=none python -c 'import psana, psana.dgram; print(psana.__file__); print(psana.dgram.__file__)'
```

The build uses Meson and installs into `install_psana`, with native build outputs
in `builddir_psana`. Both printed module paths must belong to this checkout's
`install_psana`. The generated `setup_env_perlmutter.sh` records the dependency
environment and install prefix, and can be sourced again in a fresh shell.
Before building another revision, record its SHA and give it its own build and
install directories.

To place outputs on scratch instead, choose a fresh build root and pass explicit
paths to the same builder:

```bash
build_root="$SCRATCH/psana-builds/$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
./build_psana.sh --build-type release -j 8 \
    --build-dir "$build_root/build" \
    --prefix "$build_root/install" \
    --perlmutter-setup "$build_root/setup_env_perlmutter.sh"
source "$build_root/setup_env_perlmutter.sh"
```

Retain the chosen path for subsequent activation and incremental builds. Scratch
outputs have the lifetime of that storage, so retain recipes and provenance in
durable storage.

For the default checkout-local paths, use `./build_psana.sh --python-only` after
compatible Python-only edits, or `./build_psana.sh -j 8` after native changes.
For custom paths, repeat the matching `--build-dir` and `--prefix` options.
A Python/NumPy ABI change requires a fresh native build. `--clean` removes build
outputs and, in the reviewed builder, accepts only checkout-default paths or
approved shared build/release subdirectories; use a new directory for other
scratch rebuilds.

`build_psana.sh` builds the analysis stack without psdaq and disables unrelated
CUDA subprojects by default. CuPy/KvikIO usage does not require `--with-cuda`.
Use that option only when you intend to compile those CUDA subprojects. A full
DAQ build with `build_all.sh` needs additional site dependencies and is outside
this verified procedure. The repository's SLAC `setup_env.sh` is not the
Perlmutter environment initializer.

## Verify MPI and GPU execution

First confirm the launcher plugin:

```bash
srun --mpi=list
```

The verified site configuration includes `cray_shasta`. Replace the account
placeholder below with your authorized CPU account, request a node, and activate
your installation in the allocation shell:

```bash
salloc --account=YOUR_CPU_ACCOUNT --constraint=cpu --qos=interactive \
    --nodes=1 --ntasks=3 --cpus-per-task=2 --time=00:10:00
source "$HOME/lcls2-perlmutter/setup_env_perlmutter.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --mpi=cray_shasta --ntasks=3 --cpu-bind=cores python -c \
    'from mpi4py import MPI; c=MPI.COMM_WORLD; assert c.size == 3; print(c.rank, MPI.Get_library_version(), flush=True)'
```

Expect all three ranks and a Cray MPICH library version. Use the shared setup
path or your scratch setup path instead when that is your chosen installation.
Exit the allocation shell when finished.

For the private GPU-enabled environment, request a GPU allocation from a login
shell. Replace the account placeholder with your authorized GPU account:

```bash
salloc --account=YOUR_GPU_ACCOUNT --constraint=gpu --qos=interactive \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 --time=00:10:00
source "$HOME/lcls2-perlmutter/setup_env_perlmutter.sh"
module load cudatoolkit/12.9
export KVIKIO_COMPAT_MODE=ON
srun --mpi=cray_shasta --ntasks=1 --gpu-bind=none python -c \
    'import cupy as cp; import kvikio, kvikio.defaults; cp.show_config(); assert cp.cuda.runtime.getDeviceCount() > 0; assert int(cp.arange(10).sum().get()) == 45; print("KvikIO:", kvikio.__version__, "CPU fallback:", kvikio.defaults.is_compat_mode_preferred())'
```

This smoke check intentionally forces KvikIO's CPU fallback mode. It does not
establish true GDS operation. For psana jobs that select GPUs within each rank,
`--gpu-bind=none` lets the application see the allocation's devices before
selection. Set rank/device placement before CUDA initialization and verify
physical GPU identities when multiple BDs share devices; do not assume the
one-process smoke check validates a multi-rank analysis topology.

## Data and calibration

Staged data are commonly found under:

```text
/pscratch/sd/p/psdatmgr/psdm/<instrument>/<experiment>/xtc
```

Use an explicit `dir=` and verify read access to both XTC2 files and their
`smalldata/` counterparts. Data availability and experiment permissions are
independent of software installation. For example, `mfx101232725`, run 442, was
available at verification. A CPU raw-data check inside an allocation is:

```bash
PS_PARALLEL=none python - <<'PY'
from psana import DataSource
ds = DataSource(exp='mfx101232725', run=442,
                dir='/pscratch/sd/p/psdatmgr/psdm/mfx/mfx101232725/xtc',
                detectors=['jungfrau'], max_events=1, skip_calib_load='all')
run = next(ds.runs())
det = run.Detector('jungfrau')
evt = next(run.events())
raw = det.raw.raw(evt)
assert raw is not None
print(evt.timestamp, raw.shape, raw.dtype)
PY
```

For calibration, set the service **base URL before importing psana**:

```bash
unset SIT_PSDM_OFFSITE
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
```

The reviewed `CalibConstants.py` appends `/calib_ws/` itself. Supplying that
suffix in `LCLS_CALIB_HTTP` duplicates the path. The generated Perlmutter setup
script already sets the base URL above. Raw-data checks with
`skip_calib_load='all'` do not test calibration retrieval; verify connectivity,
authorization and the required constants separately for calibrated workloads.

## Verification record and troubleshooting

The private build and compute-node verification on 2026-10-15 used:

| Component | Verified value |
| --- | --- |
| Python / NumPy | 3.11.14 / 2.3.5 |
| Compiler / build tools | GCC 13.2, Meson 1.9.2, Cython 3.2.2 |
| MPI | Source-built mpi4py 4.1.1; Cray MPICH 9.1.0.794 |
| GPU packages | CuPy 14.0.1; KvikIO and libkvikio 26.10.0 |
| CUDA runtime / NVRTC | 12.9 / 12.9, verified from CuPy |
| Device / driver | NVIDIA A100-SXM4-40GB / 580.178.04 |

Release builds and installed-module origin checks passed. Compute-node job
59827342 passed a six-rank MPI/device-placement probe, 120 native KvikIO
stream-cache contention trials, 120 byte-exact synthetic read trials and
36 synthetic gather/calibration device tests. CPU preparation job 59827901
verified 200 raw Jungfrau events and generated sampled image references for
the local run. These results establish the stated build and correctness checks;
they do not establish performance, real-data GPU A/B acceptance, true GDS,
or calibration-service access. The interactive commands above are user-facing
smoke checks; the recorded tests ran through batch jobs.

| Symptom | Check or recovery |
| --- | --- |
| Import resolves to an older checkout | Start a fresh shell, activate the intended environment/setup, and print both `psana.__file__` and `psana.dgram.__file__`. |
| MPI reports Open MPI or fails to launch | Check modules and `mpicc -show`; rebuild mpi4py from source in a private environment against Cray MPI, then verify with `srun --mpi=cray_shasta`. |
| RapidJSON header missing | Install `rapidjson` in the active Conda environment and verify `$CONDA_PREFIX/include/rapidjson/document.h`. |
| Packages cannot be downloaded on a compute node | Install them on a login node, then reuse the environment from the allocation. |
| CUDA runtime or NVRTC cannot load | Check the GPU allocation, `cupy.show_config()`, loaded CUDA module, and installed CuPy flavor. Remove unintended old shim paths from the activation procedure. |
| Calibration URL contains two `calib_ws` components | Set the base URL above before starting a new Python process. |

[Back to installation guides](README.md) · [Back to psana documentation](../README.md).
