# Prepare campaign runtimes

`prepare_runtime.py` supplies reusable `build`, `capture`, and `check-pair`
commands. It produces runtime artifacts for [the native adapter](NATIVE.md),
not a qualified campaign. The agent chooses exact revision SHAs and reviews
source API compatibility; the adapter contract remains PR298 (`199575205`).
No environment package installation, dataset access, benchmark submission or
qualification-flag change occurs. Never execute the internal `_probe` directly.

## 1. Prepare one shared dependency environment

Use an existing reviewed Python 3.9+ environment with the psana Meson build
dependencies, MPICH-compatible mpi4py, CuPy and KvikIO 26.10. Activate its build
environment on the login node. Set compiler/module/CUDA paths explicitly. Both
revisions must use this same dependency environment; do not install or upgrade
packages between builds/captures/samples.

Write a JSON object containing exactly these six keys, with their real site
values: `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `CONDA_PREFIX`, and
`LCLS_CALIB_HTTP`. PATH/library entries must be absolute with no empty entries.
Use the Python environment and site/CUDA libraries, without another psana
installation's bin/lib paths. Do not source an existing psana activate.sh.

For example, after site activation and setting the six environment variables:

```bash
python - <<'PY'
import json, os
with open('/absolute/new/environment.json', 'x') as out:
    json.dump({k: os.environ[k] for k in (
        'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
        'CONDA_PREFIX', 'LCLS_CALIB_HTTP')}, out, indent=2)
PY
```

Keep this file and compiler/module setup with the campaign. It contains selected
runtime paths, not a dump of the full process environment or credentials.

## 2. Build each exact revision on login

```bash
python -B prepare_runtime.py build \
  --repo /absolute/lcls2-repository --commit FULL_40_CHARACTER_SHA \
  --root /absolute/new/baseline --python /absolute/dependency-env/bin/python \
  --environment /absolute/new/environment.json --jobs 8
```

Repeat with the candidate SHA and a different new root. For initial A/A, build
once and use the exact same captured manifest for both sides. The root's parent
must already exist. The driver creates a detached worktree at `root/source`,
builds with that revision's `build_psana.sh` using explicit `root/build` and
`root/install`, and checks installed imports. It records source tree/commit,
build-script/tool hashes, command, selected compiler environment, Meson metadata
hashes, build log, and every installed file except bytecode caches. Uninitialized
DAQ submodules are recorded but not fetched; analysis submodules are rejected.
The selected revision must support the documented Meson build arguments.

`build-complete.json` is written only after success; its printed SHA256 is an
input to capture. `build-failed.json` and logs retain failures. There is no clean,
overwrite or resume mode: investigate a failure and choose a fresh directory.
An unrelated dirty original checkout is not used as build input. Source worktree
changes and installed-file changes after the build are rejected at capture.

## 3. Capture on an allocated GPU node

Freeze the reviewed driver and `native_contract.py` beside each other before
launch. Retain their hashes, the build receipt/hash, and launcher. In an existing
Perlmutter allocation, use one MPI rank and exactly one Slurm-visible GPU:

```bash
srun --mpi=cray_shasta --ntasks=1 --gpus-per-task=1 \
  /absolute/dependency-env/bin/python -B /absolute/frozen/prepare_runtime.py capture \
  --receipt /absolute/new/baseline/build-complete.json \
  --receipt-sha256 BUILD_RECEIPT_SHA256 \
  --output /absolute/new/baseline-runtime.json
```

The agent must request appropriate CPU/memory resources and sufficient walltime;
this command does not allocate or submit a job. Capture has a 180-second child
probe timeout by default (configurable up to 600). A fresh `.capture` directory
beside the output keeps the launch, log, CUDA cache and small probe input. Failed
attempts never publish the output manifest; use a fresh output name to retry.

Capture checks CPU imports before importing GPU modules, singleton MPICH, CUDA,
NVRTC compilation/execution and a 4 KiB KvikIO fallback read into device memory.
It hashes dependency code/extensions (including lazy-import candidates) and
observed MPI/CUDA/NVRTC/cuFile/KvikIO libraries, records versions, then exercises
the adapter's runtime verifier. It does not construct DataSource, read run51,
prove a multi-rank MPI launch, or exercise the six regression cases. Native
smoke runs must still detect any previously unobserved library or API mismatch.

New manifests declare `path_policy: isolated-install-v1`. Their six-key
`environment` is the common dependency environment. The adapter adds only the
selected install's `bin` and `lib` paths and sets its exact Python package path.
It verifies the full installed support-file inventory and rejects mixed psalg
or installed native-library origins. Legacy manifests retain previous semantics.
Do not mix path policies within a comparison.

## 4. Check the pair and hand off

```bash
python -B prepare_runtime.py check-pair \
  --baseline /absolute/new/baseline-runtime.json \
  --candidate /absolute/new/candidate-runtime.json
```

The command rejects different Python/dependency/library/version/environment
pins or different revisions sharing one install. A/A should name the same file
twice, not two separately captured manifests for the same install. Keep output
hashes and add the controller's `{ "path": "...", "sha256": "..." }` wrappers
as `baseline_runtime` and `candidate_runtime` in a new campaign profile.
`check-pair` checks manifest compatibility, not current filesystem integrity;
the adapter rechecks loaded runtime identities in native samples.

The next work remains site placement/launch qualification, six-case native
correctness and cache/byte accounting, cancellation tests, observer overhead,
component bounds, then A/A noise acceptance. Existing unit/device correctness
receipts for the chosen revisions must accompany the campaign. No successful
build or runtime probe substitutes for these receipts. Preserve executed scripts
and manifests with each campaign even as the maintained template evolves.
