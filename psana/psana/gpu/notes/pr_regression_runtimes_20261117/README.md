# Reusable runtime preparation

The maintained [runtime driver](../../scripts/performance/regression/RUNTIME.md)
now builds exact revisions into isolated prefixes, captures runtime manifests,
and checks baseline/candidate compatibility. A real build and one-rank GPU
capture passed on Perlmutter. Final independent evidence review is pending.

| Command | Agent supplies | Output |
| --- | --- | --- |
| `build` | Repository, full commit SHA, new root, absolute Python, common six-key environment | Detached source, isolated Meson build/install, log, hashed build receipt |
| `capture` | Build receipt/hash, new output, existing one-GPU compute step | Runtime manifest with installed/dependency/library hashes, versions and probe provenance |
| `check-pair` | Two driver-generated manifests (same file twice for A/A) | Compatibility result and manifest hashes; no qualification flag |

Each future campaign freezes the reviewed driver/helper, launcher, environment,
build receipts and runtime manifests. Historical campaigns keep their executed
packages unchanged. Agents should follow [RUNTIME.md](../../scripts/performance/regression/RUNTIME.md),
then the [native qualification sequence](../../scripts/performance/regression/NATIVE.md#qualification-sequence-for-the-next-agent).

## Actual validation

The driver built `1995752058bb9aaa3f16ccd86c22e84153257538` (the known adapter API
revision) into a fresh prefix with Python 3.11.14, GCC 13.2 and CUDA toolkit 12.9.
It completed in about 48 seconds and checked installed imports. This is a driver
validation revision, not a selected future PR baseline. [Build receipt summary](build-validation-01.json)
records 712 psana files and 963 total installed files.

Job **61483852** on **nid003109** finished **COMPLETED / 0:0 in 30 seconds**;
the single probe step completed in 27 seconds. [Final accounting](capture-accounting-final.txt)
records the actual full-node allocation (128 CPUs, four GPUs, 16 GiB requested
memory) and the probe step (four CPUs, one GPU). MaxRSS was reported as zero by
Slurm and is not interpreted as measured memory usage.

| Observed runtime | Value |
| --- | --- |
| Python | 3.11.14 |
| CuPy / KvikIO | 14.0.1 / 26.10.00 |
| CUDA runtime / driver API versions | 12090 / 13000 |
| Visible GPU | NVIDIA A100-SXM4-40GB; UUID in capture receipt |
| CPU import check | No GPU module or CUDA/NVRTC/cuFile mapping before GPU imports |
| GPU probes | NVRTC kernel execution; exact 4 KiB KvikIO fallback read |
| Captured coverage | 961 dependency code/extension files and six observed MPI/CUDA libraries |
| Offline verification | 71 tests PASS; build/failure/tamper/isolation/publication cases |

[Capture evidence](capture-validation-01.json) ties the resulting manifest to
reviewed frozen code, the build receipt, source/install inventories, job and step.
[Pre-submission verification](submission-verification-01.json) compares the
frozen blobs and launcher with reviewed commit `95ea0c4ed0fed477c2d42076445e48c5f315c862`.
The production driver/helper are unchanged since that review. [Round2 review](review-02.json)
passed before submission; [response](response-02.md) records subsequent checks.

Artifacts remain at:

```text
/pscratch/sd/m/monarin/psana2-gpu/pr-regression-runtimes-20261117/
  build-01/build-complete.json
  build-01/install/
  frozen-01/
  runtime-01.json
  runtime-01.json.capture/
```

The manifest SHA256 is
`6082ff8edb359bd8596cddb6e7bb070b09ddb641ec8800a26a1ace36283a5b35`.
All supporting paths/hashes are in [artifact-index.json](artifact-index.json).
Using this exact manifest for both sides passes `check-pair`; no timed A/A ran.

## Review changes and boundaries

The implementation separates the common dependency environment from each
selected install's bin/lib/package paths. The adapter pins installed support
files and rejects mixed psalg/native origins; legacy manifests keep their
existing behavior. Capture records failures, validates the child/build
association and publishes a complete manifest atomically without overwrite.
CPU import safety checks both Python modules and mapped libraries. The launcher
retains allocation/accounting evidence and embeds reviewed code hashes.

The personal activation helper initially referenced an unavailable Conda module;
no build ran in that attempt. Validation instead used available GCC/CUDA modules
and the existing unchanged dependency environment. No packages, other installs,
experiment files, frozen campaigns or qualification defaults were modified.

Coverage is deliberately explicit: the six named dependency package trees and
observed MPI/CUDA/NVRTC/cuFile/KvikIO libraries, not every transitive Python or OS
library. Arbitrary inherited environment values are hashed, not logged; site
qualification must still decide which inherited settings are allowed. Atomic
publication does not claim crash-durable storage across node/filesystem failures.

The runtime probe is not a six-case native workload check, multi-rank MPI test,
cache qualification, dataset numerical check or performance result. Next comes
allocation/placement preparation and native smoke/qualification, followed by
instrumentation/component validation and timed A/A. Default profile flags stay
false. Reuse the prepared private run51 inputs; no additional copy is needed.
