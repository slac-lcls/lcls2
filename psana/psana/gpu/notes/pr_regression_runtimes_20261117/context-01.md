# Runtime-driver independent review, round1

Review exact50eabba86f550b14fa7b1d38b1b225265ab96a3a, base66da2d989.
Public slac-lcls/lcls2 repository; read scoped code and compact non-sensitive
metadata only. No credentials, raw detector data, shell, delegation or nested
review loop. User requests reusable runtime drivers with independent review.

Read psana/psana/gpu/notes/pr_regression_runtimes_20261117/{scope.md,
validation-01.json,build-validation-01.json,capture-01.sbatch,status.json};
scripts/performance/regression/{prepare_runtime.py,test_prepare_runtime.py,
native_contract.py,RUNTIME.md,NATIVE.md}, existing test_native.py and worker as
needed. Skills: exact source/install/dependency provenance, no mixed runtime,
CPU-only import safety, GPU actions only on compute; preserve archived campaigns.

Driver build creates new detached source and explicit isolated Meson build/install
roots, rejects overwrite, records exact SHA/tree/build script/command/environment/
install hashes, completion last, failed roots retained. Real build1995752058bb9aaa3f16ccd86c22e84153257538
passed in build-validation-01.json. This validates driver, not a chosen PR baseline.
Personal Conda helper initially failed before build because its module was absent;
recovered using available GCC13.2/CUDA12.9 and the existing unchanged Python3.11
KvikIO26.10 environment. Build used no package installations or source changes.
66 offline tests PASS, including real tiny detached-build fixtures, tamper/extra
file/dirty-source/no-overwrite/failure/pair/timeout and support-library rejection.

Capture is implemented but not yet executed on hardware. It rechecks receipt,
source/install, launches the pinned Python in sanitized environment, singleton
MPICH plus CPU import check before CUDA, tiny NVRTC kernel and4KiB KvikIO fallback
read, hashes dependencies (whole package code/extensions for lazy imports) and
observed native libraries, calls adapter runtime verifier, final output only on
success. It is not six-case/dataset/throughput/native qualification. No flags change.
New policy isolated-install-v1 keeps six shared environment keys identical across
builds and prepends selected install/bin/lib at launch; full install_files pins
support files and mixed psalg/native origins. Legacy behavior retained. Check this
integration for correctness and whether it changes the existing adapter contract
safely; especially baseline/candidate isolation and provenance coverage.

Review bounded capture launcher: premium lcls_g, one node, request4CPU/1GPU/16G,
10min, srun one rank one visibleGPU, 180sec child timeout. Site may allocate a full
GPU node despite smaller request; actual allocation must be retained. No compute
job submitted yet; no production data touched. Review safety/failure semantics
and practical runtime compatibility; no invented pass for GPU execution. Default
three rounds; stable findings IDs RUN-001 etc and explicit blocking distinction.
