Independent implementation review, round 1 of 3 for a USER-REVISED scope. Review exact commit 1d1ecbf603980dee48198840e6da5969f8104a62 against prior task HEAD 1e69f3e7d. Read only notes/sim10_perlmutter_baseline_20261115. No commands, edits, delegation, nested review or submission. User explicitly authorizes scoped external Claude disclosure; AUTHORIZATION.md records continuing authorization. Prior bootstrap implementation passed its own 3-round cycle; this is a new no-export revision, not an attempt to evade unresolved findings.

Latest user: omit writing files, run Expand-Calibrate-Cube, verify dimensions and cube data not garbage (zero, NaN etc). Read scope.md, PLAN.md, SUBMISSION.md. New policy deliberately drops full-bin D2H, MPI merge/export, CPU cube reference, reference-cost probe, and full-pixel acceptance callback. User-authorized tradeoff: basic content validation on each local GPU bank, not numerical-reference correctness, no globally merged cube output. Retain small logs/JSON, exact shape (100,32,512,1024) FP32 and bin counts, finite every pixel, nonzero populated bins, zero empty bins, lifecycle/drain/byte/identity/range/affinity/headroom and fixed4workers. Check code and tests, not just prose.

Inspect cube_check.py, runner.py, audit.py, test_cube_check.py, test_audit_port.py, campaign.py, plan.py, prepare.py, run.sbatch and supporting runtime. Production cube checker must never transfer cube pixels or allocate a whole host cube; chunk predicates bounded. Host tests use numpy for small content behavior and a fake device interface for inspect_cube transfer/count/device/ownership checks; they do not establish CUDA behavior. Existing vendor MPI/CPU-reference tests are historical component regressions, not this new GPU verification. All vendor accumulation algorithms remain unchanged. Native runtime remains external exact PR298, not this checkout's old psana core. Source/install pins are rechecked by prepare/runtime.

New bootstrap02 proposal:1exclusive node4GPUs70min, two ECC checks201/1003events at fixed1BD1GPUbatch50/4BD4GPUbatch200 with fourworkers;9staging100k cases (3 eachtarget+4BD1GPUbridge), nativeGPUtests andguards. Stop firstfailure; no job submitted or approved. Limits are stop budgets, not predictions. Full same-node S/E/EC/ECC campaign still separately sized after bootstrap with own repeated controls; no automatic promotion. User asked bounded command before submission. Automatic approval rejected the first invocation before launch (not a review verdict). Coordinator completed unaffected preparation of an immutable UNREVIEWED candidate bootstrap02:46 files compared to this exact commit, seven external pins checked, all files0444/dirs0555, no job-started/results/exports. Manifest SHA2564aab0475d860f78dbb3763749ddb755c5c626e33b913b3fcd323466164d2790b. Preparation rechecked727installed/86test hashes and five count2 inputs. Judge implementation plus preparation procedure; no runtime evidence exists for this revision. Explicit external disclosure approval must be resolved before invoking this review.

Prior job61373306 failed after600s in firstECC step: all native GPU tests/guards passed, export directory empty but created after eventprocessing/postloopgates, exact blocked operation unknown. No new baseline completed. Read evidence/job61373306-preliminary.json for limits. Do not claim removal of export diagnoses/fixes root cause.

Qualified host evidence:107 tests and64subtests passed, AST parse all task Python, bash syntax run.sbatch/runtime.sh, runner/stages/device imports under exact reused native prefix. No GPU/Slurm execution. Known NERSC module Lua warning nonfatal; Codex sandbox ENOENT, checks used scoped host execution. Independent generated receipt follows; source hashes allow verification against reviewed files. Do not claim reviewer executed tests.

{
  "git_head": "1d1ecbf603980dee48198840e6da5969f8104a62",
  "task_working_status": "",
  "status": "pass",
  "exitcode": 0,
  "command": [
    "/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python",
    "-m",
    "pytest",
    "-q",
    "--basetemp=/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/gpu-check-host03/pytest-tmp",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_port.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_audit_port.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_boundaries.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_mpi_cube.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_contracts.py"
  ],
  "psana": "/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/source/install_psana/lib/python3.11/site-packages/psana/__init__.py",
  "source_hashes": {
    "runtime.py": "1fcf367da91bbe3a84d427da91f56a50f53330decc95dfe2307305d432832a2a",
    "plan.py": "0bb1793b54760eb8596c2181695c39fb709ee4c596ffd8d0285c4898e328f92b",
    "check_tests.py": "47daa0449e19653e331efa5bac7923b5489e556e49f202920e1185b1dcea8901",
    "port_sources.py": "7b068ba6e61cc12187953856bd8ea2d7cee894c24ad8362fef2ee11129de1025",
    "import_sources.py": "9ed995fd32522c7e3a9a202141ad1eca1a6d83a71ddbe3f0f59c3b72cb2f2999",
    "cube_check.py": "8e2661a3a8e4c7998fcb43dd2569ad9493fae7a835a07179e718e34feab65d19",
    "check_reuse.py": "d7a254e1e88712b925c4822e01ff571359e126c6ec392b493c5ddda4c82b7265",
    "qualified_support.py": "c92d46a65d2610adca284be5589aeffd2d51e2da482be9a6d4fcf83f1a51279f",
    "bulk_accounting.py": "7be4ad8453a27c2575000e37873f57e03c9654b5289fe9cdedf216da8060f393",
    "prepare.py": "0d56ef3b0b7e1c321d76490eee48b04d8d2523ebe698d507a5242e0b506a90ff",
    "cpu_plan.py": "fa6a756b67c1e29371d195e89e08264d49268374d0b270ecbf3de5fbfce510be",
    "device_check.py": "034065fc18c2a829be9d4fefb07f0446f889496883413df97f770c7eceeb0c1b",
    "runtime.sh": "988ae2532369f109ba6548f4faf787ba621c4d5177f862bc1301c21bd1b80aa7",
    "source-origins.json": "c10368dd283b01d024c8114069ef74b310e01e69a69ee035b86f0ff2f31522c4",
    "campaign.py": "c0c6495514d9fee489578622ea0e191c8d035db4cf8703e0dcfd71eb57b4ec79",
    "test_audit_port.py": "6ee6b60e5bb77fca03c58b2da8ba81202be869a8cc367d8f45fc7a14d2be064c",
    "memory_watch.py": "4917b0894dbd964cc07397023bab4b9b681961d6b56af752342578d24094c597",
    "expected-native.json": "6bf2c4b6e5e75100c99ccba221f81426d69e039407f330bd970a7af31ff1f816",
    "support.py": "96679115bea5bf08d58fd15228d632e7ed6a2236b32e0973f6c69a20f61be188",
    "affinity_probe.py": "dd0e35a36610cfce1d488cc7f9f56c977e6aa74dfe164dd6afad6078496e3732",
    "test_port.py": "5fb07f2058c8c94d9be0e01635b44f528a27afa1a513c60a5b7e448cd84ccddc",
    "test_cube_check.py": "5665bcc7e5cd6749bd843d97a52d9546a43f4c01edbaacc2b0d6ea178a6ea4d4",
    "storage.py": "0a4c77f99ba97b4bfd20943a6e538d140129d76a90fea2d6d6d5cfca0d029070",
    "host_memory.py": "e4142f35478934c917adf7c15739e840ade9764f0722b7233f1c9e51b0927c8d",
    "task_runtime.py": "0157be63682cbcefebd608e0faf01b0d9fde69a8ef3b78eb993dc06b194b3a03",
    "run.sbatch": "5586a985bbf069559a9a234e00f262dc67f59bc5b431771ef3b3b8c2f3a71743",
    "host_checks.py": "50e5997b868bcbca2caaaf990b2f13953b1e9d78b8356145b4d3f39156784bf2",
    "clean_exec.py": "4994a609b58950d7072d4c983bda332ea65644cbf30c6e3ad6022e009340d492",
    "audit.py": "413949312b01707685bf7004ec5b0259d63f1b500b06a90fde6234348c83dc9b",
    "reference_probe.py": "4b00ff774fe8bcef65cdc1d6fd4641e88f281cc5dfd7a57c8ca368d4a0629a42",
    "test_boundaries.py": "e4539fed1f7a7f257dad76020a5b7af52f836fbe9dd0c2263acd54a2e6424968",
    "instrument.py": "aa7d6d720bf1bd1d7a8349db3dff8601820ba6d56042bc9a3d80f3cc17f23859",
    "runner.py": "207edf2f0cef5b8c747b7e9cd6d0958a06d4367a74483ec569e5d82a80f9793b",
    "status.json": "dd911e55a094ed99a352102ab742203bcbd84d1bd94d724b784c670d55333e1c",
    "vendor/sim10_compute_cube_20261115/test_mpi_cube.py": "dddfef187b69f588afcff01284261b194cf380d5e5aef98e6bf2ea1961471f92",
    "vendor/sim10_compute_cube_20261115/stages.py": "b19ec2131eeca45189af25a5e213139277527463db779ad1e7f6f6f78cad5f69",
    "vendor/sim10_compute_cube_20261115/mpi_cube.py": "15fe0d105a3fa16c0db2a0130789281568af8e3f0f0b171ea6229e38a2cec4c0",
    "vendor/sim10_compute_cube_20261115/test_contracts.py": "c683f55a4404a1605b2b7d0349cda73eacc2d6152151f8ebf1127f7954886812",
    "vendor/sim10_compute_cube_20261115/reference.py": "d161a4940c179c7d8f982318323b1d035b43078dec603703a024e392beda213e",
    "vendor/sim10_compute_cube_20261115/contracts.py": "3d1ade042cf0c0970917ab85f36174b199f66719c66e685b5b4497106e25b856",
    "vendor/sim10_compute_cube_20261115/test_reference.py": "15a8bee8ac37183ac0188739ba4af824c7a19f5d18ce39166fe23552061d18bd",
    "vendor/mfx_expand10_stage2_20261105/adapter.py": "fe6fbc73ec3d3198955342d59af4d88a2a1811af48d882fe6a3986c7b28a993c",
    "vendor/mfx_expand10_stage2_20261105/gpu_accept.py": "7ebc741c643a0b832db03308f9d4122bb962127e6ab5c5fcfbe7f6b6f2078d4d",
    "vendor/mfx_expand10_stage2_20261105/verify.py": "b391ffd32b229963c16ee50c0eff0a7f7ec7a0bca39afeec805c0b58c6a192a3",
    "vendor/mfx_expand10_stage2_20261105/fixture.py": "7e62e50669bb6a3315bbdf7be1f1ae471d76b36973f3074e1dd86806db62a8a4",
    "vendor/mfx_expand10_stage3a_20261105/cube.py": "3b3452da6d05edb8b9b3d7ad23efd4f18d210ac34ebd0849ebf5099cabdf4415"
  },
  "limitations": [
    "CPU-only; no GPU acceptance or scheduler execution. Host checks ran outside the unavailable Codex sandbox."
  ]
}

....................................................................................... [ 81%]
....................                                              [100%]
=============================== warnings summary ===============================
notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py::SeedReaderTests::test_changed_configure_and_initial_service_rejected
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/lib/python3.11/site-packages/amitypes/hsd.py:24: DeprecationWarning: mypy_extensions.TypedDict is deprecated, and will be removed in a future version. Use typing.TypedDict or typing_extensions.TypedDict instead.
    HSDSegementPeakTimes = TypedDict(

notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py::SeedReaderTests::test_changed_configure_and_initial_service_rejected
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/lib/python3.11/site-packages/amitypes/hsd.py:52: DeprecationWarning: mypy_extensions.TypedDict is deprecated, and will be removed in a future version. Use typing.TypedDict or typing_extensions.TypedDict instead.
    HSDSegmentPeaks = TypedDict(

notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py::SeedReaderTests::test_changed_configure_and_initial_service_rejected
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/lib/python3.11/site-packages/amitypes/hsd.py:78: DeprecationWarning: mypy_extensions.TypedDict is deprecated, and will be removed in a future version. Use typing.TypedDict or typing_extensions.TypedDict instead.
    HSDSegmentWaveforms = TypedDict(

notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py: 109 warnings
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/lib/python3.11/site-packages/kafka/protocol/schemas/load_json.py:14: DeprecationWarning: read_text is deprecated. Use files() instead. Refer to https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy for migration advice.
    msg_json = importlib.resources.read_text(package, msg_type + '.json')

notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py: 109 warnings
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/lib/python3.11/importlib/resources/_legacy.py:80: DeprecationWarning: open_text is deprecated. Use files() instead. Refer to https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy for migration advice.
    with open_text(package, resource, encoding, errors) as fp:

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
107 passed, 221 warnings, 64 subtests passed in 3.36s

Authorization update: User explicitly answered "Yes, let's review" to
"May I send this task’s implementation and validation evidence to external
Claude for independent review?" The earlier authorization block is resolved
by this new trusted user instruction. No compute submission is authorized.
