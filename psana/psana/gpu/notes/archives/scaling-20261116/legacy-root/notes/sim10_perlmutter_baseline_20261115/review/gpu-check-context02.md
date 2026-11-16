Independent review round2 of3 for revised no-export Sim10 scope. Exact commit af6ebc03b86033accc6c0af34bb6e3eb2b85f35c, prior reviewed1d1ecbf603980dee48198840e6da5969f8104a62. Read only notes/sim10_perlmutter_baseline_20261115 in this detached worktree. No shell/tests/edits/delegation/nested review or submission. Explicit user permission: "Yes, let's review" responding to "May I send this task’s implementation and validation evidence to external Claude for independent review?" AUTHORIZATION.md records it. This covers this scoped implementation and review follow-up. No compute submission authorized.

Round1 passed with two nonblocking suggestions. Read review/gpu-check-round01.json and gpu-check-response01.md. Assess closure of GC-R1-01 and GC-R1-02 plus regressions; don't assume code is correct because prior review passed. No GPU results exist. Actual runtime is reused external PR298, never the old native tree in this checkout.

GC-R1-01: Per-rank progress history now contains monotonic and elapsed timestamps after processing, postloop gates, cube checks, gather, shared-memory close. To preserve processing time, the first postloop write follows the existing end barrier and timing allreduce. A hang within those collectives remains indistinguishable from processing; PLAN.md explicitly states this boundary. The prior job61373306 stall was beyond these phases. 300s remains an unmeasured stop bound, not a performance prediction or root-cause claim.
GC-R1-02: Added device-like array_checks behavior test forbidding payload get/asnumpy/implicit conversion and asserting scalar item, bounded slices/reductions and exact all-pixel coverage. Added non-open/missing-bank/unknown/duplicate/count dtype/count shape rejections to inspect_cube tests. CPU behavior tests only, not CUDA evidence.

New immutable candidate will use bootstrap02r2; unsubmitted bootstrap02 remains unchanged. prepare/campaign/run.sbatch and proposal paths agree; 4200s budget/case matrix/fixed4workers and nativepins unchanged. Candidate will be prepared after review and independently compared to this commit, with source/external hashes/modes and no-job evidence. That mechanical preparation does not imply GPU validation. Small diagnostic receipts retained, no cube payload export/MPI merge/CPU cube reference. Full same-node S/E/EC/ECC campaign still separately sized after repeated staging; same-node controls/fourworkers remain required.

Host checks on exact clean HEAD:114 tests+64subtests pass. AST/bashes/import checks pass. Reviewer did not run these. Qualified runtime from reuse proof; known NERSC module Lua warning nonfatal; sandbox startup ENOENT, scoped host execution used. Current diff, exact test receipt and pytest output follow.

diff --git a/notes/sim10_perlmutter_baseline_20261115/campaign.py b/notes/sim10_perlmutter_baseline_20261115/campaign.py
index d0ad322e5..aee2ff5d0 100644
--- a/notes/sim10_perlmutter_baseline_20261115/campaign.py
+++ b/notes/sim10_perlmutter_baseline_20261115/campaign.py
@@ -65,7 +65,7 @@ def verify_bundle(attempt):
     return bundle
 
 def main(attempt):
-    require(attempt==TASK_ROOT/'bootstrap02' and HERE==attempt/'bundle', 'Prepared immutable attempt only')
+    require(attempt==TASK_ROOT/'bootstrap02r2' and HERE==attempt/'bundle', 'Prepared immutable attempt only')
     deadline=allocation((attempt/'slurm-job.txt').read_text())
     require(os.environ.get('SLURM_JOB_NUM_NODES')=='1', 'Single node')
     output=attempt/'results';output.mkdir()
diff --git a/notes/sim10_perlmutter_baseline_20261115/prepare.py b/notes/sim10_perlmutter_baseline_20261115/prepare.py
index 4ec517644..2d30d96be 100644
--- a/notes/sim10_perlmutter_baseline_20261115/prepare.py
+++ b/notes/sim10_perlmutter_baseline_20261115/prepare.py
@@ -9,7 +9,7 @@ from support import HERE, TASK_ROOT, ROOT, PACKAGE, require, sha, write
 from plan import cases, WALLTIME, BUDGET, OUTPUT_RESERVE
 
 def main(name):
-    require(name=='bootstrap02', 'Only the revised GPU-only bootstrap is prepared')
+    require(name=='bootstrap02r2', 'Only the revised GPU-only bootstrap is prepared')
     repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
     attempt=TASK_ROOT/name
     require(not attempt.exists(), 'Never overwrite an attempt')
diff --git a/notes/sim10_perlmutter_baseline_20261115/run.sbatch b/notes/sim10_perlmutter_baseline_20261115/run.sbatch
index 4d1a9c4eb..bb6b34d87 100644
--- a/notes/sim10_perlmutter_baseline_20261115/run.sbatch
+++ b/notes/sim10_perlmutter_baseline_20261115/run.sbatch
@@ -12,7 +12,7 @@
 #SBATCH --job-name=sim10-baseline-bootstrap
 set -euo pipefail
 attempt=${1:?prepared attempt}
-[[ "$attempt" == /pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/bootstrap02 ]]
+[[ "$attempt" == /pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/bootstrap02r2 ]]
 (set -o noclobber; date -u > "$attempt/job-started.txt")
 TZ=UTC scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
 lscpu > "$attempt/cpu.txt"
diff --git a/notes/sim10_perlmutter_baseline_20261115/runner.py b/notes/sim10_perlmutter_baseline_20261115/runner.py
index 018327db1..a50bc7b81 100644
--- a/notes/sim10_perlmutter_baseline_20261115/runner.py
+++ b/notes/sim10_perlmutter_baseline_20261115/runner.py
@@ -138,7 +138,16 @@ def execute(arguments):
         cp.cuda.Device().synchronize()
     setup_seconds = time.monotonic() - started
     progress=Path(str(arguments.output)+'.rank%d.progress.json'%rank)
-    write(progress,dict(phase='setup_complete',setup_seconds=setup_seconds))
+    progress_history = []
+
+    def mark(phase, **details):
+        now = time.monotonic()
+        progress_history.append(dict(phase=phase, monotonic_seconds=now,
+                                     elapsed_seconds=now-started, **details))
+        write(progress, dict(rank=rank, phase=phase, setup_seconds=setup_seconds,
+                             history=progress_history))
+
+    mark('setup_complete')
     host_memory['before_loop'] = snapshot()
     threads_before = thread_snapshot()
     comm.Barrier()
@@ -155,8 +164,7 @@ def execute(arguments):
         observation = dict(conditioning=conditioning, initial_cache=initial, idle_before=idle_before,
                            idle_after=idle_after, before=counters(leaves),
                            conditioning_seconds=time.monotonic()-conditioning_start)
-        write(progress,dict(phase='conditioning_complete',setup_seconds=setup_seconds,
-                            conditioning_seconds=observation['conditioning_seconds']))
+        mark('conditioning_complete', conditioning_seconds=observation['conditioning_seconds'])
     comm.Barrier()
     begin = time.perf_counter()
     delivered, errors_seen, cpu_reads, io = [], 0, [0], None
@@ -188,6 +196,8 @@ def execute(arguments):
                 delivered.append(int(event.timestamp))
     comm.Barrier()
     seconds = comm.allreduce(time.perf_counter() - begin, op=MPI.MAX)
+    # Keep diagnostic file writes outside the unchanged processing timer.
+    mark('processing_complete', processing_seconds=seconds, delivered=len(delivered))
     if rank == 0:
         observation.update(seconds=seconds, after=counters(info.get('observed_leaves', [])))
         observation['final_cache'] = cache(info['files'], Path(info['directory']), 'lustre')
@@ -217,9 +227,11 @@ def execute(arguments):
         gc.collect()
         require(not any(owner() is not None for owner in task.previous), 'Full image owner survives drain')
         task.sample('drained')
+    mark('post_loop_gates_complete')
     cube_verification = inspect_cube(task.cube, delivered) if rank >= 2 and arguments.stage == 'ECC' else None
     if cube_verification is not None:
         task.sample('cube_verified')
+    mark('cube_verification_complete', checked=cube_verification is not None)
     host_memory['after_loop'] = snapshot()
     validate_host_memory(host_memory, arguments.bds, 4, 16)
     modules = ('psana', 'psana.dgram', 'psana.eventbuilder', 'psana.smdreader', 'psana.utilsdetector_ext',
@@ -240,6 +252,7 @@ def execute(arguments):
                cuda_intervals=task.interval_results() if rank >= 2 else None,
                threads_before=threads_before, threads_after=thread_snapshot(), runtime_files=identity(modules))
     rows = comm.gather(row, root=0)
+    mark('gather_complete')
     if rank == 0:
         require(not rows[0]['timestamps'] and not rows[1]['timestamps'], 'Helper delivered events')
         merged = [timestamp for entry in rows[2:] for timestamp in entry['timestamps']]
@@ -266,6 +279,7 @@ def execute(arguments):
                       end_to_end_seconds=time.monotonic() - started)
     comm.Barrier()
     run.close_shared_memory()
+    mark('shared_memory_close_complete')
     comm.Barrier()
     handles_context.__exit__(None, None, None)
     budget_context.__exit__(None, None, None)
diff --git a/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py b/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py
index 73428619f..05cda1430 100644
--- a/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py
+++ b/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py
@@ -55,7 +55,8 @@ def test_empty_owner_valid():
     assert not any(row['nonzero'] for row in check(sums, counts))
 
 
-@pytest.mark.parametrize('corruption', [None, 'counts', 'device', 'identity', 'drain'])
+@pytest.mark.parametrize('corruption', [None, 'counts', 'device', 'identity', 'drain', 'phase', 'bank',
+                                         'unknown', 'duplicate', 'counts_dtype', 'counts_shape'])
 def test_inspection_transfers_only_counts(monkeypatch, corruption):
     import sys
     from types import SimpleNamespace
@@ -85,6 +86,12 @@ def test_inspection_transfers_only_counts(monkeypatch, corruption):
     cube.drain = drain
     if corruption == 'device': bank.sums.device.id = 1
     if corruption == 'identity': cube.seen = set()
+    if corruption == 'phase': cube.phase = 'failed'
+    if corruption == 'bank': cube.bank = None
+    if corruption == 'unknown': cube.timestamps = [456]
+    if corruption == 'counts_dtype': bank.counts.dtype = np.dtype('float64')
+    if corruption == 'counts_shape': bank.counts.shape = (99,)
+    delivered = [123, 123] if corruption == 'duplicate' else [123]
     cp = SimpleNamespace(ndarray=DeviceArray, cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0)))
     monkeypatch.setitem(sys.modules, 'cupy', cp)
     calls = []
@@ -94,10 +101,61 @@ def test_inspection_transfers_only_counts(monkeypatch, corruption):
         return [dict(bin=i, finite=True, nonzero=i==0) for i in range(100)]
     monkeypatch.setattr(cube_check, 'array_checks', checks)
     if corruption:
-        with pytest.raises(ValueError): cube_check.inspect_cube(cube, [123])
+        with pytest.raises(ValueError): cube_check.inspect_cube(cube, delivered)
         assert not calls
     else:
-        receipt = cube_check.inspect_cube(cube, [123])
+        receipt = cube_check.inspect_cube(cube, delivered)
         assert calls == [True]
         from audit import audit_cube
         audit_cube(receipt, [123], [123])
+
+
+def test_device_like_reductions_are_scalar_and_chunk_bounded(monkeypatch):
+    from types import SimpleNamespace
+    monkeypatch.setattr(cube_check, 'CHECK_ELEMENTS', 2)
+    slices, reductions, host_flags = [], [], []
+    class Array:
+        def __init__(self, data): self.data = data
+        @property
+        def shape(self): return self.data.shape
+        @property
+        def dtype(self): return self.data.dtype
+        @property
+        def nbytes(self): return self.data.nbytes
+        @property
+        def flags(self): return self.data.flags
+        @property
+        def size(self): return self.data.size
+        def __getitem__(self, key):
+            result = Array(self.data[key])
+            if isinstance(key, slice): slices.append(result.size)
+            return result
+        def reshape(self, *args): return Array(self.data.reshape(*args))
+        def __ne__(self, value):
+            assert self.size <= cube_check.CHECK_ELEMENTS
+            return Array(self.data != value)
+        def all(self):
+            reductions.append(self.size)
+            assert self.size <= cube_check.CHECK_ELEMENTS
+            return Array(np.asarray(self.data.all()))
+        def item(self):
+            assert self.shape == ()
+            host_flags.append(bool(self.data.item()))
+            return self.data.item()
+        def get(self, *args, **kwargs): raise AssertionError('No payload get')
+        def __array__(self, *args, **kwargs): raise AssertionError('No implicit host array')
+    def isfinite(array):
+        assert isinstance(array, Array) and array.size <= cube_check.CHECK_ELEMENTS
+        return Array(np.isfinite(array.data))
+    def any_(array):
+        assert isinstance(array, Array) and array.size <= cube_check.CHECK_ELEMENTS
+        reductions.append(array.size)
+        return Array(np.asarray(array.data.any()))
+    def asnumpy(*args, **kwargs): raise AssertionError('No payload asnumpy')
+    cp = SimpleNamespace(isfinite=isfinite, any=any_, asnumpy=asnumpy)
+    sums, counts = fixture()
+    result = cube_check.array_checks(cp, Array(sums), counts, shape=sums.shape)
+    assert slices and max(slices) <= cube_check.CHECK_ELEMENTS
+    assert sum(slices) == sums.size
+    assert len(reductions) == len(host_flags) == 2*len(slices)
+    assert result == [dict(bin=i, finite=True, nonzero=i==0) for i in range(100)]

{
  "git_head": "af6ebc03b86033accc6c0af34bb6e3eb2b85f35c",
  "task_working_status": "",
  "status": "pass",
  "exitcode": 0,
  "command": [
    "/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python",
    "-m",
    "pytest",
    "-q",
    "--basetemp=/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/gpu-check-host05/pytest-tmp",
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
    "prepare.py": "146636bc4979c206bb357918b467a4a4e9c6c355046d8be950b0243c87b52f8a",
    "cpu_plan.py": "fa6a756b67c1e29371d195e89e08264d49268374d0b270ecbf3de5fbfce510be",
    "device_check.py": "034065fc18c2a829be9d4fefb07f0446f889496883413df97f770c7eceeb0c1b",
    "runtime.sh": "988ae2532369f109ba6548f4faf787ba621c4d5177f862bc1301c21bd1b80aa7",
    "source-origins.json": "c10368dd283b01d024c8114069ef74b310e01e69a69ee035b86f0ff2f31522c4",
    "campaign.py": "e2050bb7b2b83e500b3a7488fc6792610b8232ea835d912ee87b536b354bb799",
    "test_audit_port.py": "6ee6b60e5bb77fca03c58b2da8ba81202be869a8cc367d8f45fc7a14d2be064c",
    "memory_watch.py": "4917b0894dbd964cc07397023bab4b9b681961d6b56af752342578d24094c597",
    "expected-native.json": "6bf2c4b6e5e75100c99ccba221f81426d69e039407f330bd970a7af31ff1f816",
    "support.py": "96679115bea5bf08d58fd15228d632e7ed6a2236b32e0973f6c69a20f61be188",
    "affinity_probe.py": "dd0e35a36610cfce1d488cc7f9f56c977e6aa74dfe164dd6afad6078496e3732",
    "test_port.py": "5fb07f2058c8c94d9be0e01635b44f528a27afa1a513c60a5b7e448cd84ccddc",
    "test_cube_check.py": "9f072d0459bfc5b8765e0e89555666353f6b83b9cf455b8ac9aba813abc4d3b2",
    "storage.py": "0a4c77f99ba97b4bfd20943a6e538d140129d76a90fea2d6d6d5cfca0d029070",
    "host_memory.py": "e4142f35478934c917adf7c15739e840ade9764f0722b7233f1c9e51b0927c8d",
    "task_runtime.py": "0157be63682cbcefebd608e0faf01b0d9fde69a8ef3b78eb993dc06b194b3a03",
    "run.sbatch": "9fff150cd00e8bfb9deebadbdb5bf97d5934d007f3e6d8ab26ddacdd3caf72e4",
    "host_checks.py": "50e5997b868bcbca2caaaf990b2f13953b1e9d78b8356145b4d3f39156784bf2",
    "clean_exec.py": "4994a609b58950d7072d4c983bda332ea65644cbf30c6e3ad6022e009340d492",
    "audit.py": "413949312b01707685bf7004ec5b0259d63f1b500b06a90fde6234348c83dc9b",
    "reference_probe.py": "4b00ff774fe8bcef65cdc1d6fd4641e88f281cc5dfd7a57c8ca368d4a0629a42",
    "test_boundaries.py": "e4539fed1f7a7f257dad76020a5b7af52f836fbe9dd0c2263acd54a2e6424968",
    "instrument.py": "aa7d6d720bf1bd1d7a8349db3dff8601820ba6d56042bc9a3d80f3cc17f23859",
    "runner.py": "229c09f0bb0fed89969758b409b111985db5a56c94e46ffea37823de3359b59d",
    "status.json": "88894fbe07c95991dc32dd871a976acf70bd2c319b019fbb4528a60c11e72b6d",
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
........................................................................ [ 63%]
..........................................                                       [100%]
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
114 passed, 221 warnings, 64 subtests passed in 3.31s
