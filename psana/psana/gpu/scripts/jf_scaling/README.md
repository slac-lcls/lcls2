# Current-runtime Jungfrau staging scaling

Maintained MPI benchmark for `mfx101210926/r0387`, streams 005–009. The
historical comparison is [Jungfrau single-node scaling](../../docs/performance/jungfrau_single_node_sdf.md).
The maintained harness now measures read/parse/dense-gather staging without
calibration or `GpuTask`. It uses the established benchmark-only
`stage1_regression/input_adapter.py`; include that directory on `PYTHONPATH`
and freeze/hash the adapter. Legacy frozen campaigns still contain the original
calibration-enabled harness. Do not interpret their rate difference as a matched
workload regression or a kernel batching gain.

Default campaign: 1 GPU/1 BD, 1/4, 2/4 and 4/8; bulk off/on; cold/warm;
10,000 events; two fresh-process repetitions with reversed topology, cache and
mode order in round two. `run.py --full` selects all 11 historical topologies;
`--modes off` or `--modes on` selects a single read mode.

`run.py --include-feespec` selects a smaller **one-GPU, 1/2/4-BD** matrix:
24 timed samples and six 200-event pixel preflights. It adds physical stream
000 using the explicit benchmark-only shared-stream routing exception in
`feespec.py`; the other CPU detectors sharing that stream are not consumed.
All JF-only batch, depth, KvikIO, budget, cache and timing settings are retained.
The timed consumer additionally reads feespec `raw.hproj` through its GPU field
view and computes an int64 sum; only the compact result is copied after timing.
Timestamp-matched CPU sums validate every event across BD distributions, while
preflights compare all 200 feespec arrays and three JF raw samples.
Clear public field aliases before advancing the iterator to preserve ownership.

Generate six-stream references with `prepare_feespec_reference.py` before
freezing. It independently reads SMD offsets, preserves the prior JF timestamp
and payload references, decodes feespec on CPU, and counts adjacent requests
within batch-20 and transition boundaries. For this 10k dataset: **60,000 reads
off; 50,577 on; 335,669,114,744 bytes in both modes**. CPU helper tests include
multi-BD sum/pixel association, corrupt-result rejection and the reduced matrix.

Settings: one SMD0, one EB, eight KvikIO workers per BD, 1 MiB KvikIO task and
bulk target, batch 20, execution depth 1, no automatic D2H, and the current
automatic per-BD GPU budget. Timed event loops only read timestamps. Setup before
`run.events()`, local staging, cache preparation and teardown are excluded;
lazy setup/BeginStep work inside that iterator remains included. Separate
200-event preflight processes check three CPU-reference raw arrays
for every topology/mode. Preflight timing is not throughput evidence.

The five Jungfrau datagrams each exceed the target, so both modes must submit
50,000 reads totaling 335,571,760,000 bytes for 10,000 events. References come
from the actual SMD extents, not a rate-derived estimate.

`bench.py` pins every BD before MPI or CuPy imports. Acceptance checks global
unique timestamps against an independent SMD hash, exact bytes/read count,
zero CPU bigdata reads, distinct physical GPU bus IDs, identical CPU affinity,
correct BD peers per GPU, owned-allocation budget bounds, completion cleanup,
and the separate pixel checks. Per-BD records retain distribution and read-wait
time; `nvidia-smi` logs retain physical memory/occupancy observations.

`run.py` requires a frozen campaign root with `python`, `scripts`, hashed
calibration/reference inputs, `reference.json`, `pixels.json`, `source-commit.txt`
and `hashes.json`. It verifies hashes before/after execution, stages only five
private prefixes on local scratch, requires cold residency below 1% and warm
residency above 99%, and writes a complete summary only after every sample
passes. It removes only its own newly created local stage on exit. Warm page
allocation is NUMA-interleaved by the shared cache helper; timed workers keep
the allocation's common affinity, with no per-rank NUMA tuning.

If a prefix loses pages during initial warming, preparation allows up to three
missing-page repair passes and rechecks all prefixes after each pass. Repairs
retain NUMA interleaving, read only absent page ranges inside the measured
prefix, and record `CACHE_WARM_RETRY`, `CACHE_WARM_REPAIRED`, and `warm_retries`.
The 99% per-file gate is unchanged. Post-timing checks remain read-only.

Frozen cache helpers must include `common.py`, `warm_cache.py` and its dependency
`memory_state.py`, all recorded in `hashes.json`. Before staging, the runner
executes the real NUMA-interleaved warm-cache subprocess with no input files;
its diagnostics are saved in `cache-preflight.log` inside the job directory.

Current campaigns and comparisons are recorded in the
[user-kernel scaling report](../../docs/performance/user_kernel_scaling_20260928.md).
The pre-user-kernel campaign remains documented in the
[historical September 26 report](../../docs/performance/jungfrau_current_scaling.md).
Generated results stay on scratch.

CPU checks: all three `test_*.py` files (**26 passed**).
With an installed or
frozen psana runtime first on `PYTHONPATH`, import psana before invoking pytest
so source-package collection does not hide its compiled extensions:

```python
import psana, pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider',
    'psana/psana/gpu/scripts/jf_scaling/test_contract.py',
    'psana/psana/gpu/scripts/jf_scaling/test_cache_preflight.py']))
```
