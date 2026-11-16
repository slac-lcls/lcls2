# Single-node sample10 performance preparation

Preparation scope: a serial one-BD/one-GPU runner, before the 10k campaign.
Implementation, device smoke and independent review status are in status.json.
The reviewed expansion, calibration and accumulation CUDA sources are reused;
psana core and the frozen installed runtime are unchanged.

Independent Claude re-review passed code commit `5f4e7167b` with all four initial
findings closed and no blockers. See [review closure](review-02/response.md).

Post-review preparation validation passed in job 43548551 on sdfampere011,
A100-SXM4-40GB (COMPLETED, exit 0, 50 seconds): nine host tests, four controlled GPU cases,
and 1,000-event throughput cases with one/two streams plus a two-stream profile.
All exact event/bin counts passed; two banks handled 501/499 events. Actual
callback sizes were 1, 2, 5, 6 and 7, with 158 callbacks. The three smoke windows
contained 100 warmup and 900 measured events. These single small runs do not
establish a stable 10k rate or a stream-scaling conclusion. See
validation-summary.json, device-checks.json and artifacts.json.

Both clean and profile two-stream cases observed 15 warmup and 142 steady-window
cache trims after framework GpuMemoryPressureError, with the inherited 4 GiB
framework budget. The clean one-stream case observed zero. The frozen
runtime calls `_trim_gpu_caches` only on this admission-failure path.
The [follow-up investigation](../mfx_expand10_pressure_20261106/README.md)
corrects the initial budget attribution: the cause is a small-group credit
conflict per physical file stream, with ample byte budget and free reader slots.
Doubling the quota leaves the drain counts unchanged; changing the classification
threshold removes them but increases read-group count. Core admission behavior
remains unchanged. Initial smoke job
43548178 on sdfampere035 also passed before the review improvements; its logs
are retained and the original reports remain referenced by review-01/copy-index.json.
CUDA/CPU profile totals
remain overlapping/inclusive, not a critical-path attribution.

Default case: 10,000 original events, 100 bins (99 populated plus an empty
control bin), two CUDA streams, nominal batch 7, at least 1,000 warmup events.
The first 1,000 events are still accumulated; only their timing is excluded.
Actual warmup rounds up to a complete callback and is reported. The next
callback starts the steady timer after a one-time join of both bank streams.
The timer ends after EOF and an explicit bank join, so its denominator includes
tail delivery/drain and its numerator counts completed GPU events. Input for
the boundary callback may already have been read/parsed: this is a warmed
pipeline window, not a cold full-file I/O measurement. The uninterrupted whole
iteration is also reported (it contains the intentional warmup-boundary join).

`throughput` mode has no per-event D2H, memory sampling, CUDA timing events, or
framework timer patches. Both modes install one count-only observer on the
pressure-only cache-trim path; `admission_pressure_trims` reports warmup and
steady counts. Nonzero values flag forced pool drains that can suppress stream
overlap; do not interpret those cases as unconstrained stream scaling.
The throughput loop retains host timestamp checks, callback sizes,
bank identities and the reviewed adapter's inexpensive CuPy pool accounting.
Memory is sampled only outside iteration; those samples are not peak tracking.
All panel errors feed the existing device acceptance kernel. After completion,
each bank's bin counts must exactly match its submitted event identities.
This is event/count acceptance, not a new full-pixel correctness test. The
previous 1k full numerical acceptance and controlled device tests are its
kernel correctness basis.

`profile` mode adds CPU spans, NVTX ranges and CUDA intervals. CPU stages cover
SMD/EventBuilder acquisition, ancillary reads and dgram construction, KvikIO
submission/waits, slot submission/retirement/flush, descriptor construction and
parsing. CUDA intervals bracket expansion, calibration (after its first lazy
call), count and sum kernel launches; parse and descriptor intervals bracket
whole calls including host work and stream waits. The inherited bin-index H2D
is within `cube_bins_upload_accumulate_call`, not an isolated copy interval.
Likewise the default-stream synchronization is included in slot submission,
not separately intercepted. Use a separate Nsight Systems capture to resolve
those waits, transfers and idle gaps if these inclusive spans are significant.
Generator spans exclude time suspended at yield. Timers are nested and streams
overlap: never add these totals to infer wall time. I/O byte counters aggregate
across EventBuilder resets for the entire profiled run, not just steady events.
The clean-mode baseline intentionally does not collect those I/O deltas.

The default frozen runtime uses KvikIO compatibility mode ON (CPU fallback);
this is not GDS qualification. Report runtime paths/hashes, environment and
cache state with results. Calibration is the frozen full-panel pedestal, gain
and offset set, hash-checked before setup; no live calibration DB modifications.
The 10k preflight binds the generation-manifest SHA256 and every file size to
the committed Stage 1 receipt. File hashes are historical unless
`--verify-files` is supplied; that flag rereads all 37.7 GB before timing and
warms cache. The 1k smoke always uses its original full-hash/schema preflight.

## Run after preparation review

From this checkout, create a fresh project-scratch logs directory, then submit
one case (the batch script creates the new result directory):

```bash
task_dir="$PWD/notes/mfx_expand10_performance_20261106"
results=/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/performance_RUNID
mkdir -m 770 "$results-logs"
sbatch --output="$results-logs/%j.out" --error="$results-logs/%j.err" \
  "$task_dir/run.sbatch" "$task_dir" "$results" throughput
```

For the small preparation checks replace `throughput` with `smoke`. For separate
profiling use `profile`; pass further runner options after that argument.
Supported bounds: streams 1/2, batches 1..16, bins 2..200.
The diagnostic follow-up adds `--framework-budget-gib` (1..8, default 4) and
`--bulk-target-bytes` (1 byte..8 MiB). On November 8 the sim10 runner default
changed from 8 MiB to **512 KiB** to bypass the small-group credit restriction
for its 0.5–0.7 MiB JF datagrams. This changes read grouping, not stored data.
The framework default remains 1 MiB. See the
[512 KiB rerun](../mfx_expand10_512k_20261108/README.md) for results and limitations.
Admission reserves at least 8 GiB beyond the persistent banks; this is not a framework-wide memory
guarantee. Default 100 bins/two banks is 12.5 GiB plus counts. No multi-BD/MPI
claim. Each repetition uses a fresh process/output directory. Three repeats
and then stream/batch scaling remain the next campaign, not this preparation.

Full cube export is opt-in (`--export NEW_DIRECTORY`), checked for free space,
performed after the completed timing window, and reports join, D2H, host merge,
write/fsync times. It uses the reviewed bounded one-bin host staging. Default
performance runs export only compact reports/counts, not the multi-GB cube.
Preemption may leave `status=running`; check Slurm and logs before interpreting
that record. It is never an accepted result.
If iteration raises, the failed report currently omits partial pressure counts;
preserving that optional failure diagnostic is deferred (PERF-PREP-05).

## Wrap-up note: XTC buffers, seeds and raw gathering

KvikIO reads XTC file bytes into GPU buffers, including headers and payloads.
Our sim10 task declares a tuple field selector and uses descriptors to locate
each panel's seed array in those buffers. It does **not** first gather a
contiguous `[event, panel, sampled_pixel]` seed array. The expansion kernel
directly reads those payloads and writes contiguous uint16
`raw[event, 32, 512, 1024]`, combining assembly and tenfold expansion. Calibration
then produces float32 images consumed by the cube on the same stream.

By comparison, a normal uncompressed task declaring `inputs=['jungfrau.raw']`
uses `DenseInputPreparer` to gather the panel payloads into a contiguous raw
array. That gather is absent in sim10; timing should not invent a separate
gather stage. A future real codec adapter may decode from payload pointers or
need a packed input buffer according to its API. Ordinary cube users should
call the adapter rather than implement XTC descriptor plumbing themselves.

## Current handoff

See the [November 8 single-node baseline plan](../mfx_expand10_single_node_baseline_20261108.md)
for the consolidated concurrency conclusion, the decision to retain 512 KiB for
sim10, completed evidence and the next 10k/within-node scaling stages. The [six-case 10k baseline](../mfx_expand10_baseline10k_20261108/README.md)
has completed; the extended-compute real-JF test establishes
concurrency is permitted, without isolating the compressed workload's read and
preparation costs.
