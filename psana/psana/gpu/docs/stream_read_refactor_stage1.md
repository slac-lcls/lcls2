# Stream-interleaved read refactor: Stage 1

Follow-up review fixes, integration requirements and deferred cleanup are
tracked in [the cleanup checklist](stream_read_refactor_cleanup.md).

2026-09-24. E runtime and recent benchmark evidence were committed and pushed
as `84c49cdce` before this work. Stage 1 adds a pure CPU planner, tests and a
real-SMD preview. It does **not** change production scheduling or claim new
GPU performance. The Stage 1 additions are left for review before committing.

## Contract

`gpu_stream_read_plan.build_stream_read_plan()` consumes resolved datagrams
for one EB batch and their transition epochs. It returns independent read
groups with original event/stream/timestamp identity, file offset, byte count,
and an ordered list of source datagrams. A datagram's group-relative device
offset is its file offset minus the group's file offset.

- Dgrams below the 1 MiB target coalesce only while exactly adjacent, in the
  same physical stream, file and transition epoch, and within the target.
- A dgram at or above the target is a single request; it is not split here.
- Groups are offered by first event, then physical stream ID. JF is therefore
  event-major across five streams, rather than reading a whole stream first.
- Each small group names its preceding small group with `after_group`.
  Runtime must wait for that predecessor's **consumers**, not just its I/O,
  before submitting the successor. It must skip blocked streams and continue
  eligible work. This is not a global pipeline retirement order.
- Group IDs are batch-local; owners must use `(batch_id, group_id)`. The
  runtime's one-outstanding-small-group credit must persist across EB batches.
- Missing dgrams create no read. Zero-size rows retain metadata. Duplicate
  identities, timestamp mismatches, overlapping file ranges, out-of-batch
  events and invalid fences fail before I/O.
- The raw-input allowance is separate from the full GPU budget. Stage 1
  checks a conservative progress bound: the sum of each physical stream's
  largest request. Parser/output/scratch/cache/margin reservations and actual
  in-flight growth remain runtime responsibilities. Total planned bytes are
  not allocated at once. Larger execution groups need additional input credit.

The old adjacent-range planner and the bulk-off path remain unchanged.

## Real-data preview

Input: private Weka FFB copy of `mfx101210926/r0387`, streams 0 and 5–9,
the same first **1,000 events** as the minimum cold reproducer. Batch size
100, target 1 MiB. The diagnostic reads actual SMD offsets/sizes, requires
aligned event timestamps and transition histories, and forms ten 100-event
planning batches. It is a metadata replay, not a live EventBuilder/GPU run.

The preview cross-checks every stream's timestamp hash and useful byte count
against job 39008696's measured manifest. No planned BigData reads are issued;
there is no throughput or cache-state measurement in this preview.

### Feespec s000: first three batches

| Batch | Event range | File offset | Bytes | Dependency |
|---|---|---:|---:|---|
| 0 | [0,35) | 57,312 | 340,688 | First small group |
| 0 | [35,100) | 398,024 | 632,840 | All consumers of preceding group complete |
| 1 | [100,155) | 1,030,864 | 535,480 | Prior batch's small-stream credit available |
| 1 | [155,200) | 1,566,368 | 438,048 | All consumers of preceding group complete |
| 2 | [200,275) | 2,004,416 | 730,200 | Prior batch's small-stream credit available |
| 2 | [275,300) | 2,734,640 | 243,328 | All consumers of preceding group complete |

**Real data requires two feespec reads in each of these batches.** The ranges
are separated by a 24-byte SlowUpdate dgram (service 10). The planner preserves
the transition fence and does not read through the gap. All 100 feespec event
dgrams would otherwise fit below 1 MiB. Across all ten batches, feespec has
19 requests rather than the idealized ten.

### JF: first three batches

Each stream has 100 requests per batch: one for every event in `[0,100)`,
`[100,200)`, and `[200,300)`, respectively. Requests are interleaved across
streams; the table gives each stream's exact single-request size and the first
file offset in each batch. Later offsets and every source row are in JSON.

| Stream | Bytes/request | Batch 0 first offset | Batch 1 first offset | Batch 2 first offset |
|---|---:|---:|---:|---:|
| s005 | 6,291,972 | 156,200 | 629,353,424 | 1,258,550,648 |
| s006 | 7,340,630 | 181,502 | 734,244,526 | 1,468,307,550 |
| s007 | 5,243,314 | 130,898 | 524,462,322 | 1,048,793,746 |
| s008 | 7,340,630 | 181,502 | 734,244,526 | 1,468,307,550 |
| s009 | 7,340,630 | 181,502 | 734,244,526 | 1,468,307,550 |

The first offered requests are:

```text
feespec[0,35)
JF s005(event 0), s006(event 0), s007(event 0), s008(event 0), s009(event 0)
JF s005(event 1), s006(event 1), s007(event 1), s008(event 1), s009(event 1)
...
feespec[35,100), eligible only after consumers of feespec[0,35) complete
JF s005(event 35), s006(event 35), ...
```

Over 1,000 events the plan has **5,019 requests: 19 feespec plus 5,000 JF**,
covering exactly **33,566,911,424 bytes**, matching the old off/on baseline.
The first three batches each have 502 requests. Their conservative
one-largest-group-per-stream input bounds are 34,190,016 / 34,092,656 /
34,287,376 bytes. The diagnostic supplies a 64 MiB **raw-input-only** allowance;
these bounds are not predictions of total runtime GPU memory.

## Validation

59 tests passed: 13 Stage 1 tests plus 46 existing read-plan tests. Coverage
includes exact byte reconstruction, event-major ordering, six-stream JF/feespec
shape, small-group dependency chains, gaps, transitions, chunks, partial tails,
sparse events, zero-size rows, exact/oversized targets, separate batches,
duplicate/overlapping inputs and insufficient input allowance.

The earlier benchmark commit separately passed 24 harness/trace/timing tests
and all configured commit checks.

## Remaining stages

2. Introduce independent input-group owners and a bounded reusable pool.
   Seal planned uses; track consumer completion tokens. Group reclamation is
   independent, with small-stream credit held across batches.
3. Wire the planner into scheduling and replace whole-stream residency
   preference. Preserve batched parsing/gather/calibration. Execution groups
   must fit the available small-group event coverage: waiting to form one
   kernel batch from two mutually exclusive small groups would deadlock.
   Splitting execution at that boundary does not impose ordered retirement.
   Respect the current KvikIO small-read deferred-future behavior.
4. Validate GPU ownership, retained views, out-of-order completion, tight
   budgets, transitions, early exit and failures against CPU references.
5. Repeat the cold 1,000-event Weka comparison with eviction verification,
   native traces and untraced controls; then warm and 10,000-event acceptance.

Automatic detector-owned field materialization remains deferred.

## Reproduction and artifacts

Maintained files: `gpu_stream_read_plan.py`,
`scripts/preview_stream_read_plan.py`, and
`tests/gpu/unit/test_stream_read_plan.py` (paths relative to psana's package).

Scratch output:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-plan-stage1-20260924/reviewed/`.
`preview.log` prints only the first three batches. `plan.json` contains all ten
plans, every stream's exact requests, SMD-derived rows, hashes and provenance.

Run in an activated psana environment (the planner itself requires no GPU):

```bash
python psana/psana/gpu/scripts/preview_stream_read_plan.py \
  --planner-source psana/psana/gpu/gpu_stream_read_plan.py \
  --directory /sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc \
  --events 1000 --batch-size 100 --print-batches 3 \
  --reference-manifest /sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-cold1k-20260924/job-39008696/measured-manifest.json \
  --output /path/on/scratch/plan.json
```

The output file must be new. `--planner-source` loads this new Python module
against the frozen E installation's existing native psana components; no
native rebuild is needed for the preview.
