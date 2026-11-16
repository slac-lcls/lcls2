# Why one CUDA stream did not trigger admission drains

The cause is the one-outstanding-small-group credit per physical XTC file
stream, not exhaustion of the 4 GiB framework byte budget. This corrects the
initial budget attribution in the performance preparation notes. CUDA execution
streams and physical file streams are separate things: this fixture has five
JF files (s005–s009), regardless of whether execution uses one or two CUDA streams.

## Mechanism

At the preparation runner's 8 MiB `gpu_bulk_target_bytes`, each sim10 JF datagram
is classified as small (524,724 to 734,604 bytes). The read planner coalesces
these datagrams, and `InputGroupPool` permits only one outstanding small group
per physical file stream. Its credit remains occupied while input-window uses
or CUDA dependencies retain the group. Finishing a kernel alone does not release
an execution slot's retained input leases.

With **one CUDA stream**, the next execution slot is also the previous slot.
Normal retirement releases that execution's leases before planning the next
read; the input pool can reclaim its group and return its file-stream credit.
There is no exceptional drain. This is sequential retirement, not evidence of
input/compute overlap or a memory optimization.

With **two CUDA streams**, the next slot can be empty while the other slot
still holds the previous input group. `begin_retire_next()` retires only the next
slot. A new small group for the same file then conflicts with the existing
credit. `plan_slots()` returns None and `_issue_group_reads()` raises
`GpuMemoryPressureError('input groups await stream credit or raw slots')`
**before calling the byte-budget hold**. The generic recovery flushes the
execution pool, trims idle inputs, and retries, serializing that handoff.

See `psana/psana/gpu/gpu_stream_read_plan.py`, `gpu_input_group.py`,
`gpu_input_window.py`, `gpu_stream.py`, and `gpu_events.py`.
[runtime-callsite-check.json](runtime-callsite-check.json) records 12 successful
AST comparisons between the checkout and the frozen installed runtime for the
relevant methods/module; [execution-source-hashes.json](execution-source-hashes.json)
binds the actual executed files.

## Causal controls on A100

Job **43557748**, sdfampere010, A100-SXM4-40GB, completed successfully in 64 s.
Each fresh process used the same 1,000-event pilot, 100 bins, nominal batch 7,
100 warmup events, frozen calibration, and KvikIO CPU fallback (compatibility ON).
All event/bin checks passed; four observer tests and nine preparation tests passed.

| CUDA streams | Framework budget | Bulk target | Warmup drains | Steady drains |
|---|---|---|---|---|
| 1 | 4 GiB | 8 MiB | 0 | 0 |
| 2 | 4 GiB | 8 MiB | 15 | 142 |
| 2 | 8 GiB | 8 MiB | 15 | 142 |
| 2 | 4 GiB | 512 KiB | 0 | 0 |

The two-stream 8 MiB cases made 315 admission attempts: 158 succeeded and
157 failed then retried. Thus all but the first successful submission required
a full pool flush before admission, serializing those execution handoffs. This
does not measure overlap inside an individual submission.

All captured failures had the same stream-credit/raw-slot exception text; there
were **zero byte-budget hold failures** in every case. The observer retains four
failure snapshots per phase, not a detailed snapshot of every failure. Those
samples show five conflicting physical streams and ample free raw slots.
The [representative steady rejection](rejection-example.json) has:

- 145 free reader slots, with only five requested;
- slot 0 next/empty and slot 1 occupied;
- conflicts on all five JF streams, each prior group retaining window uses;
- 621,326,848 bytes committed (~0.579 GiB), zero held, and
  3,673,640,448 bytes (~3.421 GiB) available in the 4 GiB framework budget.

These are **framework-accounted bytes**, not total GPU memory. The two persistent
100-bin cube banks alone occupy about 12.5 GiB outside this budget; user raw and
calibrated arrays also need device space. Doubling the framework quota changes
available bytes but leaves the failure counts unchanged.

At 512 KiB, every JF datagram is classified as large, avoiding the small-group
credit rule. This eliminates the drains but increases successful planned read
groups from 790 coalesced groups to 5,000 individual file/event groups. Callback
count changes from 158 to 150 and bank counts from 501/499 to 500/500. This is a
causal control, **not a recommended tuning or a demonstrated throughput gain**.
These short instrumented runs do not measure actual CUDA overlap or stable
performance; input verification warms cache. No 10k campaign was launched.

## Implementation and evidence

`diagnose.py` wraps the existing runner with process-local observers that forward
calls/returns/exceptions and restore original methods. It snapshots scalar
ownership and budget state without additional CUDA events, waits, polling, or
retaining GPU owners. Observer overhead makes its rates diagnostic only.
`test_diagnose.py` checks credit versus slot exhaustion, large-group admission,
snapshot independence, forwarding, bounded failure records, and restoration.

The preparation runner now accepts `--framework-budget-gib` (1..8, default 4)
and `--bulk-target-bytes` (1 byte..8 MiB, default 8 MiB); its defaults and kernels are
unchanged. `run.sbatch` reproduces the four focused cases. The diagnostic wrapper
restricts those options to the actual causal-control values in the table.

Full reports and traces:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/pressure_20261106_r1/`.
[artifacts.json](artifacts.json) records paths, sizes and SHA256;
[validation-summary.json](validation-summary.json) contains compact results.
Scheduler output and accounting are preserved alongside them in this directory.
Independent restricted Claude review passed code commit `784e4fba4` with no
blocking findings. See [review and responses](review-01/response.md) and
[status.json](status.json).

## Next decision

Increasing the byte budget will not address this cause. Before using two-stream
rates as a scaling baseline, investigate allowing a bounded number of concurrent
input groups per file while preserving lease retirement and memory accounting.
Decoupling the coalescing target from that concurrency policy would avoid using
smaller I/O as an accidental way to allow overlap. This requires a separate core
change and review; no core behavior or kernels were changed here.
