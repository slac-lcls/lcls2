# Six native raw smoke cases

`run_smoke.py` exercises the existing adapter, worker and result validator using
1,000 measured events and the standard 200-event separate-process warmup. It runs
all six profile cells once: CPU events/steps with 1/2 BDs, GPU events with 1/2 BDs
sharing one A100. Settings remain batch20, one GPU stream, 8 GiB per BD, KvikIO
four workers/1 MiB tasks/16 MiB bounce, fallback ON, automatic direct I/O off.
The GPU task includes dense input preparation/gather and a callback that checks
metadata; it has no user kernel, output or payload D2H. This is not pixel validation.

Run directly in a single-node premium lcls_g batch allocation, with the reviewed
runtime's pinned Python and compiler/CUDA modules. Request 4 tasks, 16 CPUs/task,
one GPU, disable-binding, 128 GiB host memory and a 75-minute limit. Expected smoke
work is much shorter; the limit allows bounded prefix preparation (10 minutes),
four placement probes and six 360-second case deadlines with scheduler headroom.
Each native step has a 90-second timeout. Check Iris before submission, capture
final scheduler accounting, and put the EXIT trap before setup/hash checks.

Freeze `run_smoke.py`, `campaign.py`, `perlmutter_adapter.py`, `native_worker.py`,
`native_contract.py`, `prepare_inputs.py`, `prepare_site.py`, `site_probe.py`,
`prepare_runtime.py`, and `profile.json` together. No rebuild is needed. Verify
source hashes before invocation. Supply an input JSON with keys `manifest`,
`reference`, `preparation`, `plan`, and `runtime`, each containing an absolute
`path` and its reviewed `sha256`. These refer to the accepted private 10k dataset
manifest, independent event reference, input-complete receipt, original independent
SMD scan plan, and captured isolated runtime. Never substitute the shared public
input directory. Preserve that JSON alongside the execution evidence.

```bash
/absolute/pinned/python -B -s /frozen/run_smoke.py \
  --inputs /scratch/pinned-inputs.json --inputs-sha256 FULL_SHA256 \
  --output /new/scratch/smoke-JOBID
```

The output must be new. Prefix preparation derives exact ranges from the pinned
SMD plan, checks association with the accepted parent manifests, verifies all
5,000 selected headers and hashes only the selected private prefix on compute.
It creates new manifests in the output directory; it neither copies data nor
modifies the original 10k manifests/files. Input ownership, read-only mode,
metadata, link count and SMD hashes are checked. Full-file integrity still relies
on the original preparation and unchanged protected files; there is no full
560 GiB rehash in every smoke job. Prefix hashing warms cache; each measured
worker independently evicts the selected private ranges after setup and checks
zero residency twice with mincore before its loop. Warmups are separate processes.
SMD, server and other filesystem caches are uncontrolled.

Fresh site preparation runs in this same allocation. Do not reuse the manifest
from a completed job. The adapter rechecks runtime/imports, files, settings,
physical placement, exact delivered timestamps and bytes, and non-idle BDs.
All twelve workload steps (warmup/measured) must complete 0:0 with no remaining
matching step. The driver invokes campaign.validate_result and requires available
NIC counters. This is an availability-only check until traffic coverage is
qualified. In job 61503769, the hsn kernel network-device counters did not account
for the workload data traffic; RDMA/storage coverage remains unqualified. These
node-wide counters do not establish storage attribution or a link-bandwidth ceiling. Preserve indexed per-rank/cache/scheduler
records even on failure; retry in a new output directory and allocation.

This is a qualification-only entry point. It creates correctness requests directly
rather than invoking campaign.execute, because native qualification is still being
established. It retains all campaign qualification flags false and records the
remaining blockers. The component contract is explicitly unqualified. A smoke
pass is not authorization to enable those flags or claim statistical regression,
instrumentation overhead, component bounds, calibrated pixel correctness, full
EndRun/multi-step coverage, or 10k throughput. The selected prefix has one native
step. Runtime 199575205 is a known-API fixture, not a newly selected PR baseline.

After reviewing evidence, proceed to full-prefix cache/byte-accounting and
instrumentation/component qualification, then timed A/A before any PR A/B.

The input specification is decoded from the same bytes whose hash was checked,
and the prefix child verifies that hash itself. The parent verifies the child's
completion/specification association and both derived artifact hashes. Before
publishing a complete smoke receipt, a bounded direct Slurm audit requires only
batch/extern records and the live batch identity; any numeric or unexpected step
fails. Final post-job accounting remains required independently of this in-job check.
