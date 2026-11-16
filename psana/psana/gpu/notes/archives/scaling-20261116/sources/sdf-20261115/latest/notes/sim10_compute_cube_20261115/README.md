# Same-node four-stage compute/cube screen

This package prepares a job; it does not submit one. The original single-node
attempt02 passed independent review with two timing-budget advisories. The
allowlist amendment is described in `ALLOWLIST.md`; `status.json` identifies the
latest reviewed snapshot and submission command. Host tests are not CUDA/MPI
acceptance. The pinned PR298 source and inherited native build are unchanged.

## Frozen comparison

One exclusive allocation selected from the reviewed allowlist, four A100-SXM4-40GB GPUs, six MPI task
slots, six physical CPUs per slot, 192 GiB host memory, requested 24 hours.
The topology is varied within that allocation: BD1/GPU1 uses batch50 and 13
bound physical cores; BD4/GPU4 uses batch200 and 34 bound cores. Every BD has
four KvikIO workers and one CUDA stream. Keep 1 MiB tasks, 16 MiB worker bounce
buffers, 512 KiB bulk target and 6 GiB framework budget. Compatibility mode is
ON; this is not GDS. Derived Jungfrau sharing and geometry sharing are disabled.

S stages/delivers the same five-stream seed input. E also expands all 32 uint16
panels. EC also performs offset-enabled, unmasked CPU-v3-equivalent FP32
calibration. ECC also accumulates 100 dense FP32 bins, with 99 active and one
empty, using global ordinal modulo99. This remains a synthetic MFX fixture for
the XPP workflow, not scientific XPP-coordinate binning.

After device ownership/transition regression and malformed-descriptor checks,
two ECC acceptance cases exercise nominal50/200 batches, tails, reuse, and
bitwise full-image CPU-v3 comparisons. Eight 4000-event pilots precede 24 full
100000-event cells (three repeats), plus six same-topology S anchors. Pilots
have syscall tracing and CUDA stage intervals; their rates are not clean
throughput. Case order is frozen; each case is a fresh MPI process group.

## Mandatory gates

- Every case uses the same node and owned cold host-page-cache Weka stage,
  with NIC evidence. Cold server cache is not claimed. NVMe is not an automatic
  substitute for failed Weka, and no multi-stream or smaller-batch rescue occurs.
- Frozen constants enter through the smd0 base calibration hook, then the
  existing MPI calibration-XTC path. Host and device hashes are checked before
  conditioning. The GPU manager is precreated and consumed exactly once by
  the normal event iterator, moving setup out of timing without replacing it.
- Physical free memory must cover outstanding demand plus 4 GiB safety.
  Framework-resident allocations are subtracted once from its ceiling; the
  31 GiB batch200 planning subtotal is not a measured fit. Weakref and stream
  drain checks prohibit retaining a second full-image set. Runtime CUDA/pool
  observations are sampled, not exact peaks.
  A one-element setup probe verifies CuPy weakref creation and release before
  processing, and its result is retained in every BD row.
- Images use keepalive, never publication. Only uint32 `sim10.errors` may be
  published, capped at64KiB/callback with a64MiB pinned-output cap. Harness
  wrappers observe every publication/destination choice and reject ordinary
  host fallback. This is independent of the retired d2h-chunk setting.
- Exact timestamps, five datagram reads/event, bytes, callbacks, single-stream
  identity, no pressure drains/CPU payload fallback, and complete drain are
  required. MPI ownership may be uneven; global events are never replayed perBD.
- ECC merges one bin at a time in fixed BD-rank order into FP64 on rank0.
  CPU-v3 checks every pixel/bin for pilots; full runs check fixed sampled pixels
  in every panel/bin using every event, and all pixels of the empty bin. Each
  bin's hash and error/bound evidence is retained. No whole-cube MPI gather.
  Full-run sampled references use native-derived seed offsets, checking every
  record's XTC headers/Shapes, identity and panel layout. Three-event native
  parity checks show 2,932 requested bytes/event after one full witness per
  stream; this is not physical-I/O or reference-walltime acceptance.

## Storage, timing and limits

Shared scratch has approximately6.6GiB free as observed during preparation.
Only compact snapshots/logs/evidence go there. Weka holds the approximately
336GB input stage; node-local storage holds ten regenerable approximately
12.5GiB merged cubes (two acceptance, two pilots, six full). Capacity is checked
before running. No cleanup is automated. Node-local persistence after an
allocation is not guaranteed; compact evidence must survive independently.

Processing time starts at the common post-conditioning barrier and ends only
after iteration, kernels and drain. Constants/JIT, cube initialization,
independent reference, export and complete case time are recorded separately.
The processing+export rate additionally reported for ECC removes serialized
reference/hash intervals from export wall time and is labelled derived, not a
separately measured clean export benchmark. Inclusive CUDA intervals are never
summed into a throughput claim.
Every stage performs two CUDA-free/pool safety queries per callback inside the
processing interval. Results label this monitoring overhead explicitly: clean
means no syscall trace or CUDA stage timing, not an uninstrumented hot path.
Per-rank records also retain both sharing flags and the absence of all four
optional derived/geometry shared-memory/cache objects.

The inherited native receipt has a `psana/meson.build` hash mismatch against the
frozen install. Host native CPU-v3 and calibration-XTC roundtrip checks cover
the newly used native interfaces only at small shapes. They do not substitute
for the allocation's full-shape CUDA/MPI checks; no new build is claimed.

The original attempt02 pinned sdfampere036. New allowlist preparation broadens
selection but never splits the campaign between hosts. Historical inventory and
Slurm observations do not reserve a node or guarantee an immediate start.
Any gate failure leaves an incomplete diagnostic, not a performance result.

The verified command is in `submission-command.txt`; it has not been executed.
Read `REVIEW.md` for the two retained timing-budget advisories before submission.
