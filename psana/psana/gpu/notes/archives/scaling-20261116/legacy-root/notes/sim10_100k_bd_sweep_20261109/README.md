# 100k sim10: 1–4 BDs on one GPU, cold Weka FFB

**Four BDs reached 4.50 GiB/s, 3.79 x the one-BD rate, for psana-GPU descriptor
staging on one A100.** All native checks, 22 audit tests and independent
results review passed on `c12cd589b05564c4d3b6b57436f07dedcebd5246`.
[Review closure](review-02/responses.md) records the findings and limitations. Job 44033881 completed 0:0 in 43m03s on sdfampere038.
Code review passed on `fb4609a89a67e958e918fbd4c01a7dcad2db78bd` with no blocking
findings; [responses](review-01/responses.md) track the optional items.

Each full case processes 100,000 events and 335,820,000,000 JF datagram bytes
(312.756747 GiB). The accepted fixture physically repeats 10k payloads ten times,
with unique outer timestamps and one step. Only five JF streams and their SMD
files are staged on FFB; ancillary detector payloads are excluded. The public
GpuTask uses field descriptors; framework XTC walking/descriptor preparation is
included, but no dense gather, expansion, calibration, cube or payload D2H.

All configurations use two CUDA execution streams **per BD**, batch 100, SMD 1000,
512 KiB bulk target, 6 GiB budget per BD, 8 KvikIO workers per BD, 1 MiB task size,
16 MiB bounce buffers, compatibility ON and automatic direct I/O off. Same physical
A100-SXM4-40GB, 56 shared physical CPU cores and NUMA interleave across cases.
BD count increases CUDA contexts and total workers (8–32) while node CPU allowance
stays fixed; this is not a process-count-only or worker-count-only experiment.
The exclusive Slurm allocation reserved the node; only the selected GPU UUID
was used by all BD ranks, with helpers CUDA-masked empty.

Four short topology pilots precede two full cases per BD count, in order
1,2,3,4,4,3,2,1. Below are medians and ranges of two samples, not confidence
intervals. These are **staging-only** results, not full decompression/cube rates.

| BDs | KvikIO workers total | GiB/s median (range) | Events/s median | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 1.189 (1.174–1.204) | 380 | 1.00 x |
| 2 | 16 | 2.356 (2.336–2.377) | 753 | 1.98 x |
| 3 | 24 | 3.479 (3.470–3.488) | 1112 | 2.93 x |
| 4 | 32 | 4.502 (4.496–4.508) | 1439 | 3.79 x |

The 4 BD result improves on 3 BD by 29.4%. This is the best tested staging configuration,
not a measured ceiling for the node or a per-GPU entitlement in a multi-GPU job.
No timeline/utilization attribution was collected; two streams exercised does
not establish simultaneous execution or isolate the cause of scaling.

Cold means per-file fsync/FADV_DONTNEED on this campaign's owner-controlled FFB
copy, then **zero mincore-resident pages after DataSource setup** before every
case. No global cache drop or NVMe input stage. SMD and Weka backend caches are
uncontrolled. Physical NIC port enp33s0f0np0 received 332.376–341.767 GiB per full
case (1.063–1.093 x the JF bytes); the other physical port received 0 bytes. Idle RX
was 0.960–1.026 MiB/s. This supports network-served reads and does not yield exact
cache-hit fractions: NIC counters include protocol/readahead and other host
traffic. Final payload residency ranged 96.670%–100%; it is reported, not used
to retroactively label a run warm or cold.

The timer spans the start barrier, event delivery, all BD GPU joins and completion
barrier. It includes lazy per-process allocations and framework initialization;
setup, cache conditioning, validation, reporting and teardown are outside it.
NIC endpoints have separate intervals: active-port excess over the loop timer
was 1.513–1.753 s (other port 1.486–1.770 s). Payload GiB/s uses the application timer,
not the longer NIC envelope. Per-case evidence:

| Case | GiB/s | Loop s | Events by BD rank | Active NIC RX GiB | NIC envelope excess s |
|---|---:|---:|---|---:|---:|
| 00-bd1 | 1.204 | 259.73 | 100000 | 332.376 | 1.731 |
| 01-bd2 | 2.336 | 133.91 | 50400/49600 | 333.316 | 1.597 |
| 02-bd3 | 3.470 | 90.14 | 33600/32800/33600 | 337.211 | 1.649 |
| 03-bd4 | 4.508 | 69.37 | 25800/24200/24700/25300 | 340.202 | 1.566 |
| 04-bd4 | 4.496 | 69.57 | 24800/25200/24900/25100 | 341.767 | 1.653 |
| 05-bd3 | 3.488 | 89.66 | 33400/32900/33700 | 336.679 | 1.662 |
| 06-bd2 | 2.377 | 131.55 | 51600/48400 | 336.711 | 1.513 |
| 07-bd1 | 1.174 | 266.48 | 100000 | 334.816 | 1.753 |

Every full case completed 500,000 requests, exactly 500,000 completions and 100,000
events matching the exact accepted set globally, ordered within each BD. Every BD exercised both streams; global task
invocations were 1000, each with 100 events. No psana EventManager._read payload
calls or pressure drains occurred, and all managers/resources drained with the
correct peer count. Submitted-but-not-retired peaks were 500 requests/320.263 MiB
**per BD**. These are queue bookkeeping peaks, not 500 simultaneous storage reads;
per-BD maxima are not asserted to coincide. The peak equals one 100-event
subbatch even with two execution streams. The frozen controller finishes the
current subbatch read waits before issuing the next subbatch reads; more BDs
add independent issuers. Reads within a subbatch can proceed concurrently, and
this does not rule out read/compute overlap. Memory snapshots are in
[validation](validation.json), not continuous GPU-memory peaks.

The runtime comparison to the previous one-BD bulk sweep matches all five common
recorded framework modules, GPU model, CuPy 14.0.1, KvikIO 26.10.00 and effective
settings; [comparison](baseline-runtime-comparison.json). Three newly recorded
modules have no hashes in the old report. Node, stream count and dataset length
differ from that historical measurement, so rate differences cannot be assigned
to input size alone. Specifically, the earlier accepted **1-BD/1-stream, 10k**
run on **sdfampere042** measured **2.1884/2.1197 GiB/s** (median 2.1541).
The current **1-BD/2-stream, 100k** cases on **sdfampere038** were slower,
**1.1737/1.2042 GiB/s** (median 1.1889, about 44.8% below that historical median).
The headline **3.79x** compares only configurations within this sweep; relative
to the historical single-BD median, current 4-BD throughput is about 2.09x.
This comparison does not identify why the single-BD rate differs.

The fallback log's generic “NVMe” wording is not the storage
path: `findmnt` verifies `drpsrcf/ffb`. KvikIO uses the intended CPU bounce path.
Native logs retain nonfatal UCX unmatched-tag warnings during MPI teardown; all
process groups exited successfully and exact delivery/drain checks passed.

[Scope](scope.md), [status](status.json), [results](results.json),
[tests](tests.log), [exact validation commands](validation-commands.txt), [job accounting](slurm-accounting.txt) and
[artifact index](artifact-index.json) preserve the evidence. Native artifacts:

```
/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_100k_bd_sweep_20261109_r1
```

The unique temporary FFB copy was removed after the sweep; cleanup receipt passes.
The accepted 100k scratch fixture remains the reproducible input. A killed job
could leave staged files or MPI children, so inspect scheduler/process state
before owner-scoped recovery; missing complete/cleanup receipts never pass.
