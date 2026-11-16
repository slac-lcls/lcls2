# Qualification results, 2026-11-15

Job61359629 completed successfully on nid001264 using lcls_g/gpu_premium.
It ran 20:16:40–20:28:25 UTC (12:16:40–12:28:25 Pacific),705seconds.
All11numeric steps, batch, extern and allocation exited0:0. The immutable
bundle is from c3a8cb653fe6374953566c2a9245ee153856272b; no runtime code changed.
The post-job audit passed with that explicit reviewed SHA. Independent results
review passed at543240a9334c03a186263d1967f0d38cfe9d8056 with no blocking
findings. Prelaunch review passed separately at the runtime commit.

All31 exact-PR GPU integration tests passed without skips. Input validation
checked all30 files. Six100-event binding cases passed for counts1,2,4,8,16,32.
The1003-event count2 tail case passed its96-values/event CPU/GPU witness and
five200-event callbacks plus one3-event callback. The100k sizing case passed
500callbacks of200, all four BDs nonempty, exact event/range/byte checks,
zero CPU payload reads, zero pressure drains, clean final drain, actual handle
identity and cold local page-cache gates. The six ranks used34 disjoint physical
cores, masks3,3,7,7,7,7, matching the allocation probes.

| Case | Events | Timed loop seconds | Whole case seconds | GiB/s |
|---|---:|---:|---:|---:|
| Count2 tail/witness | 1003 | 1.2497 | 53.1929 | 2.5101 |
| Count2 sizing | 100000 | 61.3367 | 113.8344 | 5.0990 |

The sizing rate is1630.35events/s over312.7567GiB. Configuration: four BDs
sharing one A10040GiB, four KvikIO workers/BD, batch200, one CUDA stream/BD,
512KiB GPU byte target, KvikIO task1MiB/bounce16MiB, compatibilityON,
auto-directOFF. This measures the read/parse staging path, not a cube or GDS.
The other three allocated GPUs were unused. Short binding cases are correctness
checks and cannot rank stripes. Only count2 has a full100k observation;
no best-stripe conclusion or repeated screen result exists yet.

Sizing sampled device usage peaked at4.9804GiB with35.0196GiB minimum sampled
free. Sum of per-BD tracked budget maxima was2.5703GiB, below24GiB aggregate;
each BD stayed within6GiB. NVML sampling can miss shorter spikes, and budget
maxima are not a measurement of simultaneous total VRAM. Setup began9.56s
after Slurm start, within the current90s reserve. Nonfatal NERSC module logging
errors remain in stderr; native tests and all job steps succeeded.

Phase allocation cost is3747seconds including the four earlier attempts and
this705second job. Remaining under the six-hour cap:17853seconds (4:57:33).
This is node walltime, not Iris charged hours. No full screen has been submitted.
Before planning/submitting that screen, resolve the deferred launcher/test
items in responses-01.md and review its concrete walltime/timeout plan.

Evidence: evidence/qualify01 contains unmodified audit, Slurm accounting,
test/validation/affinity records, memory receipts and timing files. Case summaries
retain all fields except long timestamp/callback arrays, replaced by explicit
summaries. Exact raw records remain in the attempt below, and their hashes are
in artifact-sha256.json. The coordinator ran the full audit on raw records,
including range and witness hashes; a reader of compact summaries alone cannot
independently recompute those checks.

Attempt: /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/qualify01
Bundle: /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/bundle-fixed01
