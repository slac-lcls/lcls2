# Clean 10k baseline scope

User authorized the next stage of the November 8 single-node plan, with the
implement/review loop. Run only stage 1: six fresh-process serial single-worker,
one-GPU cases, alternating one/two execution streams for three repeats each in
one allocation. Use the existing reviewed runner without changing kernels,
framework, timing contract, data or calibration. Dataset is the accepted unique
10k sample10 fixture, 100 bins (99 active), batch 7, requested 1k warmup, budget 4 GiB,
read-group target 512 KiB, KvikIO CPU fallback. No artificial compute inflation,
profiling, per-event D2H, cube export, batch sweep, MPI scaling or new data.

Acceptance: every case completes 10k ordered unique reference events and exact
bin/bank counts, uses expected settings/runtime, and has zero admission drains.
Report actual warmup/subbatches, whole-iteration and callback-window rates with
medians/ranges, memory sampling limits, cache history, same-node/GPU allocation
and source hashes. Do not infer kernel overlap or 35 kHz qualification. If cases
fail or have drains, preserve them and investigate before accepting comparisons.
Existing kernel numerical acceptance remains the basis; no new full-pixel cube
validation is claimed for 10k. Independent restricted Claude review, up to three
rounds, on work/synthetic-mfx-scan; commit and push scoped records only.
