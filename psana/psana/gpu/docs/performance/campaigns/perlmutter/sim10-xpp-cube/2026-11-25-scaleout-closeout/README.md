# Sim10 campaign closeout: weak scaling did not succeed

**Closed 2026-11-25. The performance objective was not met.** Historical replay and single-node PR331 qualification passed their diagnostic screens, but four-node weak-scaling efficiency was only **53–55%**. Execution and correctness gates passed; that does not make this a successful scaling result. Stop this campaign and test multi-node Lustre reading and psana2-gpu staging first.

## Results

All rates are aggregate events/s, medians of three primary runs. Acceptance, bridge and anchor cases are excluded from these medians. All **159 executions** passed their case audits: 39 in Stage 1, 72 in Stage 2 and 48 in Stage 3. All 12 measurement allocations and their steps completed with exit 0:0. No additional benchmarks were run for closeout.

### Stage 1: reproduce the historical single-node campaign

Historical framework `67c327a148a08b2aaf972e67e1e02e367c2b298f`; one BD/GPU. Batches 50 (1 BD) and 200 (4 BDs).

| Stage | Replay 1 BD/1 GPU | Replay 4 BD/4 GPU | Change from history, 1 / 4 BD |
|---|---:|---:|---:|
| S | 669.1 | 1666.1 | +1.74% / +0.57% |
| E | 659.8 | 1664.4 | +0.72% / +0.95% |
| EC | 594.1 | 1601.3 | +1.34% / +0.85% |
| ECC | 536.9 | 1529.2 | +1.59% / +0.05% |

All eight medians were inside the agreed ±10% reproduction screen. Maximum rate CV was 2.65%, maximum S-anchor drift 2.67%. Profiles were paired within each allocation; repetitions used different nodes. This is a descriptive reproduction screen, not statistical equivalence. Stage 1 did not perform the later sampled CPU-reference numerical check.

### Stage 2: qualify pre-sync versus PR331 separately

Pre-sync P: `546764e0338225d026faa29627297a4a390d3396`; candidate C: `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad`.

| Stage | P → C, 1 BD/1 GPU | P → C, 4 BD/4 GPU | Paired time change, 1 / 4 BD |
|---|---:|---:|---:|
| S | 671.4 → 666.7 | 1661.6 → 1646.6 | +0.53% / +0.92% |
| E | 659.1 → 663.1 | 1649.0 → 1645.5 | −0.27% / +0.21% |
| EC | 591.0 → 591.4 | 1587.5 → 1606.1 | −0.07% / +0.72% |
| ECC | 544.6 → 545.0 | 1525.2 → 1527.5 | +0.28% / +0.02% |

Positive paired time change means slower: `100*(median(C_seconds/P_seconds)-1)` for same-node pairs. It need not equal a ratio of separately calculated median rates. All changes were below 1% in magnitude; no regression signal under the 10% screen. Maximum time CV 3.77%, S drift 1.36%. Three repetitions, with reversed P/C order only in repeat 2, do not establish equivalence. The historical-build bridge and formal A/A study were deferred. Sampled numerical acceptance and native device checks passed; transitive native library mappings were not exhaustively audited.

### Stage 3: 100k events/node, four BDs and four GPUs/node

Candidate C only; batch 200; one Smd0 and one EB globally. Total events were 100k/200k/400k, with 6/10/18 MPI ranks.

| Stage | 1 node | 2 nodes | 4 nodes | 2-node efficiency | 4-node efficiency |
|---|---:|---:|---:|---:|---:|
| S | 1685.8 | 2334.3 | 3586.6 | 69.2% | 53.2% |
| E | 1691.8 | 2319.8 | 3599.3 | 68.6% | 53.2% |
| EC | 1610.2 | 2288.7 | 3540.4 | 71.1% | 55.0% |
| ECC | 1566.4 | 2233.4 | 3443.6 | 71.3% | 55.0% |

Efficiency is `R(N)/(N*R(1))`. Stage 3 time CV was below 0.7% in every cell; maximum absolute S-anchor drift was 1.51%, below the 5% gate. Whole loops lasted 59–64 s, 85–90 s and 111–116 s at 1/2/4 nodes. One-node S was slightly below 60 s; approximate interior-window diagnostics do not establish a steady-state plateau.

Actual runtime records verified **four BDs and four distinct GPUs on every node**, seven nonoverlapping physical cores/BD, and nonzero work on every BD, in all 32 two-/four-node cases. Smd0/EB occupied separate three-core sets on the first node. Full-run node event counts ranged 97.8k–102.2k at two nodes and 94.4k–103.2k at four nodes. There is no evidence of BDs accidentally concentrated on one node.

| Group | Job(s) | Elapsed |
|---|---|---|
| Stage 1 repeats | 61901036, 61902144, 61903166 | 31m37s, 30m37s, 35m18s |
| Stage 2 paired allocations | 61907442, 61907941, 61908317, 61908615, 61909393, 61910183 | 38m44s, 22m57s, 23m12s, 40m56s, 43m16s, 27m51s |
| Stage 3, 1 / 2 / 4 nodes | 61945080 / 61946856 / 61949106 | 53m03s / 54m10s / 55m17s |

## Conditions and limits

S reads/stages the seed; E expands to 32 uint16 panels; EC adds offset-enabled FP32 calibration; ECC accumulates a distributed FP32 cube. No cube export or global cube reduction was timed. The maximum-rank processing interval includes delivery, drain, final synchronization and in-loop safety/memory checks; setup/JIT/constants, cache conditioning and post-loop checks are separate. Device integration and ECC numerical acceptance ran before primary timings. Stage 2/3 numerical checks sampled pixels, panels, events and bins; they were not exhaustive comparisons of all timed pixels.

All runs used KvikIO compatibility ON / POSIX fallback, not true GDS: four workers/BD, 1 MiB task, 16 MiB bounce, one CUDA stream/BD, 512 KiB bulk target, 6 GiB device budget, 64 MiB pinned D2H budget. Inputs were Lustre count 2 / 1 MiB stripes. Client cache gates passed; backend cache state and filesystem contention were uncontrolled. No storage-server throughput ceiling or isolated bottleneck is established.

Stage 1/2 used historical `mfx101555026` run 10 and its count-02 scratch view. Stage 3 used the newly qualified 400k-event `prjdatpsana` run 4, JF streams s005–s009, from the private copy of `/global/cfs/cdirs/lcls/projecta/lcls/psdm/prj/prjdatpsana/xtc/`. Only JF content was qualified, as requested. The other streams' missing data are excluded. All five JF streams were nonempty/undamaged with matching SMD offsets/timestamps and published hashes. Input repeats synthetic seed physics. Payload bandwidth uses 3,358,200 input bytes/event, not expanded-image or NIC bytes. Compare scaling to Stage 3's own one-node baseline, not the different Stage 1/2 input placement.

Earlier setup failures remain part of the record: Stage 1 host preflight timed out then passed an instrumented retry; unchanged per-group preflights subsequently passed. Input job 61939667 copied and hash-verified data, then qualification rejected a non-JF missing-data stream. JF-only qualification job 61943215 passed. Measurement launch 61944386 failed before any timing because of conflicting Slurm placement options; corrected launch-only job 61944668 passed 1/2/4-node placement. None of these attempts contributes performance samples. See [setup evidence](../../../../../../notes/archives/sim10-scaleout-20261125/setup-outcomes.json).

## Archive and next step

- [All 159 cases](runs.csv), [recomputed summaries](summary.json), [compact rank/correctness evidence](../../../../../../notes/archives/sim10-scaleout-20261125/cases.jsonl), and [scheduler accounting](../../../../../../notes/archives/sim10-scaleout-20261125/scheduler.txt).
- [Archive guide](../../../../../../notes/archives/sim10-scaleout-20261125/README.md): raw locations/hashes, original summaries, qualification and placement evidence. Large timestamp/pixel arrays and scratch logs remain outside Git and may expire; this is a compact archive, not a lossless copy of scratch.
- [Frozen scripts and tests](../../../../../../scripts/campaigns/perlmutter-sim10-scaleout-20261125/README.md): exact executed bundles, group manifests, original Stage 3 preparation/qualification tests, restoration and offline validation instructions. Archived launch paths require explicit adaptation; do not execute relocated launchers unchanged.
- [Review record](review.md) covers this final report and archive; old snapshot review statuses are historical, not the current closeout status.

**Next: measure multi-node reading/staging before further application scaling.** Compare IOR, the five-file C++ reader and psana S on the same 1/2/4/8/16/24-node allocations; screen stripe counts and actual OST coverage. The historical C++ stripe screen was single-node. Distinguish storage behavior from shared-file access, EB supply and H2D costs. See [bounded follow-up](NEXT_STEPS.md). This closeout launches no new tests and makes no Lustre root-cause claim.
