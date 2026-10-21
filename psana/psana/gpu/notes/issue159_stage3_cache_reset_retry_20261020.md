# Issue 159 stage 3: screen with one-time private-cache reset

Status: job **41400127** completed on **sdfampere030** in **30m28s**, exit `0:0`.
Both six-rank native-library preflights, four pixel/address preflights, and all
eight timed samples passed. Every measured file was 100% resident before and
after every timed sample. All 2,084 frozen-input hashes verified afterward,
and the job removed its private stage.

## Applied correction

Harness commit `688205b9e` adds
[`reset_private_stage_cache`](../scripts/feespec_bulk_benchmark/common.py).
It validates all supplied filenames, extents, ownership and regular-file/link
properties before calling the existing cold-cache preparation. Symlinks,
additional hard links, escaped paths and invalid extents are rejected before
any discard. The caller must provide its own fresh private copies.

The screen calls this helper exactly once, after all four pixel/address
preflights and before the first timed warm-up. Its target is explicitly checked
against `/lscratch/monarin/jf-issue159-${SLURM_JOB_ID}`, which the harness creates
and fills with independent copies. It flushes and discards only those five
private files, verifies the existing per-file cold gate (at most 1% resident),
and records before/after memory snapshots and the reset result.

The existing interleaved warm-up and 99% per-file pre/post timing gates are
unchanged. Later samples do not reset the cache again. The measured loop,
runtime builds, timed-worker CPU/GPU binding, event count, and A/B ordering
are unchanged. This implements the remedy validated by
[cache probe 41398320](issue159_cache_diagnosis_20261020.md).

## Checks and protocol

- **38 cache/harness tests passed** using the isolated frozen script directory,
  including the new scope/rejection checks and cold-gate failure check.
- Both A/B script trees are byte-identical.
- Syntax, shell and balanced-schedule checks passed.
- All 1,973 prior frozen inputs verified unchanged; **2,084 inputs** are frozen
  for the new attempt and checked before/after the screen.
- A0 remains accepted Stage 2, `7f29f0dc7`; B0 remains the clean native build
  of Stage 3, `f425e601e`. Both installs are reused at their original prefixes.
  Passing CPU/CUDA/sanitizer validation from job 41397249 remains applicable.
- Four BDs, one physical A100, batch 20, depth 1, KvikIO CPU fallback,
  eight workers, 1 MiB tasks, 6 GiB per BD.
- Four 200-event pixel/address preflights, then eight warm 10,000-event samples:
  two balanced A/B pairs each for bulk off/on.

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/stage3-20261020-r3/`.
The original attempts remain unchanged under `stage3-20261020-r2/`.

The [retry manifest](evidence/issue159_stage3_cache_reset_retry_20261020.json)
records builds, setup and hashes. The
[exact orchestration patch](evidence/issue159_stage3_cache_reset_harness.patch)
records the reset placement in the screen. Runtime checks, NIC/GPU monitoring,
staged input hashes, event/read/budget checks and private-stage cleanup remain
enabled.

## Completed warm comparison

A is accepted Stage 2; B is Stage 3. Each sample processes 10,000 events on
the same node with four BDs sharing one physical A100. Pair 2 reverses A/B
order. These loop times exclude staging and cache preparation.

| Bulk / pair | A events/s | B events/s | A loop seconds | B loop seconds | Paired loop change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off / 1 | 489.28 | 489.41 | 20.438 | 20.433 | -0.03% |
| Off / 2 | 489.36 | 480.74 | 20.435 | 20.801 | +1.79% |
| On / 1 | 487.51 | 496.87 | 20.513 | 20.126 | -1.89% |
| On / 2 | 490.96 | 534.74 | 20.368 | 18.701 | -8.19% |

Median paired loop changes are **+0.88% bulk off** and **-5.04% bulk on**.
The short screen shows no consistent Stage 3 slowdown across the pairs; its
variation does not establish a reliable speedup. It measures the incremental
cost of Stage 3's unused-copy bookkeeping, not future field materialization.
It does not replace the final cumulative comparison against frozen integrated A.

The one-time reset took **61.9 seconds** and left at most 0.000313% residency
in any measured prefix. Every subsequent warm preparation reached 100%
residency with **zero repair passes**. The original cache blocker is therefore
resolved in this actual screen, including all post-timing residency checks.

The [completed evidence](evidence/issue159_stage3_screen_20261020.json) preserves
all sample rates/times, paired comparisons, per-file residency, rank counters
and timestamp digests, GPU telemetry summaries, cleanup and artifact hashes.
Raw logs, GPU traces and the physical-interface counter trace remain on scratch.

Stage 3's focused validation and short screen are complete. Stage 4 has not
started. The inherited broad-suite failures remain separately tracked for
integrated correctness and are not waived by this result.
