# B-CPU establishment — 2026-07-10 (Ralph iteration 1)

New `--cpu` mode in bench_calib.py: BD ranks run det.raw.calib() on the shared
collective DataSource (MPI-capable, same rank layout as the GPU path).

## Blocker
The reference run r387 has been PURGED from FFB (rolling buffer). The only
FFB-resident run for mfx101210926 is r0215, which is incomplete (in-progress
stream s003, no fetchable Configure/BeginRun dgrams). r387 survives only on
/sdf/data Lustre. So B-CPU-on-FFB (the number that would compare to the
B-MVP FFB rows: 210 Hz / 32 BD) COULD NOT be measured this iteration.

## What was measured (Lustre r387, verification of the new code path)
srun on ralph-gpu (job 31260656, sdfampere042), mpirun --bind-to none,
dir=/sdf/data/lcls/ds/mfx/mfx101210926/xtc

| BD ranks | aggregate Hz | per-rank Hz | CPU calib ms/event |
|---:|---:|---:|---:|
| 1  | 10.1  | 10.06 | 29.75 |
| 32 | 106.5 | 3.33  | 54.48 |

Logs: cpu_lustre_bd1.log, cpu_lustre_bd32.log

## Caveats (why these are NOT the baseline)
- Lustre, not FFB. r387 is warm on Lustre now (repeatedly read), so these are
  cache-contaminated and time-variable — NOT comparable to the historical
  GPU Lustre column (1.2/6.2/.../15.0 Hz), which was measured on cold Lustre.
- Real finding that IS robust: CPU calib inflates 29.7 -> 54.5 ms/event going
  1 -> 32 ranks (32-way core/memory-bandwidth contention). So the CPU compute
  ceiling at 32 ranks is ~32/0.0545 = ~587 Hz, vs the GPU kernel ceiling
  ~3100 Hz. Compute does not scale linearly on CPU.
