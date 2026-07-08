# MPI GPU scaling benchmark scripts

SLURM scripts for the 2026-07-07/08 measurement campaign on the psana2 GPU
MVP (`psana/psana/gpu/`). Results and interpretation live in
`psana/psana/gpu/TASK.md`; feature verdicts in `psana/psana/gpu/DEFERRED.md`.

| Script | Question it answered |
|---|---|
| `sweep.sbatch` | 1-32 BD ranks on one node, Lustre then FFB (`--dir`) |
| `multinode.sbatch` | 2 and 4 nodes: is the ceiling per-node or central? |
| `ebsweep.sbatch` | PS_EB_NODES 1/2/4: is the EventBuilder the serializer? |
| `smd0_d2h.sbatch` | PS_SMD_N_EVENTS 1000/4000/16000, and D2H on/off cost |
| `r78test.sbatch` | correctness on a second experiment (mfx100852324 r78) |

All drive `psana/psana/gpu/bench_calib.py` (MPI-safe as of f8ae68984).
Raw logs from the campaign (sweep_*.log, nic_*.log, slurm-*.out) are not
committed; job IDs to retrieve accounting are listed in TASK.md.

Reproduction gotchas: always `mpirun --bind-to none` (default core binding
distorts rates and refuses >17 procs on a 17-core allocation); pass psana
env vars through mpirun with `-x`; FFB data path is
`/sdf/data/lcls/drpsrcf/ffb/<hutch>/<exp>/xtc`.
