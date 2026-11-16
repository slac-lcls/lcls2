# Perlmutter Sim10 XPP cube baseline

This entry reserves the result history for the **Verify Sim10/XPP-cube baseline** task. The archived state is dated 2026-11-16 04:46 UTC; it is a snapshot, not a live job status. The full same-node S/E/EC/ECC campaign has no accepted throughput results yet.

| Stage | Work included |
|---|---|
| S | Seed staging and delivery |
| E | S plus expansion to 32 uint16 panels |
| EC | E plus offset-enabled CPU-v3-equivalent FP32 calibration |
| ECC | EC plus persistent dense 100-bin FP32 cube per BD, resident on its GPU |

This is the synthetic MFX fixture for the XPP workflow; bins use global event ordinal modulo 99 with one empty bin, not scientific XPP-coordinate binning. Ordinary image D2H is disabled; bounded error publication remains enabled. The prepared bootstrap checks each distributed GPU cube after processing: exact shape/dtype/bytes and per-bin counts, every pixel finite, at least one nonzero pixel in each populated bin, and every empty-bin pixel zero. Only counts and scalar flags return to the host. It performs no cube export, MPI reduction, CPU cube reference or numerical-reference comparison. These content checks do not establish numerical-reference correctness.

The prepared profiles are one BD/one GPU/batch 50 and four BDs/four GPUs/batch 200, four workers/BD. Their input view currently uses **stripe count 2**, unlike the count-1 staging scaling campaign. Record any later change as a new configuration.

Bootstrap job 61373306 failed at accept-bd1-ECC and is retained as diagnostic evidence. It ran implementation `296d69985ba2cfa14e4759b035c886d0c658465f`, reviewed at `af2e612eae30f4c32f7d805a0d448200aa84b8a6`; independent failed-run result review remains pending. Export/reduction belonged to that failed bootstrap01 design. The replacement removes that phase; this does not diagnose or prove a fix for the original timeout. Any future export/reduction measurement would be separately scoped and recorded. Replacement bootstrap02r2 passed implementation/host checks and was awaiting submission approval in the archived status. Deferred low-priority item `GC-R2-02` suggests message-specific assertions in the original cube corruption tests; it remains deferred in the archived implementation review. This catalog import launches no work and does not promote failed or pending cases into measured results.

For the next result, add a new dated campaign entry with exact source and environment identities, S/E/EC/ECC timings, globally unique event counts, the actual shape/content and bin-count checks (and any separately authorized numerical-reference checks), memory and available storage-counter records, terminal accounting, and result-review disposition. Keep staging-only bandwidth and full application throughput in separate tables. Preserve this snapshot rather than overwriting it.

Evidence and scripts: [plan](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/PLAN.md) · [status snapshot](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/status.json) · [source provenance](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/source-origins.json) · [prepare script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/prepare.py) · [campaign script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/campaign.py) · [runner script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/runner.py) · [batch script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/run.sbatch) · [validation script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/cube_check.py) · [tests](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/test_cube_check.py)

[Machine-readable record](manifest.json). Paths inside the record are relative to the repository root.
