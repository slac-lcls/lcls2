# Four-stage, same-node submission preparation

Prepare (do not submit) one exclusive four-A100 allocation comparing one BD on
one GPU with four BDs on four GPUs. Stages are S, E, EC and ECC as defined in
the reviewed `../sim10_compute_cube_plan_20261115/PLAN.md`. Preserve the 100-bin
synthetic cube, 99 active bins, nominal batches 50/200, one CUDA stream, four
KvikIO workers, 1 MiB tasks, 16 MiB bounce buffers, 512 KiB bulk target and
6 GiB framework budget. Primary cohort is cold host-page-cache Weka.

No psana core changes, rebuild, commits, branches, cleanup or job submission
are authorized. Existing dirty files and frozen historical packages remain
unchanged. New maintained code and compact evidence live here; generated
snapshots/logs go to shared scratch, full cube exports to owned node-local
storage, with no automatic deletion.

Default independent review: Claude, at most three rounds, restricted read-only
tools and a frozen SHA256 manifest because commits are not authorized. Pass
requires host tests and no unresolved blocking findings on the latest snapshot.
Allocation-time CUDA/MPI/numerical/storage gates remain mandatory, not presumed
passed by host tests or code review. Any unavailable required check stops the
campaign rather than becoming a successful measurement.

## Allowlist amendment

The user authorizes preparing a hardware-backed node allowlist, not submission.
Replace the single-node pin in a new frozen snapshot, preserving attempt02.
One exclusive node still runs the entire unchanged campaign. Match inventoried
CPU/GPU topology, driver and NIC layout; require inventoried Gen4 local scratch.
Local SSD models may differ because input stays on Weka and cube export is
outside processing timing. No NVMe input or cross-node rate pooling is authorized.
The independent amendment review has its own maximum of three rounds. Carry
IMPL-R2-001 and IMPL-R2-002 forward as deferred timing risks, not closed findings.
