# Cold-Weka single-node allowlist

Prepared at the user's request; **not submitted or reserved**. See `status.json`
for the current review verdict and frozen attempt. Original attempt02 remains
unchanged and is not the allowlist implementation.

## Candidates and evidence

Nine candidates: `sdfampere010`, `011`, `012`, `013`, `014`, `015`, `020`,
`036`, `038` (all names have the `sdfampere` prefix).
`node-allowlist.json` retains exact names, October 26 raw-inventory paths and
hashes, normalized hardware, current Slurm observation and exclusion reasons.
The fleet's raw inventory includes results collected after the old summary
table was written; this selection reads those raw records, not stale PENDING
entries in that summary. Missing or malformed records cannot qualify a node.

These candidates have inventoried four A100-SXM4-40GB GPUs on the same PCI/NUMA
layout, dual EPYC 7542 CPUs with eight NUMA domains, driver 575.57.08, two
ConnectX-6 Ethernet functions plus two Intel Ethernet functions, and two-drive
Gen4 x4 RAID0 local scratch. Other NIC layouts are conservatively excluded.
The auxiliary Intel pair is X540 on 010–015/020 and X710 on 036/038; the
ConnectX-6 functions occupy the same PCI addresses. Runtime compares each node
against its own inventory. This does not assert identical Weka routing or NIC
throughput, which must be observed within the selected allocation.
010–015 and 020 use Kioxia KCM6XRUL7T68 local drives; 036/038 use Samsung
MZQL27T6HBLA-00A07 drives. This is not a claim of equal storage performance.

The scope remains **cold host-page-cache Weka input**, never NVMe input. Local
scratch holds untimed cube exports only. SSD/queue differences can affect export
and whole-job time; record them and do not pool historical rates across nodes.
A later NVMe-input campaign needs a separately reviewed storage-matched cohort.

## Scheduling and runtime checks

Slurm `--nodelist=a,b` requests both nodes; it is not an OR candidate list.
The reviewed command therefore uses `--nodes=1` and `--exclude` for the
complement of the frozen Ampere partition membership. The existing four-GPU,
exclusive, six-task, 192-GiB, 24-hour resource request remains unchanged.
Command generation re-reads the partition membership and fails if it changed;
regenerate the command with frozen `prepare.py command` immediately before an
authorized submission rather than blindly using old command text.

The runtime allowlist remains authoritative if the partition changes after
submission. An unexpected node fails before staging; no automatic retry or
node substitution is implemented. Slurm must report exactly the selected host,
the process must be on it, and whole-node exclusivity/sole occupancy must pass.
All MPI rank receipts, GPU UUID assignments, CPU affinities, output records and
expected-node environment settings refer to that actual selected node, not 036.

Before any input staging, require live CPU topology, exact GPU UUID inventory,
GPU PCI/NUMA/max-link topology, driver and NIC layout to match that candidate.
Record full hardware and the qualification receipt. Existing local filesystem,
capacity, four-GPU visibility/model/memory, physical-core, cold-Weka NIC traffic,
GPU regression, numerical acceptance and pilot gates remain mandatory. Historical
inventory does not establish current NIC speed, bandwidth or local free space.

No stage, event count, batch, worker, bounce, input budget, CUDA stream, runtime,
calibration or timing budget changes. Forty MPI cases still execute sequentially
on one node. The unmeasured reference/index timing risks IMPL-R2-001 and
IMPL-R2-002 remain deferred; a review pass is not a completion guarantee.
