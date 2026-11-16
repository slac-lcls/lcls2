# Continue on Perlmutter: controlled Sim10 cold staging comparison

The user asked to prepare a clone and transfer this task, then let their
Perlmutter agent implement and run the rest. Do not resume the SDF allocation.
SDF job44043617 was cancelled while PENDING on 2026-11-09 at16:05:24 Pacific;
it never obtained a node and produced no performance data.

## Current status: Sim10 data is already on Perlmutter

Use the transferred 100k Sim10 fixture at:
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`

Globus task `a8426223-dca2-11f1-aa1a-0affd629e535` SUCCEEDED on November 9,
2026 at 17:18:29 Pacific. All 11 files (335875429007 bytes, 312.808 GiB) arrived
with checksum verification enabled and zero faults. Destination sizes match;
independent SHA256 checks passed for all five SMD files and generation.json.
Independent bigdata SHA256 rereads and Perlmutter runtime acceptance remain.
No local Sim10 generation or additional transfer is needed for this comparison.

Transfer receipts, expected hashes and selection/relocation manifest:
`notes/sim10_globus_transfer_20261109/` in this repository, also copied to
`/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/transfer/globus-20261109/`.
The task-root `TRANSFER_UPDATE.md` records the same completed transfer.

## Objective and repository

First investigate one-BD stream count and input duration, before further BD,
batch-size or worker tuning. Run one BD/one GPU with 1 versus2 CUDA execution
streams and 10k versus100k events, two repeats each in reversed order, plus
small pilots. Use the SAME accepted100k Sim10 files for both event counts.

Prepared clone location:
`/global/homes/m/monarin/lcls2_worktree/benchmarks/sim10-stream-length-20261109`

Receiving branch: `work/perlmutter-sim10-stream-length`.
Source branch: `monarin/work/synthetic-mfx-scan`.
SDF benchmark implementation commit: `21f56de8884dc386d2d0ae65d159a795ed448614`.
Prior accepted scaling report: `notes/sim10_100k_bd_sweep_20261109/README.md`.
Task scripts and this transfer are in `notes/sim10_stream_length_20261109/`.
Use the clone HEAD as the transfer commit; review metadata distinguishes code
review from documentation-only closure commits. Existing Perlmutter checkouts
and their dirty files must remain untouched. Unrelated SDF untracked work was
not transferred.

Scratch task root:
`/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109`

Put builds, installs, generated data, compiler caches, logs and reviewer worktrees
under this root with unique attempt directories. No Perlmutter job is submitted
by this transfer. The clone contains source and provenance, not a new native build.

## Measured context and interpretation limits

SDF cold Weka,100k Sim10,2 streams/BD:1/2/3/4BD medians1.189/2.356/3.479/4.502GiB/s.
Earlier SDF1BD/1stream/10k was2.1884/2.1197GiB/s on another node and separate10k
files. None of these measurements isolates stream count, event count or node.
4.50GiB/s is the best tested SDF staging rate, not a measured hardware ceiling.

Perlmutter `/pscratch` is Lustre, verified by findmnt. This is a new within-site
controlled test; it cannot directly attribute the earlier SDF discrepancy.
The transferred fixture keeps the source run and data constant. Changes in
CPU/GPU/platform, filesystem and CUDA/runtime versions still preclude calling
it a direct Weka reproduction. Do not label Lustre as Weka.
The same-file10k prefix tests duration, not physical10k-versus100k file layouts.

## Transferred dataset and next steps

The fixture is mfx101555026 run 10, with five physical JF streams s005..s009,
32 panels, five matching SMD files under smalldata/, and unchanged generation.json.
These are the exact JF files used in the accepted SDF 100k staging sweep. Use
explicit dir= pointing to the destination above and preserve the filenames.
Use a 10k event prefix and all 100k events from these SAME files for the matrix.

The original generation.json describes the complete ten-stream SDF fixture and
contains SDF absolute paths. Keep it byte-identical for provenance; use the
transfer-manifest.json for the five-stream selection and path relocation.
Validate this deliberate JF-only subset rather than requiring absent non-JF
streams. The earlier perlmutter-dataset.json and run-442 receipts describe an
unused fallback source, not the dataset for this benchmark.

1. Verify the five bigdata SHA256 values against the transfer manifest in an
   appropriate compute allocation. File sizes and SMD/manifest hashes are
   already checked; retain the original transfer receipt.
2. Revalidate Configure/Names, panel mapping, synchronized SMD timestamps,
   event count and offsets, then traverse the relocated subset with DataSource.
   Adapt absolute SDF path, inode/mtime and ten-stream assertions explicitly;
   do not reinterpret relocation or the deliberate subset as payload corruption.
3. Port the controlled runner and independent audit for Perlmutter, including
   runtime, scheduler, cache and network-counter changes described below.
4. Run small GPU pilots and then the 1/2-stream x 10k/100k matrix. Continue the
   implement-review-loop for implementation/fixes and review native results
   separately. No benchmark acceptance is claimed by the file transfer.

This is simulated compression: each panel stores raw.reshape(-1)[::10],
52429 uint16 words for an original 512x1024 panel, with shape/expansion metadata.
Repeated payloads in the 100k fixture are intentional. Generation is complete;
writer porting and creating new run-442 fixtures are outside the next steps.

The staging benchmark uses skip_calib_load='all' and no calibration-consuming
inputs, so no calibration DB deployment is needed. Keep source/calibration
provenance without requiring unavailable SDF calibration snapshots to perform
this staging-only test. Expansion/calibration/cube correctness is later.

## Runtime and scheduler adaptation

An existing Perlmutter frozen B install is available:
`/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01/install-B`
Environment candidate:
`/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env`
Eight Python module hashes in `perlmutter-runtime.json` match the SDF baseline.
This is source identity evidence, not native/GPU/MPI ABI or complete-runtime
validation. Recheck imports, CuPy/KvikIO versions and CUDA/Cray MPICH on compute.
Existing bootstrap/build scripts in the older attempt are references only:
do not write its tmp, cache or install paths. Use new task-owned paths.

Use skills psana-perlmutter, psana2-gpu, gpu-cuda-python and implement-review-loop.
Build on login if needed with the checkout's build_psana.sh, explicit scratch
build/install prefixes; inspect --help for supported options. Source
~/goodstuffs/bashrc and ~/psana-nersc/activate_psana_build_env.sh as documented.
Never overwrite another checkout's native install. No build was needed for
preparing this source-only transfer.

Replace OpenMPI OMPI_COMM_WORLD_RANK in runner.py with a verified Perlmutter
rank mechanism (e.g. SLURM_PROCID under srun), preserving CUDA mask setup BEFORE
MPI/psana/CuPy imports. Use srun with verified Cray MPI support, explicit tasks
(SMD0+EB+1BD), GPU allocation/binding, one physical GPU UUID for the BD and empty
CUDA masks for helpers. Maintain constant CPU affinity/NUMA policy across cases;
do not copy SDF CPU IDs or topology assumptions blindly.

Fixed intended controls: batch100, SMD1000, bulk524288B,6GiB GPU budget,
KvikIO compatON,8workers,1MiB tasks,16MiB bounce, automatic direct I/O OFF,
PS_FAKESTEP_FLAG=0, PS_PARALLEL=mpi,1EB,0SRV. Record actual GPU memory size/model;
recompute admission/subbatch counts on the new input. Task receives field
metadata only: framework read/walk/descriptor preparation, no dense gather,
expansion, calibration, cube or payload D2H.

Use account lcls_g and premium QoS per standing preference; check Iris personal
and project charged balances before submission and alert if either <20%.
This transfer does not launch jobs; the receiving agent owns subsequent work.

## Cold methodology and acceptance

Use owner-controlled generated files on Lustre. Port the SDF-specific path/mount
allowlists explicitly. fsync/FADV_DONTNEED then mincore after DataSource setup
must establish zero resident input pages for local-cold labels. Never perform
global cache drops or fall back silently to NVMe. Lustre client/server caches
remain distinct; if eviction does not work, report it rather than label cold.
Inspect available Lustre llite/lctl statistics and Slingshot counters on compute;
Linux Ethernet sysfs RX counters alone may miss KFI/Lustre data traffic. Establish
counter scope/units and timestamp brackets before claiming network-served bytes.
The current physical_counters() helper aborts when ethtool physical-byte
statistics are absent; it does not degrade automatically. Replace physical_counters/
counter_delta or explicitly define an audited local-cache-only measurement mode;
fail closed on that mode's agreed evidence requirements. Sever/redirect the
network_watch -> plan_reads -> fixture_check imports that pin an SDF input ROOT,
as well as the SDF-only cache allowlist. If appropriate counters are inaccessible,
retain local-cache evidence and state that remote-traffic attribution is unverified.

Require exact ordered event identities, sizes/read counts, callbacks and
subbatch histogram, requested stream count exercised, proper GPU/CPU routing,
correct peer count, no pressure drains, complete resource retirement and cleanup.
Audit measured rates from actual bytes and loop time, not assumed seed pixels.
Record first/last wall-clock times and one job identity across cases if possible.

Carry code-review findings from review-01: fixed startup/fill/drain remains timed;
short cases can be more affected; reversed order does not equalize backend cache;
later10k cases follow full-file reads. Report individual repeats and available
traffic/payload ratios, not only medians. Preserve exact audit/test commands and
artifact roots. Original SDF24native-receipt tests have NOT run because the job
was cancelled. Port/review them for new receipts before claiming acceptance.
