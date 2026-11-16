# Perlmutter transfer: reproduce the selected Sim10 configuration

Prepared November 15, 2026. This is a task/source transfer, not a runnable
Perlmutter campaign or permission to submit compute jobs. No SDF job is
submitted or cancelled by this transfer. Use an independent implementation
review loop for the port, then report the bounded submission plan to Mona.

## Objective and evidence boundary

Establish a **new Perlmutter single-node baseline**, first for staging and then
for S/E/EC/ECC on the same allocated node. Preserve the accepted input and
selected application controls. Do not restart broad parameter sweeps or
multi-node scaling before this baseline passes correctness and result review.

The SDF parameter report's best-supported measured staging configuration is
**four BDs sharing one GPU**, batch 200, four KvikIO workers/BD. Separate SDF
cohorts measured 1,635.45 events/s (5.1150 GiB/s) and 1,529.58 events/s
(4.7839 GiB/s); these are not a pooled mean or Perlmutter performance targets.
The report's throughput-oriented one-BD control uses **eight workers**, batch
50. It is distinct from Mona's newer, deliberately fixed-four-worker comparison.

The subsequently prepared compute campaign uses **one BD/one GPU, batch 50**
and **four BDs/four GPUs, batch 200**, both with **four workers/BD**. These
compute configurations have passed SDF implementation review, not GPU execution
or full-shape memory acceptance. Do not call them measured compute optima.

Recommended receiving sequence:

1. Reuse today's qualified four-BD/one-GPU **staging-only** result below as the
   bridge. Confirm it with clean repeats in the new same-node comparison, not
   a duplicate build/qualification campaign or a new stripe sweep.
2. Establish S staging controls for the selected one-BD/one-GPU and
   four-BD/four-GPU profiles, keeping four workers/BD fixed.
3. Add E, EC, ECC for those two selected profiles. Keep all comparisons within
   one exclusive four-GPU allocation if the reviewed walltime permits.
4. An eight-worker one-BD staging control is optional and separately labelled;
   it must not silently replace the agreed four-worker compute configuration.

Do not run four full-cube BDs on one 40 GB GPU just because staging fit. All
profiles, repetitions and ordering need a fresh bounded Perlmutter manifest.

## Frozen controls and semantics

| Control | Value |
| --- | --- |
| KvikIO workers / CUDA streams | 4 / 1 per BD |
| Task / bounce per worker | 1 MiB / 16 MiB |
| Bulk target / framework input ceiling | 512 KiB / 6 GiB per BD |
| SMD batch | 1000 events |
| KvikIO | 26.10, compatibility ON, auto-direct read OFF; not GDS |
| GPU sharing of derived constants / geometry | PS_JF_SHARE_DERIVED=0 / PS_GEO_SHARE=0 |
| Main profiles | 1 BD / 1 GPU / batch 50; 4 BDs / 4 GPUs / batch 200 |
| MPI roles | one smd0, one EB, then one or four BD ranks (3 or 6 total) |
| Main full cell | 100,000 globally unique events, not 100,000 per rank |
| Bytes per event | 3,358,200 input payload bytes, five datagrams |

S = seed staging/delivery; E = S plus expansion to 32 uint16 512x1024 panels
(32 MiB/event); EC = E plus offset-enabled CPU-v3-equivalent FP32 calibration
(64 MiB/event), without masking/common-mode; ECC = EC plus a persistent dense
100-bin FP32 cube, 99 active bins and one empty bin, global event ordinal modulo
99. This is the synthetic MFX fixture for the XPP workflow, **not scientific
XPP-coordinate binning**. Framework budgets are not whole-process GPU limits.

Keep images device-resident with keepalive, no ordinary host-image publication.
Only bounded uint32 sim10.errors publication is allowed (64 KiB/callback,
64 MiB pinned-output cap). Preserve single-stream ownership and drain checks.
ECC reduces one bin at a time in fixed BD-rank order into FP64 on rank 0;
do not gather entire cubes. Keep full-image CPU-v3 checks for acceptance/pilots,
fixed sampled pixels in every panel/bin using every event for full runs, and
all-pixel empty-bin checks. Count events/timestamps and bytes independently.

Payload bandwidth = events/s * 3,358,200 / 2**30 GiB/s. Label it logical payload
bandwidth, not NIC/OST traffic or expanded/calibrated image bandwidth.

## What is in the package

- `sdf-attempt03/`: exact 77-file frozen source/dependency snapshot and original
  manifest/preparation receipt. Treat all historical status and paths as SDF-only.
- `reference/notes/`: maintained compute scripts, staging/bounce and storage
  scripts/tests, latest review receipts, parameter report/heatmaps and fixture
  transfer provenance. The preserved relative directory structure supplies the
  SDF harness dependencies. `latest/notes/` holds maintained documents that
  overlap frozen historical documents; use its newer status/review rather than
  historical status text. Neither reference tree changes the original snapshot.
- `calibration/`: original stage1 manifest and exactly three Jungfrau binary
  constants (pedestals, pixel_gain, pixel_offset), 192 MiB each. Their original
  relative names and SHA256 values are retained. Other manifest entries describe
  the larger source snapshot and are deliberately not copied.
- `sdf-native-compatibility.json`: historical provenance only, NOT a portable
  native install receipt. No SDF native binaries or virtual environment copied.
- `PACKAGE.json`: every payload file's bytes, SHA256 and origin, plus identities.
- `verify_package.py`: standard-library-only, read-only transfer verification.
- `PROMPT.md`: receiving-agent prompt.

Authoritative SDF attempt03 manifest SHA256:
`03bcd8121c9919cea0623761c0006add3c57c6658e5aa1b9fff71951a3ab4d0d`.
Independent amendment review SHA256:
`404a71f4824c701381ceab97894593b9e59f1094a89b89f50b75929d08f8f5f4`.
Pinned psana PR298 commit: `67c327a148a08b2aaf972e67e1e02e367c2b298f`.
SDF coordinator HEAD: `004b30e01` on `work/synthetic-mfx-scan`.
The in-scope untracked scripts are included explicitly, not inferred from HEAD.
Unrelated dirty `notes/mfx_expand10_single_node_baseline_20261108.md`, other
untracked work, credentials, large benchmark artifacts and native builds are
excluded. No branch, commit, existing environment or core code changed.

## Already on Perlmutter: reuse, do not overwrite

Access and scratch were checked successfully at approximately 15:01-15:06
Pacific on November 15. Verified `$SCRATCH` is `/pscratch/sd/m/monarin`, owned
by monarin. Quota reported 3,433,712,652 KiB used against a 21,474,836,480 KiB
soft quota. Recheck quota and compute-node visibility before large outputs.

Dataset:
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`

Five original JF streams s005..s009, five SMD files and unchanged generation.json
are present (11 files, 335,875,429,007 bytes). Do not regenerate, rename or
retransfer 313 GiB of data. generation.json describes the original ten-stream
SDF fixture and has SDF paths: preserve its bytes, and use the supplied transfer
manifest for deliberate five-stream selection and relocation.

The receiving task has **already independently checked all eleven SHA256s**,
Configure/Names/panel order, every SMD timestamp/offset, and full-100k DataSource
traversal in job **61065065**. This supersedes the older SDF transfer document's
pending-bigdata-hash statement. Read the current completion and input-stat
receipts before deciding whether identities still match; a historical success
does not guarantee unchanged files today. Revalidate traversal under the new
runtime, and do any required large rehash in an allocation, not on a login node.

Existing receiving task:
`/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/COMPLETION.json`
and `CONTINUE.md`. Its reviewed job 61065065 measured 100k one-BD staging
medians 2.195 / 0.969 GiB/s for one / two CUDA streams. These older settings
are not the new selected baseline and must not be rerun automatically.

Existing sources inspected, clean at the check:

- `/pscratch/sd/m/monarin/psana2-gpu/sim10-input-bookkeeping-20261110/source`
  at `3c3799829c659d739d5927511f86cc968a6fae6e`.
- `/global/homes/m/monarin/lcls2_worktree/benchmarks/sim10-stream-length-20261109`
  at `543240a9334c03a186263d1967f0d38cfe9d8056`.

**Both already contain the pinned PR298 Git object.** A clean, exact pinned
scratch worktree and native build also already exist:

- Source: `/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/source`.
- Install: that source's `install_psana/` (do not overwrite it).
- Build receipt: `/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/build-provenance.json`,
  recording PR298, the install prefix, installed/source hashes and build-log hash.
- Native qualification: `/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/qualify01/`,
  job 61359629. Read its results/audits before reusing the build; this transfer
  inspected the reviewed results summary, accounting and provenance, not all
  installed hashes or CUDA itself.
- Current receiving guidance: the receiving clone's
  `notes/psana_striping_pr298_20261111/execution-20261115/` and
  `selection-20261115/`. The parent README is an older pre-execution plan;
  do not treat its "no build/job" statement as the current state.

Inspect these first: they can avoid a rebuild and a fresh source transfer.

**Important: the selected profile has already run once on Perlmutter today.**
The current `execution-20261115/RESULTS.md` records job61359629 on nid001264,
completed 0:0 in 705 seconds, with independent result review PASS at
`543240a9334c03a186263d1967f0d38cfe9d8056`. Its full count2 sizing case used
four BDs sharing one A10040GiB, four workers/BD, batch200, one CUDA stream,
1 MiB tasks, 16 MiB bounce, 512 KiB bulk, 6 GiB/BD, compatibility ON/direct OFF:
**100,000 events / 61.3367 seconds = 1,630.35 events/s, 5.0990 GiB/s**.
It used a pre-existing **two-stripe, 1 MiB stripe-size** Lustre view, not the
original dataset's default single-stripe layout. Whole case time was 113.8344s.
All31 GPU integration tests passed, all30 stripe-copy files were checked,
six binding cases and a1003-event tail witness passed, and six-rank affinity
used34 disjoint physical cores (3,3,7,7,7,7). This is **one full sizing
observation**, not three repeated performance cells or a best-stripe result.
Cache evidence is local-page-cache cold; backend/wire traffic remains unverified.

Use this qualified count2 view as the initial PM baseline candidate after
reading `inputs.json` and the actual-open identity/relocation receipts. Hold
its stripe layout fixed across the new comparisons. Do not silently switch
between count1 and count2, select a different existing copy by its directory
name alone, or mutate original files. Preserve the separately reviewed
18-case stripe-selection plan; do not submit or rerun it as part of this
compute-baseline transfer. Full compute acceptance remains new work.

The canonical `~/lcls2`
did not resolve that object in our check. Use a suitable existing task-owned
scratch worktree after checking status, or obtain approval for a new detached
worktree from a repository that has the pin. Do not reset or repurpose existing
worktrees, copy a tree as a replacement for Git worktree creation, or silently
switch the baseline to a newer native runtime.

Also inspect existing Perlmutter `sim10-input-bookkeeping-20261110`,
`sim10-striping-20261110`, and `sim10-trace-20261110` artifacts and the receiving
clone's notes. They may supply native runtime, Cray MPI, cache-conditioning,
audit and filesystem-counter adapters. Their newer code is a porting reference,
not permission to mix runtime revisions into the baseline.

## Required adaptations before execution

Read local `scratch-workspaces`, `psana-perlmutter`, `psana2-gpu`, and
`implement-review-loop` skills plus applicable AGENTS.md files.

1. **Runtime/build:** verify PR298 Python and native source compatibility,
   installed module paths, CuPy/KvikIO source identity, CUDA/driver and MPI.
   Existing Perlmutter native builds may accelerate bootstrap only after that
   check. The SDF inherited psana/meson.build mismatch is documented evidence,
   not a waiver for this port. Build on login only when needed, with isolated
   scratch build/install/cache paths using supported local helpers; preserve
   active environments. Read `~/psana-nersc/activate_psana_build_env.sh` and
   `~/activate_psana_gpu.sh` for the actual supported path activation. Keep
   SIT_PSDM_OFFSITE unset. No live calibration lookup may replace frozen arrays.
2. **Scheduler/affinity:** replace Ampere partition, node allowlist/exclusions,
   Open MPI rank environment and mpiexec commands with reviewed Perlmutter
   mappings. Query `srun --mpi=list`, prefer supported `cray_shasta`, request
   `-A lcls_g -C gpu -q premium -N 1 --gpus-per-node=4`, exclusivity and a bounded
   walltime. These are resource guidance, NOT a ready sbatch command. Check Iris
   account/personal balances before any authorized submission and alert below
   20%. Discover physical cores and GPU/NUMA topology live; rederive disjoint
   masks rather than copying EPYC7542 masks. SDF used 13/34 bound cores for
   four-worker BD1/BD4, not evidence that the same masks fit Perlmutter.
   Keep smd0/EB out of BD GPU assignment; verify actual per-rank UUIDs. Use
   --gpu-bind=none only with validated explicit BD mapping/visibility.
3. **Storage/cache/counters:** replace Weka mount/NIC assumptions with Lustre
   evidence, and record existing stripe count/size and OST layout without
   mutating canonical input files. Record client cache residency and available
   Lustre/Slingshot traffic counters. If counters are unavailable, explicitly
   label remote traffic unverified; never fabricate Weka-equivalent NIC proof
   or claim cold servers. Recondition after reference/setup reads, outside
   timing. Prefer proven owner-scoped cache adapters from the earlier PM run.
   Do not evict another task's input pages during an active run. Use a separately
   qualified owned copy if conditioning requires isolation; budget space first.
4. **Paths/manifests:** port plan.py/reference.py/support.py/storage.py/runtime
   and launcher dependencies as necessary, not a blind global path substitution.
   Preserve the received snapshot. Emit new PM input, calibration relocation,
   runtime, hardware, budget and source manifests. SDF paths, inode assertions,
   readonly native hashes and review wrappers are not executable PM receipts.
5. **Memory/output:** discover A100 model/capacity (do not assume 40 or 80 GB),
   CPU RAM, and writable storage on the allocation. Retain measured physical
   headroom of at least 4 GiB beyond demand and aggregate per-device accounting.
   The batch200 planning subtotal (~31 GiB) is NOT a measured fit. No automatic
   smaller-batch rescue. Perlmutter local NVMe is not assumed: use qualified
   shared scratch for outputs if needed, recording export-I/O interference.
   The SDF plan retained ten ~12.5 GiB merged FP64 cubes plus allowances; budget
   ~150 GiB output capacity separately from the input, or review an explicit
   retention amendment. Do not silently drop exports or delete existing results.

## Review, acceptance and timing

The transferred SDF implementation had 84 passing host tests, eight host
native/setup checks and independent Claude PASS. **No SDF CUDA/MPI full-cube
case ran.** These receipts do not certify the PM port. Preserve mandatory
numerical, malformed-descriptor, transition, ownership, byte-count, weakref,
memory, publication and drain gates. Re-run appropriate tests on the native PM
stack. Keep code review separate from GPU acceptance and result review.

The SDF design contains historical 31 GPU tests/device, ten descriptor guards,
two ECC acceptance cases (201 / 1003 events), eight traced 4000-event pilots,
24 full 100000-event cases (three repeats) and six S anchors: 40 sequential MPI
cases in one allocation. This is a design reference, **not automatic permission
to copy a 24-hour SDF request into a costly PM job**. The additional staging
bridge requires its own accounting. Run/review a bounded bootstrap proposal
first and use measured timing to justify the full same-node campaign.

Two nonblocking SDF advisories remain unmeasured: ~500,000 ECC header-index reads
and bin-strided sampled-reference reads. Estimates of 120 seconds for index and
600 seconds for reference were not calibrated on compute; 1,980 seconds was the
enforced whole-full-case limit, not individual stage timers. Profile these
boundedly before committing the long premium campaign; fail closed on timeout.

Keep untraced processing throughput separate from traced pilot timing. Record
setup/JIT/constants, cache conditioning, processing/drain, independent reference,
hash/export, merge and whole-case/allocation time. Report per-rank and per-GPU
resource observations with sampling limitations. No cross-site pooled heatmaps,
GPU-only speedup claim for simultaneous BD+GPU changes, or empty-task rate
prediction for real expansion/cubes. Save compact accepted evidence durably;
scratch retention is not guaranteed.

## Next concrete step

Run `python3 verify_package.py .` from this package, inspect the existing native
receiving receipts and runtime, then prepare an isolated PM adapter plus a small
staging/correctness diagnostic for independent review. Report exactly what is
reusable, what changed, pending acceptance, expected cost/walltime and the
proposed submission command. Do not execute the shipped SDF run.sbatch,
submission-command.txt, prepare.py command, or cleanup-capable old campaigns.
