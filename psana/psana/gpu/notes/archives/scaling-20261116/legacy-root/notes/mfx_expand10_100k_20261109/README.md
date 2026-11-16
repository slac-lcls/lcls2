# 100,000-event sim10 fixture

Generation and all CPU validation passed as Slurm job **44025856** on sdfmilan271
in **42m26s**, exit 0:0. All 20 files passed readback, and DataSource traversed
one run/step containing 100,000 events with 3,200,000 panel-layout checks and
1,280 seed-value witnesses around repetition boundaries. Independent code review
passed with no blocking findings on `186ac6c299a4869afdbf169f8266bccb5331b864`;
results review passed on `450bde48002464f69a0917917908f58ae613cdd9`.
The fixture is **accepted**; [review findings and responses](review-round2/responses.md)
record optional hardening deferrals. The [scope](scope.md), [status](status.json),
[verification](verification.json), [file receipts](files.json) and
[job accounting](slurm-accounting.txt) record this user-authorized fixture.

Keep **gpu_bulk_target_bytes=512*1024** for subsequent performance tests.
[Small-datagram scheduling work](../../psana/psana/gpu/docs/proposals/small_dgram_scheduling_followup.md)
is deferred: one live small group per file restricts read-ahead, and group
boundaries force changes to execution size. No scheduler code changes here.

The fixture physically repeats the accepted 10,000-event sequence ten times,
retaining ten streams (five JF plus five ancillary) and all 32 panel identities.
There are 100,000 unique outer event timestamps and 10,000 source payload events.
Nanosecond arithmetic shifts outer timestamps; SMD offsets/sizes are rewritten.
All other header and payload bytes are preserved, including internal detector
timestamps/frame counters, which repeat intentionally. One native step spans the
output, with no inserted step boundaries. The manifest uses `plan` to describe
repetition source indices/cycles, rather than the old fixture's `schedule` key.
Later benchmark scripts must use this new manifest and accepted file receipts;
the old campaigns deliberately pin the 10k fixture and cannot be reused unchanged.

Target directory:

```
/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_100k
```

Evidence: same parent, `sample10_100k_evidence_20261109`.
Native log: `sample10_100k_logs_20261109/44025856.log` in that parent.
Source experiment/run identity remains mfx101555026/run 10; use explicit `dir=`.
No public-run registration or calibration DB writes. Full-shape frozen constants
retain their panel mapping; this campaign skips calibration loading and does not
claim a fresh scientific calibration or GPU correctness test.

[Preflight](preflight.json) and verified output agree on **376,806,534,528 bytes (350.928 GiB)** across
20 files, and 100,847 datagrams per stream. JF event datagrams account for
**335,820,000,000 bytes (312.757 GiB)**. Physical files contain all repetitions;
there are no sparse aliases or loops over the old file. Total volume and run
duration grow 10x. At batch 100, per-batch JF input remains about 320 MiB, so this
tests total working set/duration rather than a larger outstanding-input window.
A subsequent cold-FFB benchmark still needs FFB staging, cache evidence and NIC
counters; generating project-scratch data does not establish read bandwidth.

Validation includes four unit tests (second rollover, partial/complete cycles,
independent schedules and malformed inputs), a [23-event pilot](pilot-verification.json),
full output hashes and byte preservation against the accepted base, all SMD
references and aligned strictly ordered timelines, and serial DataSource
traversal of all 100k events with all 32 panel layouts checked each event.
DataSource seed-value witnesses check the first/last two events of each cycle;
full raw-byte preservation is checked separately for every datagram.
Validation shares base-index and XTC helper code with the writer; schedule
derivation and byte comparison use separate code paths. This establishes
consistency against the accepted base, not an adversarial independent validator.

Native validation and independent review both passed. The native job initially
left `accepted=false` with `validated_review_pending`; coordinator closure records
`complete/accepted` in the evidence status and [acceptance receipt](acceptance.json).
A nonterminal status paired with a terminated Slurm job means interruption,
not successful validation; inspect accounting and partial files before recovery.
The writer refuses existing output and evidence directories. Do not rerun into
partial files or delete them merely to obtain a clean status.
