# PR299 Perlmutter execution archive

This is the **historical execution harness**, not a new portable benchmark or
part of the production psana runtime. The files in `harness/` and `analysis/`
are byte-preserved selections from the completed Perlmutter campaign. They
were not rerun to prepare this evidence commit.

- [Reviewed results and limitations](../../notes/pr299_perlmutter_20261115.md)
- [Compact evidence](../../notes/evidence/pr299_perlmutter_20261115/)
- [Source-to-archive mapping](../../notes/evidence/pr299_perlmutter_20261115/archive-map.json)

## What was run

Baseline `11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba` and head
`acfd029a8d27898b5f165e02facd3c3bbc9c4ad2` were built locally from separate
scratch source/build/install trees. Job 61375111 completed 12 preflights and
48 timed samples of the first 10,000 events in mfx100848724 Run51. The six
cases are GPU events, CPU events, and CPU nested steps, each with one/two BD
workers. Each case has four adjacent baseline/head pairs with alternating
execution order. Run77 and the C1 fix were not included.

The original task root was:

```text
/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6
    source-base/       source-head/
    build-base/        build-head/
    install-base/      install-head/
    harness/
    manifest.json      campaign-plan.json
    pilot-estimate.json
    review-approval-campaign.json
    results/reference-51.json
    results/pilot-job-61373624/
    results/campaign-job-61375111/
    evidence/          logs/          review/
```

`harness/campaign-job.sh` contains the submitted Slurm configuration and
invokes `python -B "$TASK_ROOT/harness/perlmutter_screen.py" campaign`.
The orchestrator runs `worker.py` through `srun --mpi=cray_shasta`, with
three/four ranks, eight physical cores per rank, one GPU per step, and
`--gpu-bind=none`. It sets separate install prefixes for each variant and
checks the source, build options, install hashes, input reference, completed
sizing evidence, and campaign review approval before launching. Per-sample
`command.json` files retain actual argv, timeout, and selected environment.

| Script | Purpose |
| --- | --- |
| `harness/campaign-job.sh` | Campaign Slurm wrapper and environment activation |
| `harness/perlmutter_screen.py` | Frozen-input checks, review gate, preflights, pairs, and process status |
| `harness/make_campaign_plan.py` | Bounds derived from completed local sizing samples |
| `harness/worker.py`, `harness/common.py` | Workload, independent SMD reference, event and step validation |
| `harness/check_runtime_provenance.py`, `harness/perlmutter_support.py` | Install/library, MPI, and runtime checks |
| `harness/gpu_identity.py`, `harness/io_accounting.py`, `harness/storage.py` | Placement, completed-read accounting, and monitoring |
| `analysis/summarize_campaign.py`, `analysis/analyze_storage_progress.py` | Paired and storage/progress summaries |
| `analysis/analyze_campaign.py` | Historical driver producing campaign-summary.json and campaign-storage-progress.json from original sample.json files |
| `analysis/campaign_validation.py`, `analysis/plot_campaign.py` | Contract checks, sustained windows, CSV, and order plot |
| `analysis/write_final_report.py`, `analysis/preserve_evidence.py` | Historical report generation and compact snapshot preservation |

The other harness helpers and tests are retained to preserve dependencies.
`screen.py`, `nic.py`, and `test_nic.py` are SDF-era helpers. `job.sh` is the
earlier Perlmutter pilot/generic wrapper with a 30-minute allocation; it accepts
a campaign argument but was not this campaign's entry point. The entry point
was **`campaign-job.sh`**. Neither archived wrapper should be submitted as-is.

## Offline verification of this package

From the repository root, using Python 3.9 or newer:

```bash
python3 -B psana/psana/gpu/scripts/pr299_perlmutter/verify_archive.py
python3 -B -m unittest discover -s psana/psana/gpu/scripts/pr299_perlmutter -p test_verify_archive.py
```

`verify_archive.py` and its tests were added during packaging; they are not
part of the measured harness. Verification reads files only and requires no
GPU, MPI, psana installation, dataset access, scheduler, or network. It
checks package/source checksums, approval binding, sample contracts, paired
timings and order, provenance, sustained progress, and storage aggregation.
It cannot reproduce physical storage measurements or discarded event arrays.

Three source-format exceptions are intentionally retained: the original
`plots/paired-timings.csv` uses CRLF, `logs/slurm-61375111.err` ends with a
blank line, and `provenance/build-environment.txt` contains a space-only line.
These produce `git diff --check` warnings. Their source hashes are verified;
normalizing them would break byte-preserved provenance. Newly authored files
must pass the whitespace check without these exceptions.
The original build-options JSON files also omit a final newline. The
`trailing-whitespace` and `end-of-file-fixer` pre-commit hooks must not rewrite
these archived inputs; those two hooks are skipped for archive commits after
the scoped whitespace and source-hash checks. Other hooks remain enabled.

## Reuse is not automatic

Do **not** submit the archived batch scripts directly from this checkout.
They contain original absolute paths, account/environment choices, and
historical approval hashes. `common.ROOT` expects the original task layout.
Analysis scripts expect original `sample.json` and sometimes `storage.jsonl`,
not this archive's `sample-summary.json`; several write reports to fixed
locations. The frozen harness tests also include site/layout assumptions.
Their archived 94-test result is historical, not a claim that those tests
were rerun here.

A future run needs a new scratch task root, accessible inputs, clean builds,
reviewed site configuration, and a new manifest/approval. Preserve this
archive; adapt a separate copy rather than changing historical provenance.
No such adaptation or new performance job is part of this commit.
