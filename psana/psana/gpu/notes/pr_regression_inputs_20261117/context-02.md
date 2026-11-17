# Round2: narrow input preparation recovery
Reviewed SHA199b5091c0482d015609ed728efa1aaaab0240ab
Prior implementation df46718002367e56a7bca5c0d1d844342df2493c passed round1.
Public slac-lcls/lcls2 repository (API visibility public verified). Read only
committed code and compact metadata, no payload/credentials/private files.

User goal remains prepare real run51 inputs. First job61478366 hit30min timeout
before final3streams completed verification. All bigdata copied, streams3/8 and
SMD verified; no completion receipt, so entire input set invalid. Retain failed
attempt, no resume/overwrite. Time extension attempted too late, rejected because
job already ended. See attempt-01-failure.json. New attempt has45min limit and new
destination, same1GPU unused/8CPU/32GiB/premium lcls_g. No job submitted yet.

Only production logic change: prepare_inputs.py opens sparse header-check reads
unbuffered and calls POSIX_FADV_RANDOM, leaving full copying/hashes/reference and
all header/gap checks unchanged. First job read counters exceeded source+target
full scan bytes; avoiding readahead is a plausible improvement, not measured.
No waived checks.55offline tests pass again; bashsyntax/diffcheck pass.

Review latest prepare_inputs.py plus prepare-02.sbatch, scope.md, validation-02,
response-01 and review-01 under psana/psana/gpu/notes/pr_regression_inputs_20261117.
Also review new audit_prepared_inputs.py (metadata/receipt audit after successful
job; not yet run) for code/provenance/hash/stat/private/layout/Slurm verification.
No large data rehash on login; only actual compute-job evidence accepted.

PREP-001 handled by documented pre/post verification of frozen files against Git
blobs and exact inventory, readonly bundle; keep raw receipts to substantiate.
PREP-002 optional non-L1 timestamp check remains deferred. PREP-003 audit checks
both SMD and bigdata stripe1/1MiB. PREP-004 addressed with preserved timeout and
new longer bounded allocation. Evaluate evidence claims without claiming success
for attempt2. No framework/profile/qualification changes. No nested review/tools
beyond file reading. Return findings at exact currentSHA.
