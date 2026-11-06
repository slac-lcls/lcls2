# Mikhail's exact 50-frame command

Command tested on real `mfx100848724` run 49, with all 32 Jungfrau panels:

```bash
jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 \
  --nrecs 50 --nrecs1 50 --stepnum 2
```

Each case ran from its own fresh directory under `/tmp/jf-exact-command`.
The harness invokes the CLI's actual `do_main()` with exactly these arguments.
It observes the application's unmodified callback and raw reads. This is the
serial stage-1 gate calculation, not stage-2 pedestal deployment or an MPI test.

| Source | PS_SMD_N_EVENTS | Result | Raw events from step 2 |
| --- | --- | --- | --- |
| Unfixed | Unset (MFX default 1000) | Fails: `UnboundLocalError` for `ievt`; no gates | 0 |
| Unfixed | 10000 | Completes; both gates saved | 50 |
| Fixed | Unset (MFX default 1000) | Completes; both gates saved | 50 |

The two successful cases select precisely physical step 2's first 50 timestamps,
in order, as established by the earlier independent direct-SMD-file oracle.
Their full `(32, 512, 1024)` `gate_lo` and `gate_hi` arrays are exactly equal,
with zero differing pixels. `results.json` records the checked results and
observed chunk/callback counts. `commands.json` records exits and elapsed times;
these times are observations, not a performance benchmark.
Per-case `chunks` entries are SMD view sizes including transitions, not L1 event
counts: the 1000-event chunks have sizes 1011–1013, with a three-datagram tail
in the failing case. The analyzer's exact chunk/callback-count assertions were
added after observing these runs and apply to this dataset and parameter set;
the timestamp oracle and gate comparisons provide separate correctness checks.

At the default, the unfixed callback is recreated for each chunk and never
recognizes physical step 2 as local step index 2. Its four invocations select
nothing; the application's empty-event logging then references uninitialized
`ievt`. With 10000 this 3000-event run fits one SMD chunk, avoiding callback
restart. The fixed callback retains its iterator state across chunks and works
with the MFX default. Increasing the chunk size is a workaround for this run,
not a general fix for larger runs or other chunk boundaries.

## Source isolation and limits

- Unfixed: `71378658a54dcdd941c9137355a985fa4b637595`, detached checkout
  `/tmp/jf-callback-ab/unfixed`.
- Fixed: `b59c2fd1aad012de195b14b62401bc966e6755c3`, separate detached checkout
  `/tmp/jf-exact-command/fixed` (production fix unchanged from `a64df64d9`).
- Separate Python import overlays; same Python 3.9 environment and unchanged
  locally compiled extensions. This is source/build-overlay isolation, not a new
  native compilation. The source pair differs only in Python core/test files;
  detector, application, and native source files are unchanged between versions.
- Per-case JSON records resolved source paths and SHA-256 hashes for the CLI,
  detector calculations, datasource, callback machinery, and native extensions.
  The analyzer checks source roots and shared native hashes.
- These are our controlled before/after builds running his exact command, not a
  reconstruction of Mikhail's complete installed environment or linked revision.
- No production code changed in this experiment. Earlier callback core and
  byhand tests remain applicable; those suites were not rerun for these notes.
  The deferred long-no-yield buffering issue remains outside this experiment.

## Protection and evidence

All generated gates and application logs stay in the case's `./work1`. Keeping
that literal output argument also suppresses RepoManager's central startup log.
The harness forces `add_shortname=False`; all lookups returned the existing
`jungfrau_000005`. Calibration HTTP post/put/delete helpers raise if called;
none were called. No deploy or submission flags are present. Logbook file
discovery is stubbed to use local on-disk discovery, as in the earlier A/B test;
all cases used the same ten SMD streams. No calibration database updates occurred.

Raw logs, gates, and reports remain under `/tmp/jf-exact-command`. The independent
oracle is `/tmp/jf-callback-ab/oracle.json`; its generation script is
`../jf-workflow-ab/prepare.py` (direct-file `DataSource`, bypassing SMD callbacks).
`artifact-manifest.json` records their sizes and SHA-256 hashes. Compact reports
are committed here; temporary large artifacts are not durable repository storage.

The scripts are experiment-specific and use Mona's paths. Reproduction requires
the pinned checkouts, the prior source/native overlay template and oracle, access
to run 49, `source setup_env.sh`, and fresh case/overlay directories. Run
`run_matrix.py`, then `analyze.py`. Each job is sequential and limited to 600s;
existing output directories are rejected to prevent accidental reuse.

Independent review status and exact reviewed SHA are in `status.json` and
`round-01/`. Review passed at `b6628b0735a0723af5e2c9897d6b42605c943690` with no
blocking findings. The subsequent commit records review responses and these
documentation clarifications; the executed and reviewed scripts are unchanged.
