# Response to Claude round 1

Reviewed code: `fa695530a92b322ca96efb157ac506be4d04bfe1`. Verdict: **changes_requested**. No code changed after review. The reviewer found no additional defect in the iterator change or its regression test, but the required exact MPI test remains failing.

| Finding | Disposition | Evidence / next action |
| --- | --- | --- |
| R1-CI-EXACT-MPI-TIMEOUT | Accepted, OPEN / blocking | PR331 plus this fix times out in checks 02 and 07. Adding PR326 or equivalent transport changes is a separate integration decision, outside the explicitly requested standalone candidate. No PR326 changes were imported. A future combined candidate needs fresh verification and review; the earlier combined diagnostic is supporting evidence only. |
| R1-UNIT-RETRY-FAILURE-UNEXPLAINED | Documentation corrected; not re-reviewed | Check 06 failed only `test_core.py::test_mpi_batch_source_posts_lookahead_before_yield`. At line 986, monkeypatching `node.MPI` raises AttributeError because the harness set `PS_PARALLEL=none`. `node.py` imports MPI only when mode == "mpi". Check 08 changes the unit runner to `PS_PARALLEL=mpi`, with identical runtime hashes and the same source test-package bootstrap. Its XML confirms that exact test passes and all 1,398 selected tests pass. |

The original check-06 log and XML are preserved in `evidence/unit-retry01/results/06-fixed-gpu-unit-retry/`; the correction is line 27 of `evidence/unit-retry02/service.py`; the final XML is in `evidence/unit-retry02/results/08-fixed-gpu-unit-normal-mode/`. This is a diagnosed harness correction, not an unexplained passing retry. The one default-deselected slow case is `gpu/unit/test_gpu_step_delivery_properties.py::test_many_seeds` (the 2,000-seed sweep); the default parametrized property tests ran.

The reviewer could inspect the detached source checkout but was denied direct access to external evidence paths. The context included complete primary failure logs, successful lifetime/device logs, per-case verdicts, installed-file hashes, harness code, and terminal accounting. It did not include the failed check-06 traceback; that omission is corrected here. Raw XML and launch records were independently audited by the coordinator, not independently read by Claude. Preserve this review limitation; do not imply unrestricted raw-evidence review.

The loop stops after round 1 at the dependency decision boundary. The maximum was three rounds, but repeating review without resolving the required MPI failure cannot produce approval. No finding is disputed and no CI exception is waived. The blocking finding remains open; this is a verification handoff, not a successful CI-repair review.
