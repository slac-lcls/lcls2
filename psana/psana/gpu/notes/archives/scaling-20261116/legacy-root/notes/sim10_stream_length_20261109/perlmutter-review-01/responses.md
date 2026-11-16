# Response to implementation review

Reviewed cfadd496a10138a196f165108ef1487d6cc374de: pass for implementation
readiness only, ten nonblocking findings. No native acceptance.

- PMR-01: keep the one-hour limit for now. Attempt01 hashed all 336 GB in under
  two minutes; the actual validation and case times will determine adequacy.
  Each subprocess timeout still fails closed; no acceptance on timeout.
- PMR-02: extend native negative mutations and print the test file SHA in its
  log. Preserve existing failed checks. Native tests remain pending.
- PMR-03: replace stale README/status/preflight information during results closure.
- PMR-04: derive a separate summary with every repeat, medians and paired ratios.
- PMR-05: defer a campaign supervision rewrite. Successful native acceptance
  requires complete.json, every checked srun return, per-rank resource retirement,
  and final Slurm completion evidence. cleanup.json alone is insufficient evidence.
  Its boolean process field is a code-path assertion, not an independent process scan.
- PMR-06: fixed in 97399922b: choose from the allocation mask and verify the UUID
  is in every step's original visible set before masking. Record original masks
  and UUIDs. Slurm exclusive grants four GPUs; each MPI step requests one.
  Add inventory PCI comparison to the derived summary.
- PMR-07, SLN-01, SLN-04: retain these interpretation limits in the final report.
- BDS-08: open pending complete native artifacts and independent results review.

Native debugging found a separate issue missed by static review: Configure has
both sim10 and config Names per Jungfrau panel. f0bc14cef restricts the payload
Names assertion to sim10; all five original headers pass the corrected check.
Attempt01 job61064869 failed there after all eleven SHA256 values passed.
Attempt02 job61064979 failed before input reads because its one-GPU allocation
assertion incorrectly assumed exclusive-node grants only one GPU. 97399922b
fixes that and preserves the single-BD/single-used-GPU control.

Final native evidence supplied for round02:

- PMR-01: job61065065 completed all work in 35m58s under the one-hour limit;
  input validation took 711.973s. No timeout or orphaned step occurred.
- PMR-02: 47 native tests passed; direct first-pilot mutations exercise the
  KvikIO settings checks, and native validation mutations recompute the SHA pin
  to reach the status/hash/count checks. The test file SHA is printed in its log.
- PMR-03/04: README/status/preflight rewritten; summarize.py checks chronology
  and inventory PCI, and reports every repeat, medians and order-paired ratios.
- PMR-05: final Slurm job and every step exited 0:0; eleven-file pre-pilot and
  post-sweep stat records match; no temporary XTC input exists. Deferred broader
  failure-supervision changes do not invalidate this successfully completed job.
- PMR-06: all timed steps exposed the same single UUID, all helpers were empty,
  and the derived summary verified the physical PCI against nvidia-smi inventory.
- PMR-07/SLN-01/SLN-04: retained explicitly in the report. Absolute-path post-run
  lctl probe also failed to expose llite stats; CXI codeword/BER/pause rates have
  no audited remote-payload interpretation or timed brackets. Remote unverified.
- BDS-08: native receipts, audit, 47-test log and full result report now supplied;
  acceptance awaits the independent native-results review.
