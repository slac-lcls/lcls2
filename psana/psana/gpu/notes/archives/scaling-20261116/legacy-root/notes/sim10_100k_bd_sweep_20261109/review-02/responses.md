# Results review closure

PASS on c12cd589b05564c4d3b6b57436f07dedcebd5246, no blocking findings.
Executed harness remains fb4609a89a67e958e918fbd4c01a7dcad2db78bd; no executed
code, audit or tests change at closure. All native cases and 22 audit tests pass.

- BDS-06: fixed in README and results.json notes. Earlier accepted 1-BD/1-stream
  10k cases on sdfampere042 were 2.1884 and 2.1197 GiB/s (median 2.1541), versus
  current 1-BD/2-stream 100k median 1.1889 on sdfampere038. The 3.79x gain uses
  only this sweep's own one-BD baseline. Node, stream count and input length
  differ; neither input size nor stream count is isolated as the cause.
- BDS-07: clarified using both measured bookkeeping and frozen controller code.
  Each BD's outstanding-read peak equals one 100-event subbatch. The controller
  waits for its current subbatch reads before issuing the next, as seen in
  _issue_gpu_read/_wait_gpu_read and the subbatch loop. Multiple BDs add independent
  issuers. This says nothing against concurrency among reads within a subbatch
  or read/compute overlap; no GPU timeline was measured.
- BDS-08: exact interpreter, commands, working directory, source commits and
  native artifact root are recorded in validation-commands.txt. Original tests.log
  remains unchanged.
- BDS-01/03/04/05 closed by the reviewer. BDS-02 cancellation/timeout hardening
  remains optional and documented; the normal run and cleanup both completed.

Reviewer limitations remain explicit: read-only artifact/code inspection, no
independent rerun/hashing by Claude; long timestamp arrays were sampled. The
coordinator verified all 40 copied artifacts/runtime files by SHA, and the native
runner plus independently executed audit checked all events/counts/cold gates.
