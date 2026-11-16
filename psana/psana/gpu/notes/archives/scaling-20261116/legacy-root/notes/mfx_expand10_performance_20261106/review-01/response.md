# Response to preparation review 1

Reviewed commit: 5727e727fd18de513988acaba66251a681dda25c. Verdict: pass;
all four findings were nonblocking. We chose to address all four before the
10k campaign rather than defer them. The follow-up SHA is recorded in round 2.

| Finding | Response | Verification |
|---|---|---|
| PERF-PREP-01 | Added a count-only observer of the frozen runtime's pressure-only `_trim_gpu_caches` path in both modes; report warmup/steady counts. The recorder's timed wrappers remain profile-only. README and timing contract explain that these forced drains constrain stream scaling. | Host regression checks phase assignment, return forwarding, restoration on failure and absence of clock calls. Native rerun reports preserve the counts. |
| PERF-PREP-02 | Bypass profiling when `_metadata_open` is false, preserving the original sealed-context RuntimeError. | Host regression calls the actual framework method on a sealed synthetic context inside the profile patch. |
| PERF-PREP-03 | Add the imported Stage 2 fixture.py and verify.py plus Stage 3 xtc_fixture.py paths to runtime code hashes. | Native rerun records all three; coordinator rehashes against the follow-up source. |
| PERF-PREP-04 | Guard prepare() against an existing kernel or any previously seen events, before importing CuPy. | Host regression rejects both states without CUDA. The native runner still prepares exactly once. |

No kernel, core psana, data or calibration changes. Nine host tests pass. The
same controlled device and 1k throughput/profile smoke suite passed again in
job43548551 on sdfampere011 (50s, exit0). The count-only observer establishes
zero admission-pressure trims in the clean one-stream case, and 15 warmup/142
steady trims in both two-stream modes. Thus the original review's possible
one-stream-flush inference is not supported by this rerun; the instrumentation
now distinguishes it directly. All newly fingerprinted helpers match current
source. validation-summary.json and round-02 context carry the evidence.
