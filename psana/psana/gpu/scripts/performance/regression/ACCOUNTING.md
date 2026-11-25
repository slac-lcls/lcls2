# Full-prefix cache and traffic qualification

Use the maintained driver with `--full-prefix` and the same pinned input spec,
Python and frozen source set as [SMOKE.md](SMOKE.md). Freeze `accounting_probe.py`
as well as the existing sources. Execute directly in a fresh single-node GPU
batch allocation, with a new output directory. The default mode remains 1,000
events; this mode selects all 10,000 prepared events and retains 200-event
separate warmups and all six profile cells/settings. Use 128 GiB host RAM and a
5-hour allocation deadline to cover bounded child deadlines (prefix 600 s,
placement probes, controls 480 s, six adapters at 1950 s each). Actual runs may
be much shorter. No new dataset copy or installed runtime modification.

This mode rederives ranges from the independent SMD plan, verifies all 50,000
selected headers on compute and compares each full selected-range hash against
its accepted preparation hash. Fresh placement probes use the same allocation.
Every real case conditions the private selected ranges after setup, with at most
three eviction/check attempts. Each attempt saves two full mincore observations
before testing them; both must show zero resident bytes. Failed attempts are
retained, with one second between attempts. No timed loop has started when
conditioning retries occur. Exhausting three attempts fails the case. Successful
conditioning is followed by exact timestamp/datagram/read byte accounting. SMD and
server caches are uncontrolled; no true cold-storage claim.

## Traffic controls

The worker now saves both kernel netdev RX/TX and CXI HNI good-octet samples at
the loop boundaries. An MPI barrier after the opening snapshots prevents other
ranks starting reads before those samples. This changes instrumentation overhead,
which remains unqualified. HNI observations include physical-device mapping and
counter timestamps. Missing/unreadable samples remain unavailable; the full-prefix
qualification rejects missing counters, identity changes, resets and stale
samples. Do not add netdev counts to HNI counts or HNI totals to their subsets.

A separate bounded subprocess first uses 256 MiB from the first selected private
range. It records idle, cold, warm, cold, warm controls. Each cold read requires
two zero-residency checks; each warm read requires full residency. All four data
hashes must match. Predeclared empirical acceptance: cold HNI RX 0.9–2.0 times
requested bytes, warm and idle RX each at most 0.1 times those bytes. A 0.25-second
settling interval is included in each control. The original private metadata is
verified before/after, and the control range is evicted afterwards.

Each full cell then requires HNI RX at least 0.9 times selected bytes. This is
an empirical coverage check, not equality between datagram bytes, read-ahead and
network traffic. Preserve the ratio, per-interface samples and all evidence.
Whole-node HNI counters include protocol/background traffic and cannot prove
storage-only attribution, physical media reads or a one-OST/link ceiling. Timing
from a single instrumented correctness pass is not a regression comparison.

## MPI diagnostic

Three isolated three-rank controls run through the existing bounded launcher:
MPI/psana import plus barrier; 20 CPU raw events with return tracing of the exact
`Smd0.start` function; the same with diagnostic Cancel/Wait of its unmatched
termination receive. The request and buffer remain alive during observation.
The cancellation affects only its diagnostic process, with no package/file edit.
The six real cells retain normal installed behavior. Each control has a 60-second
step deadline; the whole control subprocess has a 480-second deadline.

Expected evidence: no warning in the MPI/import control, exactly one builtin-world
pending-reference warning and an incomplete termination receive in the observed
raw control, then successful cancellation/completion and no warning in the cancel
control. Event identities/bytes and before/after runtime hashes must match. Any
other outcome fails qualification and needs investigation. This diagnoses that
specific exit path; it does not fix production MPI cleanup or cover all abort,
termination, multi-run or full-EndRun paths.

Review code and actual hardware evidence independently. All shipped qualification
flags stay false. Instrumentation overhead, component bounds and timed A/A remain
later steps. Save failures and never overwrite or relabel a failed attempt.

Counter-window caveat: controls include a 0.25-second settling interval, while
full cells sample immediately after the ending MPI barrier. Raw per-interface
timestamps and monotonic sample times are retained; the coordinator report
compares their intervals. HPE's public driver reads a hardware register then
attaches current realtime, so timestamp advancement alone does not establish
firmware freshness. The full-case coverage floor is an independent requirement.

The MPI observe control extends the lifetime of the Python request and its
receive buffer until interpreter shutdown. Evidence is pending-at-return with a
warning versus Cancel/Wait completion without a warning; it is not a byte-for-byte
reproduction of production object lifetime. Normal full cells remain unchanged.

Timeouts are bounded fail-closed limits, not guaranteed generous margins. See
[prelaunch estimates](../../../notes/pr_regression_accounting_20261118/response-01.md).

After job 61534960 exceeded the original 300-second measured-step deadline,
the first retry used 600/1350 seconds. Job 61535808 then passed four event cases
but failed CPU steps BD1 before iteration because its cache was not fully cold;
BD2 steps did not run. The missing failure observations motivated bounded
conditioning receipts/retries. Current full-mode step/controller deadlines are
900/1950 seconds and the allocation limit is five hours. Both failures remain
preserved; no six-case pass is inferred from their partial results. Rank-0 setup-ready,
cache-ready and loop-complete receipts distinguish stages; full 10k correctness
processes also save per-rank stacks every 120 seconds. These are diagnostic
observations and are not clean-timing evidence. The installed psana is unchanged.
