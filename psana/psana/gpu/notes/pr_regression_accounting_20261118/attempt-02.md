# Partial second allocation — job 61535808

FAILED 1:0 after 36:23 on nid001292. Prefix, placement, traffic and MPI controls
passed. The four event-iterator cases passed exact 10k event/335571760000-byte,
zero-residency and hardware RX coverage checks. CPU steps BD1 failed its measured
pre-loop cold-cache check; BD2 steps did not run. This is not a six-case pass.
The adapter confirmed cleanup; final direct scontrol query found no steps.

The old helper raised before persisting its residency observations, so neither
the affected ranges nor resident-byte counts are known. Read-ahead refill or
advisory-eviction behavior are hypotheses, not established causes.

The next attempt records all observations and permits at most three conditioning
attempts, with one second between failed attempts. Both observations must still
show zero residency. Retry happens only before measurement, never in response to
a measured timing. All six cases rerun in one fresh allocation. No acceptance
threshold, profile, installed runtime or input data changes. Maximum native step
900 s, controller1950 s (>2*900+120), allocation5h. Worst bounded stages are about
4h36m including prefix600, four placement bounds936, controls480, six controllers
1950; remaining time is for setup/evidence overhead. Diagnostic timing only.

Partial audit in attempt-02-evidence.json verifies immutable source pins, input
metadata and accepted prefix receipts, placement artifacts, controls and all four
completed cases. It indexes the entire failed capture and explicitly reports
partial-failure. No qualification flags are promoted.
