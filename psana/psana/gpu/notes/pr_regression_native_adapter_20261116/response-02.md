# Round 2 response

Reviewed commit: 1af57a40073306763f013dd0876b57554fc53b2d.
Reviewer closed NAT-001 through NAT-007. NAT-008 accepted and corrected for
round 3 re-review; no unilateral pass claimed.

The observer now classifies each native Events.__next__ return using psana's
first_service. L1 (service12) and transitions by service are counted separately
for each delivery window and final drain. Classified totals must equal the
native window envelope count. Only L1 counts participate in delivered-event
coverage and first-batch timing. No counters mutate the returned envelope.

The existing GPU packet0=0/packet1=2/drain=2 stub now exercises classification
and method restoration. A new mixed CPU-window test covers transition-only
BeginStep/Enable, L1 plus EndStep, and the next step plus EndRun, in events and
steps consumption patterns. It verifies three delivered L1 events, separate
transition counts, no first-batch time from the transition-only window, and
rejection if a delivered L1 is missing. These are adapter observer tests, not
replacement framework transition/lifetime tests.

NATIVE.md and the timing receipt explicitly describe L1-only populations.
Envelope classification adds per-envelope overhead: this remains a required
native overhead measurement before any clean performance acceptance. No low
or negligible overhead claim, native execution, or qualification flag change.
