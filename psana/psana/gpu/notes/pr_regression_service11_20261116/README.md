# Service-11 compatibility closed

The regression observer now uses `TransitionId.isEvent`, bound once at setup.
Both service12 and service11 count as events in delivery windows and drains.
One production predicate and its setup wiring changed; framework code did not.
NAT-009 is closed by [Claude review](review-01.json), PASS in one round on
`0739a1ce92ae48af2d956af72a0cef88a1bf1ec5`.

All 45 offline tests pass (25 adapter, 20 controller). Updated observer tests
use the actual classifier and cover mixed services, service11-only first L1
window, transition-only window, CPU events/steps, final drain, repeated drain,
delivered-total mismatch, and restoration of all three patched methods.

[Timing](timing.json): nine alternating 100,000-event pairs on login34 with
Python3.11 measured a median paired increase of118.8ns/event (about1.19ms per
10,000events). [Reproducer](measure_observer.py) compares baseline and updated
observer wrappers on identical service12 stubs. This is synthetic local timing;
it establishes neither native throughput impact nor full observer overhead.
Native qualification remains pending. Default profile and archived campaigns
are unchanged. No dataset copy, build, MPI/GPU run or job submission occurred.

See [scope](scope.md), [validation](validation-01.json), and [response](response-01.md).
Optional extra guard tests (NAT-010) remain deferred, not qualification blockers.
