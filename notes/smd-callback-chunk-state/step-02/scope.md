# Step 2: persistent callback input iterator

Base: `516f25d4a` on `tests/smd-callback-chunk-state`.
The user requested step 2 and keeping the implementation/review loop active for
the whole session. Codex implements, Claude independently reviews each code
commit; up to three rounds with a 15-minute invocation limit.

Implement the run-scoped input/lifetime component: one callback generator and
one callback-side run across a sequence of EventBuilders, with owned event and
transition data across refills. Chunk boundaries must not reset local counters
or end a physical Step. Provide explicit early-return/close behavior and retain
scan/EPICS metadata.

Step 3 is the production integration: adapting serial/MPI batching and routing,
draining transition records while callbacks filter without yielding, enforcing
bounded buffering, and upstream MPI shutdown. Do not wire the new iterator into
production before those contracts are implemented. The step-1 production
regressions therefore remain strict xfails at the end of step 2; report that
clearly rather than claiming the runtime bug is fixed.

Required checks: real generated XTC/SMD input with multiple steps and refills,
exact callback selections/counters, scan/EPICS metadata, retained-event and
transition serialization after deliberate input buffer overwrites, one callback
invocation, input chunk counts independent of callback invocation counts,
event-only/reject-all/early-return/close/error cases, and per-run reset. Run the
existing regression matrix plus both psana core pytest groups per the psana
skill. Record environmental or pre-existing failures separately and do not
claim full validation without resolving or receiving acceptance of them.

No Jungfrau numerical processing or multi-EB global scheduling changes.
