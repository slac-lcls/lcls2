# Stage 1: 10,000 unique sample10 events

User authorized generating and validating the first 10,000 unique MFX run-10
events in prjdatpsana project scratch, then independent Claude review (up to
three rounds). Base commit f1b8bb7e5265e7044f6a27289b65c9285f09c703 on
work/synthetic-mfx-scan. Commit and push task records under existing authorization.

Keep five JF and five ancillary streams, full 32-panel calibration identities,
sample10-v1 schema and original event timestamps. Preserve source transitions
and SlowUpdates; truncated output closes using each stream's own source
Disable/EndStep/EndRun template with ordered timestamps in the next source gap.
No artificial steps, repetitions, public run assignment, DB changes, GPU kernel
changes, performance measurements, or edits to existing Stage 2 artifacts.

New fixture.py and verify.py are explicitly forked from the reviewed Stage 2
tools to preserve their historical code-hash binding. Changes are the 10k bound,
group-accessible new directory, progress logs, cached source schema and indexed
event/panel witnesses (instead of quadratic DataSource verification lookup).

Acceptance: exact size preflight, write/fsync/readback, SHA256 of all 20 files,
all 320,000 panel-event seeds compared to source and CPU expansion, ancillary
payload/scalar preservation, Names identity, aligned ordered BD/SMD transitions,
every SMD reference, one native step, exactly 10,000 unique original timestamps
through serial DataSource, and frozen pedestal/gain/offset hashes and shapes.
Only completion of independent verification accepts the fixture. Failure or
preemption leaves partial output for diagnosis; never silently overwrite it.

GPU correctness is established on the earlier 1k fixture, not re-claimed for
10k by this CPU-only stage. Performance runner updates and single-node scaling
remain separate stages. Filesystem free space is not proof of project quota;
actual successful writes/readback establish this fixture fits at execution time.
