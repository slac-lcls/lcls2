# Scope and review contract

Create 100,000 sim10 events in project scratch by physically repeating the accepted
10k fixture ten times. Preserve ten streams, 32 JF panels and source experiment/run
identity; one native step. Rewrite outer timestamps and SMD offsets only. Do not
modify existing fixtures, calibration DB, production GPU scheduler, or publish a
new run. Keep the 512 KiB baseline for later GPU tests. Record deferred small-dgram
read-ahead and execution-size coupling. Larger total volume does not enlarge the
per-batch outstanding window.

Acceptance: unit tests including second rollover, partial cycle, malformed inputs;
23-event pilot from accepted tiny fixture; exact preflight size; full output hashes,
payload/header preservation excluding declared fields, aligned timestamp/state
and SMD offsets in all streams; serial DataSource all 100k events, panels/layouts,
seed-value witnesses across every seam/tail. Source manifests pinned to previously
accepted receipts and source files freshly hashed. No new calibration or GPU
performance correctness claim. Scratch fixture validation does not establish cold
FFB read bandwidth: that is a subsequent benchmark with cache/NIC instrumentation.

Codex implements, read-only Claude reviews detached commit, maximum three rounds.
Only mark accepted after validation and no blocking review findings on latest code.
Base commit a219819ef. Existing branch work/synthetic-mfx-scan, monarin remote.
