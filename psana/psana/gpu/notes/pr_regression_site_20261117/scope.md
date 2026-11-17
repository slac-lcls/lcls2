# Allocation and placement preparation

Implement the next reusable PR-regression preparation driver on
results/psana2-gpu-scaling. Generate a fresh allocation-specific native_site
manifest only after four placement probes (CPU/GPU x 1/2 BDs) succeed through
the existing adapter launcher. Record physical-core placement, selected A100 UUID,
physical Slingshot NIC inventory/byte counters, scheduler allocation and steps,
timeout headroom, exact runtime/source hashes and inherited environment digest.

Acceptance: focused failure tests and existing offline suite; independent Claude
review (maximum three rounds); bounded premium lcls_g compute validation using
the existing isolated runtime. Preserve failed attempts. No run51 I/O, cache
conditioning, timed campaign, component/overhead qualification or profile flag
changes. Do not claim disposable timeout/signal qualification from normal exits.
Commit/push within standing authorization; final evidence review before closure.
