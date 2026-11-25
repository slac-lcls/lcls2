# Round 1 response

Reviewed d3f0956dc25bf2c85e706fc0b9ed51406e494224: pass, no blocking findings.
All suggestions addressed for a second review:

- PRR-01: require the full observed contract, compare it with the request and
  recompute its digest. Native evidence uses distinct files inside each sample;
  absolute outside paths and resolved symlink escapes fail. Tests cover echoed
  digests with changed settings and outside/reused receipts. This cannot prove
  the adapter truthfully measured its values; native qualification remains required.
- PRR-02: preserve executed audit bytes as `.executed.py.txt`, explicitly label
  them frozen investigation source and instruct copying to external scratch before
  reproduction. No new reusable audit CLI is claimed. Add selected-status-doc.json
  and a post-audit provenance readback with source SHA256, PR299 install/manifest
  SHA, Python version/executable and selection-module hash. Soften validation
  wording: this is not contemporaneous provenance. The source is intentionally
  unmodified to preserve what actually ran; its neighboring output behavior is
  now explicit. Large payload remains external.
- PRR-03: every true qualification flag requires its corresponding hashed receipt;
  missing receipts block even with all flags true. Receipt contents still require
  independent review. Tests cover each absent receipt.
- PRR-04: timeout test obtains the real child's PID and checks it no longer exists
  after termination. Relative interpreter/executable paths are rejected. Native
  Python/build identity remains in runtime receipts and per-sample provenance.
