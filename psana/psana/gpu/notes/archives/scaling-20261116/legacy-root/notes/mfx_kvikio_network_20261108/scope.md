Repeat the accepted sample10 five-JF-file read benchmark with the historical
Weka methodology: physical ethtool rx_bytes_phy/tx_bytes_phy endpoints, owner-
checked per-file fsync/FADV_DONTNEED and a zero-mincore gate, then warm repeats.
Three fresh processes, four KvikIO workers/four logical CPUs, depth35,
task4MiB/bounce16MiB, forced CPU fallback, no auto-direct. Three full passes per
process: cold then two warm. No global cache flush, staging, detector compute,
or source-data edits. Only these synthetic fixture files have cache evicted.
Preserve completed request byte accounting and GPU/CPU witnesses; report NIC
interval separately from read interval and keep each physical port separate.
Idle traffic is context, not an exact subtractable background. No exact cache
hit ratio inferred from NIC bytes; backend caches remain uncontrolled.
Acceptance: three successful cases, zero pages before every cold pass, valid
monotonic NIC endpoints, exact byte/count totals, GPU witnesses, independent
summary, unit tests for counter resets/missing devices. Review with Claude up
to three rounds on work/synthetic-mfx-scan; commit/push within existing scope.
Historical evidence: ~/lcls2/notes/weka_ffb_cold_20260918/README.md.
