# Native run51 smoke qualification

Base bce4db792c185ec8f29b5c7bf0b7d07a7246d782 on results/psana2-gpu-scaling.
Run all six maintained raw cells: CPU events/steps with 1/2 BDs, GPU events with
1/2 BDs sharing one A100. Use the prepared private real mfx100848724 run51 copy,
known API runtime 1995752058bb9aaa3f16ccd86c22e84153257538, fresh allocation
placement, exact existing settings and 200-event separate-process warmup. Use a
1,000-event measured smoke prefix derived from the pinned independent SMD plan;
retain the original 10k manifests unchanged. Hash the selected prefix and verify
its headers on compute. No copy, shared-public-data eviction or calibration.

Maintain a reusable smoke driver that calls actual adapter and controller result
validation, retaining false campaign qualification flags. The qualification-only
entry point may form correctness requests without pretending full campaign gates
are passed. Record this distinction explicitly. Each measured process must pass
private-range eviction/mincore, exact events/bytes, non-idle BDs, settings/runtime,
CPU/GPU placement, NIC availability and all native indexed evidence checks.
Warmups and measured processes must each finish COMPLETED/0:0. Any failure stops
the cohort; preserve all attempts. Final allocation and step audits mandatory.

Use bounded premium lcls_g single-node jobs after Iris check. Start with <=75-minute
allocation, 128 GiB host memory, 90-second per-step timeout and 360-second
per-case controller timeout. Fresh placement runs in the same allocation.
No statistical throughput claim, full 10k cohort, A/A or A/B, component/overhead
qualification, pixel correctness or production source changes unless required
adapter compatibility fixes. Raw output stays in scratch; compact evidence in git.
Codex implementation + independent restricted Claude review, max three rounds.
Review implementation before initial hardware, final code/evidence after fixes.
