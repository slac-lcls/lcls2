# Round2 response and hardware readback

Claude PASS at95ea0c4ed0fed477c2d42076445e48c5f315c862. No production driver/helper
changes follow this review. Job61483852 ran exactly that frozen code, passed in
30s (step27s), and published runtime-01.json. Job, batch and step all COMPLETED0:0.
RUN-001: fresh final sacct retained in capture-accounting-final.txt, distinct from
the still-running EXIT-trap snapshot. Actual allocation128CPU/4GPU/16G; step4CPU/
1GPU. Optional trap placement before checksum deferred: checksum failure retains
Slurm log and is never accepted; coordinator always collects final accounting.
RUN-005: added fake-child success and commit/install/receipt-mismatch rejection
test;71 total tests PASS. Directory fsync remains optional; atomic publication
is claimed, not crash-durable persistence across filesystem/node failure.
RUN-006: documented check-pair is only for unmodified driver-generated canonical
manifests, not arbitrary hand-edited path aliases. Same-manifest A/A check passes.
RUN-007: submission-verification-01.json checks both frozen blobs and launcher
against reviewed Git before sbatch. Post-job manifest code hashes match as well.

Raw GPU runtime probe passes CPU import guard, MPICH, NVRTC and4KiB KvikIO read.
No experiment data, six-case smoke, cache conditioning or performance run. Driver
prepared a runtime pin; source API and native campaign qualification stay pending.
