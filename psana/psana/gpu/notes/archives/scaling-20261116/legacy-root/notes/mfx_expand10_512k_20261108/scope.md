# 512 KiB sim10 read target and concurrency rerun

User authorized documenting and changing gpu_bulk_target_bytes to 512 KiB and
rerunning the existing test to assess concurrency. Change the performance
runner default only; preserve the framework default, diagnostic controls,
fixtures, calibration and GPU kernels. Run focused 1k pilot checks on one A100:
three fresh-process throughput cases each at one/two CUDA streams, a two-stream
8 MiB control, and separate Nsight traces at both targets. Keep 4 GiB framework
budget, 100 bins, batch 7 and 100 warmup events. No 10k/multi-node campaign.

Check exact events/bins, actual read target and admission drains. Use device
kernel intervals to establish whether task kernels on distinct execution streams
overlap; zero admission drains alone is not sufficient. Traced rates are not
throughput results. Preserve source hashes, runtime, trace data and compact
summaries. Run existing host/device checks and independent restricted Claude
review (up to three rounds), respond to findings, commit and push to the
existing authorized fork branch. If tracing is unavailable, report that as an
unresolved measurement limitation rather than inferring CUDA overlap.
