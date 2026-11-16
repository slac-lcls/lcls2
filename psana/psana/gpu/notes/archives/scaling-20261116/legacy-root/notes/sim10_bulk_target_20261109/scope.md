# Sim10 bulk target sweep

Authorized comparison of gpu_bulk_target_bytes=512KiB,4MiB,8MiB. Fix batch100,
one BD/GPU/execution stream,8 KvikIO workers,1MiB tasks,16MiB bounce buffer,
compatibility ON, automatic direct I/O OFF,6GiB budget,SMD1000,NUMA interleaving,
shared full physical-core affinity. Public GpuTask seed descriptors, no user
kernels, expansion, calibration, dense gather, or payload D2H.

Two 10,000-event cold-FFB runs per target, in forward then reverse order, after
200-event preflights for each target. Both repeats use the same node/GPU and
SHA-verified temporary FFB copy of the five accepted sim10 JF streams plus SMD.
Only the five bigdata payload files are evicted and mincore-checked after setup;
SMD and server cache are uncontrolled. NIC physical counters bracket the complete
MPI event loop and GPU drain. Preflights warm compiler cache; lazy per-process
allocation remains timed. No NVMe input staging.700GiB allocation.

Report actual GPU-reader/KvikIO request counts, read-size histograms, payload
bytes, callback/subbatch sizes, peaks of outstanding requests/bytes, NIC RX,
initial/final payload residency, wall time and rates. Larger target changes
small-group ownership and subbatch scheduling, so this measures the pipeline's
net effect rather than isolating storage read size. No framework behavior edits.

Acceptance: frozen exact timestamps, exact bytes and independent request-count/
size predictions from fixture offsets within batch100, callback identity, one
stream, no CPU payload reads/admission drains, resources drained, hash/runtime/
settings/affinity identity, no initial resident payload pages, expected FFB mount.
Use independent Claude review on committed code/evidence, max3rounds. Store native
artifacts on project scratch and remove only this campaign's unique FFB copy.
