# 100k sim10 cold Weka BD sweep

User authorizes 1,2,3,4 BDs sharing one GPU, two CUDA execution streams per BD,
512 KiB bulk target and best measured KvikIO fallback settings. Continue the
preceding descriptor-only psana-GPU staging baseline: no dense gather, expansion,
calibration, cube kernels, payload D2H or framework modifications.

Use accepted 100k fixture, five JF bigdata files plus their SMD, with exact copied
SHA256 on a unique owner-controlled Weka FFB directory. Preserve scratch fixture.
Total JF event datagrams per full case335820000000bytes/312.756747GiB;500000reads.
Batch100, SMD1000, 6GiB ceiling per BD, two streams, no D2H. KvikIO compatibilityON,
8 workers per BD,1MiB tasks,16MiB bounce,no auto-direct. All ranks share the same
fixed node physical-core affinity (>=40), NUMAinterleaveall, exclusive allocation.
BD count increases contexts/total workers, not the node CPU allowance. Fresh MPI
processes per case, helper masks empty, every BD masked to the same GPU UUID.

Four pilots of1000*BDcount events, then two100k cases per BDcount, order1,2,3,4,
4,3,2,1. Full-loop timer includes start barrier, event delivery, lazy allocations,
GPU drain and completion barrier; excludes setup, cache conditioning, validation,
NIC endpoint queries and teardown. Exact per-BD timestamps/request histogram,
aggregate unique partition/bytes, both streams exercised, discovered peers and
resource drain required. No CPU payload reads or pressure drains permitted.

Cold means all five staged bigdata files fsync/FADV_DONTNEED then zero pages by
mincore after DataSource setup. SMD and Weka backend caches uncontrolled. No global
cache flush and no NVMe data staging. Physical NIC ports retained separately,
with idle and loop endpoints and actual counter intervals; no exact cache-hit
fraction inferred from NIC bytes. Endpoint envelope differs from application time.

Results include GiB/s,events/s,time,NIC bytes/cache residency, per-BD distribution,
reader counts/sizes, GpuTask invocations/sizes, submitted-not-retired peaks,
runtime hashes/GPU mapping and memory snapshots (not continuous peaks). Independent
audit and negative checks, then Claude review on detached committed snapshots,
maximum3rounds. Preserve native evidence on project scratch and concise records
in this branch. Remove only the campaign's unique regenerable FFB copy on exit.
