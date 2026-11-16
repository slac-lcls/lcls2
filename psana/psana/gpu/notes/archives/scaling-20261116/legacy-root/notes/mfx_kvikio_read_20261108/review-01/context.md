Independently review implementation and evidence at commit
70359e3b894206f2e9c9f2d2abb4740805a3a87b, base4e09db0ec.
User requests a read-operation baseline on the new sample10 dataset, varying
KvikIO fallback settings to find attainable bandwidth to GPU. Session requires
Claude review for implementation. Do not edit, execute, or start a nested review.

Read notes/mfx_kvikio_read_20261108: scope.md, README.md, read_case.py,
plan_reads.py, campaign.py, audit.py, tests, tests.log, results.json,
job-43866709.out, slurm-accounting.txt. Fixture identity helper is unchanged
notes/mfx_expand10_performance_20261106/fixture_check.py with receipt at
notes/mfx_expand10_10k_20261106/files.json.

Acceptance: input files read-only, no staging/new data/calibration or psana
changes, forced KvikIO CPU fallback verified via effective defaults, bounded
buffer ownership/drain and all futures complete before timing ends. Exact
per-datagram requests vs whole-file slabs distinguished. Validate counts,
returned bytes, end-of-run GPU/CPU witnesses and independent rate/median audit.
Keep setup/warm reads/witnesses outside measured passes. No cold-cache or
physical-storage/GDS/end-to-end pipeline-rate claim. Independent confirmations
must be separated from adaptive selection. Review benchmark validity/safety and
conclusions, not just syntax; report stable IDs KR-01 etc.

Job43866709 completed0:0 in4m52s on sdfampere012 A10040GB. All33 cases and10 host
tests pass. Coordinator ran audit.py against native reports, validating accounting,
settings, source/hash identity and medians. Five queue/planning tests exercise
ownership/tails/error drains; five native-report tests exercise rejection of
bad totals/settings/witnesses/rates. No full-pixel/every-read byte hashing claim.

Evidence inside build/read-evidence is byte-identical native33 case JSON copies,
campaign summary/progress and copy-index.json (native/copy paths,size,SHA).
They are copied before reviewer launch, not just in an uncommitted root folder.
Read any needed copies, especially anchor-first, confirm-anchor-r1/r2/r3,
event-t8, direct-events, direct-slabs, confirm-best-events and confirm-best-slabs.
Originals at
/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/kvikio_read_20261108_r1/.
Hash assertions are coordinator evidence; restricted tools cannot hash/re-execute.

Key caveats to verify: first control4.14GiB/s, later repeated4-worker control
15.62median vs8-worker15.47. No statement that this is a causal speedup. Larger
reads14.61median with wider spread. direct-requested probes3.57/5.03 are single
cases; opened O_DIRECT fds don't prove all actual bytes used direct I/O. Initial
GPU6373MiB/100% activity, first-case postwarm460MiB, final0MiB/16% snapshot;
no continuous exclusivity claim. Cases use16CPUs except4CPU-affinity control.

The earlier psana baseline explicitly recorded only KvikIO compatibility mode
and worker count; its effective task/bounce defaults were not captured. The
installed frozen library defaults have now been read as4MiB/16MiB and this
benchmark sets/reports them explicitly. Challenge any wording that implies
historical effective settings were retrospectively proven. Runtime wrapper
exports KVIKIO_NTHREADS=4; read_case overrides via defaults.set before opening
any handles, and checks effective getters.
