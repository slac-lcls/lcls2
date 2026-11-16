# Independent review: sample10 admission investigation

Review commit 784e4fba4f360a51f8c591ce842e44cbd025ac11 against base
c10abb7f9ad86202aca7e08a50004b74548171c9. Read only; do not invoke another reviewer,
edit, deploy, or launch tests/jobs. This is round 1 of 3. No unresolved prior IDs.

Scope: explain why one CUDA stream reports zero admission drains while two report
142 steady drains. See notes/mfx_expand10_pressure_20261106/scope.md and README.md.
Changed code: that directory's diagnose.py, test_diagnose.py and run.sbatch;
notes/mfx_expand10_performance_20261106/runner.py adds two bounded options with
unchanged defaults. Documentation corrections in preparation README and
psana/psana/gpu/docs/proposals/synthetic_mfx_expand10.md. No core/kernel changes.

Inspect observer fidelity: original calls occur once; no added poll/wait/CUDA
allocation or lease acquisition; scalar snapshots cannot retain owners; errors
propagate and patches restore. Check count semantics, attribution to file-stream
credit versus raw slots or bytes, one-slot versus two-slot retirement, whether
claims overreach evidence, and regressions from parameter additions. Distinguish
physical XTC streams from CUDA execution streams. A finished kernel alone is not
proof all input leases are released.

Evidence: four observer tests + nine existing preparation tests passed in native
A100 job 43557748. Four 1k native cases passed event/bin checks, with independent
fresh processes on sdfampere010, compatibility mode ON (CPU fallback). Raw logs,
Slurm accounting, validation-summary.json, execution-source-hashes.json and
rejection-example.json are committed in the investigation directory. All source
hashes were rechecked; runtime-callsite-check.json compares 12 relevant methods
or modules via AST to the frozen installed runtime. Source in psana/psana/gpu can
therefore be consulted for those paths. Full report.json and admission-trace.json
for each case are copied under build/pressure-evidence/{case}/ in your checkout;
copy hashes match artifacts.json. Read them, not just coordinator conclusions.

Hypothesis/result to independently assess: one small input group per physical
file stream blocks the second CUDA slot's admission; framework flush/retry
releases leases. Sampled failures show five conflicting file streams, 145 free
raw slots, and ~0.58 GiB framework committed of 4 GiB. All counted exceptions have
stream-credit/raw-slot text; zero budget.hold failures. Detailed snapshots are
bounded to four per phase, so not every rejection has a full snapshot. Doubling
quota to 8 GiB leaves counts unchanged (15 warmup/142 steady); classifying every
JF datagram as large with a 512 KiB threshold removes drains. That control also
changes groups from 790 coalesced to 5,000 per-file/event groups and callbacks
158 to 150. It is not recommended final tuning or proof of better throughput.

Limitations: diagnostics add Python overhead; only 1k causal controls, not a 10k
performance campaign or CUDA-overlap measurement. Frozen runtime, warm cache,
no GDS. Framework budget excludes 12.5 GiB of user-owned persistent cube banks
and other user arrays. Core fixes need a separate scope/review. Earlier prep
failed-run partial trim counters remain an explicit deferred low-priority item;
this observer independently writes its own failure trace when output exists.

Return the required structured verdict with stable finding IDs and exact reviewed
SHA. Acceptance requires no unresolved blocking findings, not blanket approval
of the unchanged core admission design. No tests can run in your restricted
review; assess provided evidence and state that limitation accurately.
