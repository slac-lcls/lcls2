# Independent review: six-case clean10k baseline

Base f1b7ec7a0. Exact reviewed SHA supplied by invocation. Round1 of3. Read-only
review; no commands, edits, delegation or nested loop. Use scope.md in
notes/mfx_expand10_baseline10k_20261108. User authorized the next clean10k stage
with512KiB and independent review. Only stage1 ran: no batch tuning, new data,
core/kernel fixes, MPI scaling, artificial compute or profile.

Changes: new run.sbatch calls the existing reviewed performance runner six times
in fresh processes, s1-r1,s2-r1,s1-r2,s2-r2,s1-r3,s2-r3, in one allocation;
100bins,99active,batch7,requested1000warmup,budget4GiB,target512KiB,CPUfallback,
accepted10k fixture and frozen calibration. The runner, kernels and runtime are
unchanged and hashes rechecked across all reports (27files including harness/
analysis/test). New summarize.py checks allsix complete pass reports, settings,
fixture binding, same host/job/runtime/code/calibration, zero drains, actual
warmup/event conservation, independently reconstructed bank/bin counts from
callback order, rates and sequential process order. Tests use copies of actual
six reports: native pass, missing case, nonzero pressure, wrongbank, false rate.
Allfive passed. No new full-pixel cube validation; inherited1k numerical checks
remain the basis.10k native runner compares every delivered original timestamp.

Native job43859846 completed0:0 in4m27s on sdfampere011 A10040GB. Eachcase passes
10k identities and exact counts,1000warmup+9000steady,1500callbacks (1000x7,
500x6),zero drains; banks [10000] or[5000,5000]. Warmed rates: s1=[285.2394551,
256.6521679,257.2539490],s2=[283.8693981,282.9052524,283.5848556]. Medians257.2539490
and283.5848556,ratio1.10235375. Whole-iteration medians255.2739519 and279.2474533.
Please independently check arithmetic/claim strength. One-stream spread is
material and repeat1 matches/exceeds two-stream; no causal speedup, universal
rate or comparison with prior1k across different node/cache is claimed. Windows
last31.55–35.07s. Device sample maxima7.91/14.84GiB,hostHWM1.93–1.96GiB; no peak
tracking. Summaries explicitly retain allcases without discarding an outlier.

Important caveats: gpu-before.csv shows6373MiB/100% before firstprocess. Allsix
before_setup samples show424.125MiB and same admissionfree. At2m06s the coordinator
ran one `srun --overlap ... nvidia-smi --query-compute-apps` during s1-r2: exactly
one process, our frozen Python,8090MiB on sameUUID. This read-only query is logged
as a transcribed tool observation (gpu-activity-observation.md), not a native
continuous monitor. It could perturb that case and does not prove uninterrupted
GPU/CPU/storage exclusivity. Slurm allocated oneGPU and staticGPU inventory+same
env identify sameGPU across sequentialcases. No other benchmark instrumentation
was added; rates are unprofiled. README discloses query and pre-run activity.
Do not treat unavailable continuous monitoring as proof of isolation.

Timing is inherited: callback-boundary warmup join can omit firstmeasured input
prep, but completes after EOF+join; report warmed-window and whole-iteration
rates separately. Whole includes startup/warmup/intentionaljoin, excludes prior
setup and subsequent validation/cleanup. Cacheuncontrolled,no full-file rehash,
freshmanifesthash+allfile sizes; compiler cache shared across fresh processes.
CPUfallback,PS_PARALLEL=none. This is serial single-worker, not an actual MPI BD.

Evidence in build/baseline10k-evidence/: six byte-exact native reports in case
subdirs, native-metadata.json is a projection dropping only bulky callbacks, and
callbacks.csv maps all9000callbacks to ordinal/eventsize/stream/phase. Each native
report SHA matches committed artifacts.json and summary.json; copy-index.json
binds copies/projections. Reviewer cannot execute; use readable metadata/CSV/raw
as needed and acknowledge unverified hashes/commands. Coordinator rehashed
executed source/runtime and ran AST/shellsyntax/nativeanalysis/fivetests. Job
logs normalized only NUL/trailingwhitespace; originals hash-bound. Tests.log is
an explicit coordinator transcript. No native case failure or extra campaign.

Progress-plan and old README edits only link the completed baseline and move
next action to batchtuning/shortcompressed profiling. Do not authorize these next
stages through this review. Return exactSHA,verdict,stable findingIDs/blocking.
Missing required evidence blocks acceptance; low optional future enhancements
can be deferred. No rerun needed merely because reviewer cannot execute commands.
