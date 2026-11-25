# Disposable launcher cleanup qualification

Base: bd388844145d1bf1d1e84aed3fbf448b636574e3, results/psana2-gpu-scaling.
Implement a reusable, bounded compute-only driver using the actual shared adapter
launch/cancel path and campaign.run_command controller timeout. Inject launcher
step timeout, SIGTERM, SIGINT, SIGHUP and controller timeout. Require all target
ranks ready and a distinct, similarly named sentinel Slurm step live before
injection. Demand target failure/no success receipt, cleanup within 10-second
grace, no target remaining, sentinel heartbeat after cleanup and COMPLETED/0:0
on graceful sentinel release. Record hashes, raw logs, step IDs, scheduler audit.
The sentinel deliberately overlaps resources; this is not a performance test.
CPU-only sleeping ranks: no XTC/cache access, no CUDA initialization, no dataset
copies, native workload smoke or full timed campaign. No qualification flags.
Existing prepared runtime is reused only for its Python/environment pins.

Use a fresh <=15-minute premium lcls_g single-node allocation after Iris check.
Final allocation accounting and no active steps mandatory. Runtime/harness failures
must not be mistaken for successfully injected failures. Preserve failed attempts.
Review loop: Codex implementation, restricted independent Claude; at most 3 rounds.
Review implementation before allocation and final code/evidence after hardware.
Next separate step: six native run51 workload smoke cases.
