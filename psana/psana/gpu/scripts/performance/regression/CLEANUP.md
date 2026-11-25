# Disposable cleanup qualification

Run `qualify_cleanup.py` directly in a fresh single-node premium lcls_g GPU
batch allocation, with the prepared runtime's pinned Python and compiler/CUDA
modules (see [RUNTIME.md](RUNTIME.md)). Request 4 tasks, 16 CPUs/task, one GPU,
16 GiB RAM, `--gres-flags=disable-binding`, and 15 minutes. Check Iris balances
before submission. Install an EXIT accounting trap before module/checksum setup.
Freeze and hash together: qualify_cleanup.py, cleanup_worker.py,
perlmutter_adapter.py, native_contract.py and campaign.py. No rebuild is required.

```bash
/absolute/pinned/python -B -s /frozen/qualify_cleanup.py \
  --runtime /scratch/runtime.json --runtime-sha256 FULL_SHA256 \
  --output /new/scratch/cleanup-JOBID
```

The output directory must not exist. All five cases must pass: 60-second native
step timeout, SIGTERM, SIGINT, SIGHUP, and the real campaign controller's 60-second
timeout with its 10-second SIGTERM grace. Every target uses the actual adapter
launcher, three CPU sleeping ranks, four bound cores/rank and 16 reserved
physical CPUs/rank. No GPU or dataset work occurs. This bypasses native workload
contract qualification intentionally; it does not produce an accepted site,
sample result, or qualification profile. Existing runtime pins supply Python and
launch environment, not a new runtime provenance qualification.

Before each injection all ranks must be ready, with the target and sentinel
visible as distinct live Slurm steps. The sentinel name extends the target name
to exercise exact-name cancellation. It deliberately uses `--overlap` so it can
remain runnable during cleanup. This exception is for fault injection only;
never run a competing sentinel during performance measurements. The sentinel
must produce a fresh heartbeat after target cleanup, then finish gracefully with
sacct COMPLETED/0:0. Target failures are expected, but an unexpected cause, missing
failure receipt, lingering target, success receipt or uncertain cleanup fails.
Timeout startup delays are included: failure to establish readiness in the
bounded window is a failed attempt, not qualification.

Retain failed attempts and use a fresh path on retry. The harness's finally path
cancels only its two unique names. Sleeping ranks have an independent 240-second
safety deadline. SIGKILL, node loss, batch walltime and scheduler outages are not
qualified and can prevent receipts. Obtain final allocation sacct and direct scontrol step
records outside the job even when the driver passes. No-active-step receipts
from the driver do not replace this audit. A pass proves only these disposable
CPU launcher/controller paths on the recorded allocation, not CUDA/MPI teardown
under load, cache behavior, data correctness or throughput.

After independent evidence review, proceed to the six native workload smoke
cases. Keep default profile qualification flags false until the remaining native,
cache/accounting, overhead and component-bound work has passed review.

The shared launcher queries `scontrol --oneliner show step JOB` and requires
well-formed, unique job-scoped identities and exact names (including names with
spaces). Empty or malformed output fails closed. A live allocation normally
retains batch/extern records even after numeric steps finish. A diagnostic on
2026-11-18 showed squeue omitting live numeric steps that scontrol returned;
squeue absence alone is insufficient cleanup evidence. Query failure after an
allocation ends needs final accounting review, not conversion to an empty list.

After a zero-exit srun, wait up to six seconds for its controller record to
vanish, without cancelling it. Direct step queries can briefly retain a step
after its launcher exits; an immediate absence check failed a sentinel release
in job 61502208 even though final accounting confirmed COMPLETED/0:0. The wait
never converts malformed/failed query output into success. Fault cleanup retains
its existing six-second total deadline including cancellation and disappearance.
