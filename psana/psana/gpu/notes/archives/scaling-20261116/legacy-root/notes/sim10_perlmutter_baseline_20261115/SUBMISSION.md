# Bootstrap02r2 proposal — not submitted

The user authorized the no-export implementation revision and continuing scoped
Claude review. Prior compute approval covered only bootstrap01/job61373306,
which failed. Preserve that immutable attempt and historical submission receipt.
The new proposal requires its bounded plan/command to be shown before submitting.

Bootstrap02r2 requests one exclusive four-GPU node for 70 minutes, lcls_g/premium.
Run existing GPU tests/guards; two ECC shape/content checks at 201/1003 events;
three 100k staging repeats each for 1BD/1GPU/batch50, 4BD/4GPU/batch200 and the
staging-only 4BD/1GPU bridge. Four workers/BD and qualified count2 layout stay
fixed. No cube files, full-bin transfers, global cube merge or CPU cube reference.
Only small receipts/logs are written. Detailed gates/budget: PLAN.md.

```bash
attempt=/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/bootstrap02r2
sbatch --account=lcls_g --qos=premium --constraint=gpu --nodes=1 \
  --gpus-per-node=4 --exclusive --time=01:10:00 --chdir="$attempt" \
  --output="$attempt/slurm-%j.out" --error="$attempt/slurm-%j.err" \
  "$attempt/bundle/run.sbatch" "$attempt"
```

Refresh squeue/count2 users and Iris before authorized submission. Old stripe
job61372107 is complete; do not carry its dependency into a fresh proposal.
If another job is reading these inputs, wait before submitting. Do not cancel or
alter other campaigns. Stop on any failure; no automatic resubmission, tuning,
or full-campaign promotion. GPU checks remain pending actual execution.

Implementation review passed round2 at af6ebc03b86033accc6c0af34bb6e3eb2b85f35c; both original findings closed.
One optional test-message matching suggestion is deferred (GC-R2-02).
Host114 tests and64subtests pass on that exact clean code commit. Prepared
bootstrap02r2 has46 files matching committed bytes, all read-only, with seven
external pins verified. Manifest SHA256: d04326f27b9acd18576393392847bf7451451fb7fbfe1c01e455d16b243430c9.
Fresh reuse check passes727installed/86test hashes and five count2 inputs.
No job-started marker, results or exports directory exists. No job submitted.

Earlier unsubmitted bootstrap02 and failed bootstrap01 remain untouched.
GPU verification, memory fit and the300s ECC ceiling remain unproven until
execution. Full same-node S/E/EC/ECC throughput is still a later proposal.

Queue snapshot: PR299 campaign61375111 has left the queue; job61385304
(pr298-gpu-scale) is running. Its case manifest selects count1, not this count2
view. Refresh queue/input coordination and Iris before any authorized submit.
