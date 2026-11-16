# Comparison02 submitted as job61409171

Mona explicitly authorized submission: "Let's submit the job." Prepared from
the exact reviewed fd63dfb327b154b06e3e37bf39abf698a5495f89 commit and submitted once
as job61409171 on2026-11-16T16:07:23UTC. Preserve comparison01. Source/input,
coordination and allocation checks passed before submission.

35cases, fixed four workers, same-node100k matrix; three conditioning attempts
per case,180s conditioning deadline,5h55 allocation ceiling. See PLAN.md.

```bash
attempt=/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/comparison02
sbatch --account=lcls_g --qos=premium --constraint=gpu --nodes=1 \
  --gpus-per-node=4 --exclusive --time=05:55:00 --chdir="$attempt" \
  --output="$attempt/slurm-%j.out" --error="$attempt/slurm-%j.err" \
  "$attempt/bundle/run.sbatch" "$attempt"
```

The command above was executed once. Native build and GPU processing kernels
are unchanged.

Review passed at fd63dfb327b154b06e3e37bf39abf698a5495f89 after two rounds;
164host tests+64subtests passed. All three first-round findings are closed.
Review records: review/cache-retry-round02.json and cache-retry-response02.md.
The submitted bundle matches that exact reviewed commit:49files,7external pins,
read-only. Manifest SHA256:
2c8e5ffc3003957d9ba0b2386fb99446fee8aabf4e6c855b7fb72ef59b11da1d.

At submission, Slurm readback: RUNNING on nid001020, one exclusive node, fourA100GPUs,
lcls_g/gpu_premium,128logicalCPUs,05:55:00 limit. Start2026-11-16T16:07:29UTC;
limit-end2026-11-16T22:02:29UTC (not a predicted finish). Submission receipts
are in evidence/comparison02-submission.json and comparison02-submission-slurm.txt.
Fresh checks passed727installed/86test hashes and five input identities/layouts.
Queue was empty; count2 lock available. Iris remaining95.6355%personal and
61.6647%project. No payload rehash or new GPU qualification claimed at submission.

Completed2026-11-16T17:36:20UTC, SlurmCOMPLETED0:0,elapsed01:28:51.
All35cases passed; coordinator re-audit and independent result review passed.
See [results](RESULTS-61409171.md) and review/comparison02-results-round01.json.
