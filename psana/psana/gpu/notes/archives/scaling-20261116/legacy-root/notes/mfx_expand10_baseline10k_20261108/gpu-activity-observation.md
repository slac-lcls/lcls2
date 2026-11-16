# Initial GPU activity check

The launch's gpu-before.csv reported 6373 MiB used and 100% utilization on
GPU-b1a13715-f3ef-f7d7-d637-ffdd2c0263ae. This was before the first benchmark
process. All six completed cases report only 424.125 MiB device usage
at before_setup and 39.079345703125 GiB free at admission. Thus the initial
activity did not persist into the six setup snapshots; its owner was not identified.

At job elapsed 2m06s, while clean-s1-r2 was running, the coordinator ran this
read-only check through an overlapping Slurm step in the same allocation:

`srun --jobid=43859846 --overlap --ntasks=1 --cpus-per-task=1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv`

The tool returned exactly one compute process:

```
769500, /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python, 8090 MiB, GPU-b1a13715-f3ef-f7d7-d637-ffdd2c0263ae
```

Slurm job 43859846 was allocated one A100 on sdfampere011. This is a coordinator
observation transcribed from tool output, not a separate continuous monitoring
log. The single query happened during the third case and could perturb it;
no performance instrumentation was inserted into the benchmark itself. Neither
this spot check nor endpoint memory samples establish continuous absence of
other CPU/storage/device activity. Retain these limits with the results.

The post-campaign gpu-after.csv reports 0 MiB used and 97% utilization. This
single sample immediately follows the final case; it may reflect preceding
activity, but its source cannot be determined from these snapshots.
