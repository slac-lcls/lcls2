# GPU tools

| Family | Role |
| --- | --- |
| [performance](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/scripts/performance/README.md) | Maintained source-only catalog validation and offline SDF replay |
| [jf_scaling](jf_scaling/) | Existing scaling harness and CPU contract tests |
| [stage1_regression](stage1_regression/) | Existing stage-1 validation harness |
| [stage5c](stage5c/) | Existing experimental harness |
| [feespec_bulk_benchmark](feespec_bulk_benchmark/) | Existing feespec benchmark tools |
| [pr187_pr208](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/scripts/pr187_pr208/README.md), [pr299_perlmutter](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/scripts/pr299_perlmutter/README.md) | Frozen historical execution packages |

Read-only diagnostics such as preview_stream_read_plan.py and complexity_inventory.py
remain tools, not throughput tests. Public examples stay in ../examples/.
Frozen sources and campaign tools are source-only. Installed readers should use
the [source branch](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/scripts)
for campaign records after publication (the consolidation is currently local).
