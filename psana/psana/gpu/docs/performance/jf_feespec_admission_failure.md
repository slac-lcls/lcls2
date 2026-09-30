# JF + feespec: retained input view in the benchmark

## Failure and reproduction

Job 38995226 successfully staged the complete six-stream input on Weka FFB
and passed A's validation. E bulk-off then failed during validation, before
any clean timing cases. This was a `GpuMemoryPressureError` raised by the
8 GiB software budget. It was separate from the earlier Slurm cancellations.

Targeted probe job 38997640 reproduced the identical failure using the same
frozen Integrated runtime, batch size 100, depth 1, and budget 8 GiB. The probe
reuses staged files and deliberately skips cache conditioning; its rates are
not performance results.

The admission planner splits each 100-event batch into **52 + 48 events**.
The first 52-event input needs 1,745,479,352 logical bytes. The next batch's
52-event input needs 1,745,479,424 bytes: a **72-byte increase**. The reader
compares logical array sizes, so it requests replacement even though both
sizes round to the same 1,745,479,680-byte pool block.

The first reservation fails while reusable reader, parser, and detector
buffers remain cached. Recovery drains execution and trims caches. The
second reservation still fails, with this exact allocation inventory:

| Component | Bytes | GiB |
|---|---:|---:|
| Fixed storage | 671,161,344 | 0.6251 |
| Old input backing still referenced | 1,745,479,680 | 1.6256 |
| New input/parser/detector allocations requested | 6,985,829,376 | 6.5061 |
| Safety margin | 858,993,459 | 0.8000 |
| Total required | 10,261,463,859 | **9.5567** |
| Configured budget | 8,589,934,592 | **8.0000** |

The retry inventory contains only `fixed` and `reader` categories. The old
reader cache itself is empty, proving that the input allocation survives
outside that cache. Parser and detector caches were successfully released.

## Cause

The benchmark retained `segments` and its extracted CuPy `values` array
after `with field.on_gpu_view(...)` exited. Python's `with` block does not
clear those local variables. The next `run.events()` advance occurs before
the next loop body can overwrite them.

An 8 KiB feespec field is a slice of the shared 52-event input buffer, which
also contains all five JF streams. Its alias therefore retains the whole
1.626 GiB backing allocation through replacement. The ownership ledger is
correct to keep charging that storage. The context records GPU completion;
exiting it does not destroy arbitrary CuPy aliases held by the caller.

A's feespec H2D creates an independent small allocation. It does not retain
the combined GPU input buffer through this particular field-access path.

## Fix

The maintained benchmark now clears all three local references immediately
after leaving the field-view context:

```python
with field.on_gpu_view(cp.cuda.Stream.null) as segments:
    values = segments[0]
    result[i] = cp.sum(values, dtype=cp.int64)
    # Full-array checks occur here only in validation mode.
field = segments = values = None
```

This leaves the existing lease/completion handling in control and releases
unneeded Python aliases before requesting another event. No production GPU
runtime or budget policy was changed. The recovered allocation request then
needs approximately **7.931 GiB including fixed storage and margin**, within
the existing 8 GiB budget.

Increasing the budget is not the first fix: the planner can select larger
subbatches and leave the same ownership problem. Reducing batch size changes
the requested bulk-read experiment. Keep batch 100 and budget 8 GiB while
validating the reference-lifetime correction.

A separate possible runtime improvement is to let reader allocations use
their rounded capacity, avoiding replacement for this 72-byte increase.
That would need careful bounds/accounting tests and is not part of this fix.
Admission that adapts to intentionally retained user arrays is also separate;
it must never uncharge live storage or overwrite active consumers.

## Evidence

Probe job **38997640** reproduced the original failure. With the reference
cleanup, **both E bulk-off and E bulk-on completed 10,000 events**, with all
feespec arrays and the three JF raw/calibrated samples matching the CPU
reference. Bulk-off issued 60,000 KvikIO API requests; bulk-on issued 1,092.
After trimming at the original failure point, fixed E-off's committed bytes
were exactly 671,161,344 (fixed storage only), versus 2,416,641,024 originally.
The probe does not control cache state and supplies no accepted warm/cold rates.

The original two-round warm/cold matrix was resubmitted as **38997826**, using
`lcls:data`, normal QoS, and the same frozen A/E builds. It reuses the verified
FFB prefix after manifest/size/tier checks. Results are under
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-ffb-20260924-fixed`.

Original campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-ffb-20260924-normal/job-38995226`.

Ownership probe:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-ownership-20260924/job-38997640`.
The original/fixed scripts are frozen in its parent directory. Per-case logs
record admission requests and allocation categories. The maintained change
is in `psana/psana/gpu/scripts/feespec_bulk_benchmark/bench.py`.
