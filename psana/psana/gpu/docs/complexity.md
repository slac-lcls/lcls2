# CPU/GPU size and responsibility comparison

Measured runtime checkpoints: `31d68655a6` (September 26) and `6ba5fa586`
(current user-kernel runtime). Documentation, benchmarks and tests are excluded.
This consolidates the earlier comparison; its full history remains in Git.

## Measured size

| Scope | Before | Current | Interpretation |
| --- | ---: | ---: | --- |
| GPU runtime physical LOC | 9,455 / 24 files | 8,833 / 26 files | Net decrease of 622 lines (6.6%) |
| External calibration + integration | — | 352 LOC | 196 calibration + 156 integration; algorithm work moved outside runtime |
| Runtime plus those algorithms | — | 9,185 LOC | 270 fewer than the old runtime; old runtime did not include radial integration |
| Runtime plus all five example files | — | 9,325 LOC | Includes 140 lines of example drivers |
| CPU `Events` + `EventManager` | 589 LOC | 589 LOC | Shared event/read layer, not equivalent total functionality |
| Runtime Python function definitions | 395 | 431 | Structural count, not maintenance cost |
| Runtime Python control sites | 1,076 | 1,122 | Not cyclomatic complexity; excludes embedded CUDA |

Fewer lines do not mean fewer decisions. Removing built-in calibration,
derived-constant/MPI setup and old orchestration offsets new task/publication
machinery. This is a net checkpoint comparison, not an isolated cost of batching.

## Runtime responsibilities

| Disjoint GPU group | Before LOC | Current LOC |
| --- | ---: | ---: |
| Coordination/read scheduling | 3,192 | 2,834 |
| Field/result API and input lifetime | 1,452 | 1,425 |
| Quota/allocation | 418 | 419 |
| Parser/configuration | 2,121 | 2,121 |
| Internal detector/calibration | 1,185 | 392 |
| Task/input context/publication/D2H | 0 | 801 |
| Descriptor ABI | 487 | 487 |
| MPI/sharing helpers | 547 | 316 |
| Package exports | 53 | 38 |
| **Total** | **9,455** | **8,833** |

The new 801-line group is `gpu_task.py` (143), `gpu_task_batch.py` (223),
`gpu_producer.py` (209) and `gpu_d2h.py` (226). It includes validation, selective
constant staging, identity alignment, owner retention and bounded host delivery,
not just launch scheduling. Some responsibilities previously lived elsewhere.

Coordination, field/lifetime, quota and task/publication groups total 5,479 lines
(62% of runtime), versus 5,062 (53.5%) previously. Managing asynchronous storage
and completion remains the largest maintenance surface.

## User analysis on CPU and GPU

| Responsibility | Typical CPU event loop | Current GPU task path |
| --- | --- | --- |
| Calibration/integration submission | User invokes algorithms per event | Psana invokes one callable per execution subbatch |
| Intermediate arrays | Ordinary host values | User-owned device arrays registered before launch |
| Result delivery | Returned host value | Named grouped D2H and independent NumPy rows |
| Storage lifetime | Synchronous references in the common path | Input leases, producer/copy events, retained owners |
| Failure handling | Python exception propagation | Drain or retain owners; MPI fatal path aborts |

CPU users can implement their own batching/parallelism. The inspected CPU
read/event pair does not supply the GPU task publication and completion contract.
Its smaller size therefore does not establish equal functionality at lower cost.
GPU users still own numerical policy, stream-correct kernel code and allocation
sizes; psana moves repeated scheduling out of their event loop.

One task can launch multiple kernels. The example runs calibration followed by
integration, while per-event identity/result mapping remains. The
[user-kernel measurements](performance/user_kernels.md) quantify that workload's
benefit; they are not a matched CPU/GPU scientific throughput comparison.

## Reproduction

[Evidence](performance/evidence/complexity.json) preserves per-file counts and
AST metrics. Regenerate the same revisions with the existing inventory script:

```bash
python psana/psana/gpu/scripts/complexity_inventory.py \
  --before 31d68655a6f2fb711e2489f89d3149172b46dad3 \
  --after 6ba5fa586
```

Use the same file groups for comparisons. CPU/shared scopes outside the disjoint
GPU table overlap responsibilities and should not be added into a single ratio.
Source size and AST counts are structural evidence, not proof of runtime speed,
correctness or architectural simplicity.
