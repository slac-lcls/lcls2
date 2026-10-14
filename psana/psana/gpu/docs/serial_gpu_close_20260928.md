# Serial GPU event iterator cleanup

Reviewed the preceding geometry fallback commit `10173a263`: shared hits,
explicit collective startup, option-specific local caching, numerical tests,
and isolated-rank MPI validation remain consistent. No blocker or additional
geometry change was found. Stage 4 and geometry were already committed as
`9730a9982` and `10173a263` respectively.

Committed as `10df4c6e3`. This change addresses the remaining serial cleanup finding from the
[overall review](user_kernel_overall_review_20260928.md).

## Behavior

`RunSerial.events()` now closes the GPU manager in a `finally`, matching the
existing MPI integration. Exhaustion and explicit closure of a started event
generator are terminal for that run's GPU stream. CPU-only serial iteration
keeps its prior behavior. Use `contextlib.closing(run.events())` to guarantee
cleanup on a loop-body exception or early `break`; see the
[public guide](user_task_results.md). Closing a generator that has never started
does not execute its body/finally, as with ordinary Python generators.

Manager close first unwinds its suspended serial producer, including any active
slot-retirement window. It then drains remaining slots before closing input
resources, requested constants, and publication staging. The manager's final
delivery loop and `finish()` now execute cleanup even when interrupted. Pool
flush iterators are explicitly closed rather than relying on garbage collection
to retire the currently yielded slot.

A reentrancy guard allows these nested `finally` blocks to converge on one
resource shutdown. Successful close is idempotent. A failed drain leaves the
manager open for retry and preserves the existing slot quarantine/owner rules;
resources are not closed before execution completion is established. Retained
published rows are materialized to independent NumPy storage before the pinned
cache is released. Later serial event iteration returns no events.

Stale docstrings saying public tasks are gated until Stage 4 were updated.
This change adds no callback, kernel, or copy scheduling. No throughput claim
is made from correctness-test runtimes.

## Validation

Frozen source: parent `10173a263` plus `source.patch` and the new unit test at
`/sdf/scratch/users/m/monarin/gpu-validation/serial-gpu-close-20260928-r1`.
The manifest covers 907 Python/CUDA sources, checked against the checkout.

- Local GPU unit suite: **477 passed**.
- Main CPU suite, job **39359467**, sdfmilan001: **557 passed**, 141 skipped,
  seven deselected, 126.66 seconds.
- A100 device suite, job **39359466**, sdfampere020: **141 passed**, 184.57 seconds.
- Byhand/MPI suite, job **39359467**: **five passed**, 138.74 seconds.
- Public GPU-task MPI, job **39359466**: exclusive and hybrid modes each
  delivered 13 exact events with callback sizes 5, 5, 3 and retained host access.
- Both jobs completed with exit code 0; CPU/byhand allocation 4m39s, A100/MPI
  allocation 6m05s. Runtime and test sources still matched the frozen manifest
  after all tests. See [validation hashes](serial_gpu_close_validation_20260928.json).

New host tests cover normal exhaustion, explicit close, `closing` with break
or loop-body error, mid-retirement and final-flush interruption, producer errors,
failed-drain retention/retry, terminal/idempotent behavior, and unchanged CPU-only
semantics. Six new A100 cases use the real serial DataSource with two occupied
execution slots, pinned/ordinary-host delivery, and weak references to scratch,
published outputs, and requested constants. Their only cleanup route is the
public generator protocol. They verify owner release and retained host results.
