# CUDA Stream Practice Notes

This note summarizes the learning scripts used after the standalone MPI/MPS
benchmark. The goal is to understand how CUDA streams, events, queue depth, and
buffer reuse affect overlap between:

```text
H2D copy -> scale_kernel -> D2H copy
```

The Part I MPI/MPS benchmark summary is in:

```text
psana/psana/gpu/multiowner/gpu_context_overlap_notes.md
```

## Test Scripts

The four practice scripts are:

```text
psana/psana/gpu/multiowner/cuda_direct_practice.py
psana/psana/gpu/multiowner/cuda_row_based_practice.py
psana/psana/gpu/multiowner/cuda_direct_advance_practice.py
psana/psana/gpu/multiowner/cuda_row_based_advance_practice.py
```

They all use a CuPy `RawKernel`, pinned host arrays, device arrays, CUDA events,
and non-blocking CUDA streams.

## Direct/Per-Workload Model

The direct model gives each slot its own stream:

```text
stream[slot]: H2D -> scale_kernel -> D2H
```

This is implemented in:

```text
cuda_direct_practice.py
cuda_direct_advance_practice.py
```

Each slot owns:

```text
host_in[slot]
host_out[slot]
dev_in[slot]
dev_out[slot]
streams[slot]
```

CUDA guarantees in-order execution within one stream. So for one slot, the GPU
ordering is naturally:

```text
old H2D -> old kernel -> old D2H -> new H2D -> new kernel -> new D2H
```

This means explicit GPU-side `stream.wait_event(...)` calls are usually
redundant in the direct model. The CPU still needs event synchronizes before it
touches pinned host buffers that an async copy may still be using.

## Row-Based/Stage-Stream Model

The row-based model uses one stream per pipeline stage:

```text
h2d_stream
kernel_stream
d2h_stream
```

This is implemented in:

```text
cuda_row_based_practice.py
cuda_row_based_advance_practice.py
```

For each iteration, CUDA events connect the stages:

```text
h2d_stream records h2d_end
kernel_stream waits for h2d_end
kernel_stream records kernel_end
d2h_stream waits for kernel_end
d2h_stream records d2h_end
```

In code, the core dependency pattern is:

```python
with h2d_stream:
    dev_in[slot].set(host_in[slot], stream=h2d_stream)
    h2d_end.record()

with kernel_stream:
    kernel_stream.wait_event(h2d_end)
    kernel(...)
    kernel_end.record()

with d2h_stream:
    d2h_stream.wait_event(kernel_end)
    dev_out[slot].get(out=host_out[slot], stream=d2h_stream, blocking=False)
    d2h_end.record()
```

The row-based model is designed to overlap different stages, for example:

```text
H2D(iteration i + 1) overlaps with kernel(iteration i)
D2H(iteration i - 1) overlaps with kernel(iteration i)
```

It is not primarily designed to overlap kernels with each other when there is
only one `kernel_stream`.

## Simple Buffer Reuse

The simple scripts use one completion event per slot:

```python
done_events[slot] = d2h_end
```

Before reusing a slot:

```python
if done_events[slot] is not None:
    done_events[slot].synchronize()
```

This is correct and easy to reason about. It means the whole slot is locked
until the full chain completes:

```text
H2D -> kernel -> D2H
```

The downside is that it can wait longer than necessary. For example,
`host_in[slot]` is safe to refill as soon as H2D finishes; it does not need to
wait for kernel and D2H.

## Advanced Buffer Reuse

The advanced scripts split the reuse markers:

```text
last_h2d_done[slot]
last_kernel_done[slot]
last_d2h_done[slot]
```

The meaning is:

```text
last_h2d_done:
  previous H2D is done reading host_in[slot]

last_kernel_done:
  previous kernel is done reading dev_in[slot]

last_d2h_done:
  previous D2H is done reading dev_out[slot] and writing host_out[slot]
```

The general rule is:

```text
Before touching a buffer again, wait for the event from the previous operation
that last touched that same buffer.
```

For host buffers, the CPU must synchronize:

```python
if last_h2d_done[slot] is not None:
    last_h2d_done[slot].synchronize()
host_in[slot].fill(...)

if last_d2h_done[slot] is not None:
    last_d2h_done[slot].synchronize()
host_out[slot].fill(...)
```

For row-based GPU buffers, use stream waits instead of CPU waits:

```python
with h2d_stream:
    if last_kernel_done[slot] is not None:
        h2d_stream.wait_event(last_kernel_done[slot])
    dev_in[slot].set(...)
    last_h2d_done[slot] = h2d_end

with kernel_stream:
    kernel_stream.wait_event(h2d_end)
    if last_d2h_done[slot] is not None:
        kernel_stream.wait_event(last_d2h_done[slot])
    kernel(...)
    last_kernel_done[slot] = kernel_end

with d2h_stream:
    d2h_stream.wait_event(kernel_end)
    dev_out[slot].get(..., blocking=False)
    last_d2h_done[slot] = d2h_end
```

The two waits before the kernel protect different things:

```text
kernel_stream.wait_event(h2d_end):
  current input dev_in[slot] is ready

kernel_stream.wait_event(last_d2h_done[slot]):
  previous output dev_out[slot] is no longer being copied out
```

## What We Learned

- `kernel.compile()` removes CuPy first-use compile/module-load noise from the
  Nsight timeline.
- `dev_out.get(..., blocking=False)` is required for async D2H. Without it,
  CuPy's default `blocking=True` serializes the Python producer loop.
- Increasing `spin_iters` made kernels long enough to see H2D/kernel overlap.
- The sum of per-kernel CUDA event times can be larger than wall time when
  kernels or copies overlap.
- In the direct model, advanced GPU-side wait events do not change scheduling
  much because each slot already uses one ordered stream.
- In the row-based model, event waits are essential because H2D, kernel, and
  D2H are submitted to different streams.
- With `queue_depth = 2`, the scripts use double buffering: two complete slots,
  each with host and device input/output buffers.
- With `queue_depth > 2`, the same idea becomes N-buffering.
- Simple reuse is safer to learn first. Advanced reuse is useful for avoiding
  unnecessary CPU waits when only one buffer in a slot is still busy.

## Expected Timeline Differences

Direct simple and direct advanced should look very similar in Nsight because
same-stream ordering already protects device buffers.

Row-based simple and row-based advanced should also have the same required
stage dependencies:

```text
H2D -> kernel -> D2H
```

The advanced row-based model mainly changes when individual buffers can be
reused after the queue wraps. To see those reuse paths, use more iterations than
slots, for example:

```python
iterations = 20
queue_depth = 2
```

or:

```python
iterations = 40
queue_depth = 4
```

## Nsight Commands

Run from `~/lcls2` on a GPU node:

```bash
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
mkdir -p /pscratch/sd/m/monarin/psana2-gpu/practice
```

Direct:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_direct_practice \
  python psana/psana/gpu/multiowner/cuda_direct_practice.py
```

Row-based:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_row_based_practice \
  python psana/psana/gpu/multiowner/cuda_row_based_practice.py
```

Direct advanced reuse:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_direct_advance_practice \
  python psana/psana/gpu/multiowner/cuda_direct_advance_practice.py
```

Row-based advanced reuse:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_row_based_advance_practice \
  python psana/psana/gpu/multiowner/cuda_row_based_advance_practice.py
```
