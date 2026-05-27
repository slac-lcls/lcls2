# CUDA Overlap Model Comparison

This note summarizes the CUDA stream models used to study overlap between:

```text
H2D copy -> scale_kernel -> D2H copy
```

The larger MPI/MPS benchmark summary is in:

```text
psana/psana/gpu/multiowner/gpu_context_overlap_notes.md
```

## Goal

The goal is to compare two ways of expressing the same work:

```text
direct/per-workload:
  each workload owns one stream containing H2D -> kernel -> D2H

row-based/stage-stream:
  H2D, kernel, and D2H are submitted to separate stage streams
```

Both models can overlap copy and compute when work is long enough and copies are
asynchronous. The row-based model makes stage dependencies more explicit in the
timeline, but for the small tests used so far the CUDA HW row looked mostly the
same as the direct model.

## Simple Direct Model

Script:

```text
psana/psana/gpu/multiowner/cuda_direct_practice.py
```

The direct model creates one stream per slot:

```text
streams[slot]: H2D -> scale_kernel -> D2H
```

Each slot owns:

```text
host_in[slot]
host_out[slot]
dev_in[slot]
dev_out[slot]
streams[slot]
```

CUDA guarantees in-order execution within a stream. Therefore the kernel waits
for the H2D copy, and D2H waits for the kernel, without extra stream-wait calls:

```text
H2D -> kernel -> D2H
```

The simple direct script uses one completion event per slot:

```python
done_events[slot] = d2h_end
```

Before reusing a slot, the CPU waits for the whole previous chain:

```python
if done_events[slot] is not None:
    done_events[slot].synchronize()
```

This is conservative and correct. The whole slot is reused only after D2H has
finished.

Run:

```bash
cd ~/lcls2
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
python psana/psana/gpu/multiowner/cuda_direct_practice.py
```

Run with Nsight Systems:

```bash
mkdir -p /pscratch/sd/m/monarin/psana2-gpu/practice
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_direct_practice \
  python psana/psana/gpu/multiowner/cuda_direct_practice.py
```

## Simple Row-Based Model

Script:

```text
psana/psana/gpu/multiowner/cuda_row_based_practice.py
```

The row-based model creates one stream per stage:

```text
h2d_stream
kernel_stream
d2h_stream
```

For each iteration, CUDA events connect the stage streams:

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

The dependencies mean:

```text
kernel waits for current H2D because dev_in[slot] must be ready
D2H waits for current kernel because dev_out[slot] must be ready
```

The row-based model is useful because the timeline directly shows the H2D,
kernel, and D2H rows as separate lanes. That made it easier to reason about the
pipeline, even though the CUDA HW row was more or less the same as the direct
model for the small comparison tests.

The simple row-based script also uses:

```python
done_events[slot] = d2h_end
```

So, like the simple direct model, it reuses a slot only after the full chain has
completed.

Run:

```bash
cd ~/lcls2
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
python psana/psana/gpu/multiowner/cuda_row_based_practice.py
```

Run with Nsight Systems:

```bash
mkdir -p /pscratch/sd/m/monarin/psana2-gpu/practice
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_row_based_practice \
  python psana/psana/gpu/multiowner/cuda_row_based_practice.py
```

## Shared Buffer-Reuse Dependency

Both simple models rely on the same conservative buffer-reuse rule:

```text
do not reuse the slot until D2H is done
```

This matters because a slot contains both host and device buffers:

```text
host_in[slot]   source for H2D
dev_in[slot]    destination for H2D and source for kernel
dev_out[slot]   destination for kernel and source for D2H
host_out[slot]  destination for D2H
```

With `queue_depth = 2`, this is double buffering:

```text
slot 0 has host/device input/output buffers
slot 1 has host/device input/output buffers
```

With `queue_depth > 2`, this becomes N-buffering.

## Advanced Buffer Reuse

In addition to the simple model comparison, we added advanced buffer reuse to
both models.

The advanced reuse scripts split the single `done_events[slot]` marker into
more specific completion events:

```text
last_h2d_done[slot]
last_kernel_done[slot]
last_d2h_done[slot]
```

The rule is:

```text
Before touching a buffer again, wait for the event from the previous operation
that last touched that same buffer.
```

The event meanings are:

```text
last_h2d_done:
  previous H2D is done reading host_in[slot]

last_kernel_done:
  previous kernel is done reading dev_in[slot]

last_d2h_done:
  previous D2H is done reading dev_out[slot] and writing host_out[slot]
```

## Advanced Direct Model

Script:

```text
psana/psana/gpu/multiowner/cuda_direct_advance_practice.py
```

In the direct model, advanced reuse only needs CPU-side waits for host buffers:

```python
if last_h2d_done[slot] is not None:
    last_h2d_done[slot].synchronize()
host_in[slot].fill(...)

if last_d2h_done[slot] is not None:
    last_d2h_done[slot].synchronize()
host_out[slot].fill(...)
```

The direct model does not need extra GPU stream waits for device buffer reuse
because each slot uses one ordered stream:

```text
streams[slot]: old H2D -> old kernel -> old D2H -> new H2D -> new kernel -> new D2H
```

Same-stream ordering already protects `dev_in[slot]` and `dev_out[slot]`.

Run:

```bash
cd ~/lcls2
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
python psana/psana/gpu/multiowner/cuda_direct_advance_practice.py
```

Run with Nsight Systems:

```bash
mkdir -p /pscratch/sd/m/monarin/psana2-gpu/practice
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_direct_advance_practice \
  python psana/psana/gpu/multiowner/cuda_direct_advance_practice.py
```

## Advanced Row-Based Model

Script:

```text
psana/psana/gpu/multiowner/cuda_row_based_advance_practice.py
```

The row-based model needs explicit stream waits because H2D, kernel, and D2H
are submitted to different streams.

Before H2D overwrites `dev_in[slot]`, H2D waits for the previous kernel that
read `dev_in[slot]`:

```python
with h2d_stream:
    if last_kernel_done[slot] is not None:
        h2d_stream.wait_event(last_kernel_done[slot])
    dev_in[slot].set(...)
    last_h2d_done[slot] = h2d_end
```

Before the kernel runs, the kernel stream waits for two conditions:

```python
with kernel_stream:
    kernel_stream.wait_event(h2d_end)
    if last_d2h_done[slot] is not None:
        kernel_stream.wait_event(last_d2h_done[slot])
    kernel(...)
    last_kernel_done[slot] = kernel_end
```

These waits protect different buffers:

```text
kernel_stream.wait_event(h2d_end):
  current input dev_in[slot] is ready

kernel_stream.wait_event(last_d2h_done[slot]):
  previous output dev_out[slot] is no longer being copied out
```

D2H waits for the current kernel:

```python
with d2h_stream:
    d2h_stream.wait_event(kernel_end)
    dev_out[slot].get(..., blocking=False)
    last_d2h_done[slot] = d2h_end
```

Run:

```bash
cd ~/lcls2
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
python psana/psana/gpu/multiowner/cuda_row_based_advance_practice.py
```

Run with Nsight Systems:

```bash
mkdir -p /pscratch/sd/m/monarin/psana2-gpu/practice
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /pscratch/sd/m/monarin/psana2-gpu/practice/cuda_row_based_advance_practice \
  python psana/psana/gpu/multiowner/cuda_row_based_advance_practice.py
```

## Current Observation

For the small comparison problem used so far, roughly:

```text
iterations = 10
queue_depth = 2
```

the advanced row-based model did not show an obvious scheduling improvement over
the simple row-based model. That is expected: the problem is small, the queue is
short, and the simple model's conservative `done_events[slot] = d2h_end` wait
does not yet dominate the total time.

The advanced reuse model is still useful because it expresses the more precise
ownership rules:

```text
host_in can be reused after H2D
dev_in can be reused after kernel
dev_out and host_out can be reused after D2H
```

This is the version to keep in mind for a larger or more realistic pipeline
where CPU scheduling and buffer reuse can become visible in the timeline.

## Other Lessons

- `kernel.compile()` removes CuPy first-use compile/module-load noise from the
  Nsight timeline.
- `dev_out.get(..., blocking=False)` is required for async D2H. CuPy defaults
  to `blocking=True`, which blocks the Python producer loop.
- Increasing `spin_iters` made kernels long enough to see H2D/kernel overlap.
- The sum of per-kernel CUDA event times can be larger than wall time when work
  overlaps.
- Direct and row-based are different ways to express the same work; row-based is
  often easier to explain because the stage dependencies are explicit.
