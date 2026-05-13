# GPU Context/Stream Overlap Test Notes

This note documents the standalone overlap benchmark:

```text
psana/psana/gpu/multiowner/run_gpu_context_overlap.py
psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh
```

The benchmark is independent of psana event data. It uses NumPy/CuPy buffers,
async H2D copies, and a synthetic CUDA `spin_kernel` to study overlap across:

- one process with multiple CUDA streams
- multiple MPI ranks with one CUDA stream per rank
- multiple MPI ranks with CUDA MPS enabled

## How `run_gpu_context_overlap.py` Works

The script defines a CUDA kernel, `SPIN_KERNEL`, through CuPy `RawKernel`.
Each CUDA thread computes one output element:

```cpp
long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
if (i >= n) {
    return;
}

float x = ((float)src[i % src_n]) * 0.001f + seed + ((float)(i & 1023)) * 0.000001f;
for (int k = 0; k < spin_iters; ++k) {
    x = fmaf(x, 1.000001f, 0.000001f);
    if (x > 4096.0f) {
        x -= 4096.0f;
    }
}
dst[i] = x;
```

The important parameters are:

```text
--iterations          Timed outer iterations per rank.
--warmup              Untimed warmup outer iterations.
--data-size           Size of host input and device input buffers.
--compute-elements    Number of float32 output elements processed by the kernel.
--compute-iters       Arithmetic loop count inside each thread.
--block-size          CUDA threads per block. Default: 256.
--pipeline-depth      Number of reusable slot sets per rank.
--streams-per-rank    Number of independent CUDA streams launched per outer iteration.
```

If `--compute-elements` is not specified, the script uses:

```text
compute_elements = data_nbytes / sizeof(float)
```

For example:

```text
heavy:
  --data-size 64M
  compute_elements = 64 MiB / 4 = 16,777,216
  block_size = 256
  grid_size = 65,536 blocks

tinygrid:
  --data-size 4M
  --compute-elements 8192
  block_size = 256
  grid_size = 32 blocks
```

Each `Slot` contains:

```text
host      NumPy uint8 host buffer
dev_in    CuPy uint8 device input buffer
dev_out   CuPy float32 device output buffer
stream    non-blocking CuPy CUDA stream
```

The script allocates slots as:

```python
slot_sets = [
    [
        Slot(...) for _ in range(args.streams_per_rank)
    ]
    for _ in range(args.pipeline_depth)
]
```

So:

```text
pipeline_depth=1, streams_per_rank=1:
  [[slot0]]

pipeline_depth=1, streams_per_rank=8:
  [[slot0, slot1, ..., slot7]]

pipeline_depth=4, streams_per_rank=8:
  [[8 slots], [8 slots], [8 slots], [8 slots]]
```

For each outer iteration, the script picks:

```python
slot_set = slot_sets[iteration % args.pipeline_depth]
```

Before reusing a slot set, it synchronizes every stream in that set:

```python
for slot in slot_set:
    slot.stream.synchronize()
```

Then it loops over each slot/stream in the selected slot set:

```text
1. Fill or read host input buffer.
2. Enqueue async H2D copy on that slot's stream.
3. Enqueue `spin_kernel` on that same stream.
4. Save CUDA events for timing.
```

For `pipeline_depth=1`, the two main modes are:

```text
8 MPI ranks x 1 stream:
  8 Python processes
  each rank owns 1 slot/stream
  each rank launches 1 workload per outer iteration

1 MPI rank x 8 streams:
  1 Python process
  rank 0 owns 8 slots/streams
  rank 0 launches 8 workloads per outer iteration
```

Both produce the same timed workload count with:

```text
--iterations 80
8 ranks x 1 stream      -> 8 * 80 = 640 workloads
1 rank x 8 streams      -> 1 * 80 * 8 = 640 workloads
```

## CPU I/O Modes

The default mode is:

```text
--io-mode fill
```

In this mode, the script fills the host buffer:

```python
slot.host.fill((rank * 17 + workload_iteration) & 0xFF)
```

This does not change workload size. It just gives each rank/workload a different
byte pattern.

The optional file-read mode is:

```text
--io-mode pread --io-file <path>
```

This uses `_read_preadv()` to read `data_size` bytes from a real file into the
host buffer before the H2D copy. The recent tests used the default `fill` mode.

## Commands Used

These commands assume the usual environment:

```bash
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh
activate_psana
```

### Tinygrid: 1 Rank x 8 Streams

```bash
psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh \
  -n 1 \
  --gres-gpu 1 \
  -o /pscratch/sd/m/monarin/psana2-gpu/singleowner/gpuctx_n1_streams8_tinygrid \
  -- \
  --iterations 80 \
  --streams-per-rank 8 \
  --pipeline-depth 1 \
  --data-size 4M \
  --compute-elements 8192 \
  --compute-iters 50000 \
  --gpu-id 0
```

### Tinygrid: 8 Ranks x 1 Stream, No MPS

```bash
psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh \
  -n 8 \
  --gres-gpu 1 \
  -o /pscratch/sd/m/monarin/psana2-gpu/singleowner/gpuctx_n8_streams1_tinygrid_nomps \
  -- \
  --iterations 80 \
  --streams-per-rank 1 \
  --pipeline-depth 1 \
  --data-size 4M \
  --compute-elements 8192 \
  --compute-iters 50000 \
  --gpu-id 0
```

### Heavy: 1 Rank x 8 Streams

```bash
psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh \
  -n 1 \
  --gres-gpu 1 \
  -o /pscratch/sd/m/monarin/psana2-gpu/singleowner/gpuctx_n1_streams8_d1_single \
  -- \
  --iterations 80 \
  --streams-per-rank 8 \
  --pipeline-depth 1 \
  --data-size 64M \
  --compute-iters 4096 \
  --gpu-id 0
```

### Heavy: 8 Ranks x 1 Stream, No MPS

```bash
psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh \
  -n 8 \
  --gres-gpu 1 \
  -o /pscratch/sd/m/monarin/psana2-gpu/multiowner/nomps/gpuctx_n8_d1_nomps \
  -- \
  --iterations 80 \
  --streams-per-rank 1 \
  --pipeline-depth 1 \
  --data-size 64M \
  --compute-iters 4096 \
  --gpu-id 0
```

### MPS Setup and Checks

For MPS runs, start MPS before launching the benchmark:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}-gpuctx
export CUDA_MPS_LOG_DIRECTORY=/pscratch/sd/m/monarin/psana2-gpu/tmp/mps

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

nvidia-cuda-mps-control -d
```

Warm up CuPy so the MPS server is created:

```bash
python - <<'PY'
import cupy as cp
x = cp.arange(1)
cp.cuda.Stream.null.synchronize()
print("mps_warmup_device_count", cp.cuda.runtime.getDeviceCount())
print("mps_warmup_value", int(x.get()[0]))
PY
```

Check the MPS server:

```bash
echo get_server_list | nvidia-cuda-mps-control
```

Run the same `run_nsys_gpu_context_overlap.sh` command with the MPS environment
variables exported. Afterward, inspect:

```text
$CUDA_MPS_LOG_DIRECTORY/server.log
$CUDA_MPS_LOG_DIRECTORY/control.log
```

The MPS server log should contain lines like:

```text
Client {PID: ..., Context ID: ...} connected
```

In the pipeline-depth sweep, each MPS case wrote a `*_mps_clients.txt` file with
the number of new client connection lines observed after that case. Each case
recorded `client_connected_delta=20`.

Stop MPS when done:

```bash
echo quit | nvidia-cuda-mps-control
```

## Tinygrid vs Heavy Findings

All cases below ran 640 timed workloads.

### Tinygrid

```text
--data-size 4M
--compute-elements 8192
--compute-iters 50000
```

| Case | Wall Time (s) | Workloads/s | Sum Kernel CUDA Time (s) | CUDA Work / Wall |
|---|---:|---:|---:|---:|
| 1 rank x 8 streams | 0.246639 | 2594.9 | 0.680600 | 3.269 |
| 8 ranks x 1 stream, no MPS | 0.892681 | 716.9 | 3.740521 | 4.318 |
| 8 ranks x 1 stream, MPS | 0.116050 | 5514.9 | 0.689985 | 7.170 |

Tinygrid has only 32 blocks per kernel, so multiple kernels can co-reside on the
GPU. This is why kernel overlap is visible in both `1 rank x 8 streams` and
`8 ranks x 1 stream` with MPS.

### Heavy

```text
--data-size 64M
--compute-iters 4096
--compute-elements unset, defaults to 16,777,216
```

| Case | Wall Time (s) | Workloads/s | Sum Kernel CUDA Time (s) | CUDA Work / Wall |
|---|---:|---:|---:|---:|
| 1 rank x 8 streams | 12.575893 | 50.9 | 47.167017 | 3.909 |
| 8 ranks x 1 stream, no MPS | 10.973009 | 58.3 | 76.606539 | 7.202 |
| 8 ranks x 1 stream, MPS | 9.403605 | 68.1 | 71.774346 | 7.806 |

Heavy has 65,536 blocks per kernel, enough for one kernel to keep the GPU busy.
Kernels mostly serialize even when multiple streams or ranks submit work.

## Pipeline Depth Sweep

Pipeline-depth results are under:

```text
/pscratch/sd/m/monarin/psana2-gpu/pipeline-depth
```

Folders:

```text
singleowner
multowner_nomps
multiowner_mps
```

All cases used 640 timed workloads.

### Tinygrid Wall Time vs Pipeline Depth

| Case | Depth 1 | Depth 2 | Depth 4 | Depth 8 | Best |
|---|---:|---:|---:|---:|---:|
| 1 rank x 8 streams | 0.226393 | 0.199366 | 0.223543 | 0.233184 | depth 2 |
| 8 ranks x 1 stream, no MPS | 0.695346 | 0.370798 | 0.212219 | 0.154263 | depth 8 |
| 8 ranks x 1 stream, MPS | 0.106710 | 0.055099 | 0.072298 | 0.115513 | depth 2 |

Tinygrid benefits from queue depth, especially for multi-rank no-MPS. For MPS,
depth 2 was best; deeper queues increased copy/queue pressure and became slower.

### Heavy Wall Time vs Pipeline Depth

| Case | Depth 1 | Depth 2 | Depth 4 | Depth 8 | Best |
|---|---:|---:|---:|---:|---:|
| 1 rank x 8 streams | 9.890786 | 9.536190 | 9.701094 | 9.699368 | depth 2 |
| 8 ranks x 1 stream, no MPS | 10.955530 | 10.926527 | 11.030471 | 11.117762 | depth 2 |
| 8 ranks x 1 stream, MPS | 9.419909 | 9.346293 | 9.690539 | 9.677040 | depth 2 |

Heavy shows only small wall-time improvement from queue depth. This matches the
timeline interpretation: each heavy kernel already saturates the GPU, so deeper
queues mostly enqueue more pending work rather than creating useful kernel
overlap.

## Takeaways

- `streams_per_rank` controls how many independent CUDA streams each rank submits
  per outer iteration.
- `pipeline_depth` controls how many slot sets can be in flight before reuse.
- Tinygrid shows real kernel overlap because each kernel is small enough not to
  occupy the whole GPU.
- Heavy mostly serializes kernels because each kernel has enough blocks to fill
  the GPU by itself.
- MPS improves multi-rank behavior significantly for tinygrid and modestly for
  heavy.
- Queue depth helps tinygrid more than heavy; depth 2 is a good general starting
  point, while deeper queues are workload dependent.
