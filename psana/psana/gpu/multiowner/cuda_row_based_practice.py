import numpy as np
import cupy as cp
import cupyx
import time

SCALE_KERNEL = r"""
extern "C" __global__
void scale_kernel(const float* src, float* dst, long long n, float scale, int spin_iters)
{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float x = src[i] * scale +1.0f;
        for (int j = 0; j < spin_iters; j++) {
            x = fmaf(x, 1.000001f, 0.000001f);  // Just to add some extra work and make the kernel take more time
        }
        dst[i] = x;
    }
}
"""

def main():
    """ This example demonstrates a stage-stream/row-based approach with separate streams for H2D, kernel execution, and D2H."""
    cp.cuda.Device(0).use()

    # Tunable parameters
    nitems = 1_000_000
    block_size = 256
    grid_size = (nitems + block_size - 1) // block_size
    iterations = 10
    spin_iters = 10000  # Number of iterations in the kernel to increase its execution time
    queue_depth = 10

    # Create separate streams for each stage of the pipeline (stage-stream approach)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    kernel_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)
    kernel = cp.RawKernel(SCALE_KERNEL, "scale_kernel")
    kernel.compile()

    host_in = [cupyx.empty_pinned(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    host_out = [cupyx.empty_pinned(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    dev_in = [cp.empty(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    dev_out = [cp.empty(nitems, dtype=cp.float32) for _ in range(queue_depth)]

    done_events = [None for _ in range(queue_depth)]  # To track when each slot's operations are done

    # Each iteration contains one H2D, one kernel, and one D2H, all recorded with events for timing.
    h2d_total_s = 0.0
    kernel_total_s = 0.0
    d2h_total_s = 0.0
    wall0 = time.perf_counter()
    records = [] # keep track of all events for final synchronization and timing
    print(f"Row-based Method: Starting {iterations} iterations with {queue_depth} slots and {spin_iters} spin iterations of H2D -> kernel -> D2H...")

    for iteration in range(iterations):
        slot = iteration % queue_depth

        # Wait only if this slot has old in-flight work.
        if done_events[slot] is not None:
            done_events[slot].synchronize()  

        scale = np.float32(2.0 + iteration)  # Just to have some variation in the kernel argument
        host_in[slot].fill(np.float32(iteration))  # Fill input with some data
        host_out[slot].fill(np.float32(-1.0))  # Fill output with a known value to check results later

        h2d_start = cp.cuda.Event()
        h2d_end = cp.cuda.Event()
        kernel_start = cp.cuda.Event()
        kernel_end = cp.cuda.Event()
        d2h_start = cp.cuda.Event()
        d2h_end = cp.cuda.Event()

        # Stage 1: H2D
        with h2d_stream:
            h2d_start.record()
            dev_in[slot].set(host_in[slot], stream=h2d_stream)
            h2d_end.record()
        
        # Stage 2: Kernel execution (depends on H2D completion)
        with kernel_stream:
            kernel_stream.wait_event(h2d_end)  # Ensure kernel waits for H2D to complete
            kernel_start.record()
            kernel(
                (grid_size,),
                (block_size,),
                (dev_in[slot], dev_out[slot], np.int64(nitems), scale, np.int32(spin_iters)),
                stream=kernel_stream,
            )
            kernel_end.record()
        
        # Stage 3: D2H (depends on kernel completion)
        with d2h_stream:
            d2h_stream.wait_event(kernel_end)  # Ensure D2H waits for kernel to complete
            d2h_start.record()
            dev_out[slot].get(out=host_out[slot], stream=d2h_stream, blocking=False)
            d2h_end.record()
            done_events[slot] = d2h_end  # Mark this slot as having work that will be done at d2h_end
        
        records.append((h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end))

    for event in done_events:
        if event is not None:
            event.synchronize()  # Ensure all work is done before final timing
    
    # Now calculate total times from all recorded events
    for h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end in records:
        h2d_total_s += cp.cuda.get_elapsed_time(h2d_start, h2d_end) / 1000.0
        kernel_total_s += cp.cuda.get_elapsed_time(kernel_start, kernel_end) / 1000.0
        d2h_total_s += cp.cuda.get_elapsed_time(d2h_start, d2h_end) / 1000.0   
    
    wall_s = time.perf_counter() - wall0
    print(f"Total H2D time: {h2d_total_s:.4f}s, Total Kernel time: {kernel_total_s:.4f}s, Total D2H time: {d2h_total_s:.4f}s, Total wall time: {wall_s:.4f}s")

if __name__ == "__main__":
    main()
    