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
    cp.cuda.Device(0).use()

    # Tunable parameters
    nitems = 1_000_000
    block_size = 256
    grid_size = (nitems + block_size - 1) // block_size
    iterations = 10
    spin_iters = 10000  # Number of iterations in the kernel to increase its execution time
    queue_depth = 2
    
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(queue_depth)]
    kernel = cp.RawKernel(SCALE_KERNEL, "scale_kernel")
    kernel.compile()

    host_in = [cupyx.empty_pinned(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    host_out = [cupyx.empty_pinned(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    dev_in = [cp.empty(nitems, dtype=cp.float32) for _ in range(queue_depth)]
    dev_out = [cp.empty(nitems, dtype=cp.float32) for _ in range(queue_depth)]

    last_h2d_done = [None for _ in range(queue_depth)]  # To track when each slot's H2D is done
    last_d2h_done = [None for _ in range(queue_depth)]  # To track when each slot's D2H is done

    # Each iteration contains one H2D, one kernel, and one D2H, all recorded with events for timing.
    h2d_total_s = 0.0
    kernel_total_s = 0.0
    d2h_total_s = 0.0
    wall0 = time.perf_counter()
    records = [] # keep track of all events for final synchronization and timing
    print(f"Direct Method: Starting {iterations} iterations with {queue_depth} slots and {spin_iters} spin iterations of H2D -> kernel -> D2H...")

    for iteration in range(iterations):
        slot = iteration % queue_depth

        scale = np.float32(2.0 + iteration)  # Just to have some variation in the kernel argument

        """
        Advance reuse concept
        before refilling host_in: wait for last H2D
        before refilling host_out: wait for last D2H
        """
        
        # Wait only if this slot has old in-flight work.
        if last_h2d_done[slot] is not None:
            last_h2d_done[slot].synchronize()  
        host_in[slot].fill(np.float32(iteration))  # Fill input with some data

        if last_d2h_done[slot] is not None:
            last_d2h_done[slot].synchronize()  # Ensure previous D2H is done before reusing the slot
        host_out[slot].fill(np.float32(-1.0))  # Fill output with a known value to check results later

        h2d_start = cp.cuda.Event()
        h2d_end = cp.cuda.Event()
        kernel_start = cp.cuda.Event()
        kernel_end = cp.cuda.Event()
        d2h_start = cp.cuda.Event()
        d2h_end = cp.cuda.Event()
        with streams[slot]:
            h2d_start.record()
            dev_in[slot].set(host_in[slot], stream=streams[slot])
            h2d_end.record()
            last_h2d_done[slot] = h2d_end  # Mark this slot's H2D as done at h2d_end

            kernel_start.record()
            kernel(
                (grid_size,),
                (block_size,),
                (dev_in[slot], dev_out[slot], np.int64(nitems), scale, np.int32(spin_iters)),
            )
            kernel_end.record()

            d2h_start.record()
            dev_out[slot].get(out=host_out[slot], stream=streams[slot], blocking=False)
            d2h_end.record()
            last_d2h_done[slot] = d2h_end  # Mark this slot as having work that will be done at d2h_end

            records.append((h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end))

    for event in last_d2h_done:
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
