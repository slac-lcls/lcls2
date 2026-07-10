#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/tooling_probe.log
#SBATCH --job-name=tool_probe

# Inventory of NVIDIA benchmarking/profiling tooling on an ampere node.

echo "=== node: $(hostname) ==="

echo "=== PATH tools ==="
for t in nvidia-smi nsys ncu nvprof dcgmi nvitop; do
    printf "%-12s" "$t:"; command -v $t 2>/dev/null || echo "not found"
done

echo "=== common install locations ==="
ls -d /usr/local/cuda* /opt/nvidia/nsight* /usr/local/NVIDIA* 2>/dev/null
ls /usr/local/cuda/bin 2>/dev/null | head -30

echo "=== nvidia-smi basics ==="
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1

echo "=== dmon sample (PCIe rx/tx MB/s, SM/mem util) 5s ==="
timeout 6 nvidia-smi dmon -s put -d 1 -c 5 2>&1

echo "=== DCGM ==="
command -v dcgmi && dcgmi discovery -l 2>&1 | head -5

echo "=== IB counters present? ==="
ls /sys/class/infiniband/ 2>/dev/null

echo "=== CUDA events via cupy (overlap-safe timing primitive) ==="
cd /sdf/scratch/users/a/ajshack/lcls2 && source setup_env.sh >/dev/null 2>&1
python - <<'EOF'
import cupy as cp
a = cp.arange(10**6, dtype=cp.float32)
s, e = cp.cuda.Event(), cp.cuda.Event()
s.record(); b = a * 2; e.record(); e.synchronize()
print("cupy.cuda.Event elapsed:", cp.cuda.get_elapsed_time(s, e), "ms  OK")
import cupyx.profiler as p
print("cupyx.profiler.benchmark:", p.benchmark(lambda: a*2, n_repeat=10).gpu_times.mean()*1e3, "ms  OK")
try:
    h = cp.cuda.alloc_pinned_memory(1024)
    print("pinned memory alloc: OK")
except Exception as ex:
    print("pinned memory alloc FAILED:", ex)
EOF
