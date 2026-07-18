# Four-node EB topology benchmark

This report summarizes the four Perlmutter Jungfrau GPU benchmark runs in
`tmp/20260717/03`.  The runs were collected in ABBA order on 2026-07-17:

1. spread,
2. no spread,
3. no spread (try 2),
4. spread (try 2).

## Test configuration

| Item | Value |
|---|---|
| Slurm job | `56057984` |
| Nodes | `nid001076`, `nid001077`, `nid001080`, `nid001132` |
| Experiment | `mfx101589626` |
| Run | `31` |
| XTC directory | `/pscratch/sd/p/psdatmgr/psdm/mfx/mfx101589626/xtc` |
| MPI ranks | 132: 1 SMD0 + 12 EB + 119 BD |
| Tasks per node | 33 |
| Benchmark | `bench_calib.py -n 250 --warmup 10 --wait-split` |
| Detector | Jungfrau, 16,777,216 pixels/event |
| Raw detector payload | 33,554,432 bytes/event |
| CXI counter | `hni_sts_rx_ok_octets` through `net_bandwidth.py --source cxi` |

The complete `run.events()` loop yielded 33,034 events in every run.  The GPU
aggregate calculation used 31,725 post-warmup events across the 119 BD ranks.
The difference is 1,309 events, or 11 excluded events per BD rank.

The reported rates have different definitions:

- **Loop rate** is `33,034 / maximum MPI-rank loop time`.  It covers the complete
  `run.events()` iteration, including warmup.
- **Aggregate rate** is the sum of the 119 individual BD post-warmup rates.
  It is not `31,725 / maximum loop time`, so it is expected to be larger than
  the loop rate.
- **Init time** is the maximum MPI-rank time from the pre-DataSource barrier to
  the beginning of that rank's `run.events()` loop.

## EB and BD placement

Both layouts used 12 EBs and 119 BDs.  Rank 0, the SMD0 rank, was on
`nid001076`.

| Topology | Node | EB world ranks | EBs | BDs |
|---|---|---|---:|---:|
| Spread | `nid001076` | 1-3 | 3 | 29 |
| Spread | `nid001077` | 33-35 | 3 | 30 |
| Spread | `nid001080` | 66-68 | 3 | 30 |
| Spread | `nid001132` | 99-101 | 3 | 30 |
| No spread | `nid001076` | 1-12 | 12 | 20 |
| No spread | `nid001077` | none | 0 | 33 |
| No spread | `nid001080` | none | 0 | 33 |
| No spread | `nid001132` | none | 0 | 33 |

The spread placement therefore balanced the BD population at approximately
30 BDs per node.  The no-spread placement left only 20 BDs on the SMD0/EB node
and placed 33 BDs on each of the other nodes.

## Initialization, loop, and aggregate rates

| Order | Topology | Log | Init time | Loop time | Loop rate | Aggregate rate |
|---:|---|---|---:|---:|---:|---:|
| 1 | Spread | `gpu_bench_spread_4nodes.log` | 70.69 s | 64.44 s | 512.6 Hz | 792.5 Hz |
| 2 | No spread | `gpu_bench_nospread_4nodes.log` | 69.82 s | 65.34 s | 505.6 Hz | 778.7 Hz |
| 3 | No spread, try 2 | `gpu_bench_nospread_4nodes_try2.log` | 77.50 s | 70.69 s | 467.3 Hz | 694.1 Hz |
| 4 | Spread, try 2 | `gpu_bench_spread_4nodes_try2.log` | 70.75 s | 63.27 s | 522.1 Hz | 829.5 Hz |
|  | **Spread mean** |  | **70.72 s** | **63.86 s** | **517.4 Hz** | **811.0 Hz** |
|  | **No-spread mean** |  | **73.66 s** | **68.02 s** | **486.5 Hz** | **736.4 Hz** |

On the two-run means, spread was 6.4% faster by complete-loop rate and 10.1%
faster by aggregate post-warmup rate.  The second no-spread run was visibly
slower than the other three runs, so two repetitions are not sufficient to
separate topology from storage-window variability.

## Per-node events and CXI receive bytes

The nominal payload is `events * 33,554,432 bytes`.  CXI RX is the physical
Slingshot receive traffic measured with `hni_sts_rx_ok_octets`; it includes
Lustre/KFI protocol traffic, packet overhead, MPI traffic, metadata, and any
other node traffic during the sampled interval.  It is therefore not expected
to equal the nominal Jungfrau payload exactly.

The CXI values below use synchronized timestamp groups for which all four node
samples were present.  The first spread and first no-spread logs each had four
incomplete timestamp groups, so their CXI byte totals are conservative.  Both
try-2 logs had complete timestamp groups.

| Run | Node | BDs | Loop events | Event share | Nominal payload | CXI RX bytes |
|---|---|---:|---:|---:|---:|---:|
| Spread 1 | `nid001076` | 29 | 9,000 | 27.24% | 301.990 GB | 363,618,959,790 |
| Spread 1 | `nid001077` | 30 | 9,000 | 27.24% | 301.990 GB | 362,739,339,250 |
| Spread 1 | `nid001080` | 30 | 8,034 | 24.32% | 269.576 GB | 320,935,573,090 |
| Spread 1 | `nid001132` | 30 | 7,000 | 21.19% | 234.881 GB | 277,349,737,230 |
| **Spread 1 total** |  | **119** | **33,034** | **100%** | **1,108.437 GB** | **1,324,643,609,360** |
| No spread 1 | `nid001076` | 20 | 7,481 | 22.65% | 251.021 GB | 312,157,834,970 |
| No spread 1 | `nid001077` | 33 | 8,392 | 25.40% | 281.589 GB | 351,652,119,690 |
| No spread 1 | `nid001080` | 33 | 8,894 | 26.92% | 298.433 GB | 376,415,596,480 |
| No spread 1 | `nid001132` | 33 | 8,267 | 25.03% | 277.394 GB | 336,680,009,370 |
| **No spread 1 total** |  | **119** | **33,034** | **100%** | **1,108.437 GB** | **1,376,905,560,510** |
| No spread 2 | `nid001076` | 20 | 6,720 | 20.34% | 225.486 GB | 288,638,760,520 |
| No spread 2 | `nid001077` | 33 | 9,531 | 28.85% | 319.807 GB | 405,138,877,010 |
| No spread 2 | `nid001080` | 33 | 8,096 | 24.51% | 271.657 GB | 342,990,721,760 |
| No spread 2 | `nid001132` | 33 | 8,687 | 26.30% | 291.487 GB | 350,889,448,530 |
| **No spread 2 total** |  | **119** | **33,034** | **100%** | **1,108.437 GB** | **1,387,657,807,820** |
| Spread 2 | `nid001076` | 29 | 8,000 | 24.22% | 268.435 GB | 322,026,308,350 |
| Spread 2 | `nid001077` | 30 | 9,000 | 27.24% | 301.990 GB | 357,307,015,090 |
| Spread 2 | `nid001080` | 30 | 7,034 | 21.29% | 236.022 GB | 282,822,986,050 |
| Spread 2 | `nid001132` | 30 | 9,000 | 27.24% | 301.990 GB | 355,172,087,240 |
| **Spread 2 total** |  | **119** | **33,034** | **100%** | **1,108.437 GB** | **1,317,328,396,730** |

The event allocation was dynamic rather than evenly partitioned.  Individual
node shares ranged from 20.34% to 28.85%, and nodes processing more events
generally received more CXI bytes.

## CXI throughput summary

The active average is the mean cluster RX rate over synchronized samples where
the cluster rate was at least 1 GB/s.

| Run | Synchronized CXI RX | Active average RX | Peak cluster RX |
|---|---:|---:|---:|
| Spread 1 | 1,324.64 GB | 21.25 GB/s | 34.20 GB/s |
| No spread 1 | 1,376.91 GB | 21.75 GB/s | 33.96 GB/s |
| No spread 2 | 1,387.66 GB | 20.30 GB/s | 31.68 GB/s |
| Spread 2 | 1,317.33 GB | 22.24 GB/s | 32.32 GB/s |
| **Spread mean** | **1,320.99 GB** | **21.74 GB/s** |  |
| **No-spread mean** | **1,382.28 GB** | **21.03 GB/s** |  |

The synchronized no-spread mean contains approximately 61.30 GB more CXI RX
than the spread mean.  This must not be interpreted as EB-to-BD payload: EBs
send only small-data messages, typically kilobytes, to BDs.  The CXI counter is
node-wide and does not distinguish Lustre reads from MPI, protocol overhead,
read amplification, or unrelated traffic.

## Reproduction commands

### Activate psana and CuPy

Run the environment setup from the repository checkout:

```bash
cd ~/lcls2
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh ~/.conda-envs/psana-build
activate_psana
activate_psana2_gpu_cupy

unset SIT_PSDM_OFFSITE
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/calib_ws/
mkdir -p tmp/20260717/03
```

The benchmark and `net_bandwidth.py` are executed directly from the source
tree, so edits confined to those scripts do not require rebuilding.  The EB
override is in `psana.datasource`, however, and must be rebuilt if the active
import resolves to an installed copy that predates the change.  Verify the
runtime before launching:

```bash
python - <<'PY'
import psana
import psana.datasource

print("psana:", psana.__file__)
print("datasource:", psana.datasource.__file__)
PY
```

If necessary, update `install_psana` from the login node with
`./build_psana.sh -j 8`, reactivate the environment, and verify the paths again.

### Request the four-node allocation

```bash
salloc --nodes 4 --qos interactive --time 02:00:00 \
  -C gpu -A lcls --gpus-per-node=1
```

The recorded allocation was job `56057984` on
`nid[001076-001077,001080,001132]`.  Inside the allocation:

```bash
export JOB_ID="${SLURM_JOB_ID:-56057984}"
scontrol show hostnames "$SLURM_JOB_NODELIST"
```

### Verify MPI, CuPy, and CXI

```bash
srun --overlap --jobid="$JOB_ID" \
  -N 4 -n 4 --ntasks-per-node=1 \
  --distribution=block:block --label \
  python -u -c '
from mpi4py import MPI
import cupy as cp

print("host:", MPI.Get_processor_name())
print("MPI:", MPI.Get_library_version())
print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("GPUs visible:", cp.cuda.runtime.getDeviceCount())
'

srun --overlap --jobid="$JOB_ID" \
  -N 4 -n 4 --ntasks-per-node=1 \
  --distribution=block:block --label \
  python -u psana/psana/debugtools/net_bandwidth.py \
    --source cxi --include 'hsn*' --list
```

### Start and stop CXI monitoring

The following helpers run one monitor rank per node and preserve the labeled
one-second samples in a log file:

```bash
start_cxi_monitor() {
  local output_log="$1"

  srun --overlap --jobid="$JOB_ID" \
    -N 4 -n 4 --ntasks-per-node=1 \
    --distribution=block:block --label \
    python -u psana/psana/debugtools/net_bandwidth.py \
      --source cxi \
      --include 'hsn*' \
      --interval 1 \
      --samples 0 \
      --summary-only \
      --no-header \
    > "$output_log" 2>&1 &

  netmon_pid=$!
  sleep 2
}

stop_cxi_monitor() {
  kill "$netmon_pid"
  wait "$netmon_pid" || true
}
```

The `kill` is expected to leave Slurm cancellation messages at the end of the
CXI log because `--samples 0` means monitor until stopped.

### Spread benchmark command

This is the spread command used for runs 1 and 4.  `PS_EB_NODE_LOCAL=1` and
`PS_EB_PER_NODE=3` place three EBs on each node.

```bash
run_spread() {
  local output_log="$1"

  env \
    PS_EB_NODES=12 \
    PS_EB_NODE_LOCAL=1 \
    PS_EB_PER_NODE=3 \
  srun --overlap --jobid="$JOB_ID" \
    --mpi=cray_shasta \
    -N 4 -n 132 --ntasks-per-node=33 \
    --distribution=block:block \
    -c 2 --cpu-bind=cores \
    --gpus-per-node=4 --gpu-bind=none \
    --kill-on-bad-exit=1 \
    python -u psana/psana/gpu/bench_calib.py \
      -e mfx101589626 -r 31 \
      -n 250 --warmup 10 --wait-split \
      --dir /pscratch/sd/p/psdatmgr/psdm/mfx/mfx101589626/xtc \
    > "$output_log" 2>&1
}
```

### No-spread benchmark command

This is the no-spread command used for runs 2 and 3.  All 12 EBs are the
contiguous ranks immediately after SMD0 and therefore land on the first node.

```bash
run_nospread() {
  local output_log="$1"

  env -u PS_EB_PER_NODE \
    PS_EB_NODES=12 \
    PS_EB_NODE_LOCAL=0 \
  srun --overlap --jobid="$JOB_ID" \
    --mpi=cray_shasta \
    -N 4 -n 132 --ntasks-per-node=33 \
    --distribution=block:block \
    -c 2 --cpu-bind=cores \
    --gpus-per-node=1 --gpu-bind=none \
    --kill-on-bad-exit=1 \
    python -u psana/psana/gpu/bench_calib.py \
      -e mfx101589626 -r 31 \
      -n 250 --warmup 10 --wait-split \
      --dir /pscratch/sd/p/psdatmgr/psdm/mfx/mfx101589626/xtc \
    > "$output_log" 2>&1
}
```

### Execute the four runs in ABBA order

```bash
# 1. Spread
start_cxi_monitor tmp/20260717/03/cxi_rx_spread_4nodes.log
run_spread tmp/20260717/03/gpu_bench_spread_4nodes.log
stop_cxi_monitor

# 2. No spread
start_cxi_monitor tmp/20260717/03/cxi_rx_nospread_4nodes.log
run_nospread tmp/20260717/03/gpu_bench_nospread_4nodes.log
stop_cxi_monitor

# 3. No spread, second repetition
start_cxi_monitor tmp/20260717/03/cxi_rx_nospread_4nodes_try2.log
run_nospread tmp/20260717/03/gpu_bench_nospread_4nodes_try2.log
stop_cxi_monitor

# 4. Spread, second repetition
start_cxi_monitor tmp/20260717/03/cxi_rx_spread_4nodes_try2.log
run_spread tmp/20260717/03/gpu_bench_spread_4nodes_try2.log
stop_cxi_monitor
```

Quickly extract the benchmark headline lines with:

```bash
for log in tmp/20260717/03/gpu_bench*.log; do
  echo "== $log =="
  rg '^\[EB\]|^\[total\]|^\[aggregate\]|^  aggregate rate:' "$log"
done
```

### GPU-step caveat and controlled rerun

The saved commands used `--gpus-per-node=4` for spread but
`--gpus-per-node=1` for no spread.  `psana.gpu.init_gpu_rank()` reads
`SLURM_GPUS_ON_NODE`, so this can change BD-to-GPU assignment in addition to EB
placement.  The four results above must therefore be treated as exploratory,
not as a fully controlled EB-only comparison.

For a controlled rerun, use the same `--gpus-per-node` value in both functions.
If the allocation exposes all four GPUs per node, use `--gpus-per-node=4` in
both.  If it grants only one GPU per node, use `--gpus-per-node=1` in both and
confirm `SLURM_GPUS_ON_NODE` and `CUDA_VISIBLE_DEVICES` before collecting the
ABBA sequence.
