# GPU D2H Interval and NIC Bandwidth Results

## Summary

Test date: 2026-07-06

Dataset:

```text
exp=mfx101210926
run=387
dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc
det=jungfrau
```

Topology:

```text
mpirun -n 3
PS_EB_NODES=1
1 SMD0 rank, 1 EB rank, 1 BD rank
```

The GPU runs used the CPU-fallback KvikIO path, not true GDS:

```text
storage -> CPU DRAM -> GPU VRAM
```

The main finding is that the GPU read/calib path is much faster than the CPU
path when results stay on GPU.  When every event calls `.on_cpu`, the D2H copy
is synchronous and the total rate falls back to the CPU-like range.

## Event Rate Results

All GPU runs:

```text
batch_size=1
gpu_pool_depth=2
max_events=16000
```

| Run | D2H interval | Loop time | Rate | D2H calls | D2H time |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU `ds_count_events.py` raw path | none | 419.61 s | 38.1 Hz | 0 | 0 |
| GPU `jn0` | 0 | 159.47 s | 100.3 Hz | 0 | 0 |
| GPU `jn100` | 100 | 162.10 s | 98.7 Hz | 160 | 2.63 s |
| GPU `jn10` | 10 | 185.39 s | 86.3 Hz | 1600 | 25.90 s |
| GPU `jn1` | 1 | 404.16 s | 39.6 Hz | 16000 | 246.37 s |

Subtracting the measured D2H time gives a nearly constant GPU base time:

```text
jn100: 162.10 - 2.63   = 159.47 s
jn10:  185.39 - 25.90  = 159.49 s
jn1:   404.16 - 246.37 = 157.79 s
```

This confirms that the current D2H path is not overlapped with later GPU
read/H2D/compute work.  It is effectively additive.

## D2H Cost

The full Jungfrau calibrated result is approximately:

```text
32 segments * 512 * 1024 pixels * float32 = 64 MiB/event
```

Measured per copied event:

| Run | Copied events | D2H time / copied event |
| --- | ---: | ---: |
| `jn100` | 160 | 16.4 ms |
| `jn10` | 1600 | 16.2 ms |
| `jn1` | 16000 | 15.4 ms |

The per-copy cost is very linear, around 15-16 ms for one full Jungfrau calib
result.  Current `gpu_d2h_interval=N` is a sampling knob: it copies one event
every `N` events.  It is not a batched join of `N` events.

## NIC Read Bandwidth

NIC counters were collected with `net_bandwidth.py --source ethtool`.  The
table below uses active samples with:

```text
total_recv_Bps >= 1 GB/s
```

| Run | Active NIC recv avg | NIC recv max |
| --- | ---: | ---: |
| CPU | 1.27 GB/s | 1.51 GB/s |
| GPU `jn0` | 2.31 GB/s | 3.60 GB/s |
| GPU `jn100` | 2.13 GB/s | 3.52 GB/s |
| GPU `jn10` | 2.00 GB/s | 3.20 GB/s |
| GPU `jn1` | 1.30 GB/s | 1.58 GB/s |

When D2H is rare or absent, the GPU path drives much higher read bandwidth than
the CPU path, even though true GDS is not available.  When D2H happens every
event, the BD rank stalls on the synchronous `.on_cpu` copy and the NIC read
rate falls back to CPU-like levels.

## Interpretation

The current path behaves like:

```text
GPU read/H2D + GPU calib -> GPU result
optional synchronous .on_cpu -> D2H copy for that event
```

What overlaps today:

```text
GPU read issued before CPU EventManager work
CPU EventManager can overlap with the GPU read
GPU calib is launched asynchronously through EventPool
```

What does not overlap today:

```text
.on_cpu D2H with later read/H2D/compute
batched D2H with later batches
async D2H into pinned host memory
```

This supports a future split between:

```text
gpu_pool_depth: pipeline depth for read/compute
gpu_join_size: retained GPU result count before a batched async D2H join
```

For a 40 GB A100, a full Jungfrau calib result is 64 MiB/event.  A join buffer
holding 100 events would need about 6.4 GiB per BD rank:

```text
100 * 64 MiB = 6.4 GiB
```

That is plausible for one BD per GPU.  With multiple BDs sharing one GPU, a
smaller starting point such as 16 or 32 events per BD is safer.

## Reproducing the Runs

Run the NIC monitor on the same compute node as the psana job.  Use
`--samples 0` so the monitor runs until manually stopped:

```bash
python psana/psana/debugtools/net_bandwidth.py \
  --source ethtool \
  --jsonl \
  --samples 0 \
  > tmp/20260706/02/nic_sdfampere001_gpu_bd1_bs1_pd2_jn0_evt16k.jsonl
```

Stop it with Ctrl-C after the psana command finishes.

### CPU Baseline

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  -d jungfrau \
  --batch_size 1 \
  --max_events 16000 \
  --debug_detector jungfrau \
  > tmp/20260706/02/cpu_bd1_bs1_evt16k.log
```

Suggested matching NIC output:

```bash
python psana/psana/debugtools/net_bandwidth.py \
  --source ethtool \
  --jsonl \
  --samples 0 \
  > tmp/20260706/02/nic_sdfampere001_bd1_bs1_cpu_evt16k.jsonl
```

### GPU, No D2H

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  --gpu_det jungfrau \
  --batch_size 1 \
  --gpu_pool_depth 2 \
  --max_events 16000 \
  --gpu_d2h_interval 0 \
  > tmp/20260706/02/gpu_bd1_bs1_pd2_jn0_evt16k.log
```

### GPU, D2H Every 100 Events

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  --gpu_det jungfrau \
  --batch_size 1 \
  --gpu_pool_depth 2 \
  --max_events 16000 \
  --gpu_d2h_interval 100 \
  > tmp/20260706/02/gpu_bd1_bs1_pd2_jn100_evt16k.log
```

### GPU, D2H Every 10 Events

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  --gpu_det jungfrau \
  --batch_size 1 \
  --gpu_pool_depth 2 \
  --max_events 16000 \
  --gpu_d2h_interval 10 \
  > tmp/20260706/02/gpu_bd1_bs1_pd2_jn10_evt16k.log
```

### GPU, D2H Every Event

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  --gpu_det jungfrau \
  --batch_size 1 \
  --gpu_pool_depth 2 \
  --max_events 16000 \
  --gpu_d2h_interval 1 \
  > tmp/20260706/02/gpu_bd1_bs1_pd2_jn1_evt16k.log
```

## Source Files

The result files used for this note were:

```text
tmp/20260706/02/cpu_bd1_bs1_evt16k.log
tmp/20260706/02/gpu_bd1_bs1_pd2_jn0_evt16k.log
tmp/20260706/02/gpu_bd1_bs1_pd2_jn100_evt16k.log
tmp/20260706/02/gpu_bd1_bs1_pd2_jn10_evt16k.log
tmp/20260706/02/gpu_bd1_bs1_pd2_jn1_evt16k.log
tmp/20260706/02/nic_sdfampere001_bd1_bs1_cpu_evt16k.jsonl
tmp/20260706/02/nic_sdfampere001_gpu_bd1_bs1_pd2_jn0_evt16k.jsonl
tmp/20260706/02/nic_sdfampere001_gpu_bd1_bs1_pd2_jn100_evt16k.jsonl
tmp/20260706/02/nic_sdfampere001_gpu_bd1_bs1_pd2_jn10_evt16k.jsonl
tmp/20260706/02/nic_sdfampere001_gpu_bd1_bs1_pd2_jn1_evt16k.jsonl
```
