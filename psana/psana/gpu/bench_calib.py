"""
Benchmark: GPU vs CPU Jungfrau calibration throughput.

Measures events/sec and per-stage timing (H->D, kernel, D->H) for the GPU
path compared to the CPU baseline.

Usage (MPI: 1 smd0 + 1 EB + N BD ranks, all sharing the visible GPU(s)):
    mpirun -n <N+2> python psana/psana/gpu/bench_calib.py \\
        -e mfx101572426 -r 47 -n 200 --warmup 10 \\
        --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

    -n is the per-BD-rank event target. Rank 0 prints an aggregate rate
    (sum of BD-rank rates) at the end. --compare-cpu is serial-only.

Usage (single process, no MPI ranks):
    python psana/psana/gpu/bench_calib.py \\
        -e mfx101572426 -r 47 -n 500 --compare-cpu \\
        --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

Notes on MPI structure: all ranks must construct the SAME DataSource and
iterate run.events() — smd0/EB ranks do their serving inside that loop and
yield no events. Termination comes from max_events (computed from -n and
the BD rank count); a BD rank must NOT break out early in MPI mode, since
a rank that stops requesting batches can hang smd0/EB.
"""

import argparse
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="GPU Jungfrau calib benchmark")
    p.add_argument("-e", "--exp",      required=True,       help="Experiment name")
    p.add_argument("-r", "--run",      required=True, type=int, help="Run number")
    p.add_argument("-n", "--nevents",  default=500,   type=int, help="Events to time (per BD rank)")
    p.add_argument("--det",            default="jungfrau",  help="Detector name")
    p.add_argument("--dir",            default=None,        help="XTC directory")
    p.add_argument("--warmup",         default=20,    type=int, help="Warmup events")
    p.add_argument("--d2h",            action="store_true", help="Include D->H in timing")
    p.add_argument("--profile",        action="store_true",
                   help="GPU path with per-event WALL-time attribution: buckets "
                        "wait (generator advance = bigdata read + EB/MPI) / read / "
                        "h2d / kernel, to attribute the 32-BD ceiling.")
    p.add_argument("--profile-read",   action="store_true",
                   help="Decompose the `read` (det.raw.raw) bucket to settle "
                        "storage-I/O vs in-process CPU. Splits det.raw.raw into "
                        "segments (dict lookup) / stack (per-seg .raw deserialize "
                        "+ copyto memcpy) / copy (final reshape+.copy()), and times "
                        "a SECOND det.raw.raw on the same event: read2<<read1 means "
                        "one-time cost (lazy I/O / first-touch page-in); read2~=read1 "
                        "means repeatable CPU memcpy/deserialize.")
    p.add_argument("--copy-true",      action="store_true",
                   help="GPU path only: restore det.raw.raw's default copy=True "
                        "(the pre-iter-7 baseline). The GPU path now DEFAULTS to "
                        "copy=False, which skips det.raw.raw's final 33.5 MB host->host "
                        ".copy() (iter 6: 58%% of the read bucket @32BD) — measured "
                        "+30%% @1BD and @32BD, bit-exact (iter 7). Safe because "
                        "cp.asarray(raw) copies host->device immediately, before the "
                        "reused _raw_buf is overwritten by the next event. Use this "
                        "flag only to reproduce the old A/B baseline.")
    p.add_argument("--seg-h2d",        action="store_true",
                   help="GPU path variant: copy each segment's .raw DIRECTLY "
                        "host->device into a pre-allocated device buffer, skipping "
                        "det.raw.raw's host-side `stack` memcpy (per-seg np.copyto "
                        "into the contiguous _raw_buf — 64.5 ms/event @32BD, iter 6, "
                        "the largest read component after copy=False landed). Trades "
                        "one 33.5 MB host memcpy + one big H2D for 32x 1 MB per-seg "
                        "H2Ds with no host stack. Bit-identical numerics (same segment "
                        "order + pixel layout); gate with test_jungfrau_calib.py "
                        "--seg-h2d.")
    p.add_argument("--wait-split",     action="store_true",
                   help="Split the `wait` bucket (79%% of wall @32BD, iter 9) into "
                        "EB-wait (BD blocked on the EB batch handoff: Probe+Irecv) vs "
                        "bd-read (os.pread of bigdata xtc off FFB), using cumulative "
                        "counters on run.bd_node (node.py). Runs the seg-h2d fast path "
                        "with the wait/h2d/kernel bucketing; snapshots the BD-node "
                        "counters at the warmup boundary and at the end so the split is "
                        "over the measured window only. Decides whether the serving-chain "
                        "frontier is serving-side (EB dispatch) or read-side (per-rank "
                        "xtc read under contention).")
    p.add_argument("--reader-probe",   action="store_true",
                   help="Concurrency-headroom probe for read-side prefetch (iter 13). "
                        "Runs the seg-h2d fast path, but ALSO spawns one background "
                        "daemon thread per BD rank that continuously os.pread's the "
                        "rank's bigdata xtc off FFB (16 MB chunks, advancing offset so "
                        "reads stay cold). Tests iter 12's open question: at 32 BD on a "
                        "fully-loaded node, does a background reader thread (GIL-released "
                        "in pread) overlap with the main-thread GPU work, or does it steal "
                        "the main loop's time? A GO signal (main rate ~unchanged with an "
                        "EXTRA full reader running) means real read-side prefetch — which "
                        "adds NO extra IO, only time-shifts existing reads — has room to "
                        "overlap bd_read (42%% of wall, iter 10) behind GPU work. NOTE this "
                        "is a CONSERVATIVE probe: it reads EXTRA bytes a real prefetch "
                        "would not, so it over-states cost. Reports the reader's achieved "
                        "MB/s so an IO-bandwidth limit is distinguishable from GIL/CPU "
                        "contention.")
    p.add_argument("--cpu",            action="store_true",
                   help="CPU-only mode: BD ranks run det.raw.calib() on the shared "
                        "collective DataSource (no GPU). MPI-capable — measures B-CPU "
                        "at the same rank layout as the GPU path.")
    p.add_argument("--compare-cpu",    action="store_true", help="Also run CPU calib path (serial only)")
    p.add_argument("--smd0-debug",     action="store_true",
                   help="DEBUG logging on rank 0 only: emits smd0 per-chunk stats "
                        "(eb_wait vs work time) to attribute the serving-chain ceiling")
    return p.parse_args()


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _world_size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except Exception:
        return 1


def _n_eb():
    """Number of EventBuilder ranks (psana2 default: 1). srv nodes unsupported."""
    return int(os.environ.get("PS_EB_NODES", "1"))


def _is_bd_rank():
    """True if this rank should run the benchmark body.

    psana2 MPI layout: rank 0 = smd0, ranks 1..PS_EB_NODES = EB, rest = BD.
    A single-process run does everything itself, so it is a BD rank even
    though mpi4py imports fine and reports rank 0 of a size-1 world.
    """
    size = _world_size()
    if size == 1:
        return True
    return _rank() >= 1 + _n_eb()


def run_gpu_bench(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Time the GPU path over run.events(). Does NOT create a DataSource —
    the caller's DataSource is shared by all ranks (collective)."""
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    h2d_times, kernel_times, d2h_times = [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None

    copy_raw = getattr(args, "copy_true", False)  # GPU path defaults to copy=False (iter 7)
    for evt in run.events():
        raw = det_obj.raw.raw(evt, copy=copy_raw)
        if raw is None:
            continue

        # H->D
        t0 = time.perf_counter()
        raw_gpu = cp.asarray(raw)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()

        # Kernel
        calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        # D->H (optional)
        if args.d2h:
            _ = calib_gpu.get()
            t3 = time.perf_counter()
        else:
            t3 = t2

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                h2d_times.clear(); kernel_times.clear(); d2h_times.clear()
            n_measured += 1
            continue

        h2d_times.append((t1 - t0) * 1e3)
        kernel_times.append((t2 - t1) * 1e3)
        d2h_times.append((t3 - t2) * 1e3)
        n_measured += 1

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
        "d2h_ms_mean": np.mean(d2h_times) if d2h_times else 0,
    }


def _start_reader_probe(args, rank):
    """Spawn a background daemon thread that continuously os.pread's this rank's
    bigdata xtc off FFB — the concurrency-headroom probe for read-side prefetch.

    The reader only calls os.pread (which releases the GIL), exactly the operation
    a real read-prefetch reader thread would run. Offset advances monotonically
    across the (200+ GB) file so reads stay cold (never re-hit page cache), and
    ranks fan out across the big streams round-robin so aggregate probe IO mirrors
    the real per-rank read distribution.

    Returns (stop_event, stats_dict, fd) or None if no bigdata file is found.
    stats_dict['bytes'] is the running total the caller snapshots at the warmup
    boundary and at the end to attribute bandwidth to the measured window only.
    """
    import glob
    import threading

    xdir = args.dir
    if not xdir:
        return None
    files = sorted(glob.glob(os.path.join(xdir, f"*r{args.run:04d}*.xtc2")))
    # bigdata streams only (exclude smalldata), and only the large streams that
    # actually carry detector payload (>1 GB) so the reader hits real storage.
    files = [f for f in files if ".smd." not in f and os.path.getsize(f) > (1 << 30)]
    if not files:
        return None
    fn = files[rank % len(files)]
    fd = os.open(fn, os.O_RDONLY)
    fsize = os.fstat(fd).st_size
    stop = threading.Event()
    stats = {"bytes": 0}
    CHUNK = 16 << 20
    span = max(1, fsize - CHUNK)

    # Optional per-rank rate cap (MB/s). Set PS_READER_PROBE_MBPS to make the probe
    # mimic a *faithful* read-prefetch load rather than running full-tilt. A real
    # prefetch reader only needs to stay one event ahead: 33.5 MB/event x ~19 Hz/rank
    # = ~640 MB/s/rank at 32 BD. Iter 13's uncapped reader pulled ~1115 MB/s/rank
    # (1.75x that), so its -50% overstated the contention a faithful prefetch imposes.
    # Capping isolates the contention question from the over-read confound.
    try:
        cap_mbps = float(os.environ.get("PS_READER_PROBE_MBPS", "0") or "0")
    except ValueError:
        cap_mbps = 0.0
    cap_bps = cap_mbps * 1e6 if cap_mbps > 0 else 0.0

    def _loop():
        # stagger start offsets so co-located ranks on the same file don't lock-step
        off = (rank * CHUNK * 7) % span
        t0 = time.perf_counter()
        read_bytes = 0
        while not stop.is_set():
            n = os.pread(fd, CHUNK, off)
            stats["bytes"] += len(n)
            read_bytes += len(n)
            off += CHUNK
            if off + CHUNK >= fsize:
                off = 0
            if cap_bps:
                # sleep until this rank's cumulative rate falls back to the cap
                target = t0 + read_bytes / cap_bps
                dt = target - time.perf_counter()
                if dt > 0:
                    stop.wait(dt)

    t = threading.Thread(target=_loop, daemon=True, name="reader-probe")
    t.start()
    return stop, stats, fd


def run_gpu_bench_seg_h2d(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """GPU path variant: skip det.raw.raw's host `stack` memcpy by copying each
    segment's .raw DIRECTLY host->device into a pre-allocated device buffer.

    The established path (run_gpu_bench) is:
        raw     = det.raw.raw(evt, copy=False)   # host `stack` np.copyto loop
        raw_gpu = cp.asarray(raw)                # one 33.5 MB H2D of _raw_buf
    After iter 7 landed copy=False, the host `stack` (per-seg deserialize +
    np.copyto into the contiguous _raw_buf, 64.5 ms/event @32BD, iter 6) is the
    largest remaining read component. This variant removes it entirely:
        segs = det.raw._segments(evt)            # {seg_id: seg_obj}
        for idx, sid: raw_gpu_buf[idx].set(segs[sid].raw)   # 32x per-seg H2D
    trading one host memcpy + one big H2D for 32x 1 MB per-seg H2Ds with no host
    stack. Numerics are bit-identical: same segment order (_segment_numbers),
    same per-seg row-major pixel layout, and fused_calib_gpu ravels the buffer
    exactly as it ravels cp.asarray(raw). The single `stack`+`h2d` cost is folded
    into the reported H->D bucket here (that IS the point — compare it against the
    baseline's stack(iter6) + H->D).

    Synchronous (device sync after the per-seg transfer loop), so the wall-clock
    rate is a valid headline — no overlap is introduced, so the sync-timing trap
    (PROMPT.md §6) does not apply.
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    raw_det = det_obj.raw
    seg_nums = raw_det._segment_numbers

    h2d_times, kernel_times, d2h_times = [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    raw_gpu_buf = None
    raw_gpu_3d = None

    # Concurrency-headroom probe: background os.pread load on this rank's bigdata.
    probe = None
    reader_bytes_at_start = 0
    if getattr(args, "reader_probe", False):
        probe = _start_reader_probe(args, _rank())

    for evt in run.events():
        segs = raw_det._segments(evt)
        if segs is None:
            continue

        # H->D: per-segment host->device straight into the device buffer.
        t0 = time.perf_counter()
        if raw_gpu_buf is None:
            s0 = segs[seg_nums[0]].raw
            # per-seg raw is (1, 512, 1024); buffer is (n_seg, *seg_shape) and a
            # reshape_to_3d view (n_seg, 512, 1024) matches det.raw.raw's shape.
            raw_gpu_buf = cp.empty((len(seg_nums),) + s0.shape, dtype=s0.dtype)
            raw_gpu_3d = raw_gpu_buf.reshape(-1, s0.shape[-2], s0.shape[-1])
        for idx, sid in enumerate(seg_nums):
            # ascontiguousarray is a no-op (no copy) when the segment view is
            # already C-contiguous — it only guards .set() against a stray
            # non-contiguous segment, without adding cost in the common case.
            raw_gpu_buf[idx].set(np.ascontiguousarray(segs[sid].raw))
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()

        # Kernel
        calib_gpu = fused_calib_gpu(raw_gpu_3d, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        # D->H (optional)
        if args.d2h:
            _ = calib_gpu.get()
            t3 = time.perf_counter()
        else:
            t3 = t2

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                if probe is not None:
                    reader_bytes_at_start = probe[1]["bytes"]
                h2d_times.clear(); kernel_times.clear(); d2h_times.clear()
            n_measured += 1
            continue

        h2d_times.append((t1 - t0) * 1e3)
        kernel_times.append((t2 - t1) * 1e3)
        d2h_times.append((t3 - t2) * 1e3)
        n_measured += 1

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0

    reader_mbps = 0.0
    if probe is not None:
        stop, stats, fd = probe
        reader_bytes = stats["bytes"] - reader_bytes_at_start
        stop.set()
        try:
            os.close(fd)
        except OSError:
            pass
        if t_wall > 0:
            reader_mbps = (reader_bytes / 1e6) / t_wall

    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
        "d2h_ms_mean": np.mean(d2h_times) if d2h_times else 0,
        "reader_mbps": reader_mbps,
    }


def run_gpu_bench_seg_h2d_profile(args, run, det_obj, peds_gpu, gmask_gpu,
                                  allow_break):
    """Like run_gpu_bench_seg_h2d, but buckets per-event WALL time into
    wait / h2d / kernel so the post-iter-8 ceiling can be attributed.

    Both landed host memcpys (iter 7 copy=False, iter 8 seg-h2d) are gone here,
    so this measures the seg-h2d fast path itself. The critical bucket is `wait`
    = time to advance run.events() to the next event, which in psana2 folds the
    bigdata read + EB/MPI serving; iter 8 predicted it is now the largest bucket.
    `h2d` here is the per-seg host->device loop (the only remaining host->device
    ingestion). wait + h2d + kernel should sum to ~wall.

    Synchronous (device sync after the transfer loop and after the kernel), so
    the wall rate is a valid headline — no overlap, so §6's sync-timing trap
    does not apply.
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    raw_det = det_obj.raw
    seg_nums = raw_det._segment_numbers

    wait_times, h2d_times, kernel_times = [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    t_prev_end = None
    raw_gpu_buf = None
    raw_gpu_3d = None

    events = run.events()
    while True:
        t_wait0 = time.perf_counter()
        try:
            evt = next(events)
        except StopIteration:
            break
        t_got = time.perf_counter()
        wait = (t_got - t_prev_end) * 1e3 if t_prev_end is not None \
            else (t_got - t_wait0) * 1e3

        segs = raw_det._segments(evt)
        if segs is None:
            t_prev_end = time.perf_counter()
            continue

        # H->D: per-segment host->device straight into the device buffer.
        t0 = time.perf_counter()
        if raw_gpu_buf is None:
            s0 = segs[seg_nums[0]].raw
            raw_gpu_buf = cp.empty((len(seg_nums),) + s0.shape, dtype=s0.dtype)
            raw_gpu_3d = raw_gpu_buf.reshape(-1, s0.shape[-2], s0.shape[-1])
        for idx, sid in enumerate(seg_nums):
            raw_gpu_buf[idx].set(np.ascontiguousarray(segs[sid].raw))
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()

        calib_gpu = fused_calib_gpu(raw_gpu_3d, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                wait_times.clear(); h2d_times.clear(); kernel_times.clear()
            n_measured += 1
            t_prev_end = time.perf_counter()
            continue

        wait_times.append(wait)
        h2d_times.append((t1 - t0) * 1e3)
        kernel_times.append((t2 - t1) * 1e3)
        n_measured += 1
        t_prev_end = time.perf_counter()

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "wait_ms_mean": np.mean(wait_times) if wait_times else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
    }


def run_gpu_bench_wait_split(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Like run_gpu_bench_seg_h2d_profile, but ALSO splits the `wait` bucket into
    EB-wait vs bigdata-read using cumulative counters on run.bd_node.

    iter 9 measured `wait` (generator advance) = 79% of the 32-BD wall and named
    it the frontier, but could not say whether that time is the BD rank BLOCKED on
    the EB batch handoff (serving-side) or READING its bigdata xtc off FFB once
    handed a batch (read-side). node.py's BigDataNode now accumulates both:
      total_eb_wait_ns = Probe+Irecv block on the EB batch (Phase A)
      total_bd_read_ns = os.pread of bigdata bytes (Phase B, from on_batch_end)
    Snapshotting them at the warmup boundary and at the end attributes the `wait`
    bucket over the measured window only (excludes smd0/EB startup + warmup).

    Numeric path identical to run_gpu_bench_seg_h2d (already gated bit-exact);
    only timing instrumentation added. Synchronous -> wall rate is a valid
    headline (no overlap; §6 trap does not apply).
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    raw_det = det_obj.raw
    seg_nums = raw_det._segment_numbers
    bd_node = getattr(run, "bd_node", None)   # None in single-process mode

    def _snap():
        if bd_node is None:
            return (0, 0, 0)
        return (bd_node.total_eb_wait_ns, bd_node.total_bd_read_ns,
                bd_node.total_events)

    wait_times, h2d_times, kernel_times = [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    t_prev_end = None
    raw_gpu_buf = None
    raw_gpu_3d = None
    snap0 = (0, 0, 0)

    events = run.events()
    while True:
        t_wait0 = time.perf_counter()
        try:
            evt = next(events)
        except StopIteration:
            break
        t_got = time.perf_counter()
        wait = (t_got - t_prev_end) * 1e3 if t_prev_end is not None \
            else (t_got - t_wait0) * 1e3

        segs = raw_det._segments(evt)
        if segs is None:
            t_prev_end = time.perf_counter()
            continue

        t0 = time.perf_counter()
        if raw_gpu_buf is None:
            s0 = segs[seg_nums[0]].raw
            raw_gpu_buf = cp.empty((len(seg_nums),) + s0.shape, dtype=s0.dtype)
            raw_gpu_3d = raw_gpu_buf.reshape(-1, s0.shape[-2], s0.shape[-1])
        for idx, sid in enumerate(seg_nums):
            raw_gpu_buf[idx].set(np.ascontiguousarray(segs[sid].raw))
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()

        calib_gpu = fused_calib_gpu(raw_gpu_3d, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                snap0 = _snap()      # baseline: exclude startup + warmup
                wait_times.clear(); h2d_times.clear(); kernel_times.clear()
            n_measured += 1
            t_prev_end = time.perf_counter()
            continue

        wait_times.append(wait)
        h2d_times.append((t1 - t0) * 1e3)
        kernel_times.append((t2 - t1) * 1e3)
        n_measured += 1
        t_prev_end = time.perf_counter()

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    snap1 = _snap()
    d_eb_wait_ns = snap1[0] - snap0[0]
    d_bd_read_ns = snap1[1] - snap0[1]
    d_events = snap1[2] - snap0[2]
    # ms/event over the measured window (bd_node counts every event the rank
    # advanced past, incl. skipped transitions; use its own event delta).
    ev = d_events if d_events > 0 else max(n, 1)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "wait_ms_mean": np.mean(wait_times) if wait_times else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
        "eb_wait_ms_mean": (d_eb_wait_ns / 1e6) / ev,
        "bd_read_ms_mean": (d_bd_read_ns / 1e6) / ev,
        "bd_events": int(d_events),
    }


def run_gpu_bench_profile(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Like run_gpu_bench, but also buckets the per-event WALL time into
    wait / read / h2d / kernel so the 32-BD ceiling can be attributed.

    The critical extra bucket is `wait` = the time to advance run.events()
    to the next event. In psana2 the BD rank reads its bigdata during that
    generator advance (smd0 -> EB batch -> BD reads xtc), so `wait` folds
    together the bigdata read + EB/MPI serving latency; `det.raw.raw(evt)`
    itself (the `read` bucket) is only the cheap array-locate inside the
    already-read dgram. wait + read + h2d + kernel should sum to ~wall,
    which is the sanity check that nothing is unattributed.

    Uses the same intra-event syncs as run_gpu_bench (valid: this is the
    synchronous path). Kept separate so the established B-MVP timing loop
    stays byte-identical.
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    wait_times, read_times, h2d_times, kernel_times = [], [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    t_prev_end = None

    events = run.events()
    while True:
        t_wait0 = time.perf_counter()
        try:
            evt = next(events)
        except StopIteration:
            break
        t_got = time.perf_counter()
        wait = (t_got - t_prev_end) * 1e3 if t_prev_end is not None \
            else (t_got - t_wait0) * 1e3

        t0 = time.perf_counter()
        raw = det_obj.raw.raw(evt)
        t1 = time.perf_counter()
        if raw is None:
            t_prev_end = time.perf_counter()
            continue

        raw_gpu = cp.asarray(raw)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t3 = time.perf_counter()

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                wait_times.clear(); read_times.clear()
                h2d_times.clear(); kernel_times.clear()
            n_measured += 1
            t_prev_end = time.perf_counter()
            continue

        wait_times.append(wait)
        read_times.append((t1 - t0) * 1e3)
        h2d_times.append((t2 - t1) * 1e3)
        kernel_times.append((t3 - t2) * 1e3)
        n_measured += 1
        t_prev_end = time.perf_counter()

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "wait_ms_mean": np.mean(wait_times) if wait_times else 0,
        "read_ms_mean": np.mean(read_times) if read_times else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
    }


def run_gpu_bench_profile_read(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Decompose the `read` (det.raw.raw) bucket to settle storage-I/O vs CPU.

    iter 3 measured det.raw.raw = 119 ms/event @ 32 BD (31% of wall, 15x its
    1-BD cost). iter 5's recommended decisive experiment: is that lazy bigdata
    I/O (storage) or in-process CPU deserialize/reshape? This mode answers it two
    independent ways per event:

    1. Internal breakdown via the `evt._det_raw_timing` hook that already exists
       in AreaDetectorRaw.raw:
         seg   = self._segments(evt) dict lookup (should be ~0)
         stack = the per-segment `segs[id].raw` deserialize + np.copyto memcpy loop
         copy  = residual (read1 - seg - stack) = final reshape_to_3d + arr.copy()
       stack is where both the lazy per-segment deserialize AND the 33.5 MB stack
       memcpy live; copy is the second 33.5 MB host memcpy the GPU path doesn't need.

    2. A SECOND det.raw.raw(evt) on the same event (read2), UN-instrumented:
         read2 << read1  -> the expensive part is one-time-per-event: lazy storage
                            I/O or first-touch page-in of the dgram bytes (bytes are
                            resident/cached the second time) -> STORAGE-bound.
         read2 ~= read1  -> the cost repeats every call: pure CPU memcpy/deserialize
                            -> CPU-bound (its super-linear scaling w/ rank count is
                            then host memory-bandwidth contention, not the FS).

    Uses the same intra-event syncs as run_gpu_bench (synchronous path). GPU
    numeric path byte-identical -> correctness gate not triggered.
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    wait_t, read1_t, seg_t, stack_t, copy_t, read2_t, h2d_t, kernel_t = \
        [], [], [], [], [], [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    t_prev_end = None

    events = run.events()
    while True:
        t_wait0 = time.perf_counter()
        try:
            evt = next(events)
        except StopIteration:
            break
        t_got = time.perf_counter()
        wait = (t_got - t_prev_end) * 1e3 if t_prev_end is not None \
            else (t_got - t_wait0) * 1e3

        # read1: instrumented det.raw.raw (the hook splits segments vs stack)
        hook = {'segments': 0.0, 'stack': 0.0}
        evt._det_raw_timing = hook
        t0 = time.perf_counter()
        raw = det_obj.raw.raw(evt)
        t1 = time.perf_counter()
        try:
            del evt._det_raw_timing
        except AttributeError:
            pass
        if raw is None:
            t_prev_end = time.perf_counter()
            continue

        # read2: second, un-instrumented det.raw.raw on the SAME event
        t1b = time.perf_counter()
        raw2 = det_obj.raw.raw(evt)
        t1c = time.perf_counter()

        read1 = (t1 - t0) * 1e3
        seg = hook['segments'] * 1e3
        stack = hook['stack'] * 1e3
        copy = read1 - seg - stack            # reshape_to_3d + arr.copy()
        read2 = (t1c - t1b) * 1e3

        raw_gpu = cp.asarray(raw)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t3 = time.perf_counter()

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                for lst in (wait_t, read1_t, seg_t, stack_t, copy_t,
                            read2_t, h2d_t, kernel_t):
                    lst.clear()
            n_measured += 1
            t_prev_end = time.perf_counter()
            continue

        wait_t.append(wait)
        read1_t.append(read1)
        seg_t.append(seg)
        stack_t.append(stack)
        copy_t.append(copy)
        read2_t.append(read2)
        h2d_t.append((t2 - t1c) * 1e3)
        kernel_t.append((t3 - t2) * 1e3)
        n_measured += 1
        t_prev_end = time.perf_counter()

        if allow_break and len(read1_t) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(read1_t)
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "wait_ms_mean": m(wait_t),
        "read1_ms_mean": m(read1_t),
        "seg_ms_mean": m(seg_t),
        "stack_ms_mean": m(stack_t),
        "copy_ms_mean": m(copy_t),
        "read2_ms_mean": m(read2_t),
        "h2d_ms_mean": m(h2d_t),
        "kernel_ms_mean": m(kernel_t),
    }


def run_cpu_bench_mpi(args, run, det_obj, allow_break):
    """Time the CPU calib path over the shared run.events(). Mirrors
    run_gpu_bench's loop structure (collective DataSource, warmup, no early
    break in MPI mode) so B-CPU and B-MVP are measured at IDENTICAL rank
    layouts. Does NOT create a DataSource — the caller's is shared by all
    ranks.

    det.raw.calib(evt) does the raw read + pedestal/gain/common-mode on CPU;
    this is the end-to-end CPU analogue of the GPU path's raw()+H2D+kernel.
    """
    calib_times = []
    warmup_done = False
    n_measured = 0
    t_start = None

    for evt in run.events():
        t0 = time.perf_counter()
        calib = det_obj.raw.calib(evt)
        t1 = time.perf_counter()
        if calib is None:
            continue

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                calib_times.clear()
            n_measured += 1
            continue

        calib_times.append((t1 - t0) * 1e3)
        n_measured += 1

        if allow_break and len(calib_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(calib_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "calib_ms_mean": np.mean(calib_times) if calib_times else 0,
    }


def run_cpu_bench(args, max_events):
    """Serial-only: creates its own second DataSource, which is safe only
    when there are no server ranks to coordinate with."""
    from psana import DataSource

    kwargs = dict(exp=args.exp, run=args.run, max_events=max_events)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det_obj = run.Detector(args.det)

    n_warmup = 0
    n_measured = 0
    t_start = None

    for evt in run.events():
        calib = det_obj.raw.calib(evt)
        if calib is None:
            continue
        if n_warmup < args.warmup:
            n_warmup += 1
            continue
        if t_start is None:
            t_start = time.perf_counter()
        n_measured += 1
        if n_measured >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    return {
        "n": n_measured,
        "wall_s": t_wall,
        "rate_hz": n_measured / t_wall if t_wall > 0 else 0,
    }


def _print_occupancy(rank, peds_gpu):
    import cupy as cp

    npixels = int(peds_gpu.size // 3)
    blocks = (npixels + 255) // 256
    try:
        attrs = cp.cuda.Device(0).attributes
        n_sm  = attrs.get("MultiProcessorCount", 108)
        bpsm  = min(attrs.get("MaxBlocksPerMultiprocessor", 32),
                    attrs.get("MaxThreadsPerMultiProcessor", 2048) // 256)
        sat   = n_sm * bpsm
        print(f"[rank {rank}] npixels={npixels:,}  blocks/event={blocks:,}  "
              f"saturation={sat}  occupancy={blocks/sat*100:.0f}%")
    except Exception:
        print(f"[rank {rank}] npixels={npixels:,}  blocks/event={blocks:,}")


def _report_aggregate(result, size):
    """Gather per-BD-rank results on rank 0 and print the aggregate rate.

    Collective over COMM_WORLD: every rank must call this. smd0/EB ranks
    pass result=None; they arrive here after their serving loop ends.
    """
    if size <= 1:
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    results = comm.gather(result, root=0)
    if comm.Get_rank() != 0:
        return
    bd = [r for r in results if r and r.get("n")]
    if not bd:
        print("\n[aggregate] no BD rank produced measurements")
        return
    agg = sum(r["rate_hz"] for r in bd)
    n_total = sum(r["n"] for r in bd)
    print(f"\n[aggregate] BD ranks: {len(bd)}   events measured: {n_total}")
    print(f"  aggregate rate:   {agg:.1f} Hz")
    print(f"  per-rank rate:    {agg / len(bd):.2f} Hz (mean)")
    if any("calib_ms_mean" in r for r in bd):
        print(f"  CPU calib:        {np.mean([r.get('calib_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")
    elif any("read2_ms_mean" in r for r in bd):
        g = lambda k: np.mean([r.get(k, 0) for r in bd])
        per_rank_wall = 1000.0 * len(bd) / agg
        print(f"  wait:             {g('wait_ms_mean'):.3f} ms/event (gen advance = bigdata read + EB/MPI)")
        print(f"  read1:            {g('read1_ms_mean'):.3f} ms/event (det.raw.raw 1st call)")
        print(f"    seg:            {g('seg_ms_mean'):.3f} ms/event (_segments dict lookup)")
        print(f"    stack:          {g('stack_ms_mean'):.3f} ms/event (per-seg .raw deserialize + copyto)")
        print(f"    copy:           {g('copy_ms_mean'):.3f} ms/event (reshape + final .copy())")
        print(f"  read2:            {g('read2_ms_mean'):.3f} ms/event (2nd call same evt: <<read1=>I/O, ~=read1=>CPU)")
        print(f"  H->D:             {g('h2d_ms_mean'):.3f} ms/event")
        print(f"  kernel:           {g('kernel_ms_mean'):.3f} ms/event")
        print(f"  (per-rank wall:   {per_rank_wall:.3f} ms/event)")
    elif any("eb_wait_ms_mean" in r for r in bd):
        g = lambda k: np.mean([r.get(k, 0) for r in bd])
        per_rank_wall = 1000.0 * len(bd) / agg
        wait = g('wait_ms_mean')
        eb_wait = g('eb_wait_ms_mean')
        bd_read = g('bd_read_ms_mean')
        h2d = g('h2d_ms_mean')
        ker = g('kernel_ms_mean')
        print(f"  wait:             {wait:.3f} ms/event (gen advance = EB-wait + bd-read)")
        print(f"    eb_wait:        {eb_wait:.3f} ms/event (BD blocked on EB batch: Probe+Irecv)")
        print(f"    bd_read:        {bd_read:.3f} ms/event (os.pread bigdata xtc off FFB)")
        print(f"    (eb_wait+bd_read: {eb_wait + bd_read:.3f} ms/event vs wait {wait:.3f})")
        print(f"  H->D:             {h2d:.3f} ms/event")
        print(f"  kernel:           {ker:.3f} ms/event")
        print(f"  sum:              {wait + h2d + ker:.3f} ms/event  "
              f"(vs {per_rank_wall:.3f} ms/event per-rank wall)")
    elif any("wait_ms_mean" in r for r in bd):
        wait = np.mean([r.get('wait_ms_mean', 0) for r in bd])
        read = np.mean([r.get('read_ms_mean', 0) for r in bd])
        h2d  = np.mean([r.get('h2d_ms_mean', 0) for r in bd])
        ker  = np.mean([r.get('kernel_ms_mean', 0) for r in bd])
        per_rank_wall = 1000.0 * len(bd) / agg  # ms/event per rank
        print(f"  wait:             {wait:.3f} ms/event (gen advance = bigdata read + EB/MPI)")
        print(f"  read:             {read:.3f} ms/event (det.raw.raw array-locate)")
        print(f"  H->D:             {h2d:.3f} ms/event")
        print(f"  kernel:           {ker:.3f} ms/event")
        print(f"  sum:              {wait + read + h2d + ker:.3f} ms/event  "
              f"(vs {per_rank_wall:.3f} ms/event per-rank wall)")
    else:
        print(f"  H->D:             {np.mean([r.get('h2d_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")
        print(f"  kernel:           {np.mean([r.get('kernel_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")
        if any(r.get("reader_mbps") for r in bd):
            rmb = [r.get("reader_mbps", 0) for r in bd]
            print(f"  reader-probe:     {np.sum(rmb):.0f} MB/s aggregate background read "
                  f"({np.mean(rmb):.0f} MB/s/rank mean over {len(bd)} ranks)")


def main():
    args = parse_args()
    rank = _rank()
    size = _world_size()
    is_bd = _is_bd_rank()

    from psana.gpu import init_gpu_rank, prep_calib_constants

    if is_bd and not args.cpu:
        init_gpu_rank()

    from psana import DataSource

    # Global event budget: in MPI mode the stream must end on its own
    # (see module docstring), so size it to feed every BD rank its warmup
    # + measurement quota, with 10% slack for skipped/None events.
    n_bd = max(1, size - 1 - _n_eb()) if size > 1 else 1
    max_events = int(1.1 * (args.warmup + args.nevents) * n_bd)

    kwargs = dict(exp=args.exp, run=args.run, max_events=max_events)
    if args.dir:
        kwargs["dir"] = args.dir
    if args.smd0_debug and rank == 0:
        # Rank 0 alone runs at DEBUG so smd0's per-chunk stats (eb_wait vs
        # work time) are emitted without 33 other ranks flooding stdout and
        # distorting the timing. DataSource kwargs may differ across ranks.
        kwargs["log_level"] = "DEBUG"
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det = run.Detector(args.det)

    result = None
    if is_bd and args.cpu:
        result = run_cpu_bench_mpi(args, run, det, allow_break=(size == 1))
        print(f"\n[rank {rank}] CPU results ({result['n']} events, "
              f"warmup={args.warmup}):")
        print(f"  rate:        {result['rate_hz']:.1f} Hz")
        print(f"  CPU calib:   {result['calib_ms_mean']:.3f} ms/event")
    elif is_bd:
        peds_gpu, gmask_gpu = prep_calib_constants(det)

        if size == 1 or rank == 1 + _n_eb():   # first BD rank only
            _print_occupancy(rank, peds_gpu)

        if args.wait_split:
            result = run_gpu_bench_wait_split(args, run, det, peds_gpu,
                                              gmask_gpu, allow_break=(size == 1))
        elif args.seg_h2d and args.profile:
            result = run_gpu_bench_seg_h2d_profile(args, run, det, peds_gpu,
                                                   gmask_gpu, allow_break=(size == 1))
        elif args.seg_h2d or args.reader_probe:
            # --reader-probe runs the seg-h2d fast path (the baseline it compares
            # against) with a background reader thread; see _start_reader_probe.
            result = run_gpu_bench_seg_h2d(args, run, det, peds_gpu, gmask_gpu,
                                           allow_break=(size == 1))
        elif args.profile_read:
            result = run_gpu_bench_profile_read(args, run, det, peds_gpu, gmask_gpu,
                                                allow_break=(size == 1))
        elif args.profile:
            result = run_gpu_bench_profile(args, run, det, peds_gpu, gmask_gpu,
                                           allow_break=(size == 1))
        else:
            result = run_gpu_bench(args, run, det, peds_gpu, gmask_gpu,
                                   allow_break=(size == 1))

        print(f"\n[rank {rank}] GPU results ({result['n']} events, "
              f"warmup={args.warmup}, d2h={'yes' if args.d2h else 'no'}):")
        print(f"  rate:        {result['rate_hz']:.1f} Hz")
        if args.profile_read:
            print(f"  wait:        {result['wait_ms_mean']:.3f} ms/event "
                  f"(gen advance = bigdata read + EB/MPI)")
            print(f"  read1:       {result['read1_ms_mean']:.3f} ms/event "
                  f"(det.raw.raw, 1st call = seg+stack+copy)")
            print(f"    seg:       {result['seg_ms_mean']:.3f} ms/event "
                  f"(_segments dict lookup)")
            print(f"    stack:     {result['stack_ms_mean']:.3f} ms/event "
                  f"(per-seg .raw deserialize + copyto memcpy)")
            print(f"    copy:      {result['copy_ms_mean']:.3f} ms/event "
                  f"(reshape + final .copy())")
            print(f"  read2:       {result['read2_ms_mean']:.3f} ms/event "
                  f"(2nd det.raw.raw, same evt: <<read1=>I/O, ~=read1=>CPU)")
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
        elif args.wait_split:
            tot = (result['wait_ms_mean'] + result['h2d_ms_mean']
                   + result['kernel_ms_mean'])
            print(f"  wait:        {result['wait_ms_mean']:.3f} ms/event "
                  f"(gen advance = EB-wait + bd-read)")
            print(f"    eb_wait:   {result['eb_wait_ms_mean']:.3f} ms/event "
                  f"(BD blocked on EB batch: Probe+Irecv)")
            print(f"    bd_read:   {result['bd_read_ms_mean']:.3f} ms/event "
                  f"(os.pread bigdata xtc off FFB)")
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event "
                  f"(per-seg host->device, no host stack)")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
            print(f"  sum:         {tot:.3f} ms/event  "
                  f"(vs {1000.0/result['rate_hz']:.3f} ms/event wall; "
                  f"bd_events={result['bd_events']})")
        elif args.seg_h2d and args.profile:
            tot = (result['wait_ms_mean'] + result['h2d_ms_mean']
                   + result['kernel_ms_mean'])
            print(f"  wait:        {result['wait_ms_mean']:.3f} ms/event "
                  f"(gen advance = bigdata read + EB/MPI)")
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event "
                  f"(per-seg host->device, no host stack)")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
            print(f"  sum:         {tot:.3f} ms/event  "
                  f"(vs {1000.0/result['rate_hz']:.3f} ms/event wall)")
        elif args.profile:
            tot = (result['wait_ms_mean'] + result['read_ms_mean']
                   + result['h2d_ms_mean'] + result['kernel_ms_mean'])
            print(f"  wait:        {result['wait_ms_mean']:.3f} ms/event "
                  f"(gen advance = bigdata read + EB/MPI)")
            print(f"  read:        {result['read_ms_mean']:.3f} ms/event "
                  f"(det.raw.raw array-locate)")
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
            print(f"  sum:         {tot:.3f} ms/event  "
                  f"(vs {1000.0/result['rate_hz']:.3f} ms/event wall)")
        else:
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
        if result.get("reader_mbps"):
            print(f"  reader-probe:{result['reader_mbps']:.0f} MB/s background read (this rank)")
        if args.d2h:
            print(f"  D->H:        {result['d2h_ms_mean']:.3f} ms/event")

        if args.compare_cpu:
            if size == 1:
                cpu = run_cpu_bench(args, max_events)
                print(f"\n[rank {rank}] CPU results ({cpu['n']} events):")
                print(f"  rate:        {cpu['rate_hz']:.1f} Hz")
                if cpu["rate_hz"] > 0:
                    print(f"  GPU speedup: {result['rate_hz'] / cpu['rate_hz']:.2f}x")
            elif rank == 1 + _n_eb():
                print(f"[rank {rank}] --compare-cpu is serial-only; skipped in MPI mode")
    else:
        # smd0/EB ranks serve data from inside the events() generator and
        # yield nothing; they must drive the same loop or BD ranks starve.
        for _ in run.events():
            pass

    _report_aggregate(result, size)


if __name__ == "__main__":
    main()
