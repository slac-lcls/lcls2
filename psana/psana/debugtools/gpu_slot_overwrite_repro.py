#!/usr/bin/env python
"""Reproduce late-consumer registration overwriting a GPU execution slot.

This is a diagnostic for the current GPU DataSource iterator ordering.  It
uses a pool depth and batch size of one, disables automatic D2H, and launches
a deliberately delayed external CUDA consumer for event 0.  Advancing the
normal ``run.events()`` iterator submits event 1 into the same calibration
slot before the external consumer's completion event is honored.

The normal Jungfrau path assembles calibration into a newly allocated full
detector array during routing.  That allocation masks execution-slot reuse,
so this diagnostic monkey-patches only ``_apply_full_routing`` to return the
underlying slot-backed calibration view.  DataSource, EventPool, GPUDetector,
SlotLease, ``on_gpu_view()``, and the event iterator remain unchanged.

Expected result on the affected implementation::

    slot pointers match:       True
    event 0 before != after:   True
    event 0 after == event 1:  True
    RESULT: OVERWRITE REPRODUCED

Run with three MPI ranks (SMD0, EB, BD/GPU), for example::

    mpirun -n 3 python -u \
      psana/psana/debugtools/gpu_slot_overwrite_repro.py \
      --exp mfx100848724 --run 51 \
      --dir /sdf/data/lcls/ds/prj/public01/xtc
"""

import argparse
import sys

from mpi4py import MPI


_PROBE_SOURCE = r"""
extern "C" __global__
void psana_slot_overwrite_probe(
    const volatile unsigned int* src,
    unsigned long long n_words,
    unsigned long long n_samples,
    unsigned long long delay_cycles,
    unsigned long long* hashes)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned long long count = n_samples < n_words ? n_samples : n_words;
    unsigned long long step = 11400714819323198485ULL;
    unsigned long long h0 = 1469598103934665603ULL;
    for (unsigned long long i = 0; i < count; ++i) {
        unsigned long long idx = (i * step) % n_words;
        h0 ^= (unsigned long long)src[idx];
        h0 *= 1099511628211ULL;
    }
    hashes[0] = h0;

    unsigned long long started = clock64();
    while (clock64() - started < delay_cycles) {
        // The repeated clock64() hardware reads keep this polling loop from
        // being optimized away.  Do not use inline PTX "nop": it is not a
        // recognized PTX instruction in the CUDA 12 NVRTC toolchain.
    }

    // src is volatile so these are fresh global-memory loads after the delay.
    unsigned long long h1 = 1469598103934665603ULL;
    for (unsigned long long i = 0; i < count; ++i) {
        unsigned long long idx = (i * step) % n_words;
        h1 ^= (unsigned long long)src[idx];
        h1 *= 1099511628211ULL;
    }
    hashes[1] = h1;
}
"""


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce EventPool slot overwrite after late consumer registration."
    )
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--gpu-det", default="jungfrau")
    parser.add_argument("--delay-cycles", type=int, default=500_000_000,
                        help="GPU probe delay in SM clock cycles (default: 500M)")
    parser.add_argument("--samples", type=int, default=65_536,
                        help="Number of float32 words included in each hash")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--expect", choices=("overwrite", "safe"), default="overwrite",
        help="Expected outcome; use 'safe' to validate the retirement fix",
    )
    return parser.parse_args()


def _disable_full_routing_copy():
    """Expose GPUDetector's execution-slot view through the normal context."""
    import psana.gpu.gpu_events as gpu_events

    def identity_routing(gpu_results, evt, gpu_detectors, router):
        return gpu_results

    gpu_events._apply_full_routing = identity_routing


def _hash_now(cp, kernel, arr, samples):
    hashes = cp.zeros(2, dtype=cp.uint64)
    words = arr.view(cp.uint32).ravel()
    kernel(
        (1,),
        (1,),
        (words, words.size, samples, 0, hashes),
    )
    cp.cuda.Stream.null.synchronize()
    return int(hashes[0].item())


def _run_probe(first_evt, event_iter, args):
    import cupy as cp

    kernel = cp.RawKernel(_PROBE_SOURCE, "psana_slot_overwrite_probe")
    probe_stream = cp.cuda.Stream(non_blocking=True)
    hashes_gpu = cp.zeros(2, dtype=cp.uint64)

    result0 = first_evt.gpu.get("calib")
    with result0.on_gpu_view(probe_stream) as arr0:
        ptr0 = int(arr0.data.ptr)
        words0 = arr0.view(cp.uint32).ravel()
        with probe_stream:
            kernel(
                (1,),
                (1,),
                (words0, words0.size, args.samples,
                 args.delay_cycles, hashes_gpu),
            )

    # Resuming the real iterator submits event 1 calibration into pool slot 0.
    # Event 1 itself is yielded once event 2 causes that slot to retire again.
    second_evt = next(event_iter)
    result1 = second_evt.gpu.get("calib")
    arr1 = result1._arr  # Diagnostic inspection of the execution-slot view.
    ptr1 = int(arr1.data.ptr)

    # The delayed event-0 consumer should now have observed event 1 in-place.
    probe_stream.synchronize()
    before0, after0 = (int(v) for v in hashes_gpu.get())
    hash1 = _hash_now(cp, kernel, arr1, args.samples)

    same_pointer = ptr0 == ptr1
    changed_while_consumed = before0 != after0
    after_matches_event1 = after0 == hash1
    reproduced = same_pointer and changed_while_consumed and after_matches_event1

    return {
        "timestamp0": int(first_evt.timestamp),
        "timestamp1": int(second_evt.timestamp),
        "ptr0": ptr0,
        "ptr1": ptr1,
        "hash0_before": before0,
        "hash0_after": after0,
        "hash1": hash1,
        "same_pointer": same_pointer,
        "changed_while_consumed": changed_while_consumed,
        "after_matches_event1": after_matches_event1,
        "reproduced": reproduced,
    }


def main():
    args = _parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    _disable_full_routing_copy()

    from psana import DataSource

    ds = DataSource(
        exp=args.exp,
        run=args.run,
        dir=args.dir,
        gpu_det=args.gpu_det,
        n_gpu_streams=1,
        gpu_d2h_chunk_size=0,
        batch_size=1,
        max_events=3,
        log_level=args.log_level,
    )
    run = next(ds.runs())
    event_iter = iter(run.events())

    local_result = None
    try:
        first_ctx = next(event_iter)
    except StopIteration:
        # SMD0/EB ranks participate in DataSource but do not receive user events.
        pass
    else:
        local_result = _run_probe(first_ctx, event_iter, args)
        # Drain the final event so all MPI roles terminate cleanly.
        for _ in event_iter:
            pass

    gathered = comm.gather(local_result, root=0)
    failure = 0
    if rank == 0:
        results = [r for r in gathered if r is not None]
        if len(results) != 1:
            print(f"ERROR: expected one BD/GPU result, got {len(results)}", flush=True)
            failure = 2
        else:
            r = results[0]
            print(f"timestamps:                {r['timestamp0']} -> {r['timestamp1']}")
            print(f"slot pointers:             0x{r['ptr0']:x} -> 0x{r['ptr1']:x}")
            print(f"event 0 hash before:       0x{r['hash0_before']:016x}")
            print(f"event 0 hash after:        0x{r['hash0_after']:016x}")
            print(f"event 1 hash:              0x{r['hash1']:016x}")
            print(f"slot pointers match:       {r['same_pointer']}")
            print(f"event 0 changed in use:    {r['changed_while_consumed']}")
            print(f"event 0 after == event 1:  {r['after_matches_event1']}")
            if r["reproduced"]:
                print("RESULT: OVERWRITE REPRODUCED")
            else:
                print("RESULT: overwrite not observed")
            expected = r["reproduced"] if args.expect == "overwrite" else not r["reproduced"]
            if not expected:
                print(f"ERROR: expected {args.expect} outcome")
                failure = 1

    failure = comm.bcast(failure, root=0)
    return failure


if __name__ == "__main__":
    sys.exit(main())
