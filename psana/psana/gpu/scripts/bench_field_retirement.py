#!/usr/bin/env python3
"""Measure EventPool field retirement with blocking/deferred input windows.

Synthetic host locators exclude I/O and parsing. Deferred windows represent the
bulk-on lifetime policy, not a full bulk-reader workload. Materialization is
completed before timing; a separate stream consumes the cached field after a
controlled device delay. Report host retirement and residual drain separately.
"""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def summary(values):
    return dict(median=statistics.median(values),
                p10=float(np.percentile(values, 10)),
                p90=float(np.percentile(values, 90)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--aligned-only', action='store_true')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, str(args.fixture_dir))
    from field_view_fixture import field_case
    from psana.gpu.gpu_input import InputSlotLease
    from psana.gpu.gpu_stream import EventPool, _EventSlot
    import cupy as cp

    budget_supported = 'budget' in inspect.signature(InputSlotLease).parameters
    consumer = cp.cuda.Stream(non_blocking=True)
    pool = EventPool(n=1)
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void consume(unsigned long long ticks,
        const unsigned long long* source, unsigned long long* output,
        unsigned long long count) {
        unsigned long long start = clock64();
        while (clock64() - start < ticks) {}
        output[0] = source[0]; output[1] = source[count - 1];
    }''', 'consume')
    kernel.compile()
    # Warm launch and allocations before any delayed consumer is submitted.
    warm = cp.zeros(2, dtype=cp.uint64)
    output = cp.empty(2, dtype=cp.uint64)
    kernel((1,), (1,), (np.uint64(0), warm, output, np.uint64(2)), stream=consumer)
    consumer.synchronize()
    reports = []
    for shape in ((), (512, 1024)):
        for offset in ((8,) if args.aligned_only else (8, 1)):
            for deferred in (False, True):
                for ticks in (0, 10000000):
                    records = []
                    for iteration in range(args.iterations + args.warmup):
                        case = field_case(cp, type_id=3, shapes=(shape,), offset=offset,
                            lease_budget=budget_supported, defer_retirement=deferred)
                        case.lease.result_ready.synchronize()
                        with case.result.on_gpu_view(consumer) as values:
                            pass
                        consumer.synchronize()
                        expected = case.expected[9].reshape(-1)[[0, -1]]
                        retained = case.budget.committed()
                        assert (retained > 0) == (offset == 1)
                        slot = _EventSlot(0, {}, [], pool.next_stream,
                                          [case.lease], {})
                        pool._slots[0] = slot
                        assert pool.begin_retire_next() is slot
                        # Close acquisitions while the root use still protects storage.
                        assert not case.owner.close()
                        begin, end = cp.cuda.Event(), cp.cuda.Event()
                        begin.record(consumer)
                        with case.result.on_gpu_view(consumer) as values:
                            array = values.only()
                            assert array.data.ptr % 8 == 0
                            kernel((1,), (1,), (np.uint64(ticks), array, output,
                                np.uint64(array.size)), stream=consumer)
                            end.record(consumer)
                        del array, values
                        pending = not end.done
                        start = time.perf_counter_ns()
                        pool.finish_retire_next()
                        finish_us = (time.perf_counter_ns() - start) / 1000
                        done_at_return = bool(end.done)
                        released_at_return = case.owner.released
                        assert not released_at_return or done_at_return
                        assert case.budget.committed() == 0
                        start = time.perf_counter_ns()
                        assert case.owner.drain()
                        drain_us = (time.perf_counter_ns() - start) / 1000
                        end.synchronize()
                        np.testing.assert_array_equal(output.get(), expected)
                        assert case.owner.references == 0
                        assert pool._slots[0] is None
                        if iteration >= args.warmup:
                            records.append(dict(finish_us=finish_us, drain_us=drain_us,
                                consumer_span_us=cp.cuda.get_elapsed_time(begin, end)*1000,
                                pending_at_finish_entry=pending,
                                done_at_finish_return=done_at_return,
                                released_at_finish_return=released_at_return,
                                retained_bytes_before=retained))
                    if ticks:
                        assert any(r['pending_at_finish_entry'] for r in records), \
                            'delay did not exercise pending retirement'
                    reports.append(dict(shape=shape, nbytes=case.expected[9].nbytes,
                        aligned=offset == 8, defer_retirement=deferred,
                        delay_ticks=ticks, samples=records,
                        **{key:summary([r[key] for r in records]) for key in
                           ('finish_us', 'drain_us', 'consumer_span_us')}))
    print(json.dumps(dict(iterations=args.iterations, warmup=args.warmup,
        scope='synthetic cached field; EventPool retirement; no I/O',
        results=reports), indent=2))


if __name__ == '__main__':
    main()
