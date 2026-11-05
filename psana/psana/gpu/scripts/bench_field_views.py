#!/usr/bin/env python3
"""Compare public aligned field access; report unaligned costs separately.

Run the same script/fixture on both frozen installs. The old install is eligible
only for --aligned-only: unaligned typed views are not a valid A reference.
Locator rows are host-resident synthetic metadata; parser/locator D2H costs are
excluded. Host entry+exit timing excludes setup, cleanup, and correctness reads.
CUDA stream elapsed includes host submission gaps; isolated byte-copy timing is
reported separately behind a device delay so enqueue gaps do not dominate it.
"""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--aligned-only', action='store_true')
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    sys.path.insert(0, str(args.fixture_dir))
    from field_view_fixture import field_case, retire
    from psana.gpu.gpu_input import GpuFieldResult, InputSlotLease
    import cupy as cp

    budget_supported = 'budget' in inspect.signature(InputSlotLease).parameters
    stream = cp.cuda.Stream(non_blocking=True)
    delay = cp.RawKernel(r'''
    extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long start=clock64(); while(clock64()-start<ticks) {}
    }''', 'delay')
    delay.compile()
    reports = []
    for shape in ((), (512, 1024)):
        for offset in ((8,) if args.aligned_only else (8, 1)):
            case = field_case(cp, type_id=3, shapes=(shape,), offset=offset,
                              lease_budget=budget_supported)
            case.lease.result_ready.synchronize()
            for cached in (False, True):
                hosts, spans, charges = [], [], []
                for iteration in range(args.iterations + 20):
                    kwargs = {'budget': case.budget} if budget_supported else {}
                    lease = InputSlotLease(case.lease.result_ready, (case.owner,), **kwargs)
                    field = GpuFieldResult(case.result.binding, case.sources, lease)
                    if cached:
                        with field.on_gpu_view(stream):
                            pass
                        stream.synchronize()
                    begin, end = cp.cuda.Event(), cp.cuda.Event()
                    begin.record(stream)
                    start = time.perf_counter_ns()
                    with field.on_gpu_view(stream) as values:
                        pass
                    host_us = (time.perf_counter_ns() - start) / 1000
                    end.record(stream)
                    end.synchronize()
                    elapsed_us = cp.cuda.get_elapsed_time(begin, end) * 1000
                    if iteration == 0:
                        assert values.only().get().tobytes() == case.expected[9].tobytes()
                    if iteration >= 20:
                        hosts.append(host_us)
                        spans.append(elapsed_us)
                        charges.append(case.budget.committed())
                    del values
                    lease.wait_until_safe_to_reuse()
                    assert case.budget.committed() == 0
                def summary(samples):
                    return {'median': statistics.median(samples),
                            'p10': float(np.percentile(samples, 10)),
                            'p90': float(np.percentile(samples, 90))}
                report = dict(shape=shape, nbytes=case.expected[9].nbytes,
                    aligned=offset == 8, cached=cached, host_us=summary(hosts),
                    cuda_stream_elapsed_us=summary(spans), retained_bytes=max(charges),
                    materializations_per_access=int(offset != 8 and not cached))
                reports.append(report)
            if offset != 8:
                # Same byte operation as field materialization, measured in
                # isolation. One destination remains live for all repetitions.
                source = case.sources[0].data_gpu
                destination = cp.empty(shape, dtype=cp.uint64)
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                delay((1,), (1,), (np.uint64(300000000),), stream=stream)
                begin.record(stream)
                for _ in range(100):
                    cp.cuda.runtime.memcpyAsync(destination.data.ptr, source.data.ptr + offset,
                        destination.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                end.record(stream)
                end.synchronize()
                reports.append(dict(shape=shape, nbytes=destination.nbytes,
                    isolated_byte_copy_us=cp.cuda.get_elapsed_time(begin, end) * 1000 / 100))
            retire(case)
    print(json.dumps(dict(iterations=args.iterations, warmup=20, cupy=cp.__version__,
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        results=reports), indent=2))


if __name__ == '__main__':
    main()
