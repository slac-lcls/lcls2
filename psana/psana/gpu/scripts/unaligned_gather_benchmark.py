"""Time full-panel Jungfrau gather; fixture construction and validation are untimed.

Run this identical script in frozen before/after environments. Use --shift 0
for both revisions and --shift 1 only with the alignment-safe revision. This
isolates the kernel; it does not measure KvikIO, parsing or four-BD contention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import cupy as cp
import numpy as np

from psana.gpu import gpu_detector as gd
from psana.gpu.gpudgram import parser as p


def measure(batch, shift, samples, iterations, warmup):
    segments, height, width = 32, 512, 1024
    pixels = height * width
    nrows = batch * segments
    nbytes = nrows * pixels * 2
    stream = cp.cuda.Stream(non_blocking=True)
    pattern = np.arange(pixels, dtype=np.uint16)
    with stream:
        memory = cp.cuda.Memory(nbytes + shift)
        storage = cp.ndarray((memory.size,), dtype=cp.uint8,
                             memptr=cp.cuda.MemoryPointer(memory, 0))
        data = storage[shift:]
        source = cp.tile(cp.asarray(pattern), nrows)
        cp.copyto(data, source.view(cp.uint8))
        stream.synchronize()
        del source
        assert data.data.ptr % 2 == shift
        locations = np.zeros((segments, batch, p.LOC_NCOLS), np.uint64)
        for segment in range(segments):
            for event in range(batch):
                locations[segment, event, [p.LOC_STATUS, p.LOC_TYPE, p.LOC_RANK,
                                           p.LOC_OFFSET, p.LOC_NBYTES,
                                           p.LOC_DIM0, p.LOC_DIM0 + 1]] = (
                    p.STATUS_FOUND, 1, 2, (event * segments + segment) * pixels * 2,
                    pixels * 2, height, width)
        locators = cp.asarray(locations)
        owners = cp.asarray(np.array([[data.data.ptr, data.nbytes,
                                       locators.data.ptr, batch, batch]], np.uint64))
        plan = cp.asarray(np.array([(0, s, 1, 2) for s in range(segments)], np.uint64))
        rows = cp.asarray(np.array([(0, e) for e in range(batch)], np.uint64))
        out = cp.empty((nrows, pixels), cp.uint16)
        present = cp.empty(nrows, cp.uint8)
        tiles = (pixels + 255) // 256
        kernel = gd._batched_gather_kernel(np.dtype(np.uint16))
        args = (owners, np.uint64(1), plan, rows, np.uint64(1), np.uint64(segments),
                np.uint64(pixels), np.uint64(tiles), np.uint64(height), np.uint64(width),
                out, present)

        def launch():
            kernel((tiles * nrows,), (256,), args, stream=stream)

        kernel.compile()
        launch()
        stream.synchronize()
        np.testing.assert_array_equal(out.get(), np.broadcast_to(pattern, out.shape))
        np.testing.assert_array_equal(present.get(), np.ones(nrows, np.uint8))
        for _ in range(warmup):
            launch()
        stream.synchronize()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        durations = []
        for _ in range(samples):
            start.record(stream)
            for _ in range(iterations):
                launch()
            end.record(stream)
            end.synchronize()
            durations.append(cp.cuda.get_elapsed_time(start, end) / iterations)
        np.testing.assert_array_equal(out.get(), np.broadcast_to(pattern, out.shape))
        np.testing.assert_array_equal(present.get(), np.ones(nrows, np.uint8))
        median = statistics.median(durations)
        return dict(batch=batch, segments=segments, panel_shape=[height, width],
                    dtype="uint16", base_shift=shift, address_residue=data.data.ptr % 2,
                    payload_bytes=nbytes, warmup_launches=warmup,
                    samples=samples, iterations_per_sample=iterations,
                    gather_ms=durations, median_gather_ms=median,
                    useful_payload_gb_per_s=nbytes / (median * 1e6),
                    throughput_definition="input pixel bytes per second; not physical DRAM traffic",
                    validated_pixels=nrows * pixels, kernel_attributes=kernel.attributes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=256,
                        help="untimed launches after compilation and pixel validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches + [args.samples, args.iterations, args.warmup]) < 1:
        parser.error("batches, samples, iterations and warmup must be positive")
    device = cp.cuda.Device()
    source = Path(gd.__file__).resolve()
    report = dict(device=device.id, pci_bus=device.pci_bus_id, cupy=cp.__version__,
                  runtime=cp.cuda.runtime.runtimeGetVersion(),
                  driver=cp.cuda.runtime.driverGetVersion(), module=str(source),
                  module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  measurements=[measure(b, args.shift, args.samples, args.iterations, args.warmup)
                                for b in args.batches])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
