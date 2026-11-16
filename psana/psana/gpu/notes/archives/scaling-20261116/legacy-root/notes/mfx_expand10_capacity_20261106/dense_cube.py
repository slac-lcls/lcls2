"""Configurable dense Cube with bounded host export; reviewed kernels unchanged."""
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np

NOTES = Path(__file__).resolve().parents[1]
REFERENCE_FILE = NOTES/'mfx_expand10_stage3a_20261105/cube.py'
spec = importlib.util.spec_from_file_location('capacity_reference_cube', REFERENCE_FILE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
adapter, STAGE2 = reference.adapter, reference.STAGE2


class Timings:
    """CUDA interval owners retained to final join; no waits on the hot path."""
    def __init__(self):
        self.records = []
        self.seen = set()

    def call(self, name, execution_stream, function, *args, **kwargs):
        import cupy as cp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        key = name, int(execution_stream.ptr)
        record = dict(name=name, stream=int(execution_stream.ptr), first=key not in self.seen,
                      start=start, end=end, submitted=False)
        self.records.append(record)  # Own events before recording asynchronous work.
        self.seen.add(key)
        start.record(execution_stream)
        result = function(*args, **kwargs)
        end.record(execution_stream)
        record['submitted'] = True
        return result

    def summary(self):
        import cupy as cp
        rows = []
        for r in self.records:
            if not r['submitted']:
                raise RuntimeError('Incomplete timed submission')
            r['end'].synchronize()
            rows.append(dict(stage=r['name'], stream=r['stream'], first_on_stream=r['first'],
                             ms=float(cp.cuda.get_elapsed_time(r['start'], r['end']))))
        stages = {}
        for row in rows:
            s = stages.setdefault(row['stage'], dict(calls=0, total_ms=0., warm_calls=0, warm_ms=0.))
            s['calls'] += 1; s['total_ms'] += row['ms']
            if not row['first_on_stream']:
                s['warm_calls'] += 1; s['warm_ms'] += row['ms']
        return dict(stages=stages, intervals=rows,
                    interpretation='Summed CUDA intervals may overlap; calibration includes uploads/host launch gaps. Warm excludes first call per stage/stream, not a throughput window.')


class TimedKernel:
    def __init__(self, kernel, name, timings):
        self.kernel, self.name, self.timings = kernel, name, timings

    def __call__(self, *args, **kwargs):
        return self.timings.call(self.name, kwargs['stream'], self.kernel, *args, **kwargs)


class TimedCalibration:
    def __init__(self, calibration, timings):
        self.calibration, self.timings = calibration, timings

    def calibrate(self, batch, stream, **kwargs):
        return self.timings.call('calibration_call', stream, self.calibration.calibrate, batch, stream, **kwargs)


class Cube(reference.Cube):
    """Same add/identity/bank kernels, new explicit budget and streamed finish API."""
    def __init__(self, timestamps, *, nbins, active_bins, max_state_bytes,
                 shape=(32,512,1024), max_streams=2, memory_sample=None):
        stamps = tuple(int(t) for t in timestamps)
        integers = (nbins, active_bins, max_streams, max_state_bytes, *shape)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in integers):
            raise ValueError('Integer dimensions and budgets required')
        if (len(stamps) != len(set(stamps)) or not 1 <= active_bins <= nbins <= np.iinfo(np.int32).max
                or max_streams not in (1,2) or not shape or any(n <= 0 for n in shape)
                or max_state_bytes <= 0):
            raise ValueError('Invalid cube configuration')
        self.shape, self.nbins, self.active_bins = tuple(shape), nbins, active_bins
        self.state_bytes_per_stream = nbins * (math.prod(shape)*4 + 8)
        self.required_state_bytes = max_streams*self.state_bytes_per_stream
        if self.required_state_bytes > max_state_bytes:
            raise ValueError('Persistent cube exceeds explicit state cap')
        self.max_state_bytes, self.max_streams = max_state_bytes, max_streams
        self.ordinal = {t:i for i,t in enumerate(stamps)}
        self.seen, self.banks = set(), []
        self.sample10 = adapter.Sample10()
        self.inputs, self.calibconst = self.sample10.inputs, self.sample10.calibconst
        self.phase = 'open'
        self.kernel = self.accept_kernel = None
        self.batch_sizes = []
        self.export_count = self.pool_peak_bytes = 0
        self.timings = Timings()
        self.memory_sample = memory_sample or (lambda label: None)
        self.export_result = self.export_path = None

    def prepare_profile(self):
        """Compile unchanged expansion/count/sum kernels before timed iteration."""
        import cupy as cp
        if self.kernel is not None or self.seen:
            raise RuntimeError('Prepare once before processing')
        for name, source, symbol, owner, attr, options in (
            ('expand_kernel', adapter.kernel_source(), 'expand10', self.sample10, 'kernel', ('--std=c++17',)),
            ('sum_kernel', reference.KERNEL, 'accumulate', self, 'kernel', ('--std=c++17','--fmad=false')),
            ('count_kernel', reference.KERNEL, 'accept_events', self, 'accept_kernel', ('--std=c++17',))):
            kernel = cp.RawKernel(source, symbol, options=options)
            kernel.compile()
            setattr(owner, attr, TimedKernel(kernel, name, self.timings))
        self.sample10.calibration = TimedCalibration(self.sample10.calibration, self.timings)

    def __call__(self, batch, stream):
        if self.shape != (32,512,1024):
            raise ValueError('Native sample10 requires full JF layout')
        self._identify(batch.timestamps)
        capture = reference.Capture(batch)
        self.sample10(capture, stream)
        self.memory_sample('after_calibration')
        self.add(capture.arrays['jungfrau.calib'], capture.arrays['sim10.errors'],
                 batch.timestamps, stream, batch.keepalive, batch.step_generation)
        self.memory_sample('after_accumulation_submission')

    def finish(self):
        raise RuntimeError('Use finish_to(directory); full host cube export is disabled')

    def finish_to(self, directory):
        """Join, merge/write one bin at a time, return metadata only.

        Files are immutable in a new directory. A complete manifest is committed
        only after all writes; failures leave phase=failed and no success manifest.
        """
        directory = Path(directory).resolve()
        if self.phase == 'finished':
            if directory != self.export_path:
                raise RuntimeError('Cannot re-export to another directory')
            return self.export_result
        if self.phase != 'open' or len(self.seen) != len(self.ordinal):
            raise RuntimeError('Cannot export closed or incomplete cube')
        directory.mkdir(mode=0o700, exist_ok=False)
        self.phase = 'exporting'
        begin = time.monotonic()
        timing = dict(join_seconds=0., transfer_seconds=0., merge_seconds=0., write_seconds=0.)
        try:
            t = time.monotonic()
            for bank in self.banks:
                bank.stream.synchronize()
            timing['join_seconds'] = time.monotonic()-t
            counts = np.zeros(self.nbins, np.uint64)
            for bank in self.banks:
                counts += bank.counts.get()
            # Exactly one float64 bin and one float32 transfer buffer, reused.
            merged = np.empty(self.shape, np.float64)
            transfer = np.empty(self.shape, np.float32)
            records = []
            for b in range(self.nbins):
                merged.fill(0)
                for bank in self.banks:
                    t = time.monotonic(); bank.sums[b].get(out=transfer, blocking=True)
                    timing['transfer_seconds'] += time.monotonic()-t
                    t = time.monotonic(); merged += transfer
                    timing['merge_seconds'] += time.monotonic()-t
                path = directory/f'bin-{b:05d}.npy'
                t = time.monotonic()
                with path.open('xb') as f:
                    np.save(f, merged, allow_pickle=False); f.flush(); os.fsync(f.fileno())
                timing['write_seconds'] += time.monotonic()-t
                records.append(dict(bin=b, count=int(counts[b]), file=path.name, bytes=path.stat().st_size))
                self.memory_sample('export_bin')
            result = dict(status='complete', nbins=self.nbins, shape=list(self.shape), dtype='float64',
                          counts=counts.tolist(), bins=records, timings=timing,
                          reusable_host_array_bytes=merged.nbytes+transfer.nbytes+counts.nbytes,
                          elapsed_seconds=time.monotonic()-begin)
            temporary = directory/'manifest.json.partial'
            with temporary.open('x') as f:
                json.dump(result, f, indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            temporary.rename(directory/'manifest.json')
            self.export_result, self.export_path = result, directory
            self.phase = 'finished'; self.export_count += 1
            return result
        except BaseException:
            self.phase = 'failed'
            raise
