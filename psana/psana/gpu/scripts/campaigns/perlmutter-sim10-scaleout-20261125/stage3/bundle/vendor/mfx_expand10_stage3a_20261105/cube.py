"""Bounded application-owned fly-scan cube; no automatic psana lifecycle hooks."""
from pathlib import Path
import sys
import numpy as np

STAGE2 = Path(__file__).resolve().parents[1] / 'mfx_expand10_stage2_20261105'
sys.path.insert(0, str(STAGE2))
import adapter


KERNEL = r'''
extern "C" __global__ void accept_events(const unsigned int* errors,
    const int* bins, unsigned char* accepted, unsigned long long* counts,
    int n, int panels) {
    // One thread updates scalar counts exactly once per accepted event.
    if (blockIdx.x || threadIdx.x) return;
    for (int e=0; e<n; ++e) {
        bool good = true;
        for (int s=0; s<panels; ++s) good = good && errors[e*panels+s] == 0;
        accepted[e] = good;
        if (good) counts[bins[e]] += 1;
    }
}
extern "C" __global__ void accumulate(const float* images, const int* bins,
    const unsigned char* accepted, float* sums, unsigned long long pixels, int n) {
    // One thread owns a pixel across all bins. Repeated bin IDs cannot race.
    for (unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
         p<pixels; p+=(unsigned long long)blockDim.x*gridDim.x) {
        for (int e=0; e<n; ++e)
            if (accepted[e]) sums[(unsigned long long)bins[e]*pixels+p] += images[(unsigned long long)e*pixels+p];
    }
}
'''


class Bank:
    def __init__(self, stream, nbins, shape):
        import cupy as cp
        self.stream = stream  # Retain the actual stream object for this run.
        with stream:
            self.sums = cp.empty((nbins, *shape), cp.float32)
            self.counts = cp.empty(nbins, cp.uint64)
        self.events = []  # Host identities only, never borrowed batch/context.
        self.step_labels = set()


class Capture:
    """Keep reviewed intermediate arrays alive without publishing full images."""
    def __init__(self, batch):
        self.batch = batch
        self.arrays = {}

    def __getattr__(self, key):
        return getattr(self.batch, key)

    def publish(self, name, array):
        self.batch.keepalive(array)
        self.arrays[name] = array
        if name == 'sim10.errors':
            self.batch.publish(name, array)


class Cube:
    def __init__(self, timestamps, *, nbins=30, active_bins=29,
                 shape=(32,512,1024), max_streams=2, max_state_bytes=5<<30):
        stamps = tuple(int(t) for t in timestamps)
        if len(stamps) != len(set(stamps)) or not 1 <= active_bins <= nbins <= 30:
            raise ValueError('Unique timestamps and 1 <= active_bins <= nbins <= 30 required')
        if max_streams not in (1,2) or not shape or any(int(n)<=0 for n in shape):
            raise ValueError('Invalid bounded layout')
        self.shape, self.nbins, self.active_bins = tuple(shape), nbins, active_bins
        self.state_bytes_per_stream = nbins*(int(np.prod(shape))*4+8)
        if max_streams*self.state_bytes_per_stream > max_state_bytes:
            raise ValueError('Persistent cube exceeds explicit state cap')
        self.max_streams = max_streams
        self.ordinal = {t:i for i,t in enumerate(stamps)}
        self.seen = set()
        self.banks = []
        self.sample10 = adapter.Sample10()
        self.inputs, self.calibconst = self.sample10.inputs, self.sample10.calibconst
        self.phase = 'open'
        self.kernel = self.accept_kernel = None
        self.batch_sizes = []
        self.export_count = 0
        self.pool_peak_bytes = 0

    def _identify(self, timestamps):
        if self.phase != 'open':
            raise RuntimeError('Cube is not open')
        ts = tuple(int(t) for t in timestamps)
        if not ts or len(ts) != len(set(ts)) or any(t in self.seen or t not in self.ordinal for t in ts):
            raise ValueError('Unknown, repeated or empty event identities')
        return ts, np.array([self.ordinal[t] % self.active_bins for t in ts], np.int32)

    def _bank(self, stream):
        for bank in self.banks:
            if bank.stream is stream:
                return bank
        if len(self.banks) >= self.max_streams:
            raise RuntimeError('More execution streams than explicitly budgeted')
        bank = Bank(stream, self.nbins, self.shape)
        self.banks.append(bank)  # Retain before asynchronous zero initialization.
        bank.sums.fill(0)
        bank.counts.fill(0)
        return bank

    def add(self, images, errors, timestamps, stream, keepalive, step_label=0):
        """Consume arrays on stream; keepalive registers owners before CUDA work."""
        import cupy as cp
        ts, host_bins = self._identify(timestamps)
        if (images.shape != (len(ts), *self.shape) or images.dtype != cp.float32 or
                not images.flags.c_contiguous or errors.shape != (len(ts), self.shape[0]) or
                errors.dtype != cp.uint32 or not errors.flags.c_contiguous):
            raise ValueError('Unexpected images/error layout')
        with stream:
            bank = self._bank(stream)
            # Retain host source and destination before asynchronous H2D.
            bins = cp.empty(host_bins.shape, cp.int32)
            accepted = cp.empty(len(ts), cp.uint8)
            keepalive(images, errors, bins, accepted, host_bins, bank.sums, bank.counts)
            bins.set(host_bins, stream=stream)
            if self.kernel is None:
                self.kernel = cp.RawKernel(KERNEL, 'accumulate', options=('--std=c++17','--fmad=false'))
                self.accept_kernel = cp.RawKernel(KERNEL, 'accept_events', options=('--std=c++17',))
            self.accept_kernel((1,), (1,), (errors, bins, accepted, bank.counts, np.int32(len(ts)), np.int32(self.shape[0])), stream=stream)
            pixels = int(np.prod(self.shape))
            self.kernel((min(65535,(pixels+255)//256),), (256,),
                        (images,bins,accepted,bank.sums,np.uint64(pixels),np.int32(len(ts))), stream=stream)
        self.seen.update(ts)
        bank.events.extend(self.ordinal[t] for t in ts)
        bank.step_labels.add(int(step_label))
        self.batch_sizes.append(len(ts))
        self.pool_peak_bytes = max(self.pool_peak_bytes, cp.get_default_memory_pool().total_bytes())

    def __call__(self, batch, stream):
        if self.shape != (32,512,1024):
            raise ValueError('Native sample10 requires full JF layout')
        self._identify(batch.timestamps)  # Reject identity misuse before launching sample10.
        capture = Capture(batch)
        self.sample10(capture, stream)
        self.add(capture.arrays['jungfrau.calib'], capture.arrays['sim10.errors'],
                 batch.timestamps, stream, batch.keepalive, batch.step_generation)

    def finish(self):
        """Application calls only after successful full run iteration. Joins all banks.

        Idempotent after successful export. Missing submissions cannot produce a
        successful cube. No implicit EndRun callback is claimed.
        """
        import cupy as cp
        if self.phase == 'finished':
            return self.result
        if self.phase != 'open' or len(self.seen) != len(self.ordinal):
            raise RuntimeError('Cannot finalize a closed or incomplete cube')
        for bank in self.banks:
            bank.stream.synchronize()
        # Export and merge one bin at a time: bounded staging, float64 host merge.
        sums = np.zeros((self.nbins,*self.shape), np.float64)
        counts = np.zeros(self.nbins, np.uint64)
        for bank in self.banks:
            counts += bank.counts.get()
            for b in range(self.nbins):
                sums[b] += bank.sums[b].get()
        self.result = sums, counts
        self.phase = 'finished'
        self.export_count += 1
        return self.result

    def close(self):
        """Abort/success cleanup joins device work before releasing bank ownership."""
        for bank in self.banks:
            bank.stream.synchronize()
        self.banks.clear()
        self.phase = 'closed'
        self.result = None
