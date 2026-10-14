"""Benchmark-only dense input exposure for the real DataSource user loop."""
import hashlib
import struct

import numpy as np


class PublicLoopWork:
    def __init__(self, variant, checks, pixels, diagnostic):
        import cupy as cp
        self.variant, self.checks, self.pixels = variant, checks, pixels
        self.diagnostic = diagnostic
        self.expected = {}
        self.callback_sizes = []
        self.digest = hashlib.sha256()
        self.count = 0
        self.kernel = cp.RawKernel('''extern "C" __global__ void scalar(
            const unsigned short* raw, unsigned int* out,
            unsigned long long n, unsigned long long stride) {
            unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
            if(i<n) out[i]=raw[i*stride+300]+1;
        }''', 'scalar')
        self.kernel.compile()
        raw, out = cp.zeros(301, cp.uint16), cp.empty(1, cp.uint32)
        self.kernel((1,), (1,), (raw,out,np.uint64(1),np.uint64(301)))
        cp.cuda.get_current_stream().synchronize()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.host = None
        if variant == 'event_loop':
            pointer = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(4096),0)
            self.host = np.ndarray((),np.uint32,buffer=pointer)

    def callback(self, batch, stream):
        import cupy as cp
        if self.diagnostic:
            self.callback_sizes.append(batch.size)
        raw = batch.input('jungfrau.raw')
        out = cp.empty(batch.size,cp.uint32)
        batch.publish('benchmark',out)
        self.kernel(((batch.size+127)//128,), (128,),
                    (raw,out,np.uint64(batch.size),np.uint64(raw.size//batch.size)),stream=stream)

    def install(self):
        from psana.gpu.context import SlotLease
        from psana.gpu.gpu_events import GpuEventManager
        from common import digest
        if self.variant == 'event_loop':
            from input_adapter import install
            install([], {})  # identical dense preparation, no callback/output
        original = GpuEventManager._submit_gpu
        work = self
        def submitted(manager, *args, **kwargs):
            record = original(manager,*args,**kwargs)
            if not record.prepared_inputs:
                return record
            assert len(record.prepared_inputs)==1
            prepared = next(iter(record.prepared_inputs.values()))
            if work.variant == 'event_loop':
                ready = next(iter(record.input_leases_by_ts.values())).result_ready
                lease = SlotLease(ready)
                record.leases.insert(0,lease)
                for i,event in enumerate(prepared.events):
                    record.gpu_results_by_ts[event.timestamp]={'bench.input':prepared.data[i]}
                    record.leases_by_ts[event.timestamp]={'bench.input':lease}
            if work.diagnostic:
                record.stream.synchronize()
                for i,event in enumerate(prepared.events):
                    if event.timestamp in work.pixels:
                        raw = prepared.data[i].get()
                        work.checks.append(dict(timestamp=event.timestamp,raw=digest(raw)))
                        work.expected[event.timestamp] = int(raw.flat[300])+1
            return record
        GpuEventManager._submit_gpu = submitted

    def consume(self, event):
        if self.variant == 'event_loop':
            import cupy as cp
            with event.gpu.get('bench.input').on_gpu_view(self.stream) as raw:
                out = cp.empty(1,cp.uint32)
                self.kernel((1,),(1,),(raw,out,np.uint64(1),np.uint64(raw.size)),stream=self.stream)
                out.reshape(()).get(out=self.host,stream=self.stream,blocking=True)
                value = self.host.copy()
        else:
            value = event.gpu.get('benchmark').on_cpu
        assert value.shape==() and value.dtype==np.uint32
        stamp, number = int(event.timestamp), int(value)
        if self.diagnostic and stamp in self.expected:
            assert number==self.expected[stamp], (stamp,number,self.expected[stamp])
        self.digest.update(struct.pack('<QI',stamp,number))
        self.count += 1

    def summary(self):
        if self.diagnostic and self.variant=='batched_task':
            assert sum(self.callback_sizes)==self.count
            assert all(0<n<=20 for n in self.callback_sizes)
        return dict(events=self.count,output_sha256=self.digest.hexdigest(),
                    callback_sizes=self.callback_sizes,
                    reference_pinned_bytes=4096 if self.host is not None else 0)
