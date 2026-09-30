"""Benchmark-only matched dense-input exposure; no production runtime changes."""
import hashlib
import struct
import time
import numpy as np


class LocalBatch:
    def __init__(self, raw, present, segments, constants, timestamps):
        self.raw, self.presence, self.segments, self.constants = raw, present, segments, constants
        self.timestamps = timestamps
        self.size = len(raw)
        self.owners, self.outputs = [], {}
    def input(self, name): return self.raw
    def present(self, name): return self.presence
    def segment_ids(self, name): return self.segments
    def calibconst(self, det, key): return self.constants[key]
    def keepalive(self, *owners): self.owners.extend(owners)
    def publish(self, name, array): self.outputs[name] = array


class Work:
    def __init__(self, variant, checks, pixels, diagnostic, bins, batch_size):
        import cupy as cp
        from psana.gpu.examples.jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration
        with np.load(bins, allow_pickle=False) as data:
            self.analysis = JungfrauAzimuthalIntegration(data['bin_ids'],len(data['edges'])-1,
                use_offset=True,status_bits=(1<<64)-1)
        self.variant,self.checks,self.pixels,self.diagnostic=variant,checks,pixels,diagnostic
        self.batch_size=batch_size
        self.expected={};self.host_constants=None
        self.pending={};self.outputs=[];self.sizes=[];self.launches=[];self.gpu_times=[]
        self.callbacks=0;self.submission_s=0.;self.exposure_s=0.;self.first_submission_s=None
        self.device_pool_peak=0;self.kernel=None;self.managers=[]
        self.stream=cp.cuda.Stream(non_blocking=True)
        self.host=None
        if variant=='event_loop':
            pin=cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(4096),0)
            self.host=np.ndarray((1,3,self.analysis.nbins),np.float64,buffer=pin)
        if diagnostic:
            original=cp.RawKernel
            def instrumented(source,name,**kwargs):
                kernel=original(source,name,**kwargs)
                if name not in ('calibrate','integrate'):return kernel
                kernel.compile()  # Keep NVRTC compilation outside CUDA event intervals.
                def launch(grid,block,args,**kw):
                    before,after=cp.cuda.Event(),cp.cuda.Event()
                    before.record(kw['stream'])
                    result=kernel(grid,block,args,**kw)
                    after.record(kw['stream'])
                    self.launches.append(name)
                    self.gpu_times.append((name,before,after))
                    return result
                return launch
            cp.RawKernel=instrumented
        self.peak_pinned=0;self.copy_groups=0;self.d2h_times=[]

    def invoke(self, batch, stream):
        import cupy as cp
        start=time.perf_counter()
        self.analysis(batch,stream)
        elapsed=time.perf_counter()-start
        self.submission_s+=elapsed
        if self.first_submission_s is None:self.first_submission_s=elapsed
        if self.diagnostic:
            from psana.tests.gpu.user_calibration_reference import calibrate
            from psana.tests.gpu.user_integration_reference import integrate
            for i,stamp in enumerate(batch.timestamps):
                if int(stamp) not in self.pixels: continue
                stream.synchronize()
                if self.host_constants is None:
                    self.host_constants={k:batch.calibconst(d,k).get() for d,k in self.analysis.calibconst}
                raw=batch.input('jungfrau.raw')[i:i+1].get()
                present=batch.present('jungfrau.raw')[i:i+1].get()
                segments=tuple(batch.segment_ids('jungfrau'))
                image=calibrate(raw,present,segments,self.host_constants,use_offset=True,status_bits=(1<<64)-1)
                self.expected[int(stamp)]=integrate(image,raw,present,segments,self.host_constants,
                    self.analysis.bin_ids,self.analysis.nbins,status_bits=(1<<64)-1)[0]
            self.device_pool_peak=max(self.device_pool_peak,cp.get_default_memory_pool().used_bytes())

    def callback(self, batch, stream):
        self.callbacks+=1
        self.sizes.append(batch.size)
        if self.variant=='batched_task':
            self.invoke(batch,stream)
        else:
            start=time.perf_counter()
            raw=batch.input('jungfrau.raw');present=batch.present('jungfrau.raw')
            segments=tuple(batch.segment_ids('jungfrau'))
            constants={k:batch.calibconst(d,k) for d,k in self.analysis.calibconst}
            for i,stamp in enumerate(batch.timestamps):
                self.pending[int(stamp)]=LocalBatch(raw[i:i+1],present[i:i+1],segments,constants,(int(stamp),))
            self.exposure_s+=time.perf_counter()-start

    def install(self):
        from psana.gpu.context import SlotLease
        from psana.gpu.gpu_events import GpuEventManager
        from common import digest
        work=self;original=GpuEventManager._submit_gpu
        def submitted(manager,*args,**kwargs):
            record=original(manager,*args,**kwargs)
            if not record.prepared_inputs:return record
            prepared=next(iter(record.prepared_inputs.values()))
            if work.variant=='event_loop':
                ready=next(iter(record.input_leases_by_ts.values())).result_ready
                lease=SlotLease(ready);record.leases.insert(0,lease)
                for i,event in enumerate(prepared.events):
                    record.gpu_results_by_ts[event.timestamp]={'bench.input':prepared.data[i]}
                    record.leases_by_ts[event.timestamp]={'bench.input':lease}
            if work.diagnostic:
                record.stream.synchronize()
                for i,event in enumerate(prepared.events):
                    if event.timestamp in work.pixels:
                        work.checks.append(dict(timestamp=event.timestamp,raw=digest(prepared.data[i].get())))
                work.copy_groups+=len(record.publication_batches)
            return record
        GpuEventManager._submit_gpu=submitted
        if self.diagnostic:
            from psana.gpu.gpu_d2h import PublicationD2H
            import cupy as cp
            enqueue=PublicationD2H.enqueue
            def copied(delivery,record):
                if not record.publication_batches:return enqueue(delivery,record)
                if delivery._stream is None:delivery._stream=cp.cuda.Stream(non_blocking=True)
                ready=record.publication_batches[0].lease.result_ready
                delivery._stream.wait_event(ready)
                before,after=cp.cuda.Event(),cp.cuda.Event()
                before.record(delivery._stream)
                result=enqueue(delivery,record)
                after.record(delivery._stream)
                work.d2h_times.append((before,after))
                work.peak_pinned=max(work.peak_pinned,delivery.pinned_bytes)
                return result
            PublicationD2H.enqueue=copied

    def consume(self,event):
        import cupy as cp
        if self.variant=='event_loop':
            batch=self.pending.pop(int(event.timestamp))
            with event.gpu.get('bench.input').on_gpu_view(self.stream) as raw:
                assert raw.data.ptr==batch.raw.data.ptr
                self.invoke(batch,self.stream)
                out=batch.outputs[self.analysis.output]
                if self.diagnostic:
                    before,after=cp.cuda.Event(),cp.cuda.Event();before.record(self.stream)
                out.get(out=self.host,stream=self.stream,blocking=True)
                if self.diagnostic:
                    after.record(self.stream);self.d2h_times.append((before,after));self.copy_groups+=1
                value=self.host[0].copy()
        else:
            value=event.gpu.get(self.analysis.output).on_cpu
        assert value.shape==(3,self.analysis.nbins) and value.dtype==np.float64
        assert np.isfinite(value).all() and np.all(value[2]>=0)
        if int(event.timestamp) in self.expected:
            expected=self.expected.pop(int(event.timestamp))
            np.testing.assert_array_equal(value[2],expected[2])
            np.testing.assert_allclose(value[:2],expected[:2],rtol=1e-12,atol=1e-9)
        self.outputs.append((int(event.timestamp),hashlib.sha256(value.tobytes()).hexdigest()))

    def summary(self):
        import cupy as cp
        cp.cuda.Device().synchronize()
        assert not self.pending and not self.expected
        operations=len(self.outputs) if self.variant=='event_loop' else self.callbacks
        if self.diagnostic:
            assert self.launches==['calibrate','integrate']*operations
            assert self.copy_groups==operations
        durations={name:sum(cp.cuda.get_elapsed_time(a,b) for n,a,b in self.gpu_times if n==name) for name in ('calibrate','integrate')}
        return dict(events=len(self.outputs),outputs=self.outputs,callback_sizes=self.sizes,
            analysis_calls=self.analysis.calls,framework_callbacks=self.callbacks,
            actual_kernel_launches=len(self.launches) if self.diagnostic else None,
            copy_groups=self.copy_groups if self.diagnostic else None,
            submission_s=self.submission_s,first_submission_s=self.first_submission_s,
            baseline_exposure_s=self.exposure_s,kernel_ms=durations if self.diagnostic else None,
            d2h_ms=sum(cp.cuda.get_elapsed_time(a,b) for a,b in self.d2h_times) if self.diagnostic else None,
            device_pool_peak=self.device_pool_peak if self.diagnostic else None,
            measured_output_pinned_peak=self.peak_pinned if self.variant=='batched_task' else 4096,
            reference_pinned_bytes=4096 if self.host is not None else 0,
            bin_table_device_bytes=sum(t[0].nbytes+t[1].nbytes for t in self.analysis._tables.values()))
