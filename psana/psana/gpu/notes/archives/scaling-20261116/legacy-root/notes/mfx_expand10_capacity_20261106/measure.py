"""Process-scoped measurements; no changes to installed psana files."""
from contextlib import contextmanager
import time
from unittest.mock import patch


class Memory:
    def __init__(self):
        self.samples = []

    def sample(self, label):
        import cupy as cp
        status = {}
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key,value,_ = line.split();status[key[:-1]+'_bytes'] = int(value)*1024
        free,total = cp.cuda.runtime.memGetInfo()
        self.samples.append(dict(label=label, monotonic_seconds=time.monotonic(),
                                 device_free_bytes=free, device_total_bytes=total,
                                 device_used_bytes=total-free,
                                 cupy_pool_total_bytes=cp.get_default_memory_pool().total_bytes(),
                                 cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes(), **status))

    def summary(self):
        fields=('device_used_bytes','cupy_pool_total_bytes','cupy_pool_used_bytes','VmRSS_bytes','VmHWM_bytes')
        return dict(samples=self.samples, peaks={key:max(s[key] for s in self.samples) for key in fields},
                    interpretation='Device usage and pool values sampled at callback/delivery/export boundaries; not continuous exact peaks. VmHWM is process lifetime host RSS high-water, including CPU validation. Device usage includes other device users.')


@contextmanager
def read_stats():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    submit, wait = KvikioGpuReader._submit_read, KvikioGpuReader.wait_batch
    stats = dict(submit_calls=0, submit_wall_seconds=0., total_requests=0, requested_bytes=0,
                 total_bytes=0, useful_bytes=0, total_ns=0, issue_to_complete_ns=0)

    def measured_submit(reader, *args, **kwargs):
        before=reader.io_stats();start=time.monotonic()
        try:
            return submit(reader, *args, **kwargs)
        finally:
            after=reader.io_stats();stats['submit_calls']+=1
            stats['submit_wall_seconds']+=time.monotonic()-start
            for key in ('total_requests','requested_bytes'):
                delta=after[key]-before[key]
                if delta < 0: raise RuntimeError('Reader reset during submission')
                stats[key]+=delta
            stats['io_path'],stats['compat_mode']=after['io_path'],after['compat_mode']

    def measured_wait(reader, *args, **kwargs):
        before=reader.io_stats()
        try:
            return wait(reader, *args, **kwargs)
        finally:
            after=reader.io_stats()
            for key in ('total_bytes','useful_bytes','total_ns','issue_to_complete_ns'):
                delta=after[key]-before[key]
                if delta < 0: raise RuntimeError('Reader reset during completion')
                stats[key]+=delta

    with patch.object(KvikioGpuReader,'_submit_read',measured_submit), patch.object(KvikioGpuReader,'wait_batch',measured_wait):
        yield stats
    stats['interpretation']='Reader wait ns and submission-to-observed-completion ns are summed; concurrent reads overlap, and neither is standalone I/O throughput. Counters accumulated across framework resets.'
