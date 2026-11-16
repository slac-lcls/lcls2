"""Sample whole-device NVML usage without creating a CUDA context.

Sampled peaks can miss sub-interval spikes; framework budget hooks complement
them. Global device usage includes contexts, cached allocations and other users.
"""
import ctypes
import threading
import time
from support import require

class Nvml:
    def __init__(self, uuid):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle=ctypes.c_void_p()
        self.lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
        self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(self.handle)))
        class Memory(ctypes.Structure):
            _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
        self.Memory=Memory
        self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory)]
    def check(self,code):
        require(code==0,'NVML call failed: '+str(code))
    def sample(self):
        m=self.Memory();self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m)))
        return {'time':time.time(),'total':m.total,'free':m.free,'used':m.used}
    def close(self):self.check(self.lib.nvmlShutdown())

def validate_memory(receipt, started=None, finished=None):
    samples=receipt['samples'];require(not receipt['errors'] and len(samples)>=2,'Memory sampler failed')
    require(len({s['total'] for s in samples})==1 and all(0<=s['used']<=s['total'] and 0<=s['free']<=s['total'] for s in samples),'Invalid NVML samples')
    require(all(a['time']<=b['time'] for a,b in zip(samples,samples[1:])),'Memory sample order')
    require(min(s['free'] for s in samples)>=4*2**30,'Less than4GiB sampled device headroom')
    require(24*2**30+4*2**30<=samples[0]['total'],'Aggregate budget cannot fit')
    require(receipt['max_used']==max(s['used'] for s in samples) and receipt['min_free']==min(s['free'] for s in samples),'Memory extrema mismatch')
    if started is not None:
        require(samples[0]['time']<=started and samples[-1]['time']>=finished,'Memory does not bracket event loop')
        require(any(started<=s['time']<=finished for s in samples),'No memory sample inside event loop')
    require(max(b['time']-a['time'] for a,b in zip(samples,samples[1:]))<=2.0,'Memory sampling gap exceeds2s')

class Watch:
    def __init__(self,uuid):
        self.uuid=uuid;self.nvml=Nvml(uuid);self.samples=[];self.errors=[];self.stop_event=threading.Event()
    def start(self):
        self.samples.append(self.nvml.sample())
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        try:
            while not self.stop_event.wait(.1):self.samples.append(self.nvml.sample())
        except Exception as e:self.errors.append(repr(e))
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=5)
        require(not self.thread.is_alive(),'Memory sampler did not stop')
        try:self.samples.append(self.nvml.sample())
        finally:self.nvml.close()
        return {'uuid':self.uuid,'interval_seconds':.1,'samples':self.samples,'errors':self.errors,
                'max_used':max(s['used'] for s in self.samples),'min_free':min(s['free'] for s in self.samples),
                'scope':'sampled whole-device NVML usage; not a guaranteed instantaneous peak; includes all device consumers'}
