"""Qualification-only sampled NumPy oracle; never installed on timed callbacks."""
import numpy as np
from reference import SeedReader,CpuCalibration,samples,PIXELS,require

class NumericalReference:
 def __init__(self,info,timestamps,constants):
  self.selected=samples()
  self.reader=SeedReader(info['files'],timestamps)
  self.packed=CpuCalibration(constants,self.selected).constants
  self.ordinal={int(t):i for i,t in enumerate(timestamps)}
  self.seen=set();self.counts=np.zeros(100,np.uint64)
  self.sums=np.zeros((100,32,len(self.selected)),np.float64)
  self.absolute=np.zeros_like(self.sums);self.tolerance=np.zeros_like(self.sums)
  self.max_calibration_error=0.0;self.gain_codes=set()
 def __call__(self,timestamps,raw,calibrated,errors):
  require(not np.any(errors.get()),'Expansion error flags')
  for i,timestamp in enumerate(timestamps):
   timestamp=int(timestamp)
   require(timestamp in self.ordinal and timestamp not in self.seen,'Numerical identity')
   expected_raw=self.reader.raw(timestamp,self.selected)
   actual_raw=raw[i].reshape(32,PIXELS)[:,self.selected].get()
   require(np.array_equal(actual_raw,expected_raw),'Sampled expansion mismatch')
   flat=expected_raw.reshape(-1);codes=(flat>>14).astype(np.intp)
   self.gain_codes.update(map(int,np.unique(codes)))
   constants=self.packed[codes,np.arange(flat.size)]
   expected=((flat&0x3fff).astype(np.float32)-constants[:,0])*constants[:,1]
   expected=expected.reshape(expected_raw.shape)
   observed=calibrated[i].reshape(32,PIXELS)[:,self.selected].get()
   tolerance=1e-5+8*np.finfo(np.float32).eps*np.abs(expected.astype(np.float64))
   error=np.abs(observed.astype(np.float64)-expected)
   require(np.isfinite(observed).all() and np.all(error<=tolerance),'Sampled calibration mismatch')
   self.max_calibration_error=max(self.max_calibration_error,float(error.max()))
   index=self.ordinal[timestamp]%99
   self.sums[index]+=expected;self.absolute[index]+=np.abs(expected.astype(np.float64));self.tolerance[index]+=tolerance
   self.counts[index]+=1;self.seen.add(timestamp)
 def finish(self,cube,delivered):
  require(self.seen==set(delivered),'Numerical delivery coverage')
  cube.drain();maximum=0.0
  with cube.bank.stream:
   counts=cube.bank.counts.get()
   require(np.array_equal(counts,self.counts),'Numerical cube counts')
   for index,count in enumerate(self.counts):
    actual=cube.bank.sums[index].reshape(32,PIXELS)[:,self.selected].get()
    gamma=int(count)*2**-24/(1-int(count)*2**-24)
    bound=gamma*(self.absolute[index]+self.tolerance[index])+self.tolerance[index]+1e-5
    error=np.abs(actual.astype(np.float64)-self.sums[index])
    require(np.isfinite(actual).all() and np.all(error<=bound),'Sampled cube sum mismatch')
    if not count:require(not np.any(actual),'Empty numerical bin')
    maximum=max(maximum,float((error/bound).max()))
  self.reader.close()
  return dict(status='pass',events=len(self.seen),panels=32,pixel_indices=self.selected.tolist(),bins=100,gain_codes_seen=sorted(self.gain_codes),max_calibration_error=self.max_calibration_error,max_cube_bound_ratio=maximum,scope='Deterministic sampled pixels on all panels/events/bins; not exhaustive numerical agreement',calibration_tolerance='1e-5 + 8*eps32*abs(reference)',cube_tolerance='gamma_n*(sum_abs+sum_calibration_tolerances)+sum_calibration_tolerances+1e-5')
