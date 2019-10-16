from psana import *

dsource = MPIDataSource('exp=amox27716:run=100:smd')
acq = Detector('ACQ1')

smldata = dsource.small_data('hexanode.h5',gather_interval=100)

partial_run_sum = None
for nevt,evt in enumerate(dsource.events()):

   wfs = acq.waveform(evt)
   times = acq.wftime(evt)
   if wfs is None: continue
   print(wfs.shape,times.shape)
   smldata.event(waveforms=wfs,times=times)
   if nevt>100: break

smldata.save()
