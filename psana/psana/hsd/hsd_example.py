from psana import DataSource
import numpy as np

def my_evt_filter(smalldata_evt): # throw away bad events using small data
    return True

def my_realtime_monitor(user_data_dict): # watch user-defined data from smd.event() below
    print(user_data_dict)

ds = DataSource(exp='tstx00517', run=49, dir='/ffb01/data/tst/tstx00517/xtc', filter=my_evt_filter)

smd = ds.smalldata(filename='run49.h5', batch_size=5, callbacks=[my_realtime_monitor])
run = next(ds.runs())
det = run.Detector('tmohsd') # several high speed digitizers

mysum = 0.0; hsd_num = 0; chan_num = 0
for i,evt in enumerate(run.events()):
    waveforms = det.raw.waveforms(evt)
    peaks     = det.raw.peaks(evt)
    if waveforms is None:
        continue
    mypeak = waveforms[hsd_num][chan_num][12:15] # select "peak" in waveform 0
    myarea = np.sum(mypeak)      # integrate the peak area
    smd.event(evt, mypeak=mypeak, myarea=myarea)
    mysum += myarea

if smd.summary:
    smd.sum(mysum)
    smd.save_summary({'summary_data' : mysum}, summary_int=1)
smd.done()
