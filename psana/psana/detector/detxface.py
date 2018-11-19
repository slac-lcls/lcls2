from psana import DataSource
import matplotlib.pyplot as plt
import time

doPlot = 1

ds = DataSource('/reg/neh/home/yoon82/hsd_nov06_ps3.xtc') # TODO: add an example xtc file to git

tstart = time.time()
for nevt,evt in enumerate(ds.events()):
    raw = evt.xpphsd.hsd._raw()
    fex = evt.xpphsd.hsd._fex()
    samples = evt.xpphsd.hsd._samples()
    streams = evt.xpphsd.hsd._streams()
    channels = evt.xpphsd.hsd._channels()
    sync = evt.xpphsd.hsd._sync()
    waveform = evt.xpphsd.hsd.waveforms()
    pks = evt.xpphsd.hsd.peaks()
    print("raw, fex, samples, streams, channels, sync, waveform: ",
          raw, fex, samples, streams, channels, sync, waveform, pks)

    if doPlot and len(pks['chan00'][0]) > 0:
        if waveform['chan00'].size > 0:
            plt.subplot(121)
            plt.plot(waveform['chan00'], 'x-')
            plt.title("chan00 waveform: " + str(nevt))
            plt.subplot(122)
        plt.plot(pks['chan00'][1][0], 'x-')
        plt.title("chan00 fex: " + str(nevt))
        plt.show()
    if nevt == 1: break
print((nevt)/(time.time()-tstart))
