
import numpy as np
import os 

from psana import DataSource

def test_hsd():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_hsd.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')
    seg_chans = det.hsd._seg_chans()
    iseg = 0
    ichan = 0

    for nevt,evt in enumerate(myrun.events()):
        wfs = det.hsd.waveforms(evt)
        peaks = det.hsd.peaks(evt)
        times = wfs[iseg]['times']
        waveform = wfs[iseg][ichan]
        assert np.array_equal(times,np.arange(1600))
        assert len(waveform)== len(times)
        starttimes,peaks = peaks[iseg][ichan]
        # for this test-pattern data there is only one peak found: the
        # entire waveform starting at time 0.
        assert np.array_equal(peaks[0],waveform)
        assert starttimes[0] == 0
    assert(nevt==3)
