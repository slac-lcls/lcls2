
import numpy as np
import os 

from psana import DataSource

def test_hsd():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_hsd.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')

    for nevt,evt in enumerate(myrun.events()):
        wfs = det.hsd.waveforms(evt)
        fex = det.hsd.peaks(evt)
        nwf = 0
        for digitizer,wf in wfs.items():
            times = wfs[digitizer]['times']
            assert np.array_equal(times,np.arange(1600))
            for channel,waveform in wf.items():
                if type(channel) is int: # skip over the 'times'
                    assert len(waveform)== len(times)
                    nwf+=1
        assert nwf == 1
        nfex = 0
        for digitizer,fexdata in fex.items():
            for channel,fexchan in fexdata.items():
                starttimes,peaks = fexchan
                # for this test-pattern data there is only one peak found: the
                # entire waveform starting at time 0.
                assert np.array_equal(peaks[0],waveform)
                assert len(starttimes) == 1
                assert starttimes[0] == 0
                nfex+=1
        assert nfex == 1
    assert(nevt==3)
