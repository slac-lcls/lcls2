
import numpy as np
import os 

from psana2 import DataSource

def test_hsd_padonly():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_hsd.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')

    for nevt,evt in enumerate(myrun.events()):
        pad = det.hsd.padded(evt)

        # make sure we return None if there are no entries
        if not pad: assert pad is None

        if pad:
            for ndigi,(digitizer,wfsdata) in enumerate(pad.items()):
                times = wfsdata['times']
                nwf = 0
                for channel,waveform in wfsdata.items():
                    if type(channel) is int: # skip over the 'times'
                        nwf+=1
                        assert len(waveform)== len(times)
                assert nwf == 1, nwf # counting from one
            assert ndigi == 1, ndigi # enumerate counting from zero
        if nevt == 20: break # stop early since this xtc file has incomplete dg
    assert(nevt>0) # make sure we received events

def test_hsd():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_hsd.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')

    for nevt,evt in enumerate(myrun.events()):
        wfs = det.hsd.waveforms(evt)
        pad = det.hsd.padded(evt)
        fex = det.hsd.peaks(evt)

        # make sure we return None if there are no entries
        if not wfs: assert wfs is None
        if not pad: assert pad is None
        if not fex: assert fex is None

        if wfs:
            for ndigi,(digitizer,wfsdata) in enumerate(wfs.items()):
                times = wfsdata['times']
                nwf = 0
                for channel,waveform in wfsdata.items():
                    if type(channel) is int: # skip over the 'times'
                        nwf+=1
                        assert len(waveform)== len(times)
                assert nwf == 1, nwf # counting from one
            assert ndigi == 1, ndigi # enumerate counting from zero
        if pad:
            for ndigi,(digitizer,wfsdata) in enumerate(pad.items()):
                times = wfsdata['times']
                nwf = 0
                for channel,waveform in wfsdata.items():
                    if type(channel) is int: # skip over the 'times'
                        nwf+=1
                        assert len(waveform)== len(times)
                assert nwf == 1, nwf # counting from one
            assert ndigi == 1, ndigi # enumerate counting from zero
        if fex:
            for ndigi,(digitizer,fexdata) in enumerate(fex.items()):
                for nfex,(channel,fexchan) in enumerate(fexdata.items()):
                    startpos,peaks = fexchan
                    # check consistency with raw data
                    if (wfs):
                        for npeak,(start,peak) in enumerate(zip(startpos,peaks)):
                            peaklen = len(peak)
                            raw = wfs[digitizer][channel][start:start+peaklen]
                            assert (peak==raw).all(), (peak, raw)
                assert nfex == 0, nfex # enumerate counting from zero
            assert ndigi == 1, ndigi # enumerate counting from zero
        if nevt == 20: break # stop early since this xtc file has incomplete dg
    assert(nevt>0) # make sure we received events

if __name__ == "__main__":
    test_hsd()
    test_hsd_padonly()
