#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------

doPlot = 0

import hsd as Hsd
import numpy as np
import matplotlib.pyplot as plt

def OpaqueRawData(name, config):
    sw = getattr(config, 'software')
    det = getattr(sw, name)
    df = eval(str(det.dettype) + '._Factory()')
    return df.create(name, config)

class OpaqueRawDataBase(object):
    def __init__(self, name, config):
        self.detectorName = name
        self.config = config

class hsd(OpaqueRawDataBase):
    """
    hsd reader
    """
    def __init__(self, name, config):
        super(hsd, self).__init__(name, config)
        self.name = name
        sw = getattr(config, 'software')
        detcfg = getattr(sw, name)
        assert detcfg.dettype == 'hsd'
        assert detcfg.hsd.software == 'hsd'

    def __call__(self, evt):
        # FIXME: discover how many channels there are
        chan0 = evt.dgrams[0].xpphsd.hsd.chan0
        chan1 = evt.dgrams[0].xpphsd.hsd.chan1
        chan2 = evt.dgrams[0].xpphsd.hsd.chan2
        chan3 = evt.dgrams[0].xpphsd.hsd.chan3
        chans = [chan0, chan1, chan2, chan3]
        nonOpaqueHsd = Hsd.hsd("1.0.0", chans)  # make an object per event
        return nonOpaqueHsd

    class _Factory:
        def create(self, name, config): return hsd(name, config)

#------------------------------

def test_hsd():
    from psana.dgrammanager import DgramManager
    ds = DgramManager('/reg/neh/home/yoon82/Software/lcls2/hsd_052218b.xtc')

    rawData=OpaqueRawData('xpphsd',ds.configs[0])

    waveformStack = []
    chanNum = 0
    for i in range(5):
        print("### Event: ",i)
        evt = ds.next()
        myrawhsd = rawData(evt)
        raw = myrawhsd.raw()
        listOfPeaks, sPos = myrawhsd.fex(chanNum)

        if doPlot:
            plt.plot(raw[0],'o-')
            plt.title('Event {}, Waveform channel: {}'.format(i,chanNum))
            plt.show()

        print("FEX: ", len(listOfPeaks), sPos)
        recon = -999*np.ones((1600,))
        xaxis = np.zeros((1600,))
        for j in range(len(listOfPeaks)):
            recon[sPos[j]:sPos[j]+len(listOfPeaks[j])] = listOfPeaks[j]

        if doPlot:
            plt.plot(recon,'o-')
            plt.plot(xaxis,'r-')
            plt.title('Event {}, Reconstructed Fex channel: {}'.format(i,chanNum))
            plt.show()

        if waveformStack is None:
            waveformStack = raw[0]
        else:
            waveformStack.append(raw[0])

    print("waveform: ", waveformStack)

#------------------------------

def psalg() :
    test_hsd()

#------------------------------

if __name__ == '__main__':
    psalg()

#------------------------------