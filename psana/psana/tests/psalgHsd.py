import hsd as Hsd
import numpy as np
import sys
import os
cwd = os.path.abspath(os.path.dirname(__file__))

doPlot = int(sys.argv[1])
if doPlot: import matplotlib.pyplot as plt

# det = ds.Detector('xpphsd') #<----- high level interface
# hsd = det(evt)

def OpaqueRawData(name, config):
    sw = getattr(config,'software')
    det = getattr(sw, name)
    print('sw, det: ', sw, det)
    df = eval(str(det.dettype) + '._Factory()')
    return df.create(name, config)

class OpaqueRawDataBase(object):
    def __init__(self, name, config):
        self.detectorName = name
        self.config = config
        print("detName,config: ", name, config)

class hsd(OpaqueRawDataBase):
    """
    hsd reader
    """
    def __init__(self, name, config):
        super(hsd, self).__init__(name, config)
        self.name = name
        sw = getattr(config, 'software')
        detcfg = getattr(sw, name)
        print("det, software, version: ", detcfg.dettype, detcfg.hsd.software, detcfg.hsd.version)
        assert detcfg.dettype      == 'hsd'
        assert detcfg.hsd.software == 'hsd'


    def __call__(self, evt):
        # FIXME: discover how many channels there are
        chan0 = evt._dgrams[0].xpphsd.hsd.chan0
        chan1 = evt._dgrams[0].xpphsd.hsd.chan1
        chan2 = evt._dgrams[0].xpphsd.hsd.chan2
        chan3 = evt._dgrams[0].xpphsd.hsd.chan3
        chans = [chan0, chan1, chan2, chan3]
        nonOpaqueHsd = Hsd.hsd("1.2.3", chans)  # make an object per event
        return nonOpaqueHsd

    class _Factory:
        def create(self, name, config): return hsd(name, config)


from psana.dgrammanager import DgramManager
ds = DgramManager(os.path.join(cwd,'hsd_061918_n3.xtc'))

rawData=OpaqueRawData('xpphsd',ds.configs[0])

waveformStack = []
myPeaks = []
chanNum = 2
for i in range(5):
    print("### Event: ",i)
    evt = ds.next()
    myrawhsd = rawData(evt)
    print(dir(myrawhsd))
    raw = myrawhsd.waveform()
    listOfPeaks, sPos = myrawhsd.peaks(chanNum)

    plt.plot(raw[0],'o-')
    plt.title('Event {}, Waveform channel: {}'.format(i,chanNum))
    plt.show()

    print("FEX: ", len(listOfPeaks), sPos)
    recon = -999*np.ones((1600,))
    xaxis = np.zeros((1600,))
    for j in range(len(listOfPeaks)):
        recon[sPos[j]:sPos[j]+len(listOfPeaks[j])] = listOfPeaks[j]
    plt.plot(recon,'o-')
    plt.plot(xaxis,'r-')
    plt.title('Event {}, Reconstructed Fex channel: {}'.format(i,chanNum))
    plt.show()

    if waveformStack is None:
        waveformStack = raw[0]
    else:
        waveformStack.append(raw[0])
    if i >=1: break

print("waveform: ", waveformStack)

