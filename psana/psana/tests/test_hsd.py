#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------

doPlot = 0

#------------------------------

def test_hsd():
    import psana
    import numpy as np
    import matplotlib.pyplot as plt

    ds = psana.DataSource('/reg/neh/home/yoon82/Software/lcls2/hsd_052218b.xtc')
    det = ds.Detector("xpphsd")

    chanNum = 2
    for i, evt in enumerate(ds.events()):
        hsd = det(evt)
        waveform = hsd.waveform()
        listOfPeaks, sPos = hsd.peaks(chanNum) # TODO: return times

        print("waveform: ", waveform)
        print("list of peaks:", listOfPeaks)
        print("list of pos:", sPos)

        if doPlot:
            plt.plot(waveform[0], 'o-')
            plt.title('Event {}, Waveform channel: {}'.format(i, chanNum))
            plt.show()

        recon = -999 * np.ones((1600,))
        xaxis = np.zeros((1600,))
        for j in range(len(listOfPeaks)):
            recon[sPos[j]:sPos[j] + len(listOfPeaks[j])] = listOfPeaks[j]

        if doPlot:
            plt.plot(recon, 'o-')
            plt.plot(xaxis, 'r-')
            plt.title('Event {}, Reconstructed Fex channel: {}'.format(i, chanNum))
            plt.show()
        break

#------------------------------

def psalg() :
    test_hsd()

#------------------------------

if __name__ == '__main__':
    psalg()

#------------------------------