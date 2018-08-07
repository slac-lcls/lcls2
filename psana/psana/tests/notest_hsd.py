#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------
import os
cwd = os.path.abspath(os.path.dirname(__file__))

doPlot = 0

#------------------------------

def test_hsd():
    import psana
    import numpy as np
    if doPlot: import matplotlib.pyplot as plt

    ds = psana.DataSource(os.path.join(cwd,'hsd_061918_n3.xtc'))
    det = ds.Detector("xpphsd")

    chanNum = 3
    for i, evt in enumerate(ds.events()):
        waveforms = det.waveforms(evt)
        peaks, startPos = det.peaks(evt, chanNum)
        print("waveform: ", waveforms)
        print("list of peaks:", peaks)
        print("list of pos:", startPos)

        if doPlot:
            plt.plot(waveforms[0], 'o-')
            plt.title('Event {}, Waveform channel: {}'.format(i, chanNum))
            plt.show()

            recon = -999 * np.ones((1600,))
            xaxis = np.zeros((1600,))
            for j in range(len(peaks)):
                recon[startPos[j]:startPos[j] + len(peaks[j])] = peaks[j]

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
