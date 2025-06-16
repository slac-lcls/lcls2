from psana import DataSource
import os
import numpy as np
import pytest

@pytest.mark.skip(reason="to be debugged when mikhail returns from vacation in June 2025")
def test_jungfrau05M_calib():
    correctanswer=np.array(([4.0649414, 9.653076, 15.843018, -12.125977, -24.302979],\
                            [0.06494141, -6.346924, -27.156982, -3.1259766, -15.3029785]))

    dirscr = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirscr,'test_data/detector/test_jungfrau05M_calib.xtc2')
    print('DEBUG path to data:', path)
    ds = DataSource(files=path)
    myrun = next(ds.runs())
    odet = myrun.Detector('jungfrau')
    for nevt,evt in enumerate(myrun.events()):
        if nevt>1: break
        image=odet.raw.image(evt)
        calibsample=image[10][0:5]
        print('DEBUG calibsample  ', calibsample)
        print('DEBUG correctanswer', correctanswer[nevt][0:5])
        assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_jungfrau05M_calib()
