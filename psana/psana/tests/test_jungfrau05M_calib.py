from psana import DataSource
import os
import numpy as np

def test_jungfrau05M_calib():
    correctanswer=np.array(([4.0649414, 9.653076, 15.843018, -12.125977, -24.302979],\
                            [0.06494141, -6.346924, -27.156982, -3.1259766, -15.3029785],\
                            [15.064941, 14.653076, 13.843018, 25.874023, 4.6970215]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_data/detector/test_jungfrau05M_calib.xtc2'))
    myrun = next(ds.runs())
    epix = myrun.Detector('jungfrau')
    for nevt,evt in enumerate(myrun.events()):
        if nevt>1: break
        image=epix.raw.image(evt)
        calibsample=image[10][0:5]
        #print('DEBUG calibsample  ', calibsample)
        #print('DEBUG correctanswer', correctanswer[nevt][0:5])
        assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_jungfrau05M_calib()
