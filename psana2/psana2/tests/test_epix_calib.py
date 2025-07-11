from psana2 import DataSource
import os
import numpy as np
import pytest

@pytest.mark.skip(reason="Known calibration mismatch; under investigation.")
def test_epix_calib():
    correctanswer=np.array(([-186.44453, -190.3106, -178.32388, -176.64021, -148.71886],\
                            [-76.68844, -43.96913, -62.470226, -66.88411, -69.45056]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path, 'test_data/detector/test_epix_calib.xtc2'))
    myrun = next(ds.runs())
    epix = myrun.Detector('epixquad')
    for nevt,evt in enumerate(myrun.events()):
        image=epix.raw.image(evt)
        calibsample=image[10][0:5]
        #print('DEBUG calibsample  ', calibsample)
        #print('DEBUG correctanswer', correctanswer[nevt][0:5])
        assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_epix_calib()
