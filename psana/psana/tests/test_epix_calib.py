from psana import DataSource
import os
import numpy as np

def test_epix_calib():
    correctanswer=np.array(([-177.67036, -187.09805, -171.72614, -171.77377, -142.82375], [-67.91426, -40.756596, -55.872475, -62.017673, -63.55546]))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('*** before datasource')
    ds = DataSource(files=os.path.join(dir_path,'test_epix_calib.xtc2'))
    print('*** before runs()')
    myrun = next(ds.runs())
    print('*** before Detector')
    epix = myrun.Detector('epixquad')
    for nevt,evt in enumerate(myrun.events()):
        print('*** before image',nevt)
        image=epix.raw.image(evt)
        calibsample=image[10][0:5]
        assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_epix_calib()
