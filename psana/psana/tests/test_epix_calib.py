from psana import DataSource
import os
import numpy as np

def test_epix_calib():
#    correctanswer=np.array(([-177.67036, -187.09805, -171.72614, -171.77377, -142.82375], [-67.91426, -40.756596, -55.872475, -62.017673, -63.55546]))
    correctanswer=np.array(([-186.44453, -190.3106, -178.32388, -176.64021, -148.71886], [-76.68844, -43.96913, -62.470226, -66.88411, -69.45056]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_epix_calib.xtc2'))
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
