import numpy as np
import os 

from psana import DataSource

def test_timetool():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_timetool.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('tmotimetool')

    for nevt,evt in enumerate(myrun.events()):
        image = det.timetool.image(evt)

        #print("image shape = ",image.shape)
        assert image.shape == (2224,)

    assert nevt==5

if __name__ == "__main__":
    test_timetool()
