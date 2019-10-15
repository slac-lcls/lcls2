import numpy as np
import os 

from psana import DataSource

def test_timetool():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_timetool.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('tt_detector_name_placeholder')

    for nevt,evt in enumerate(myrun.events()):
        image = det.tt_algorithm_placeholder.image(evt)

        #print("image shape = ",image.shape)
        assert image.shape == (2224,) or image.shape == (144,) or image.shape == (4304,)

    assert nevt==5

if __name__ == "__main__":
    test_timetool()
