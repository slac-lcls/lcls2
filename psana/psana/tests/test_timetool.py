import numpy as np
import os 

from psana import DataSource

def test_timetool():
    return
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

"""
import numpy as np
import os 

from psana import DataSource
dir_path = os.path.dirname("psana/psana/tests/")
ds = DataSource(files=os.path.join(dir_path,'test_timetool.xtc2'))

myrun = next(ds.runs())
det = myrun.Detector('tt_detector_name_placeholder')
my_events = myrun.events()
evt = next(my_events)
det.tt_algorithm_placeholder.parsed_frame(evt)

pFrame = det.tt_algorithm_placeholder.parsed_frame(evt)
pFrame.background_frame
pFrame.prescaled_frame
plt.plot([int(i) for i in pFrame.background_frame.data],'.')
plt.plot([int(i) for i in pFrame.background_frame],'.')
plt.plot([int(i) for i in pFrame.prescaled_frame],'.')
history



"""
