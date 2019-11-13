import numpy as np
import os 

from psana import DataSource

def test_timetool():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_timetool.xtc2'))

    myrun = next(ds.runs())
    tt_detector_object = myrun.Detector('tt_detector_name_placeholder')

    for nevt,evt in enumerate(myrun.events()):
        parsed_frame_object = tt_detector_object.tt_algorithm_placeholder.parsed_frame(evt)


        image = tt_detector_object.tt_algorithm_placeholder._image(evt)

        #print("image shape = ",image.shape)
        assert image.shape == (2208,) or image.shape == (144,) or image.shape == (4272,)

    assert nevt==598

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
