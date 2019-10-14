import numpy as np
import os 

from psana import DataSource

def test_ts():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds = DataSource(files=os.path.join(dir_path,'test_ts.xtc2'))

    myrun = next(ds.runs())
    det = myrun.Detector('xppts')

    for nevt,evt in enumerate(myrun.events()):
        info = det.ts.info(evt)
        seqinfo = det.ts.sequencer_info(evt)
    assert nevt==1

if __name__ == "__main__":
    test_ts()
