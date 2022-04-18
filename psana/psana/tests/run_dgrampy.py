from dgrampy import DgramPy, AlgDef, DetectorDef, PyXtcFileIterator
from dgrampy import datadef as DataDef
import os
import numpy as np
from psana import DataSource

def create_array(dtype):
    if dtype in (np.float32, np.float64):
        arr = np.stack([np.zeros(3, dtype=dtype)+np.finfo(dtype).min, 
            np.zeros(3, dtype=dtype)+np.finfo(dtype).max])
    else:
        arr = np.stack([np.arange(np.iinfo(dtype).min, np.iinfo(dtype).min+3, dtype=dtype), 
                np.arange(np.iinfo(dtype).max-2, np.iinfo(dtype).max+1, dtype=dtype)])
    return arr

def test_output(fname):
    print(f"TEST OUTPUT by reading {fname} using DataSource")
    ds = DataSource(files=[fname])
    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')
    for evt in myrun.events():
        det.fex.show(evt)


if __name__ == "__main__":
    ifname = '/cds/data/drpsrcf/users/monarin/tmolv9418/tmolv9418-r0175-s000-c000.xtc2'
    fd = os.open(ifname, os.O_RDONLY)
    pyiter = PyXtcFileIterator(fd, 0x1000000)
    
    # Defines detector and alg.
    # Below example settings become hsd_fex_4_5_6 for its detector interface.
    # TODO: Think about initializing these by adding another example
    # where we create xtc from scratch.
    alg = AlgDef("fex", 4, 5, 6)
    det = DetectorDef("xpphsd", "hsd", "detnum1234")     # detname, dettype, detid
    
    # Define data formats
    datadef_dict = {
            "valFex": (np.float32, 0),
            "strFex": (str, 1),
            "arrayFex0": (np.uint8, 2),
            "arrayFex1": (np.uint16, 2),
            "arrayFex2": (np.uint32, 2),
            "arrayFex3": (np.uint64, 2),
            "arrayFex4": (np.int8, 2),
            "arrayFex5": (np.int16, 2),
            "arrayFex6": (np.int32, 2),
            "arrayFex7": (np.int64, 2),
            "arrayFex8": (np.float32, 2),
            "arrayFex9": (np.float64, 2),
            "arrayString": (str, 1),
            }
    datadef = DataDef(datadef_dict)
    
    # Open output file for writing
    ofname = 'out.xtc2'
    xtc2file = open(ofname, "wb")

    names0 = None
    for i in range(5):
        print(f"\nPYTHON NEW DGRAM {i}")
        pydg = pyiter.next()

        # Add new Names to config
        if i == 0:
            config = DgramPy(pydg)
            print(f"PYTHON GOT CONFIG")
            names0 = config.addnames(det, alg, datadef)
            print(f"PYTHON CONFIG DONE ADD NAME")
            config.save(xtc2file)

        # Add new Data to L1
        elif i == 4:
            dgram = DgramPy(pydg, config=config)
            print(f"PYTHON L1ACCEPT")
            data = {
                    "valFex": 1600.1234,
                    "strFex": "hello string",
                    "arrayFex0": create_array(np.uint8),
                    "arrayFex1": create_array(np.uint16),
                    "arrayFex2": create_array(np.uint32),
                    "arrayFex3": create_array(np.uint64),
                    "arrayFex4": create_array(np.int8),
                    "arrayFex5": create_array(np.int16),
                    "arrayFex6": create_array(np.int32),
                    "arrayFex7": create_array(np.int64),
                    "arrayFex8": create_array(np.float32),
                    "arrayFex9": create_array(np.float64),
                    "arrayString": np.array(['hello string array']),
                    }
            if names0:
                dgram.adddata(names0, datadef, data) # either change to add
            print(f"PYTHON L1ACCEPT DONE ADDDATA")
            dgram.save(xtc2file)
        
        # Other transitions
        else: 
            dgram = DgramPy(pydg, config=config)
            dgram.save(xtc2file)

    xtc2file.close()

    test_output(ofname)
