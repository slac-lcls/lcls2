import dgrampy as dp
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
    ds = DataSource(files=[fname])
    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')
    for evt in myrun.events():
        det.fex.show(evt)


if __name__ == "__main__":
    ifname = '/cds/data/drpsrcf/users/monarin/tmolv9418/tmolv9418-r0175-s000-c000.xtc2'
    fd = os.open(ifname, os.O_RDONLY)
    pyiter = dp.PyXtcFileIterator(fd, 0x1000000)
    
    # Defines detector and alg.
    # Below example settings become hsd_fex_4_5_6 for its detector interface.
    # TODO: Think about initializing these by adding another example
    # where we create xtc from scratch.
    alg = dp.alg("fex", 4, 5, 6)
    det = dp.det("xpphsd", "hsd", "detnum1234")     # detname, dettype, detid
    
    # Define data formats
    datadef_dict = {
            "valFex": (dp.DataType.FLOAT, 0),
            "strFex": (dp.DataType.CHARSTR, 1),
            "arrayFex0": (dp.DataType.UINT8, 2),
            "arrayFex1": (dp.DataType.UINT16, 2),
            "arrayFex2": (dp.DataType.UINT32, 2),
            "arrayFex3": (dp.DataType.UINT64, 2),
            "arrayFex4": (dp.DataType.INT8, 2),
            "arrayFex5": (dp.DataType.INT16, 2),
            "arrayFex6": (dp.DataType.INT32, 2),
            "arrayFex7": (dp.DataType.INT64, 2),
            "arrayFex8": (dp.DataType.FLOAT, 2),
            "arrayFex9": (dp.DataType.DOUBLE, 2),
            "arrayString": (dp.DataType.CHARSTR, 1),
            }
    datadef = dp.datadef(datadef_dict)
    
    # Let dp know the file to write to
    ofname = 'out.xtc2'
    dp.creatextc2(ofname)

    names0 = None
    for i in range(5):
        print(f"\nPYTHON NEW DGRAM {i}")
        pydg = pyiter.next()

        # Add new Names to config
        if i == 0:
            print(f"PYTHON GOT CONFIG")
            names0 = dp.names(pydg, det, alg, datadef)
            print(f"PYTHON CONFIG DONE ADD NAME")

        # Add new Data to L1
        if i == 4:
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
                dp.adddata(pydg, names0, datadef, data) # either change to add
            print(f"PYTHON L1ACCEPT DONE ADDDATA")
        
        # Copy the event to buffer
        dp.save(pydg)

    dp.closextc2()

    test_output(ofname)
