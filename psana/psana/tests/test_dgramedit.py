from psana.dgramedit import DgramEdit, AlgDef, DetectorDef, PyXtcFileIterator
import os, sys, io
import numpy as np
from psana import DataSource
import pytest

@pytest.fixture
def output_filename(tmp_path):
    #fname = str(tmp_path / 'out-dgramedit-test.xtc2')
    fname = 'out-dgramedit-test.xtc2'
    return fname

def create_array(dtype):
    if dtype in (np.float32, np.float64):
        arr = np.stack([np.zeros(3, dtype=dtype)+np.finfo(dtype).min, 
            np.zeros(3, dtype=dtype)+np.finfo(dtype).max])
    else:
        arr = np.stack([np.arange(np.iinfo(dtype).min, np.iinfo(dtype).min+3, dtype=dtype), 
                np.arange(np.iinfo(dtype).max-2, np.iinfo(dtype).max+1, dtype=dtype)])
    return arr

def check_output(fname):
    print(f"TEST OUTPUT by reading {fname} using DataSource")
    ds = DataSource(files=[fname])
    myrun = next(ds.runs())
    det = myrun.Detector('xpphsd')
    det2 = myrun.Detector('xppcspad')
    for evt in myrun.events():
        det.fex.show(evt)
        arrayRaw = det2.raw.raw(evt)
        knownArrayRaw = create_array(np.float32)
        # Convert the known array to the reformatted shape done by
        # the detecter interface.
        knownArrayRaw = np.reshape(knownArrayRaw, [1] + list(knownArrayRaw.shape))  
        assert np.array_equal(arrayRaw, knownArrayRaw)
        print(f'det2 arrayRaw: {arrayRaw}')
        
        # Currently only checking one field from the second detector
        #assert np.array_equal(arrayRaw, create_array(np.float32))

@pytest.mark.skipif(sys.platform == 'darwin', reason="check_output fails on macos")
def test_run_dgramedit(output_filename):
    # Test with output writing to a file
    run_dgramedit(output_filename)

    # Test with output writing to a bytearray then dump that out to a file
    run_dgramedit(output_filename, as_file=False)
    
def run_dgramedit(output_filename, as_file=True):
    ifname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'dgramedit-test.xtc2')
    fd = os.open(ifname, os.O_RDONLY)
    pyiter = PyXtcFileIterator(fd, 0x1000000)
    
    # Defines detector and alg.
    # Below example settings become hsd_fex_4_5_6 for its detector interface.
    algdef = AlgDef("fex", 4, 5, 6)
    detdef = DetectorDef("xpphsd", "hsd", "detnum1234")     # detname, dettype, detid

    # Define data formats
    datadef = {
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
              }

    algdef2 = AlgDef("raw", 2, 3, 42)
    detdef2 = DetectorDef("xppcspad", "cspad", "detnum1234")
    datadef2 = {"arrayRaw": (np.float32, 2), }
    
    # We test both output as file and as bytearray modes
    if as_file:
        xtc2buf = open(output_filename, "wb")
    else:
        xtc2buf = bytearray(64000000)

    # This offset is passed in at save() but only used in bytearray output mode
    offset = 0

    names0 = None
    for i in range(6):
        pydg = pyiter.next()

        # Add new Names to config
        if i == 0:
            config = DgramEdit(pydg)
            det = config.Detector(detdef, algdef, datadef)
            det2 = config.Detector(detdef2, algdef2, datadef2)
            config.save(xtc2buf, offset=offset)
            de_size = config.savedsize
            offset += de_size

        # Add new Data to L1
        elif i >= 4:
            dgram = DgramEdit(pydg, config=config)

            # Fill in data for previously given datadef (all fields
            # must be completed)
            det.fex.valFex = 1600.1234
            det.fex.strFex = "hello string"
            det.fex.arrayFex0 = create_array(np.uint8)
            det.fex.arrayFex1 = create_array(np.uint16)
            det.fex.arrayFex2 = create_array(np.uint32)
            det.fex.arrayFex3 = create_array(np.uint64)
            det.fex.arrayFex4 = create_array(np.int8)
            det.fex.arrayFex5 = create_array(np.int16)
            det.fex.arrayFex6 = create_array(np.int32)
            det.fex.arrayFex7 = create_array(np.int64)
            det.fex.arrayFex8 = create_array(np.float32)
            det.fex.arrayFex9 = create_array(np.float64)
            dgram.adddata(det.fex)
            
            det2.raw.arrayRaw = create_array(np.float32)
            dgram.adddata(det2.raw)
            
            if i == 4:
                dgram.removedata("hsd","raw") # per event removal 

            dgram.save(xtc2buf, offset=offset)
            de_size = dgram.savedsize
            offset += de_size
        
        # Other transitions
        else: 
            dgram = DgramEdit(pydg, config=config)
            dgram.save(xtc2buf, offset=offset)
            de_size = dgram.savedsize
            offset += de_size

    if not as_file:
        with open(output_filename, "wb") as ofile:
            ofile.write(xtc2buf[:offset])
    else:
        xtc2buf.close()
    
    # Open the generated xtc2 file and test the value inside
    check_output(output_filename)

if __name__ == "__main__":
    test_run_dgramedit()
