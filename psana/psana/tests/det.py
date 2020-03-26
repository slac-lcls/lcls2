import sys
from psana import DataSource
import numpy as np
import vals
from psana.app.detnames import detnames

def det(files):
    ds = DataSource(files=files)
    for run in ds.runs(): # Detector is created based on per-run config. 
        hsd = run.Detector('xpphsd')
        cspad = run.Detector('xppcspad')
        for evt in run.events():
            assert(hsd.raw.calib(evt).shape==(5,))
            assert(hsd.fex.calib(evt).shape==(6,))
            padarray = vals.padarray
            assert(np.array_equal(cspad.raw.calib(evt),np.stack((padarray,padarray))))
            assert(np.array_equal(cspad.raw.image(evt),np.vstack((padarray,padarray))))

def calib():
    # Test calib_constants here prior to user.py, which uses mpi
    # and tends to hang without error...
    # Use cxid9114 with run 96 (known to work) as a test case.
    exp = "cxid9114"
    run_no = 96
    det_str = "cspad_0002"
    from psana.pscalib.calib.MDBWebUtils import calib_constants
    
    pedestals, _ = calib_constants(det_str, exp=exp, ctype='pedestals', run=run_no)
    assert pedestals.shape == (32, 185, 388)

    common_mode, _ = calib_constants(det_str, exp=exp, ctype='common_mode', run=run_no)
    assert commmon_mode.shape == (32, 185, 388)

    geometry_string, _ = calib_constants(det_str, exp=exp, ctype='geometry', run=run_no)
    try:
        if not isinstance(geometry_string, str) and geometry_string is not None:
            import unicodedata
            geometry_string = unicodedata.normalize('NFKD', geometry_string).encode('ascii','ignore')
    except Exception as e:
        raise("Error getting geometry from calib_constants: %s"%e)


class MyCustomArgs(object):
    raw = False
    epics = False
    scan = False
    def __init__(self, dsname, option):
        self.dsname = dsname
        if option == "-r":
            self.raw = True
        elif option == "-e":
            self.epics = True
        elif option == "-s":
            self.scan = True

def detnames(xtc_file):
    ds = DataSource(files=xtc_file)
    myrun = next(ds.runs())

    assert ('xppcspad', 'cspad', 'raw', '2_3_42') in myrun.xtcinfo
    assert myrun.epicsinfo[('HX2:DVD:GCC:01:PMON', 'raw')] == 'raw'
    assert myrun.scaninfo[('motor1', 'raw')] == 'raw'


if __name__ == '__main__':
    det()
    calib()
