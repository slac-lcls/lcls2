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

def detnames_cmd(dsname):
    args = MyCustomArgs(dsname, "-r")
    check_against = [('xppcspad', 'cspad', 'raw', '2_3_42'), ('xpphsd', 'hsd', 'fex', '4_5_6'), ('xpphsd', 'hsd', 'raw', '0_0_0'), ('epics', 'epics', 'raw', '2_0_0'), ('runinfo', 'runinfo', 'runinfo', '0_0_1'), ('scan', 'scan', 'raw', '2_0_0')]
    detnames(args, check_against)

    args = MyCustomArgs(dsname, "-e")
    check_against = {('HX2:DVD:GCC:01:PMON', 'raw'): 'raw', ('HX2:DVD:GPI:01:PMON', 'raw'): 'raw'}
    detnames(args, check_against)

    args = MyCustomArgs(dsname, "-s")
    check_against = {('motor1', 'raw'): 'raw', ('motor2', 'raw'): 'raw'}
    detnames(args, check_against)


if __name__ == '__main__':
    det()
    calib()
