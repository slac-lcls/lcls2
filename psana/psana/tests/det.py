import sys
from psana import DataSource
import numpy as np

def det():
    ds = DataSource('data.xtc')
    for run in ds.runs(): # Detector is created based on per-run config. 
        det = ds.Detector('xppcspad')

'''
    for evt in ds.events():
        raw = det.raw(evt.__next__())
        break

    print('Raw values and shape:' )
    print(raw, raw.shape)
    assert(np.sum(raw)==9*17)
    assert(raw.shape==(2,3,3))
    assert(ds._configs[0].software.xppcspad.dettype == 'cspad')
    assert(ds._configs[0].software.xppcspad.detid == 'detnum1234')
'''

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


if __name__ == '__main__':
    det()
    calib()
