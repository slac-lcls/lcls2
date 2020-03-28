"""
Comparison of original xtcav calibration constants from file
with the same constants existing in CDB, created by the command like 
  > kinit
  > cdb convert -e xcsm9816 -u dubrovin

  > l /reg/d/psdm/xcs/*/calib/*/*/lasingoffreference # xcsm9816
  > l /reg/d/psdm/xpp/*/calib/*/*/lasingoffreference # xpph6015, xpptut15, xppx22715, xppz0216
  > l /reg/d/psdm/amo/*/calib/*/*/lasingoffreference # amoz0116, amox23616, amon0816, amolr9316, amolr2516, ...
"""
import sys

from psana.pscalib.calib.XtcavUtils import Load, dict_from_xtcav_calib_object
from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
from psana.pscalib.calib.MDBConvertUtils import compare_dicts

#------------------------------

def usage() :
    msg = 'Usage: python lcls2/psana/psana/pscalib/examples/ex08-cdb-xtcav.py [<test-number>]'
    print(msg)

#------------------------------

def compare_for(tname) :
    fname0 = '/reg/d/psdm/XCS/xcsm9816/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/500-end.data'
    fname1 = '/reg/d/psdm/XCS/xcsm9816/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/499-end.data'
    fname2 = '/reg/d/psdm/XPP/xpptut15/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/300-302.data'
    fname3 = '/reg/d/psdm/XPP/xpptut15/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/101-102.data'
    return (fname0, 'xcsm9816', 'opal1000_0059', 500, 'lasingoffreference', None) if tname=='0' else\
           (fname1, 'xcsm9816', 'opal1000_0059', 500, 'pedestals',          None) if tname=='1' else\
           (fname2, 'xpptut15', 'opal1000_0059', 302, 'lasingoffreference', None) if tname=='2' else\
           (fname3, 'xpptut15', 'opal1000_0059', 101, 'pedestals',          None)

#------------------------------

def test_xtcav_calib_constants(tname) :
    fname, exp, det, run, ctype, cvers = compare_for(tname)
    print('LCLS1 Xtcav calibration file: %s' % fname)
    print('Parameters form path: exp:%s det:%s ctype:%s run:%s'%\
          (exp, det, ctype, run))

    o1 = Load(fname)
    d1 = dict_from_xtcav_calib_object(o1)
    print('Xtcav calibration constants from file:\n', d1)

    r = calib_constants_all_types(det, exp, run) #, time_sec=None, vers=None, url=cc.URL)
    d2 = r[ctype][0]
    print('Xtcav calibration constants from cdb:\n', d2)

    print('\nCompare dictionaries for Xtcav calib objects loaded directly from calib file and passed through the CDB')
    compare_dicts(d1,d2)

#------------------------------

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    test_xtcav_calib_constants(tname)
    usage()
    sys.exit('END OF TEST %s'%tname)

#------------------------------
