#!/usr/bin/env python
""" Test access to calibration parameters for xtcav.
"""
import sys
print('e.g.: [python] %s' % sys.argv[0])

import logging
logger = logging.getLogger(__name__)
#fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
fmt='[%(levelname).1s] L%(lineno)04d : %(message)s'
logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

#----------

def test_01() :

    from psana import DataSource
    import psana.xtcav.Utils as xtu

    ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0131-e000200-xtcav-v2.xtc2')
    run = next(ds.runs()) 

    xtcav_camera = run.Detector('xtcav')
    dark_data, dark_meta = xtcav_camera.calibconst.get('pedestals')

    dark_background = xtu.xtcav_calib_object_from_dict(dark_data)

    #print('==== dark_data:\n%s'% str(dark_data))
    #print('==== ROI:\n%s' % str(dark_data['ROI']))

    print('==== dir(dark_background):\n', dir(dark_background))
    print('==== ROI: %s' % str(dark_background.ROI))
    print('==== image:\n%s' % str(dark_background.image))
    
    print('==== dark_meta:\n%s' % str(dark_meta))

#----------

def test_02() :
    from psana.pscalib.calib.MDBWebUtils import calib_constants
    data, doc = calib_constants('xtcav', exp='amox23616', ctype='pedestals', run=131) #, time_sec=t0_sec, vers=None)

    print('data:', data)
    print('metadata:')
    for k,v in doc.items() : print('%16s : %s' % (k,v))


#----------
#----------
#----------

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    if   tname == '1' : test_01()
    elif tname == '2' : test_02()
    else : print('Test %s IS NOT IMPLEMENTED' % tname)
    sys.exit('END OF TEST %s' % sys.argv[0])

#----------
