
"""
run this test by the commnad:
  pytest test_jungfrau05M_calib.py # prints do not workfor pytest
  python test_jungfrau05M_calib.py # to print all messages and logger.<method>
test executed by:
  pytest psana/psana/tests in lcls2/run_travis.sh
"""

from psana import DataSource
import os
import numpy as np
import pytest
from psana.detector.NDArrUtils import info_ndarr

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO)

#@pytest.mark.skip(reason="to be debugged when mikhail returns from vacation in June 2025")
def test_jungfrau05M_calib():
    correctanswer=np.array(([0.142126, -0.518305, 0.285803, 0.088172, 0.141285],\
                            [-0.186004, -0.307364, -0.370457, -0.146206, 0.492853]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, 'test_data/detector/test_jungfrau05M_calib.xtc2')
    #ds = DataSource(files=fname)
    #ds_kwa = {'files': '/sdf/home/d/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/tests/test_data/detector/test_jungfrau05M_calib.xtc2',}
    ds_kwa = {'files': fname,}
    ds = DataSource(**ds_kwa)
    myrun = next(ds.runs())

#   mkwa = {'status':False, 'status_bits':0xffff, 'stextra_bits':(1<<64)-1, 'gain_range_inds':(0,),\
#       'neighbors':False, 'rad':3, 'ptrn':'r',\
#       'edges':False, 'width':0, 'edge_rows':10, 'edge_cols':5,\
#       'center':False, 'wcenter':0, 'center_rows':5, 'center_cols':3,\
#       'calib':False,\
#       'umask':None,\
#       'force_update':True} #, 'dtype':DTYPE_MASK}

    mkwa = {'status': True, 'gain_range_inds': (0,),
            'logmet_init': logger.debug} # info

    odet = myrun.Detector('jungfrau', **mkwa)
    cc = odet.raw._calibconstants()
    #print(cc.info_calibconst())

    mask = odet.raw._mask() # mask.shape = (1, 512, 1024)
    logger.debug(info_ndarr(mask, 'odet.raw._mask():', last=10, vfmt='%0.6f', spa=', '))
    grinds=(0,)
    mask_stat = odet.raw._mask_from_status(gain_range_inds=grinds)
    logger.debug(info_ndarr(mask_stat, 'odet.raw._mask_from_status(gain_range_inds=%s):'%str(grinds), last=10, vfmt='%0.6f', spa=', '))

    for nevt,evt in enumerate(myrun.events()):
        if nevt>2: break
        image=odet.raw.image(evt)
        calibsample=image[0][0:5] # [row][cols]
        print(info_ndarr(image, 'evt: %d == calibsample:'%nevt, last=10, vfmt='%0.6f', spa=', '))
        print('       == correctanswer:%s' % (42*' '), correctanswer[nevt][0:5])
        assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_jungfrau05M_calib()

#EOF
