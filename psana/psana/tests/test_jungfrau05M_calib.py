from psana import DataSource
import os
import numpy as np
import pytest
from psana.detector.NDArrUtils import info_ndarr

#import logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

"""
run this test by the commnad:
  pytest test_jungfrau05M_calib.py
  python test_jungfrau05M_calib.py # to print all messages
test executed by:
  pytest psana/psana/tests in lcls2/run_travis.sh
"""

@pytest.mark.skip(reason="to be debugged when mikhail returns from vacation in June 2025")
def test_jungfrau05M_calib():
    correctanswer=np.array(([4.0649414, 9.653076, 15.843018, -12.125977, -24.302979],\
                            [0.06494141, -6.346924, -27.156982, -3.1259766, -15.3029785]))

    dirscr = os.path.dirname(os.path.realpath(__file__))
    #path = os.path.join(dirscr,'test_data/detector/test_jungfrau05M_calib.xtc2')
    #print('DEBUG path to data:', path)
    #ds = DataSource(files=path)
    kwa = {'files': '/sdf/home/d/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/tests/test_data/detector/test_jungfrau05M_calib.xtc2',\
           'gain_range_inds': (0,)}
    ds = DataSource(**kwa)
    myrun = next(ds.runs())

#    kwa = {'status':False, 'status_bits':0xffff, 'stextra_bits':(1<<64)-1, 'gain_range_inds':(0,),\
#       'neighbors':False, 'rad':3, 'ptrn':'r',\
#       'edges':False, 'width':0, 'edge_rows':10, 'edge_cols':5,\
#       'center':False, 'wcenter':0, 'center_rows':5, 'center_cols':3,\
#       'calib':False,\
#       'umask':None,\
#       'force_update':True} #, 'dtype':DTYPE_MASK}

    odet = myrun.Detector('jungfrau') #, logmet_init=logger.info)
    cc = odet.raw._calibconstants()  # prints cc.info_calibconst()
    print('cc.info_calibconst:', cc.info_calibconst())

    mask = odet.raw._mask() #; mask.shape = (512, 1024)      [0, 10, 20:30]
    print(info_ndarr(mask, 'XXX _mask:', last=10, vfmt='%0.10f', spa=', '))

    mask1 = odet.raw._mask_from_status()
    print(info_ndarr(mask1, 'XXX _mask_from_status:', last=10, vfmt='%0.10f', spa=', '))


    for nevt,evt in enumerate(myrun.events()):
        if nevt>2: break
        image=odet.raw.image(evt)
        #calibsample=image[200][0:5]
        calibsample=image
        print(info_ndarr(image, 'DEBUG calibsample:', last=10, vfmt='%0.10f', spa=', '))
        #print('DEBUG calibsample  ', image) # calibsample)
        print('DEBUG correctanswer', correctanswer[nevt][0:5])
        #assert np.allclose(correctanswer[nevt][0:5], calibsample, rtol=.001)

if __name__ == "__main__":
    test_jungfrau05M_calib()
