#!/usr/bin/env python
"""
2021-09-27 test separated from SegGeometryStore.py
           due to wired behavior (doub le import) of the __main__ in psana env
"""

import logging
logger = logging.getLogger(__name__)

import sys
from time import time
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

from psana2.pscalib.geometry.SegGeometryStore import sgs # SegGeometryStore
from psana2.pscalib.geometry.test_SegGeometryArchonV1 import detector_simulator, np

def test_mask(sg, width=6, wcenter=4):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
import psana2.pyalgos.generic.Graphics as gg
    iX, iY = sg.get_seg_xy_maps_pix_with_offset()
    #mask[mask==0]=3
    mask=sg.pixel_mask_array()
    mask_arr = None
    if sg._name == 'SegGeometryArchonV1':
        mask_arr = mask
    else:
        iX, iY = (iX+0.25).astype(int), (iY+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX, iY, W=mask+1, mask_arr=mask_arr)
    sh = sg.shape()
    h_in = 9.
    w_in = h_in*sh[0]/sh[1]*1.2
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=(w_in, h_in), window=(0.15, 0.04, 0.76, 0.94))
    gg.show()

def test_segname(segname, **kwa):
    t0_sec = time()
    #sgs1 = SegGeometryStore()
    sg = sgs.Create(segname=segname, **kwa)
    if segname=='MTRXANY:V1':
        sg.init_matrix_parameters(shape=(512,1024), pix_size_rcsd_um=(75,75,75,400))
    dt_sec = time()-t0_sec
    sg.print_seg_info(pbits=0o377)
    logger.info('Consumed time to create = %.6f sec' % dt_sec)
    test_mask(sg)

def usage():
    import inspect
    return '\n Usage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname==" in s])

def selector():
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif(tname=='1'): sg = test_segname('SENS2X1:V1')
    elif(tname=='2'): sg = test_segname('EPIX100:V1')
    elif(tname=='3'): sg = test_segname('PNCCD:V1')
    elif(tname=='4'): sg = test_segname('EPIX10KA:V1')
    elif(tname=='5'): sg = test_segname('JUNGFRAU:V1')
    elif(tname=='6'): sg = test_segname('JUNGFRAU:V2')
    elif(tname=='7'): sg = test_segname('MTRX:512:512:54:54')
    elif(tname=='8'): sg = test_segname('MTRX:V2:512:512:54:54')
    elif(tname=='9'): sg = test_segname('EPIXHR2X2:V1')
    elif(tname=='10'):sg = test_segname('EPIXHR1X4:V1')
    elif(tname=='11'):sg = test_segname('ARCHON:V1', detector=detector_simulator())
    elif(tname=='12'):sg = test_segname('EPIXMASIC:V1')
    elif(tname=='13'):sg = test_segname('EPIXUHRASIC:V1')
    elif(tname=='14'):sg = test_segname('MTRXANY:V1')
    elif(tname=='99'):sg = sgs.Create(segname='ABRACADABRA:V1');\
        logger.info('Return for non-existent segment name: %s' % sg)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage())
    sys.exit('END OF TEST')

if __name__ == "__main__":
    selector()

# EOF
