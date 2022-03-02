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

from psana.pscalib.geometry.SegGeometryStore import sgs # SegGeometryStore

def test_mask(sg, width=6, wcentral=4):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import psana.pyalgos.generic.Graphics as gg
    X, Y = sg.get_seg_xy_maps_pix_with_offset()
    mask = 1 + sg.pixel_mask_array(width=width, wcentral=wcentral)
    #mask[mask==0]=3
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    sh = sg.shape()
    h_in = 9.
    w_in = h_in*sh[0]/sh[1]*1.2
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=(w_in, h_in))
    gg.show()

def test_segname(segname):
    t0_sec = time()
    #sgs1 = SegGeometryStore()
    sg = sgs.Create(segname=segname)
    dt_sec = time()-t0_sec
    sg.print_seg_info(pbits=0o377)
    logger.info('Consumed time to create = %.6f sec' % dt_sec)
    test_mask(sg)

def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - SENS2X1:V1'
    if tname in ('0','2'): s+='\n 2 - EPIX100:V1'
    if tname in ('0','3'): s+='\n 3 - PNCCD:V1'
    if tname in ('0','4'): s+='\n 4 - EPIX10KA:V1'
    if tname in ('0','5'): s+='\n 5 - JUNGFRAU:V1'
    if tname in ('0','6'): s+='\n 6 - JUNGFRAU:V2'
    if tname in ('0','7'): s+='\n 7 - MTRX:512:512:54:54'
    if tname in ('0','8'): s+='\n 8 - MTRX:V2:512:512:54:54'
    if tname in ('0','9'): s+='\n 9 - EPIXHR2X2:V1'
    if tname in ('0','10'):s+='\n10 - ABRACADABRA:V1'
    return s

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
elif(tname=='10'):
    sg = sgs.Create(segname='ABRACADABRA:V1')
    logger.info('Return for non-existent segment name: %s' % sg)
else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
if len(sys.argv)>1: logger.info(usage(tname))
sys.exit('END OF TEST')

# EOF
