#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`SegGeometryStore` is a factory class/method
============================================================

Switches between different device-dependent segments/sensors
to access their pixel geometry using :py:class:`SegGeometry` interface.

Usage::

    from psana.pscalib.geometry.SegGeometryStore import sgs

    sg = sgs.Create(segname='SENS2X1:V1')
    sg = sgs.Create(segname='EPIX100:V1')
    sg = sgs.Create(segname='EPIX10KA:V1')
    sg = sgs.Create(segname='PNCCD:V1')
    sg = sgs.Create(segname='JUNGFRAU:V1')
    sg = sgs.Create(segname='JUNGFRAU:V2')
    sg = sgs.Create(segname='MTRX:512:512:54:54')

    sg.print_seg_info(pbits=0o377)
    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = sg.pixel_scale_size()
    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask(mbits=0o377)
    sizeX    = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()
    X        = sg.pixel_coord_array('X')
    X,Y,Z    = sg.pixel_coord_array()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')
    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_max()
    ...

See:
 * :py:class:`GeometryObject`, 
 * :py:class:`SegGeometry`, 
 * :py:class:`SegGeometryCspad2x1V1`, 
 * :py:class:`SegGeometryEpix100V1`, 
 * :py:class:`SegGeometryEpix10kaV1`, 
 * :py:class:`SegGeometryJungfrauV1`, 
 * :py:class:`SegGeometryMatrixV1`, 
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from psana.pscalib.geometry.SegGeometryCspad2x1V1 import cspad2x1_one, cspad2x1_wpc
from psana.pscalib.geometry.SegGeometryEpix100V1  import epix2x2_one, epix2x2_wpc
from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one, epix10ka_wpc
from psana.pscalib.geometry.SegGeometryMatrixV1   import SegGeometryMatrixV1, segment_one, matrix_pars
from psana.pscalib.geometry.SegGeometryJungfrauV1 import jungfrau_one
from psana.pscalib.geometry.SegGeometryJungfrauV2 import jungfrau_front

#------------------------------

class SegGeometryStore():
    """Factory class for SegGeometry-base objects of different detectors"""

#------------------------------

    def __init__(sp):
        pass

#------------------------------

    def Create(sp, **kwa):
        """ Factory method returns device dependent SINGLETON object with interface implementation  
        """
        segname = kwa.get('segname', 'SENS2X1:V1')
        wpc     = kwa.get('use_wide_pix_center', False)

        if segname=='SENS2X1:V1' : return cspad2x1_wpc if wpc else cspad2x1_one # SegGeometryCspad2x1V1(use_wide_pix_center=False)
        if segname=='EPIX100:V1' : return epix2x2_wpc  if wpc else epix2x2_one  # SegGeometryEpix100V1 (use_wide_pix_center=False)
        if segname=='EPIX10KA:V1': return epix10ka_wpc if wpc else epix10ka_one # SegGeometryEpix10kaV1(use_wide_pix_center=False)
        if segname=='PNCCD:V1'   : return segment_one  # SegGeometryMatrixV1()
        if segname[:4]=='MTRX'   :
            rows, cols, psize_row, psize_col = matrix_pars(segname)
            return SegGeometryMatrixV1(rows, cols, psize_row, psize_col,\
                                       pix_size_depth=100,\
                                       pix_scale_size=min(psize_row, psize_col))
        if segname=='JUNGFRAU:V1': return jungfrau_one    # SegGeometryJungfrauV1()
        if segname=='JUNGFRAU:V2': return jungfrau_front  # SegGeometryJungfrauV2()
        #if segname=='ANDOR3D:V1': return seg_andor3d     # SegGeometryMatrixV1()
        return None

#------------------------------

sgs = SegGeometryStore()

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__":

  import sys
  from time import time
  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

#----------

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
    if tname in ('0','8'): s+='\n 8 - ABRACADABRA:V1'
    return s

#----------

  def test_segname(segname):
    t0_sec = time()
    sg = sgs.Create(segname=segname)
    dt_sec = time()-t0_sec
    sg.print_seg_info(pbits=0o377)
    logger.info('Consumed time to create = %.6f sec' % dt_sec)

#----------

if __name__ == "__main__":

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif(tname=='1'): sg = test_segname('SENS2X1:V1')
    elif(tname=='2'): sg = test_segname('EPIX100:V1')
    elif(tname=='3'): sg = test_segname('PNCCD:V1')
    elif(tname=='4'): sg = test_segname('EPIX10KA:V1')
    elif(tname=='5'): sg = test_segname('JUNGFRAU:V1')
    elif(tname=='6'): sg = test_segname('JUNGFRAU:V2')
    elif(tname=='7'): sg = test_segname('MTRX:512:512:54:54')
    elif(tname=='8'):
        sg = sgs.Create(segname='ABRACADABRA:V1')
        logger.info('Return for non-existent segment name: %s' % sg)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

#------------------------------
