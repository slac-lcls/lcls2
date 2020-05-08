####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`SegGeometryStore` is a factory class/method
============================================================

Switches between different device-dependent segments/sensors
to access their pixel geometry using :py:class:`SegGeometryBase` interface.

Usage::

    from psana.pscalib.geometry.SegGeometryStore import sgs

    sg = sgs.Create('SENS2X1:V1')
    sg2= sgs.Create('EPIX100:V1')
    sg3= sgs.Create('PNCCD:V1')
    sg4= sgs.Create('ANDOR3D:V1')
    sg5= sgs.Create('JUNGFRAU:V1')
    sg6= sgs.Create('EPIX10KA:V1')

    sg.print_seg_info(pbits=0o377)
    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = sg.pixel_scale_size()
    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask(mbits=0377)    
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
 * :py:class:`SegGeometryBase`
 * :py:class:`SegGeometryCspad2x1V1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryEpix10kaV1`,
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryJungfrauV1`
 * :py:class:`SegGeometryStore`
 * :py:class:`GeometryAccess`
 * :py:class:`GeometryObject`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-01
"""
#------------------------------

import logging
logger = logging.getLogger('SegGeometryStore')

from psana.pscalib.geometry.SegGeometryCspad2x1V1 import cspad2x1_one
from psana.pscalib.geometry.SegGeometryEpix100V1  import epix2x2_one
from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one
from psana.pscalib.geometry.SegGeometryMatrixV1   import SegGeometryMatrixV1, segment_one, matrix_pars
from psana.pscalib.geometry.SegGeometryJungfrauV1 import jungfrau_one

#------------------------------

class SegGeometryStore() :
    """Factory class for SegGeometryBase objects of different detectors"""

    def __init__(self) :
        pass

    def Create(self, segname='SENS2X1:V1') :
        """ Factory method returns device dependent SINGLETON object with interface implementation  
        """        
        if segname=='SENS2X1:V1' : return cspad2x1_one # SegGeometryCspad2x1V1(use_wide_pix_center=False)
        if segname=='EPIX100:V1' : return epix2x2_one  # SegGeometryEpix100V1(use_wide_pix_center=False)
        if segname=='EPIX10KA:V1': return epix10ka_one # SegGeometryEpix10kaV1(use_wide_pix_center=False)
        if segname=='PNCCD:V1'   : return segment_one  # SegGeometryMatrixV1()
        if segname[:4]=='MTRX'   :
            rows, cols, psize_row, psize_col = matrix_pars(segname)
            return SegGeometryMatrixV1(rows, cols, psize_row, psize_col,\
                                       pix_size_depth=100,\
                                       pix_scale_size=min(psize_row, psize_col))
        if segname=='JUNGFRAU:V1': return jungfrau_one  # SegGeometryJungfrauV1()
        #if segname=='ANDOR3D:V1' : return seg_andor3d  # SegGeometryMatrixV1()

        #logger.warning('Segment geometry is not implemented for segname=%s. '
        #               'Check segment name in the geometry file.' % str(segname))

        # This is a part of algorithm, return None if requested object is not inplemented.
        return None

#------------------------------

sgs = SegGeometryStore()

#------------------------------
#----------- TEST -------------
#------------------------------

def usage() :
    print('For test(s) use command: python', sys.argv[0], '<test-number=1-7>')

def test_seggeom() :
    from time import time
    t0_sec = time()

    if len(sys.argv)==1   : usage()

    elif(sys.argv[1]=='1') :
        sg = sgs.Create('SENS2X1:V1')
        sg.print_seg_info(pbits=0o377)
        
    elif(sys.argv[1]=='2') :
        sg = sgs.Create('EPIX100:V1')
        sg.print_seg_info(pbits=0o377)

    elif(sys.argv[1]=='3') :
        sg = sgs.Create('PNCCD:V1')
        sg.print_seg_info(pbits=0o377)

    elif(sys.argv[1]=='4') :
        sg = sgs.Create('MTRX:512:512:54:54')
        print('Consumed time for MTRX:512:512:54:54 (sec) =', time()-t0_sec)
        sg.print_seg_info(pbits=0o377)
  
    elif(sys.argv[1]=='5') :
        sg = sgs.Create('JUNGFRAU:V1')
        sg.print_seg_info(pbits=0o377)

    elif(sys.argv[1]=='6') :
        sg = sgs.Create('EPIX10KA:V1')
        sg.print_seg_info(pbits=0o377)

    elif(sys.argv[1]=='7') :
        logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.DEBUG) #filename='example.log', filemode='w'
        sg = sgs.Create('ABRACADABRA:V1')

    else : 
        print('Non-expected arguments: sys.argv=', sys.argv)
        usage()

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    test_seggeom()
    sys.exit('End of test.')

#------------------------------
