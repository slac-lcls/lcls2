#!/usr/bin/env python
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
    sg = sgs.Create(segname='EPIXHR2X2:V1')
    sg = sgs.Create(segname='PNCCD:V1')
    sg = sgs.Create(segname='JUNGFRAU:V1')
    sg = sgs.Create(segname='JUNGFRAU:V2')
    sg = sgs.Create(segname='MTRX:512:512:54:54')
    sg = sgs.Create(segname='MTRX:V2:512:512:54:54')

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
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryJungfrauV1`,
 * :py:class:`SegGeometryMatrixV1`,
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""

import logging
logger = logging.getLogger(__name__)

def segment_geometry(**kwa):
    """Factory method returns segment geomentry object for specified segname."""
    segname = kwa.get('segname', 'SENS2X1:V1')
    wpc     = kwa.get('use_wide_pix_center', False)
    logger.debug('segment geometry of %s is requested, use_wide_pix_center=%s' % (segname, str(wpc)))

    if segname=='EPIX10KA:V1':
        from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one, epix10ka_wpc
        return epix10ka_wpc if wpc else epix10ka_one
    elif segname=='EPIXHR2X2:V1':
        from psana.pscalib.geometry.SegGeometryEpixHR2x2V1 import epixhr2x2_one, epixhr2x2_wpc
        return epixhr2x2_wpc if wpc else epixhr2x2_one
    elif segname[:7]=='MTRX:V2':
        from psana.pscalib.geometry.SegGeometryMatrixV2 import SegGeometryMatrixV2, matrix_pars_v2
        rows, cols, psize_row, psize_col = matrix_pars_v2(segname)
        return SegGeometryMatrixV2(rows, cols, psize_row, psize_col,\
                                   pix_size_depth=100,\
                                   pix_scale_size=min(psize_row, psize_col))
    elif segname[:4]=='MTRX':
        from psana.pscalib.geometry.SegGeometryMatrixV1 import SegGeometryMatrixV1, matrix_pars, segment_one
        rows, cols, psize_row, psize_col = matrix_pars(segname)
        return SegGeometryMatrixV1(rows, cols, psize_row, psize_col,\
                                   pix_size_depth=100,\
                                   pix_scale_size=min(psize_row, psize_col))
    elif segname=='JUNGFRAU:V1':
        from psana.pscalib.geometry.SegGeometryJungfrauV1 import jungfrau_one
        return jungfrau_one
    elif segname=='JUNGFRAU:V2':
        from psana.pscalib.geometry.SegGeometryJungfrauV2 import jungfrau_front
        return jungfrau_front
    elif segname=='PNCCD:V1':
        from psana.pscalib.geometry.SegGeometryMatrixV1 import segment_one
        return segment_one
    elif segname=='EPIX100:V1':
        from psana.pscalib.geometry.SegGeometryEpix100V1 import epix2x2_one, epix2x2_wpc
        return epix2x2_wpc if wpc else epix2x2_one
    elif segname=='SENS2X1:V1':
        from psana.pscalib.geometry.SegGeometryCspad2x1V1 import cspad2x1_one, cspad2x1_wpc
        return cspad2x1_wpc if wpc else cspad2x1_one
    #elif segname=='ANDOR3D:V1': return seg_andor3d # SegGeometryMatrixV1()
    else:
        logger.debug('segment "%s" gometry IS NOT IMPLEMENTED' % segname)
        return None


class SegGeometryStore():
    def __init__(sp):
        pass

    def Create(sp, **kwa):
        return segment_geometry(**kwa)


sgs = SegGeometryStore() # singleton

# EOF - See test_SegGeometryStore.py
