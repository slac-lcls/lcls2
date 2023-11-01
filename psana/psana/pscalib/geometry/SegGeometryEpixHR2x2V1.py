#!/usr/bin/env python
"""
Class :py:class:`SegGeometryEpixHR2x2V1` describes the EpixHR2x2V1 sensor geometry
===================================================================================

In this class we use natural matrix notations like in data array
\n We assume that
\n * sensor consists of 2x2 ASICs has 288 rows and 384 columns,
\n * Epix10ka has a pixel size 100x100um, wide pixel size 100x225um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::

   (Xmin,Ymax)      ^ Y          (Xmax,Ymax)
   (0,0)            |            (0,383)
      ------------------------------
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
    --|-------------+--------------|----> X
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      |             |              |
      ------------------------------
   (287,0)          |           (287,383)
   (Xmin,Ymin)                  (Xmax,Ymin)


Usage::

    from SegGeometryEpixHR2x2V1 import epix10ka_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask = sg.pixel_mask_array(width=5, wcenter=5)
    mask = sg.pixel_mask_array(width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=1, center_cols=1)

    sizeX = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()

    X     = sg.pixel_coord_array('X')
    X,Y,Z = sg.pixel_coord_array()
    logger.info('X.shape =' + str(X.shape))

    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_max()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')

    # global method for rotation of numpy arrays:
    Xrot, Yrot = rotation(X, Y, C, S)
    ...

See:
 * :py:class:`GeometryObject`
 * :py:class:`SegGeometry`
 * :py:class:`SegGeometryCspad2x1V1`
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryEpix10kaV1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-09-24 by Mikhail Dubrovin
"""
from psana.pscalib.geometry.SegGeometryEpix10kaV1 import *
logger = logging.getLogger(__name__)


class SegGeometryEpixHR2x2V1(SegGeometryEpix10kaV1):
    """Self-sufficient class for generation of Epix10ka sensor (2x2 ASICs) pixel coordinate array"""

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryEpixHR2x2V1'
        logger.debug('%s.__init__()'%sp._name)

        sp._rows  = 288     # Number of rows in 2x2
        sp._cols  = 384     # Number of cols in 2x2
        sp._pixs  = 100     # Pixel size in um (micrometer)
        sp._pixw  = 225     # Wide pixel size in um (micrometer)
        sp._pixd  = 400.00  # Pixel depth in um (micrometer)

        sp._colsh = sp._cols//2
        sp._rowsh = sp._rows//2
        sp._pixsh = sp._pixs/2
        sp._pixwh = sp._pixw/2

        sp._arows = sp._rowsh
        sp._acols = sp._colsh

        sp._nasics_in_rows = 2 # Number of ASICs in row direction
        sp._nasics_in_cols = 2 # Number of ASICs in column direction

        sp._asic0indices = ((0, 0), (0, sp._colsh), (sp._rowsh, 0), (sp._rowsh, sp._colsh))

        SegGeometryEpix10kaV1.__init__(sp, **kwa)


epixhr2x2_one = SegGeometryEpixHR2x2V1(use_wide_pix_center=False)
epixhr2x2_wpc = SegGeometryEpixHR2x2V1(use_wide_pix_center=True)

# EOF
