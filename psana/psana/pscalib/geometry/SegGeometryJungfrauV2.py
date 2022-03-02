#!/usr/bin/env python
"""
Class :py:class:`SegGeometryJungfrauV2` describes the Jungfrau V2 sensor geometry
=================================================================================

Data array for Jungfrau 512x1024 segment is shaped as (1,512,1024),
has a matrix-like numeration for rows and columns with gaps between 2x4 ASICs
\n We assume that
\n * 1x1 ASICs has 256 rows and 256 columns,
\n * Jungfrau has a pixel size 75x75um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the bottom left corner, has coordinates (xmin,ymin), as shown below
\n ::

   (Xmin,Ymax)                          ^ Y                          (Xmax,Ymax)
   (511,0)                              |                             (1023,1023)
     ----------------- -----------------|----------------- -----------------
     |               | |               |||               | |               |
     |     ASIC      | |               |||               | |               |
     |    256x256    | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     ----------------- -----------------|----------------- -----------------
   -------------------------------------+-------------------------------------> X
     ----------------- -----------------|----------------- -----------------
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     ----------------- -----------------|----------------- -----------------
   (0,0)                                |                               (0,1023)
   (Xmin,Ymin)                                                        (Xmax,Ymin)


Usage::

    from psana.pscalib.geometry.SegGeometryJungfrauV2 import jungfrau_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask_array(mbits=0o377, width=1)
    # where mbits = +1-edges, +2-wide pixels

    sizeX = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()

    X     = sg.pixel_coord_array('X')
    X,Y,Z = sg.pixel_coord_array()
    print 'X.shape =', X.shape

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
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryJungfrauV1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2017-10-12 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""

from psana.pscalib.geometry.SegGeometryJungfrauV1 import * # SegGeometryJungfrauV1, logging, np
logger = logging.getLogger(__name__)


class SegGeometryJungfrauV2(SegGeometryJungfrauV1):
    """Self-sufficient class for generation of Jungfrau 2x4 sensor pixel coordinate array"""

    _name = 'SegGeometryJungfrauV2'


    def __init__(sp, **kwa):
        logger.debug('SegGeometryJungfrauV2.__init__()')
        SegGeometryJungfrauV1.__init__(sp)


    def make_pixel_coord_arrs(sp):
        """Makes [512,1024] maps of x, y, and z pixel coordinates
        with origin in the center of 2x4
        """
        x_asic = np.arange(sp._casic)*sp._pixs
        x0 = np.array((-512-2.5, -256.5, 1.5, 256+3.5))*sp._pixs
        sp.x_arr_um = np.hstack([x_asic+x0[0], x_asic+x0[1], x_asic+x0[2], x_asic+x0[3]])

        y_asic = np.arange(sp._rasic)*sp._pixs
        y0 = np.array((-256.5, 1.5))*sp._pixs
        sp.y_arr_um = np.hstack([y0[0]+y_asic, y0[1]+y_asic])

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows, sp._cols))


    def get_xyz_min_um(sp):
        return sp.x_arr_um[0], sp.y_arr_um[0], 0

    def get_xyz_max_um(sp):
        return sp.x_arr_um[-1], sp.y_arr_um[-1], 0

jungfrau_front = SegGeometryJungfrauV2()

# EOF

