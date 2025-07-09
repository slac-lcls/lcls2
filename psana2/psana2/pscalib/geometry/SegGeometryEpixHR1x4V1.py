#!/usr/bin/env python
"""
Class :py:class:`SegGeometryEpixHR1x4V1` describes the EpixHR1x4V1 sensor geometry
===================================================================================

In this class we use natural matrix notations like in data array
\n We assume that
\n * sensor consists of 1x4 ASICs has 144 rows and 768 columns,
\n * Epix10ka has a pixel size 100x100um, wide pixel size 100x225um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::

   (Xmin,Ymax)                     ^ Y                        (Xmax,Ymax)
   (0,0)                           |                          (0,767)
      -----------------------------------------------------------
      |             |              |             |              |
      |             |              |             |              |
      |             |              |             |              |
    --|-------------+--------------|-------------+--------------|----> X
      |             |              |             |              |
      |             |              |             |              |
      |             |              |             |              |
      -----------------------------------------------------------
   (143,0)                         |                          (143,767)
   (Xmin,Ymin)                                                (Xmax,Ymin)


Usage::

    from SegGeometryEpixHR1x4V1 import epix10ka_one as sg

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
 * :py:class:`SegGeometryEpixHR1x4V1`
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryEpix10kaV1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the LCLS-II project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2023-04-26 by Mikhail Dubrovin
"""
from psana2.pscalib.geometry.SegGeometryEpix10kaV1 import *
logger = logging.getLogger(__name__)


class SegGeometryEpixHR1x4V1(SegGeometryEpix10kaV1):
    """Self-sufficient class for generation of Epix10ka sensor (1x4 ASICs) pixel coordinate array"""

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryEpixHR1x4V1'
        logger.debug('%s.__init__()'%sp._name)

        sp._rows  = 144     # Number of rows in 1x4
        sp._cols  = 768     # Number of cols in 1x4
        sp._pixs  = 100     # Pixel size in um (micrometer)
        sp._pixw  = 225     # Wide pixel size in um (micrometer)
        sp._pixd  = 400.00  # Pixel depth in um (micrometer)

        sp._colsh = sp._cols//2
        sp._rowsh = sp._rows//2
        sp._pixsh = sp._pixs/2
        sp._pixwh = sp._pixw/2
        sp._colsq = sp._cols//4 # 192

        sp._arows = sp._rowsh
        sp._acols = sp._colsh

        sp._nasics_in_rows = 1 # Number of ASICs in row direction
        sp._nasics_in_cols = 4 # Number of ASICs in column direction

        sp._asic0indices = ((0, 0), (0, sp._colsq*1), (0, sp._colsq*2), (0, sp._colsq*3))

        SegGeometryEpix10kaV1.__init__(sp, **kwa)


    def make_pixel_coord_arrs(sp):
        """Makes [144,768] maps of x, y, and z 1x4 pixel coordinates
        with origin in the center of 1x4
        """
        #x_rhs = np.arange(sp._colsh)*sp._pixs + sp._pixw - sp._pixsh
        #if sp.use_wide_pix_center: x_rhs[0] = sp._pixwh # set x-coordinate of the wide pixel in its geometry center
        #sp.x_arr_um = np.hstack([-x_rhs[::-1], x_rhs])

        x_asic = np.arange(sp._colsq)*sp._pixs - sp._pixsh  # offset provide 0-s pixel has x=0
        x_asic_r1 = x_asic + sp._pixw
        x_asic_r2 = x_asic + sp._pixw*3  + (sp._colsq-1)*sp._pixs
        x_rhs =  np.hstack([x_asic_r1, x_asic_r2])
        sp.x_arr_um = np.hstack([-x_rhs[::-1], x_rhs])

        y_rhs = np.arange(sp._rowsh)*sp._pixs - sp._pixsh #+ sp._pixw
        #if sp.use_wide_pix_center: y_rhs[0] = sp._pixwh # set y-coordinate of the wide pixel in its geometry center
        sp.y_arr_um = np.hstack([y_rhs[::-1], -y_rhs]) # reverse sign (+y is opposite to y index)

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))


    def make_pixel_size_arrs(sp):
        """Makes [144,768] maps of x, y, and z 1x4 pixel size
        """
        if sp.pix_area_arr is not None: return

#        x_rhs_size_um = np.ones(sp._colsh)*sp._pixs
#        x_rhs_size_um[0] = sp._pixw
#        x_arr_size_um = np.hstack([x_rhs_size_um[::-1],x_rhs_size_um])

        x_arr_size_um = np.ones(sp._cols)*sp._pixs

#        y_rhs_size_um = np.ones(sp._rowsh)*sp._pixs
#        y_rhs_size_um[0] = sp._pixw
#        y_arr_size_um = np.hstack([y_rhs_size_um[::-1],y_rhs_size_um])

        y_arr_size_um = np.ones(sp._rows)*sp._pixs

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp._pixd

        factor = 1./(sp._pixs*sp._pixs)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor


    def pixel_mask_array(sp, width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa):

        mask = SegGeometryEpix10kaV1.pixel_mask_array(sp, width=width, wcenter=0,\
                                  edge_rows=edge_rows, edge_cols=edge_cols, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa)

        if wcenter>0: center_cols = wcenter

        if center_cols>0: # mask 3 central columns
            w = center_cols
            zero_col = np.zeros((sp._rows,w), dtype=dtype)
            for igap in (1,2,3):
                g = sp._colsq*igap
                mask[:,g-w:g] = zero_col # mask central-left  columns
                mask[:,g:g+w] = zero_col # mask central-right columns

        return mask

epixhr1x4_one = SegGeometryEpixHR1x4V1(use_wide_pix_center=False)
epixhr1x4_wpc = SegGeometryEpixHR1x4V1(use_wide_pix_center=True)

# EOF
