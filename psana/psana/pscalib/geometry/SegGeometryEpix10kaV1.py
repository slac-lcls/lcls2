#!/usr/bin/env python
"""
Class :py:class:`SegGeometryEpix10kaV1` describes the Epix100 V1 sensor geometry
===============================================================================

In this class we use natural matrix notations like in data array
\n We assume that
\n * sensor consists of 2x2 ASICs has 352 rows and 384 columns,
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
   (351,0)          |           (351,383)
   (Xmin,Ymin)                  (Xmax,Ymin)


Usage::

    from SegGeometryEpix10kaV1 import epix10ka_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask_array(width=5, wcentral=5)

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
 * :py:class:`SegGeometryEpix10kaV1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-11-14 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""

from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)

DTYPE_MASK = np.uint8


class SegGeometryEpix10kaV1(SegGeometry):
    """Self-sufficient class for generation of Epix10ka sensor (2x2 ASICs) pixel coordinate array"""

    _name = 'SegGeometryEpix10kaV1'

    _rows  = 352     # Number of rows in 2x2
    _cols  = 384     # Number of cols in 2x2
    _pixs  = 100     # Pixel size in um (micrometer)
    _pixw  = 225     # Wide pixel size in um (micrometer)
    _pixd  = 400.00  # Pixel depth in um (micrometer)

    _colsh = _cols//2
    _rowsh = _rows//2
    _pixsh = _pixs/2
    _pixwh = _pixw/2

    _arows = _rowsh
    _acols = _colsh

    _nasics_in_rows = 2 # Number of ASICs in row direction
    _nasics_in_cols = 2 # Number of ASICs in column direction

    _asic0indices = ((0, 0), (0, _colsh), (_rowsh, 0), (_rowsh, _colsh))


    def __init__(sp, **kwa):
        sp.use_wide_pix_center = kwa.get('use_wide_pix_center', True)
        logger.debug('SegGeometryEpix10kaV1.__init__() use_wide_pix_center: %s' % sp.use_wide_pix_center)

        SegGeometry.__init__(sp)

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()


    def make_pixel_coord_arrs(sp):
        """Makes [352,384] maps of x, y, and z 2x2 pixel coordinates
        with origin in the center of 2x2
        """
        x_rhs = np.arange(sp._colsh)*sp._pixs + sp._pixw - sp._pixsh
        if sp.use_wide_pix_center: x_rhs[0] = sp._pixwh # set x-coordinate of the wide pixel in its geometry center
        sp.x_arr_um = np.hstack([-x_rhs[::-1], x_rhs])

        y_rhs = np.arange(sp._rowsh)*sp._pixs + sp._pixw - sp._pixsh
        if sp.use_wide_pix_center: y_rhs[0] = sp._pixwh # set y-coordinate of the wide pixel in its geometry center
        sp.y_arr_um = np.hstack([y_rhs[::-1], -y_rhs]) # reverse sign (+y is opposite to y index)

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))


    def make_pixel_size_arrs(sp):
        """Makes [352,384] maps of x, y, and z 2x2 pixel size
        """
        if sp.pix_area_arr is not None: return

        x_rhs_size_um = np.ones(sp._colsh)*sp._pixs
        x_rhs_size_um[0] = sp._pixw
        x_arr_size_um = np.hstack([x_rhs_size_um[::-1],x_rhs_size_um])

        y_rhs_size_um = np.ones(sp._rowsh)*sp._pixs
        y_rhs_size_um[0] = sp._pixw
        y_arr_size_um = np.hstack([y_rhs_size_um[::-1],y_rhs_size_um])

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp._pixd

        factor = 1./(sp._pixs*sp._pixs)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor


    def print_member_data(sp):
        s = 'SegGeometryEpix10kaV1.print_member_data()'\
          + '\n    _rows : %d'    % sp._rows\
          + '\n    _cols : %d'    % sp._cols\
          + '\n    _pixs : %7.2f' % sp._pixs\
          + '\n    _pixw : %7.2f' % sp._pixw\
          + '\n    _pixd : %7.2f' % sp._pixd\
          + '\n    _colsh: %d'    % sp._colsh\
          + '\n    _pixsh: %7.2f' % sp._pixsh\
          + '\n    _pixwh: %7.2f' % sp._pixwh
        logger.info(s)


    def print_pixel_size_arrs(sp, rowslice=slice(171,181)):
        sp.make_pixel_size_arrs()
        s = 'SegGeometryEpix10kaV1.print_pixel_size_arrs()'\
          + '\n  rowslice=%s' % str(rowslice)\
          + '\n  sp.x_pix_size_um[rowslice,187:197]:\n'+ str(sp.x_pix_size_um[rowslice,187:197])\
          + '\n  sp.x_pix_size_um.shape = '           + str(sp.x_pix_size_um.shape)\
          + '\n  sp.y_pix_size_um:\n'                 + str(sp.y_pix_size_um)\
          + '\n  sp.y_pix_size_um.shape = '           + str(sp.y_pix_size_um.shape)\
          + '\n  sp.z_pix_size_um:\n'                 + str(sp.z_pix_size_um)\
          + '\n  sp.z_pix_size_um.shape = '           + str(sp.z_pix_size_um.shape)\
          + '\n  sp.pix_area_arr[rowslice,187:197]:\n'+ str(sp.pix_area_arr[rowslice,187:197])\
          + '\n  sp.pix_area_arr.shape  = '           + str(sp.pix_area_arr.shape)
        logger.info(s)


    def print_maps_seg_um(sp):
        s = 'SegGeometryEpix10kaV1.print_maps_seg_um()'\
          + '\n  x_pix_arr_um =\n'      + str(sp.x_pix_arr_um)\
          + '\n  x_pix_arr_um.shape = ' + str(sp.x_pix_arr_um.shape)\
          + '\n  y_pix_arr_um =\n'      + str(sp.y_pix_arr_um)\
          + '\n  y_pix_arr_um.shape = ' + str(sp.y_pix_arr_um.shape)\
          + '\n  z_pix_arr_um =\n'      + str(sp.z_pix_arr_um)\
          + '\n  z_pix_arr_um.shape = ' + str(sp.z_pix_arr_um.shape)
        logger.info(s)


    def print_xy_1darr_um(sp):
        s = 'SegGeometryEpix10kaV1.print_xy_1darr_um()'\
          + '\n  x_arr_um:\n'       + str(sp.x_arr_um)\
          + '\n  x_arr_um.shape = ' + str(sp.x_arr_um.shape)\
          + '\n  y_arr_um:\n'       + str(sp.y_arr_um)\
          + '\n  y_arr_um.shape = ' + str(sp.y_arr_um.shape)
        logger.info(s)


    def print_xyz_min_max_um(sp):
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        s = 'SegGeometryEpix10kaV1.print_xyz_min_max_um()'\
          + '\n  In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f'\
            % (xmin, xmax, ymin, ymax, zmin, zmax)
        logger.info(s)


    def get_xyz_min_um(sp):
        return sp.x_arr_um[0], sp.y_arr_um[-1], 0

    def get_xyz_max_um(sp):
        return sp.x_arr_um[-1], sp.y_arr_um[0], 0

    def get_seg_xy_maps_um(sp):
        return sp.x_pix_arr_um, sp.y_pix_arr_um

    def get_seg_xyz_maps_um(sp):
        return sp.x_pix_arr_um, sp.y_pix_arr_um, sp.z_pix_arr_um

    def get_seg_xy_maps_um_with_offset(sp):
        if  sp.x_pix_arr_um_offset is None:
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset

    def get_seg_xyz_maps_um_with_offset(sp):
        if  sp.x_pix_arr_um_offset is None:
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
            sp.z_pix_arr_um_offset = sp.z_pix_arr_um - z_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset, sp.z_pix_arr_um_offset

    def get_pix_size_um(sp):
        return sp._pixs

    def get_pixel_size_arrs_um(sp):
        sp.make_pixel_size_arrs()
        return sp.x_pix_size_um, sp.y_pix_size_um, sp.z_pix_size_um

    def get_pixel_area_arr(sp):
        sp.make_pixel_size_arrs()
        return sp.pix_area_arr

    def get_seg_xy_maps_pix(sp):
        sp.x_pix_arr_pix = sp.x_pix_arr_um/sp._pixs
        sp.y_pix_arr_pix = sp.y_pix_arr_um/sp._pixs
        return sp.x_pix_arr_pix, sp.y_pix_arr_pix

    def get_seg_xy_maps_pix_with_offset(sp):
        X, Y = sp.get_seg_xy_maps_pix()
        xmin, ymin = X.min(), Y.min()
        return X-xmin, Y-ymin


# INTERFACE METHODS

    def print_seg_info(sp, pbits=0):
        """ Prints segment info for selected bits
        pbits = 0 - nothing,
        +1 - member data,
        +2 - coordinate maps in um,
        +4 - min, max coordinates in um,
        +8 - x, y 1-d pixel coordinate arrays in um.
        """
        if pbits & 1: sp.print_member_data()
        if pbits & 2: sp.print_maps_seg_um()
        if pbits & 4: sp.print_xyz_min_max_um()
        if pbits & 8: sp.print_xy_1darr_um()


    def size(sp):
        """ Returns number of pixels in segment
        """
        return sp._rows*sp._cols


    def rows(sp):
        """ Returns number of rows in segment
        """
        return sp._rows


    def cols(sp):
        """ Returns number of cols in segment
        """
        return sp._cols


    def shape(sp):
        """ Returns shape of the segment (rows, cols)
        """
        return (sp._rows, sp._cols)


    def pixel_scale_size(sp):
        """ Returns pixel size in um for indexing
        """
        return sp._pixs


    def pixel_area_array(sp):
        """ Returns pixel area array of shape=(rows, cols)
        """
        return sp.get_pixel_area_arr()


    def pixel_size_array(sp, axis=None):
        """ Returns numpy array of pixel sizes in um for AXIS
        """
        return sp.return_switch(sp.get_pixel_size_arrs_um, axis)


    def pixel_coord_array(sp, axis=None):
        """ Returns numpy array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_seg_xyz_maps_um, axis)


    def pixel_coord_min(sp, axis=None):
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_min_um, axis)


    def pixel_coord_max(sp, axis=None):
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_max_um, axis)


    def pixel_ones_array(sp, dtype=DTYPE_MASK):
        return np.ones((sp._rows,sp._cols), dtype=dtype)


    def pixel_mask_array(sp, width=0, wcentral=0, edge_rows=0, edge_cols=0, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,
        """
        mask = sp.pixel_ones_array()

        if width>0: edge_rows = edge_cols = width
        if wcentral>0: center_rows = center_cols = wcentral

        if edge_rows>0: # mask edge rows
            w = edge_rows
            zero_row = np.zeros((w,sp._cols),dtype=dtype)
            mask[0:w,:] = zero_row # mask top    edge rows
            mask[-w:,:] = zero_row # mask bottom edge rows

        if edge_cols>0: # mask edge cols
            w = edge_cols
            zero_col = np.zeros((sp._rows,w),dtype=dtype)
            mask[:,0:w] = zero_col # mask left  edge columns
            mask[:,-w:] = zero_col # mask right edge columns

        if center_rows>0: # mask central rows
            w = center_rows
            g = sp._rowsh
            zero_row = np.zeros((w,sp._cols),dtype=dtype)
            mask[g-w:g,:] = zero_row # mask central-low  rows
            mask[g:g+w,:] = zero_row # mask central-high rows

        if center_cols>0: # mask central rows
            w = center_cols
            g = sp._colsh
            zero_col = np.zeros((sp._rows,w),dtype=dtype)
            mask[:,g-w:g] = zero_col # mask central-left  columns
            mask[:,g:g+w] = zero_col # mask central-right columns

        return mask


# 2020-07 added for converter

    def asic0indices(sp):
        """ Returns list of ASIC (0,0)-corner indices in panel daq array.
        """
        return sp._asic0indices

    def asic_rows_cols(sp):
        """ Returns ASIC number of rows, columns.
        """
        return sp._arows, sp._acols

    def number_of_asics_in_rows_cols(sp):
        """ Returns ASIC number of ASICS in the panal in direction fo rows, columns.
        """
        return sp._nasics_in_rows, sp._nasics_in_cols

    def name(sp):
        """ Returns segment name.
        """
        return sp._name


epix10ka_one = SegGeometryEpix10kaV1(use_wide_pix_center=False)
epix10ka_wpc = SegGeometryEpix10kaV1(use_wide_pix_center=True)

# EOF

