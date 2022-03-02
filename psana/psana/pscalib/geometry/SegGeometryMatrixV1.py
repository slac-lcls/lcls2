#!/usr/bin/env python

"""
Class :py:class:`SegGeometryMatrixV1` defines the matrix V1 (pnCCD, 512x512) sensor pixel coordinates in its local frame
========================================================================================================================

Default constructor parameters are set for pnCCD; 512x512 pixels with 75x75um pixel size.
In this class we use natural matrix notations like in data array
(that is different from the DAQ notations where rows and cols are swapped).
\n We assume that
\n * segment has 512 rows and 512 columns,
\n * X-Y coordinate system origin is in the top left corner,
\n * ixel (r,c)=(0,0) is in the top left corner of the matrix which has coordinates (Xmin,Ymin) - is in origin.
\n ::

  MatrixV1 sensor coordinate frame has a matrix-style coordinate system:

  @code
    (Xmin,Ymin)        (Xmin,Ymax)
    (0,0)              (0,511)
       +-----------------+----> Y
       |                 |
       |                 |
       |                 |
       |                 |
       |                 |
       |                 |
       |                 |
       |                 |
       +-----------------+
       |
     X V
    (511,0)           (511,511)
    (Xmax,Ymin)       (Xmax,Ymax)
  @endcode


Usage of interface methods::

    from SegGeometryMatrixV1 import cspad2x1_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area  = sg.pixel_area_array()
    mbits=0o377
    mask = sg.pixel_mask_array(mbits)
    # where mbits = +1-edges, +2-wide pixels, +4-non-bonded pixels, +8-neighbours of non-bonded

    sizeX = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()

    X     = sg.pixel_coord_array('X')
    X,Y,Z = sg.pixel_coord_array()
    print('X.shape =', X.shape)

    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_max()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')

    # global method for rotation of numpy arrays:
    Xrot, Yrot = rotation(X, Y, C, S)
    ...

See:
 * :py:class:`GeometryObject`,
 * :py:class:`SegGeometry`,
 * :py:class:`SegGeometryCspad2x1V1`,
 * :py:class:`SegGeometryEpix100V1`,
 * :py:class:`SegGeometryMatrixV1`,
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""

from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)

def matrix_pars(segname):
    """Returns the matrix sensor parameters from its string-name, ex: MTRX:512:512:54:54
    """
    fields = segname.split(':')
    if len(fields)<5:
        raise IOError('Matrix-sensor specification %s has less than 4 numeric fields' % segname)

    rows, cols, psize_row, psize_col = int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])
    #print('matrix sensor %s parameters:' % (segname), rows, cols, psize_row, psize_col)
    return rows, cols, psize_row, psize_col


class SegGeometryMatrixV1(SegGeometry):
    """Self-sufficient class for generation of sensor pixel coordinate array"""

    _name = 'SegGeometryMatrixV1'

    #_rows  = 512    # Number of rows in panel at rotation 0
    #_cols  = 512    # Number of cols in panel at rotation 0
    #_pixs  = 75.00  # Pixel size in um (micrometer)
    #_pixd  = 400.00 # Pixel depth in um (micrometer)

    _asic0indices = ((0, 0),)
    _nasics_in_rows = 1
    _nasics_in_cols = 1


    def __init__(sp, rows=512, cols=512, pix_size_rows=75, pix_size_cols=75, pix_size_depth=400, pix_scale_size=75):

        SegGeometry.__init__(sp)

        sp._rows = rows
        sp._cols = cols
        sp._pix_size_rows  = pix_size_rows
        sp._pix_size_cols  = pix_size_cols
        sp._pix_size_depth = pix_size_depth
        sp._pixs           = pix_scale_size

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()


    def make_pixel_coord_arrs(sp):
        """Makes maps of x, y, and z of segment pixel coordinates
        """
        sp.x_arr_um = np.arange(sp._rows)*sp._pix_size_rows
        sp.y_arr_um = np.arange(sp._cols)*sp._pix_size_cols

        # Arguments x and y are swapped in order to get grids for "matrix" coordinate system
        # where X is directed from up to down, Y from left to right
        sp.y_pix_arr_um, sp.x_pix_arr_um = np.meshgrid(sp.y_arr_um, sp.x_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))


    def make_pixel_size_arrs(sp):
        """Makes maps of x, y, and z segment pixel size
        """
        if sp.pix_area_arr is not None: return

        x_arr_size_um = np.ones(sp._rows) * sp._pix_size_rows
        y_arr_size_um = np.ones(sp._cols) * sp._pix_size_cols

        sp.y_pix_size_um, sp.x_pix_size_um = np.meshgrid(y_arr_size_um, x_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp._pix_size_depth

        sp.pix_area_arr = np.ones((sp._rows,sp._cols))


    def print_member_data(sp):
        s = 'SegGeometryMatrixV1.print_member_data()'\
          + '\n    _rows : %d'    % sp._rows\
          + '\n    _cols : %d'    % sp._cols\
          + '\n    _pixs : %7.2f' % sp._pixs\
          + '\n    _pix_size_rows : %7.2f' % sp._pix_size_rows \
          + '\n    _pix_size_cols : %7.2f' % sp._pix_size_cols \
          + '\n    _pix_size_depth: %7.2f' % sp._pix_size_depth
        logger.info(s)


    def print_pixel_size_arrs(sp):
        sp.make_pixel_size_arrs()
        sp.make_pixel_coord_arrs()
        s = 'SegGeometryMatrixV1.print_pixel_size_arrs()'\
          + '\n  sp.x_pix_size_um[0:10,190:198]:\n'+ str(sp.x_pix_size_um[0:10,190:198])\
          + '\n  sp.x_pix_size_um.shape = '        + str(sp.x_pix_size_um.shape)\
          + '\n  sp.y_pix_size_um:\n'              + str(sp.y_pix_size_um)\
          + '\n  sp.y_pix_size_um.shape = '        + str(sp.y_pix_size_um.shape)\
          + '\n  sp.z_pix_size_um:\n'              + str(sp.z_pix_size_um)\
          + '\n  sp.z_pix_size_um.shape = '        + str(sp.z_pix_size_um.shape)\
          + '\n  sp.pix_area_arr[0:10,190:198]:\n' + str(sp.pix_area_arr[0:10,190:198])\
          + '\n  sp.pix_area_arr.shape  = '        + str(sp.pix_area_arr.shape)
        logger.info(s)


    def print_maps_seg_um(sp):
        s = 'SegGeometryMatrixV1.print_maps_seg_um()'\
          + '\n  x_pix_arr_um =\n'      + str(sp.x_pix_arr_um)\
          + '\n  x_pix_arr_um.shape = ' + str(sp.x_pix_arr_um.shape)\
          + '\n  y_pix_arr_um =\n'      + str(sp.y_pix_arr_um)\
          + '\n  y_pix_arr_um.shape = ' + str(sp.y_pix_arr_um.shape)\
          + '\n  z_pix_arr_um =\n'      + str(sp.z_pix_arr_um)\
          + '\n  z_pix_arr_um.shape = ' + str(sp.z_pix_arr_um.shape)
        logger.info(s)


    def print_xy_1darr_um(sp):
        s = 'SegGeometryyMatrixV1.print_xy_1darr_um()'\
          + '\n  x_arr_um:\n'       + str(sp.x_arr_um)\
          + '\n  x_arr_um.shape = ' + str(sp.x_arr_um.shape)\
          + '\n  y_arr_um:\n'       + str(sp.y_arr_um)\
          + '\n  y_arr_um.shape = ' + str(sp.y_arr_um.shape)
        logger.info(s)


    def print_xyz_min_max_um(sp):
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        s = 'SegGeometryMatrixV1.print_xyz_min_max_um()'\
          + '\n  In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f'\
            % (xmin, xmax, ymin, ymax, zmin, zmax)
        logger.info(s)


    def get_xyz_min_um(sp):
        return sp.x_arr_um[0], sp.y_arr_um[0], 0

    def get_xyz_max_um(sp):
        return sp.x_arr_um[-1], sp.y_arr_um[-1], 0

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
        pbits=0 - nothing,
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


    def pixel_mask_array(sp, width=0, edge_rows=0, edge_cols=0, dtype=DTYPE_MASK, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,
        """
        if width>0: edge_rows = edge_cols = width
        mask = np.ones((sp._rows, sp._cols), dtype=dtype)

        if edge_rows>0: # mask edge rows
            w = edge_rows
            zero_row = np.zeros((w, sp._cols), dtype=dtype)
            mask[0:w,:] = zero_row # mask top    edge rows
            mask[-w:,:] = zero_row # mask bottom edge rows

        if edge_cols>0: # mask edge cols
            w = edge_cols
            zero_col = np.zeros((sp._rows, w), dtype=dtype)
            mask[:,0:w] = zero_col # mask left  edge columns
            mask[:,-w:] = zero_col # mask right edge columns

        return mask


# 2020-08 added for converter

    def asic0indices(sp):
        """ Returns list of ASIC (0,0)-corner indices in panel daq array.
        """
        return sp._asic0indices

    def asic_rows_cols(sp):
        """ Returns ASIC number of rows, columns.
        """
        return sp._rows, sp._cols

    def number_of_asics_in_rows_cols(sp):
        """ Returns ASIC number of ASICS in the panal in direction fo rows, columns.
        """
        return sp._nasics_in_rows, sp._nasics_in_cols

    def name(sp):
        """ Returns segment name.
        """
        return sp._name

segment_one = SegGeometryMatrixV1()
#seg_andor3d = SegGeometryMatrixV1(rows=2048, cols=2048, pix_size_rows=13.5,\
#                pix_size_cols=13.5, pix_size_depth=50, pix_scale_size=13.5)

# EOF

