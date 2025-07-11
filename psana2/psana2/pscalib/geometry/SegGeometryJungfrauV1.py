#!/usr/bin/env python
"""
Class :py:class:`SegGeometryJungfrauV1` describes the Jungfrau V1 sensor geometry
=================================================================================

Data array for Jungfrau 512x1024 segment is shaped as (1,512,1024),
has a matrix-like numeration for rows and columns with gaps between 2x4 ASICs
\n We assume that
\n * 1x1 ASICs has 256 rows and 256 columns,
\n * Jungfrau has a pixel size 75x75um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::

   (Xmin,Ymax)                          ^ Y                          (Xmax,Ymax)
   (0,0)                                |                               (0,1023)
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
   (511,0)                              |                             (1023,1023)
   (Xmin,Ymin)                                                        (Xmax,Ymin)


Usage::

    from psana2.pscalib.geometry.SegGeometryJungfrauV1 import jungfrau_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area = sg.pixel_area_array()
    mask = sg.pixel_mask_array(width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa)

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

from psana2.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)


class SegGeometryJungfrauV1(SegGeometry):
    """Self-sufficient class for generation of Jungfrau 2x4 ASICs pixel coordinate array"""

    _name = 'SegGeometryJungfrauV1'

    _rasic =  256    # Number of rows in ASIC
    _casic =  256    # Number of cols in ASIC
    _rows  =  512    # Number of rows in 2x4
    _cols  = 1024    # Number of cols in 2x4
    _pixs  =   75    # Pixel size in um (micrometer)
    _pixd  = 400.00  # Pixel depth in um (micrometer)

    _arows = _rasic
    _acols = _casic

    _nasics_in_rows = 2 # Number of ASICs in row direction
    _nasics_in_cols = 4 # Number of ASICs in column direction

    _asic0indices = ((0, 0), (0, _casic), (0, 2*_casic), (0, 3*_casic),
        (_rasic, 0), (_rasic, _casic), (_rasic, 2*_casic), (_rasic, 3*_casic))


    def __init__(sp, **kwa):
        logger.debug('SegGeometryJungfrauV1.__init__()')
        #sp.arg = kwa.get('arg', True)

        SegGeometry.__init__(sp)

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()


    def make_pixel_coord_arrs(sp):
        """Makes [512,1024] maps of x, y, and z pixel coordinates
        with origin in the center of 2x4
        """
        x_asic = np.arange(sp._casic)*sp._pixs
        x0 = np.array((-512-2.5, -256.5, 1.5, 256+3.5))*sp._pixs
        sp.x_arr_um = np.hstack([x_asic+x0[0], x_asic+x0[1], x_asic+x0[2], x_asic+x0[3]])

        y_asic = np.arange(sp._rasic)*sp._pixs
        y0 = np.array((256.5, -1.5))*sp._pixs
        sp.y_arr_um = np.hstack([y0[0]-y_asic, y0[1]-y_asic])

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows, sp._cols))


    def make_pixel_size_arrs(sp):
        """Makes [512,1024] maps of x, y, and z 2x2 pixel size
        """
        if sp.pix_area_arr is None:
           sh = (sp._rows, sp._cols)

           sp.x_pix_size_um = np.ones(sh)*sp._pixs
           sp.y_pix_size_um = np.ones(sh)*sp._pixs
           sp.z_pix_size_um = np.ones(sh)*sp._pixd
           sp.pix_area_arr  = np.ones(sh)


    def print_member_data(sp):
        s = 'SegGeometryJungfrauV1.print_member_data()'\
          + '\n    _rows : %d'    % sp._rows\
          + '\n    _cols : %d'    % sp._cols\
          + '\n    _pixs : %7.2f' % sp._pixs\
          + '\n    _pixd : %7.2f' % sp._pixd\
          + '\n    _rasic: %d'    % sp._rasic\
          + '\n    _casic: %d'    % sp._casic
        logger.info(s)


    def print_pixel_size_arrs(sp):
        sp.make_pixel_size_arrs()
        s = 'SegGeometryJungfrauV1.print_pixel_size_arrs()'\
          + '\n  sp.x_pix_size_um.shape = ' + str(sp.x_pix_size_um.shape)\
          + '\n  sp.y_pix_size_um:\n'       + str(sp.y_pix_size_um)\
          + '\n  sp.y_pix_size_um.shape = ' + str(sp.y_pix_size_um.shape)\
          + '\n  sp.z_pix_size_um:\n'       + str(sp.z_pix_size_um)\
          + '\n  sp.z_pix_size_um.shape = ' + str(sp.z_pix_size_um.shape)\
          + '\n  sp.pix_area_arr.shape  = ' + str(sp.pix_area_arr.shape)
        logger.info(s)


    def print_maps_seg_um(sp):
        s = 'SegGeometryJungfrauV1.print_maps_seg_um()'\
          + '\n  x_pix_arr_um =\n'      + str(sp.x_pix_arr_um)\
          + '\n  x_pix_arr_um.shape = ' + str(sp.x_pix_arr_um.shape)\
          + '\n  y_pix_arr_um =\n'      + str(sp.y_pix_arr_um)\
          + '\n  y_pix_arr_um.shape = ' + str(sp.y_pix_arr_um.shape)\
          + '\n  z_pix_arr_um =\n'      + str(sp.z_pix_arr_um)\
          + '\n  z_pix_arr_um.shape = ' + str(sp.z_pix_arr_um.shape)
        logger.info(s)


    def print_xy_1darr_um(sp):
        s = 'SegGeometryJungfrauV1.print_xy_1darr_um()'\
          + '\n  x_arr_um:\n'       + str(sp.x_arr_um)\
          + '\n  x_arr_um.shape = ' + str(sp.x_arr_um.shape)\
          + '\n  y_arr_um:\n'       + str(sp.y_arr_um)\
          + '\n  y_arr_um.shape = ' + str(sp.y_arr_um.shape)
        logger.info(s)


    def print_xyz_min_max_um(sp):
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        s = 'SegGeometryJungfrauV1.print_xyz_min_max_um()'\
          + '\n  In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f' \
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

    def get_pix_depth_um(sp):
        return sp._pixd

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


    def pixel_mask_array(sp, width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked.

        Parameters
        ----------

        - width (uint) - width in pixels of masked edge
        - wcenter (uint) - width in pixels of masked central rows and columns
        - edge_rows (uint) - width in pixels of masked edge rows
        - edge_cols (uint) - width in pixels of masked edge columns
        - center_rows (uint) - width in pixels of masked central rows
        - center_cols (uint) - width in pixels of masked central columns

        Return
        ------

        np.array (dtype=np.uint8) - mask array shaped as data
        """

        if width>0: edge_rows = edge_cols = width
        if wcenter>0: center_rows = center_cols = wcenter

        mask = np.ones((sp._rows,sp._cols),dtype=np.dtype)

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

        if center_cols>0: # mask central colums
            u = center_cols
            zero_col = np.zeros((sp._rows,u),dtype=dtype)
            for i in range(1,4):
                g = sp._casic*i
                mask[:,g-u:g] = zero_col # mask central-left  column
                mask[:,g:g+u] = zero_col # mask central-right column

        if center_rows>0: # mask central rows
            w = center_rows
            g = sp._rasic
            zero_row = np.zeros((w,sp._cols),dtype=dtype)
            mask[g-w:g,:] = zero_row # mask central-low   row
            mask[g:g+w,:] = zero_row # mask central-high  row

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

jungfrau_one = SegGeometryJungfrauV1()

# EOF
