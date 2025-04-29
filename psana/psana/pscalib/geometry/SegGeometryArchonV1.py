#!/usr/bin/env python
"""
Class :py:class:`SegGeometryArchonV1` describes the ArchonV1 sensor geometry
===================================================================================

In this class we use natural matrix notations like in data array
\n We assume that
\n * sensor consists of 16 banks 1/75/150/300/600/1200 rows and 4800 columns,
\n * bank size is (<number-of-rows>,300) with 264 real and 36 fake pixeels
\n * Archon pixel size 100x100um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::

   (Xmin,Ymax)                     ^ Y                        (Xmax,Ymax)
   (0,0)                           |                          (0,4799)
      -----------------------------------------------------------
      |             |              |             |              |
      |             |              |             |              |
      |             |              |             |              |
    --|-------------+--------------|-------------+--------------|----> X
      |             |              |             |              |
      |             |              |             |              |
      |             |              |             |              |
      -----------------------------------------------------------
   (1199,0)                        |                          (1199,4799)
   (Xmin,Ymin)                                                (Xmax,Ymin)


Usage::

    from SegGeometryArchonV1 import epix10ka_one as sg

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
#from psana.pscalib.geometry.SegGeometryEpix10kaV1 import *
from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)
from psana.detector.NDArrUtils import info_ndarr

DTYPE_MASK = np.uint8

class SegGeometryArchonV1(SegGeometry):
    """Self-sufficient class for generation of ArchonV1 sensor pixel coordinate array"""

    _nbanks = 16     # Number of banks
    _rows  = 300     # Number of rows 1/75/150/300/600/1200
    _colsr = 264     # Number of cols in a single of 16 banks
    _colst = 300     # Number of cols in a single of 16 banks
    _colsf = 36      # Number of cols in a single of 16 banks
    _pixsr = 20.     # Pixel size in um (micrometer) in row
    _pixsc = 10.     # Pixel size in um (micrometer) in col
    _pixd  = 400.00  # Pixel depth in um (micrometer)

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryArchonV1'
        #logger.debug('%s.__init__()'%sp._name)

        #sp._nasics_in_rows = 1 # Number of ASICs in row direction
        #sp._nasics_in_cols = 1 # Number of ASICs in column direction
        #sp._asic0indices = ((0, 0), (0, sp._colsq*1), (0, sp._colsq*2), (0, sp._colsq*3))

        SegGeometry.__init__(sp, **kwa)
        det = kwa.get('detector', None)
        assert det is not None
        shape = kwa.get('shape', None)
        if shape is None:
            assert 'pedestals' in det._calibconst.keys(), 'unavailable constants in DB'
            peds = det._calibconst['pedestals'][0]
            shape = peds.shape
        logger.debug('__init__(): segment shape=%s' % str(shape))
        sp._rows = shape[0]
        sp.make_pixel_coord_arrs()
        sp.pix_area_arr = None
        sp.x_pix_arr_pix = None
        sp.x_pix_arr_um_offset = None

    def make_pixel_coord_arrs(sp, dtype=np.float64):
        """Makes [<nrows>,4800] maps of x, y, and z pixel coordinates with origin in the center"""
        x_bank = np.array(tuple(np.arange(sp._colsr)*sp._pixsc) + sp._colsf*(np.nan,), dtype=dtype)
        w = x_bank[-1-sp._colsf] - x_bank[0] + sp._pixsc
        y_bank = -np.arange(sp._rows, dtype=dtype)*sp._pixsr
        sp.x_arr_um = np.hstack([x_bank+i*w for i in range(sp._nbanks)])
        sp.x_arr_um -= (sp.x_arr_um[0]+sp.x_arr_um[-1-sp._colsf])/2 # offset to origin in center
        #print('x_arr_um:', sp.x_arr_um)
        sp.y_arr_um = y_bank - (y_bank[0]+y_bank[-1])/2 # offset to origin in center
        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros(sp.x_pix_arr_um.shape)
        #logger.debug('x_arr_um:\n%s...\ny_arr_um:\n%s' % (str(sp.x_arr_um[0:-sp._colsf+1]), str(sp.y_arr_um)))

    def make_pixel_size_arrs(sp, dtype=np.float64):
        """Makes maps of x, y, and z pixel size and normalized (all 1) pixel area"""
        if sp.pix_area_arr is not None: return # member data are already evaluated
        x_arr_size_um = np.array(sp._nbanks*(sp._colsr*(sp._pixsc,) + sp._colsf*(np.nan,),), dtype=dtype)
        y_arr_size_um = np.ones(sp._rows, dtype=dtype)*sp._pixsr
        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones(sp.x_pix_size_um.shape, dtype=dtype) * sp._pixd
        factor = 1./(sp._pixsr*sp._pixsc)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor

    def mask_fake(sp, dtype=np.uint8, **kwa):
        """returns mask of shape=(<nrows>,4800), with fake pixels of all banks set to 0"""
        fake1bank = np.zeros((sp._rows, sp._colsf), dtype=dtype)
        mask = np.ones(sp.x_pix_arr_um.shape, dtype=dtype)
        sr, st = sp._colsr, sp._colst # = 264, 300
        for i in range(sp._nbanks):
             mask[:,st*i+sr:st*(i+1)] = fake1bank
        return mask

    def get_seg_xy_maps_um(sp):
        return sp.x_pix_arr_um, sp.y_pix_arr_um

    def get_xyz_min_um(sp):
        return [sp.pixel_coord_min(a) for a in sp.AXIS]

    def get_xyz_max_um(sp):
        return [sp.pixel_coord_max(a) for a in sp.AXIS]

    def get_seg_xy_maps_um_with_offset(sp):
        """returns x and y pixel array coordinates in um"""
        if  sp.x_pix_arr_um_offset is None:
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset

    def get_seg_xy_maps_pix_with_offset(sp):
        """returns ix and iy pixel array indices with offset minimum to 0"""
        x, y = sp.get_seg_xy_maps_um_with_offset()
        notnan = ~np.isnan(x)
        ix = -np.ones(x.shape, dtype=np.int32)
        iy = -np.ones(y.shape, dtype=np.int32)
        ix[notnan] = x[notnan]/sp._pixsc
        iy[notnan] = y[notnan]/sp._pixsr
        return ix, iy

    get_seg_xy_maps_pix = get_seg_xy_maps_pix_with_offset

    # SegGeometry interface methods implementation

    def print_seg_info(sp, **kwa):
        """ Prints segment info for selected bits"""
        from psana.detector.NDArrUtils import info_ndarr
        print(info_ndarr(sp.x_arr_um, 'x_arr_um:'))
        print(info_ndarr(sp.y_arr_um, 'y_arr_um:'))

    def size(sp):
        """ Returns segment size - total number of pixels in segment"""
        return sp.x_pix_arr_um.size

    def rows(sp):
        """ Returns number of rows in segment"""
        return sp.shape()[0]

    def cols(sp):
        """ Returns number of cols in segment"""
        return sp.shape()[1]

    def shape(sp):
        """ Returns shape of the segment [rows, cols]"""
        return sp.x_pix_arr_um.shape

    def pixel_scale_size(sp):
        """ Returns pixel size in um for indexing"""
        return sp._pixsc # for cols 10um,  for rows 20um...

    def pixel_area_array(sp):
        """ Returns array of pixel relative areas of shape=[rows, cols]"""
        sp.make_pixel_size_arrs()
        return sp.pix_area_arr

    def get_pixel_size_arrs_um(sp):
        sp.make_pixel_size_arrs()
        return sp.x_pix_size_um,\
               sp.y_pix_size_um,\
               sp.z_pix_size_um

    def get_seg_xyz_maps_um(sp):
        return sp.x_pix_arr_um,\
               sp.y_pix_arr_um,\
               sp.z_pix_arr_um

    def get_xyz_min_um(sp):
        return sp.x_arr_um[0],\
               sp.y_arr_um[-1],\
               sp.z_pix_arr_um[0,0]

    def get_xyz_max_um(sp, axis=None):
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS"""
        return sp.x_arr_um[-1-sp._colsf],\
               sp.y_arr_um[0],\
               sp.z_pix_arr_um[0,0]

    def pixel_mask_array(sp, **kwa):
        """ Returns array of masked pixels which content depends on control bitword mbits"""
        #sp.print_warning('pixel_mask_array(mask_bits)')
        return sp.mask_fake(**kwa)

    def return_switch(sp, meth, axis=None):
        """ Returns three x,y,z arrays if axis=None, or single array for specified axis"""
        assert axis in sp.AXIS + (None,)
        return meth() if axis is None else\
               meth()[sp.DIC_AXIS[axis]]

    def pixel_size_array(sp, axis=None):
        """ Returns numpy array of pixel sizes in um for AXIS"""
        return sp.return_switch(sp.get_pixel_size_arrs_um, axis)

    def pixel_coord_array(sp, axis=None):
        """ Returns numpy array of segment pixel coordinates in um for AXIS"""
        return sp.return_switch(sp.get_seg_xyz_maps_um, axis)

    def pixel_coord_min(sp, axis=None):
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS"""
        return sp.return_switch(sp.get_xyz_min_um, axis)

    def pixel_coord_max(sp, axis=None):
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS"""
        return sp.return_switch(sp.get_xyz_max_um, axis)

    def pixel_ones_array(sp, dtype=DTYPE_MASK):
        return np.ones((sp._rows, sp._cols), dtype=dtype)


# for converter

    def asic0indices(self): self.print_warning('asic0indices')
    def asic_rows_cols(self): self.print_warning('asic_rows_cols')
    def number_of_asics_in_rows_cols(self): self.print_warning('number_of_asics_in_rows_cols')
    def name(self): self.print_warning('name')

#archon_one = SegGeometryArchonV1(use_wide_pix_center=False)

# EOF
