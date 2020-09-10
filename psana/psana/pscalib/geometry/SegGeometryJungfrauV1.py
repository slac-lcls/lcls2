#!/usr/bin/env python
#------------------------------
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

    from psana.pscalib.geometry.SegGeometryJungfrauV1 import jungfrau_one as sg

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
#------------------------------

from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)

#------------------------------

class SegGeometryJungfrauV1(SegGeometry):
    """Self-sufficient class for generation of Jungfrau 2x4 sensor pixel coordinate array"""

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

#------------------------------

    def __init__(sp, **kwa):
        logger.debug('SegGeometryJungfrauV1.__init__()')
        #sp.arg = kwa.get('arg', True)

        SegGeometry.__init__(sp)

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()

#------------------------------

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
        
#------------------------------

    def make_pixel_size_arrs(sp):
        """Makes [512,1024] maps of x, y, and z 2x2 pixel size 
        """        
        if sp.pix_area_arr is None:
           sh = (sp._rows, sp._cols)

           sp.x_pix_size_um = np.ones(sh)*sp._pixs
           sp.y_pix_size_um = np.ones(sh)*sp._pixs
           sp.z_pix_size_um = np.ones(sh)*sp._pixd
           sp.pix_area_arr  = np.ones(sh)
 
#------------------------------

    def print_member_data(sp):
        s = 'SegGeometryJungfrauV1.print_member_data()'\
          + '\n    _rows : %d'    % sp._rows\
          + '\n    _cols : %d'    % sp._cols\
          + '\n    _pixs : %7.2f' % sp._pixs\
          + '\n    _pixd : %7.2f' % sp._pixd\
          + '\n    _rasic: %d'    % sp._rasic\
          + '\n    _casic: %d'    % sp._casic
        logger.info(s)

#------------------------------

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

#------------------------------

    def print_maps_seg_um(sp):
        s = 'SegGeometryJungfrauV1.print_maps_seg_um()'\
          + '\n  x_pix_arr_um =\n'      + str(sp.x_pix_arr_um)\
          + '\n  x_pix_arr_um.shape = ' + str(sp.x_pix_arr_um.shape)\
          + '\n  y_pix_arr_um =\n'      + str(sp.y_pix_arr_um)\
          + '\n  y_pix_arr_um.shape = ' + str(sp.y_pix_arr_um.shape)\
          + '\n  z_pix_arr_um =\n'      + str(sp.z_pix_arr_um)\
          + '\n  z_pix_arr_um.shape = ' + str(sp.z_pix_arr_um.shape)
        logger.info(s)

#------------------------------

    def print_xy_1darr_um(sp):
        s = 'SegGeometryJungfrauV1.print_xy_1darr_um()'\
          + '\n  x_arr_um:\n'       + str(sp.x_arr_um)\
          + '\n  x_arr_um.shape = ' + str(sp.x_arr_um.shape)\
          + '\n  y_arr_um:\n'       + str(sp.y_arr_um)\
          + '\n  y_arr_um.shape = ' + str(sp.y_arr_um.shape)
        logger.info(s)

#------------------------------

    def print_xyz_min_max_um(sp):
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        s = 'SegGeometryJungfrauV1.print_xyz_min_max_um()'\
          + '\n  In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f' \
              % (xmin, xmax, ymin, ymax, zmin, zmax)
        logger.info(s)

#------------------------------

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

#------------------------------
# INTERFACE METHODS
#------------------------------

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


    def pixel_mask_array(sp, mbits=0o377, width=1, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,

        Parameters

        mbits:
            +1 - mask edges,
            +2 - mask central columns 

        width (uint) - width in pixels of masked edge
        """
        w = width
        #mbits = kwargs.get('mbits', 0o377)
        zero_col = np.zeros((sp._rows,w),dtype=np.uint8)
        zero_row = np.zeros((w,sp._cols),dtype=np.uint8)
        mask     = np.ones((sp._rows,sp._cols),dtype=np.uint8)

        if mbits & 1:
        # mask edges
            mask[0:w,:] = zero_row # mask top    edge
            mask[-w:,:] = zero_row # mask bottom edge
            mask[:,0:w] = zero_col # mask left   edge
            mask[:,-w:] = zero_col # mask right  edge

        if mbits & 2:
        # mask central rows and colums - gaps edges
            for i in range(1,4):
                g = sp._casic*i
                mask[:,g-w:g] = zero_col # mask central-left  column
                mask[:,g:g+w] = zero_col # mask central-right column

            g = sp._rasic
            mask[g-w:g,:]     = zero_row # mask central-low   row
            mask[g:g+w,:]     = zero_row # mask central-high  row

        return mask

#----------
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

#------------------------------

jungfrau_one = SegGeometryJungfrauV1()

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__":
  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg # For test purpose in main only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_xyz_min_max():
    w = jungfrau_one
    w.print_xyz_min_max_um() 
    logger.info('\nYmin = ' + str(w.pixel_coord_min('Y'))\
              + '\nYmax = ' + str(w.pixel_coord_max('Y')))

#------------------------------

  def test_xyz_maps():
    w = jungfrau_one
    w.print_maps_seg_um()
    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]):
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i])
        gg.move(200*i,100*i)
    gg.show()

#------------------------------

  def test_jungfrau_img():

    t0_sec = time()
    w = jungfrau_one
    logger.info('Consumed time for coordinate arrays (sec) = %.3f' % (time()-t0_sec))

    X,Y = w.get_seg_xy_maps_pix()

    w.print_seg_info(0o377)

    logger.info('X.shape =' + str(X.shape))

    xmin, ymin, zmin = w.get_xyz_min_um()
    xmax, ymax, zmax = w.get_xyz_max_um()
    xmin /= w.pixel_scale_size()
    xmax /= w.pixel_scale_size()
    ymin /= w.pixel_scale_size()
    ymax /= w.pixel_scale_size()

    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    logger.info('\n  xsize = %d' % xsize\
               +'\n  ysize = %d' % ysize)

#------------------------------

  def test_jungfrau_img_easy():
    o = jungfrau_one
    X, Y = o.get_seg_xy_maps_pix()
    xmin, xmax, ymin, ymax  = X.min(), X.max(), Y.min(), Y.max()
    Xoff, Yoff = X-xmin, Y-ymin
    iX, iY = (Xoff+0.25).astype(int), (Yoff+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,X+3*Y)
    gg.plotImageLarge(img, amp_range=(xmin+3*ymin, xmax+3*ymax), figsize=(14,6))
    gg.show()

#------------------------------

  def test_pix_sizes():
    w = jungfrau_one
    w.print_pixel_size_arrs()
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    s = 'test_pix_sizes():'\
      + '\n  area_arr[348:358,378:388]:\n'  + str(area_arr[348:358,378:388])\
      + '\n  area_arr.shape:'               + str(area_arr.shape)\
      + '\n  size_arrX[348:358,378:388]:\n' + str(size_arrX[348:358,378:388])\
      + '\n  size_arrX.shape:'              + str(size_arrX.shape)\
      + '\n  size_arrY[348:358,378:388]:\n' + str(size_arrY[348:358,378:388])\
      + '\n  size_arrY.shape:'              + str(size_arrY.shape)
    logger.info(s)

#------------------------------

  def test_jungfrau_mask(mbits=0o377, width=1):
    o = jungfrau_one
    X, Y = o.get_seg_xy_maps_pix_with_offset()
    mask = o.pixel_mask_array(mbits=mbits, width=width)
    mask[mask==0]=4
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,mask)
    gg.plotImageLarge(img, amp_range=(-1, 4), figsize=(14,6))
    gg.show()

#------------------------------

  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max()'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps()'
    if tname in ('0','3'): s+='\n 3 - test_jungfrau_img()'
    if tname in ('0','4'): s+='\n 4 - test_jungfrau_img_easy()'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes()'
    if tname in ('0','6'): s+='\n 6 - test_jungfrau_mask(mbits=1+2)'
    if tname in ('0','7'): s+='\n 7 - test_jungfrau_mask(mbits=1+2, width=10)'
    return s

#------------------------------
 
if __name__ == "__main__":

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname in ('1',): test_xyz_min_max()
    elif tname in ('2',): test_xyz_maps()
    elif tname in ('3',): test_jungfrau_img()
    elif tname in ('4',): test_jungfrau_img_easy()
    elif tname in ('5',): test_pix_sizes()
    elif tname in ('6',): test_jungfrau_mask(mbits=1+2)
    elif tname in ('7',): test_jungfrau_mask(mbits=1+2, width=10)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

#------------------------------
