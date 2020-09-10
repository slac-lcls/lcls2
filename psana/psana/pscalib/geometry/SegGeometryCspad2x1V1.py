#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`SegGeometryCspad2x1V1` describes the CSPAD 2x1 V1 sensor geometry
==================================================================================

In this class we use natural matrix notations like in data array
(that is different from the DAQ notations where rows and cols are swapped).
\n We assume that
\n * 2x1 has 185 rows and 388 columns,
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::

   (Xmin,Ymax)      ^ Y          (Xmax,Ymax)
   (0,0)            |            (0,387)
      ------------------------------
      |             |              |
      |             |              |
      |             |              |
    --|-------------+--------------|----> X
      |             |              |
      |             |              |
      |             |              |
      ------------------------------
   (184,0)          |           (184,387)
   (Xmin,Ymin)                  (Xmax,Ymin)


Usage::

    from SegGeometryCspad2x1V1 import cspad2x1_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area  = sg.pixel_area_array()
    mask = sg.pixel_mask_array(mbits=0o377)
    # where mbits = +1  - edges,
    #               +2  - wide pixels,
    #               +4  - non-bonded pixels,
    #               +8  - nearest four neighbours of non-bonded
    #               +16 - eight neighbours of non-bonded

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
#------------------------------

from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)

#------------------------------

class SegGeometryCspad2x1V1(SegGeometry):
    """Self-sufficient class for generation of CSPad 2x1 sensor pixel coordinate array"""

    _name = 'SegGeometryCspad2x1V1'

    _rows  = 185    # Number of rows in 2x1 at rotation 0
    _cols  = 388    # Number of cols in 2x1 at rotation 0
    _pixs  = 109.92 # Pixel size in um (micrometer)
    _pixw  = 274.80 # Wide pixel size in um (micrometer)
    _pixd  = 400.00 # Pixel depth in um (micrometer)

    _colsh = _cols//2
    _pixsh = _pixs/2
    _pixwh = _pixw/2

    _arows = _rows
    _acols = _colsh

    _nasics_in_rows = 1 # Number of ASICs in row direction
    _nasics_in_cols = 2 # Number of ASICs in column direction

    _asic0indices = ((0, 0), (0, _colsh))

#------------------------------

    def __init__(sp, **kwa):
        logger.debug('SegGeometryCspad2x1V1.__init__()')

        SegGeometry.__init__(sp)
        sp.use_wide_pix_center = kwa.get('use_wide_pix_center', True)

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()

#------------------------------

    def make_pixel_coord_arrs(sp):
        """Makes [185,388] maps of x, y, and z 2x1 pixel coordinates
        with origin in the center of 2x1
        """        
        x_rhs = np.arange(sp._colsh)*sp._pixs + sp._pixw - sp._pixsh
        if sp.use_wide_pix_center: x_rhs[0] = sp._pixwh # set x-coordinate of the wide pixel in its geometry center
        sp.x_arr_um = np.hstack([-x_rhs[::-1],x_rhs])

        sp.y_arr_um = -np.arange(sp._rows) * sp._pixs
        sp.y_arr_um -= sp.y_arr_um[-1]/2 # move origin to the center of array

        #sp.x_arr_pix = sp.x_arr_um/sp._pixs
        #sp.y_arr_pix = sp.y_arr_um/sp._pixs

        #sp.x_pix_arr_pix, sp.y_pix_arr_pix = np.meshgrid(sp.x_arr_pix, sp.y_arr_pix)
        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))
        
#------------------------------

    def make_pixel_size_arrs(sp):
        """Makes [185,388] maps of x, y, and z 2x1 pixel size 
        """        
        if sp.pix_area_arr is not None: return

        x_rhs_size_um = np.ones(sp._colsh)*sp._pixs
        x_rhs_size_um[0] = sp._pixw
        x_arr_size_um = np.hstack([x_rhs_size_um[::-1],x_rhs_size_um])
        y_arr_size_um = np.ones(sp._rows) * sp._pixs

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp._pixd
        
        factor = 1./(sp._pixs*sp._pixs)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor

#------------------------------

    def print_member_data(sp):
        s = 'SegGeometryCspad2x1V1.print_member_data()'\
          + '\n    _rows : %d'    % sp._rows\
          + '\n    _cols : %d'    % sp._cols\
          + '\n    _pixs : %7.2f' % sp._pixs\
          + '\n    _pixw : %7.2f' % sp._pixw\
          + '\n    _pixd : %7.2f' % sp._pixd\
          + '\n    _colsh: %d'    % sp._colsh\
          + '\n    _pixsh: %7.2f' % sp._pixsh\
          + '\n    _pixwh: %7.2f' % sp._pixwh
        logger.info(s)

#------------------------------

    def print_pixel_size_arrs(sp):
        sp.make_pixel_size_arrs()
        s = 'SegGeometryCspad2x1V1.print_pixel_size_arrs()'\
          + '\n  sp.x_pix_size_um[0:10,190:198]:\n'+ str(sp.x_pix_size_um[0:10,190:198])\
          + '\n  sp.x_pix_size_um.shape = '        + str(sp.x_pix_size_um.shape)\
          + '\n  sp.y_pix_size_um:\n'              + str(sp.y_pix_size_um)\
          + '\n  sp.y_pix_size_um.shape = '        + str(sp.y_pix_size_um.shape)\
          + '\n  sp.z_pix_size_um:\n'              + str(sp.z_pix_size_um)\
          + '\n  sp.z_pix_size_um.shape = '        + str(sp.z_pix_size_um.shape)\
          + '\n  sp.pix_area_arr[0:10,190:198]:\n' + str(sp.pix_area_arr[0:10,190:198])\
          + '\n  sp.pix_area_arr.shape  = '        + str(sp.pix_area_arr.shape)
        logger.info(s)

#------------------------------

    def print_maps_seg_um(sp):
        s = 'SegGeometryCspad2x1V1.print_maps_seg_um()'\
          + '\n  x_pix_arr_um =\n'      + str(sp.x_pix_arr_um)\
          + '\n  x_pix_arr_um.shape = ' + str(sp.x_pix_arr_um.shape)\
          + '\n  y_pix_arr_um =\n'      + str(sp.y_pix_arr_um)\
          + '\n  y_pix_arr_um.shape = ' + str(sp.y_pix_arr_um.shape)\
          + '\n  z_pix_arr_um =\n'      + str(sp.z_pix_arr_um)\
          + '\n  z_pix_arr_um.shape = ' + str(sp.z_pix_arr_um.shape)
        logger.info(s)

#------------------------------

    def print_xy_1darr_um(sp):
        s = 'SegGeometryCspad2x1V1.print_xy_1darr_um()'\
          + '\n  x_arr_um:\n'       + str(sp.x_arr_um)\
          + '\n  x_arr_um.shape = ' + str(sp.x_arr_um.shape)\
          + '\n  y_arr_um:\n'       + str(sp.y_arr_um)\
          + '\n  y_arr_um.shape = ' + str(sp.y_arr_um.shape)
        logger.info(s)

#------------------------------

    def print_xyz_min_max_um(sp):
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        s = 'SegGeometryCspad2x1V1.print_xyz_min_max_um()'\
          + '\n  In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f'\
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


    def pixel_mask_array(sp, mbits=0o377, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,
        mbits=1 - mask edges,
        +2 - mask two central columns, 
        +4 - mask non-bonded pixels,
        +8 - mask nearest four neighbours of nonbonded pixels,
        +16- mask eight neighbours of nonbonded pixels.
        """
        zero_col = np.zeros(sp._rows,dtype=np.uint8)
        zero_row = np.zeros(sp._cols,dtype=np.uint8)
        mask     = np.ones((sp._rows,sp._cols),dtype=np.uint8)

        if mbits & 1: 
        # mask edges
            mask[0,:] = zero_row # mask top    edge
            mask[-1,:] = zero_row # mask bottom edge
            mask[:, 0] = zero_col # mask left   edge
            mask[:,-1] = zero_col # mask right  edge

        if mbits & 2: 
        # mask two central columns
            mask[:,sp._colsh-1] = zero_col # mask central-left  column
            mask[:,sp._colsh]   = zero_col # mask central-right column

        if mbits & 4 or mbits & 8 or mbits & 16: 
        # mask non-bonded pixels
            for p in range(0, sp._rows, 10):
                h = sp._colsh
                
                if mbits & 16:
                # mask eight neighbours of nonbonded pixels
                    if p==0:
                        mask[0:2,0:2] = 0
                        mask[0:2,h:2+h] = 0
                    else:
                        mask[p-1:p+2,p-1:p+2] = 0
                        mask[p-1:p+2,p-1+h:p+2+h] = 0

                elif mbits & 8:
                # mask nearest four neighbours of nonbonded pixels
                    if p==0:
                        mask[1,0] = 0
                        mask[0,1] = 0
                        mask[1,0+h] = 0
                        mask[0,1+h] = 0
                    else:
                        mask[p-1:p+2,p] = 0
                        mask[p,p-1:p+2] = 0                        
                        mask[p-1:p+2,p+h] = 0
                        mask[p,p+h-1:p+h+2] = 0                        

                elif mbits & 4:
                # mask nonbonded pixels
                    mask[p,p] = 0
                    mask[p,p+h] = 0

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

cspad2x1_one = SegGeometryCspad2x1V1(use_wide_pix_center=False)
cspad2x1_wpc = SegGeometryCspad2x1V1(use_wide_pix_center=True)

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__":

  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_xyz_min_max():
    w = SegGeometryCspad2x1V1()
    w.print_xyz_min_max_um() 
    s = 'test_xyz_min_max'\
      + '\n  Ymin = %.1f' % w.pixel_coord_min('Y')\
      + '\n  Ymax = %.1f' % w.pixel_coord_max('Y')
    logger.info(s)

#------------------------------

  def test_xyz_maps():
    w = SegGeometryCspad2x1V1()
    w.print_maps_seg_um()
    titles = ['X map','Y map']
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i])
        gg.move(200*i,100*i)
    gg.show()

#------------------------------

  def test_2x1_img():

    w = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    X,Y = w.get_seg_xy_maps_pix()

    w.print_seg_info(0o377)

    xmin, ymin, zmin = w.get_xyz_min_um()
    xmax, ymax, zmax = w.get_xyz_max_um()
    xmin /= w.pixel_scale_size()
    xmax /= w.pixel_scale_size()
    ymin /= w.pixel_scale_size()
    ymax /= w.pixel_scale_size()

    xsize = int(xmax - xmin + 1)
    ysize = int(ymax - ymin + 1)

    H, Xedges, Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize],\
                          range=[[xmin, xmax], [ymin, ymax]], normed=False, weights=X.flatten()+Y.flatten()) 

    s = 'test_img'\
      + '\n  X.shape:' + str(X.shape)\
      + '\n  xsize = %.1f' % xsize\
      + '\n  ysize = %.1f' % ysize\
      + '\n  Xedges:'  + str(Xedges)\
      + '\n  Yedges:'  + str(Yedges)\
      + '\n  H.shape:' + str(H.shape)
    logger.info(s)

    gg.plotImageLarge(H, amp_range=(-250, 250), figsize=(8,10)) # range=(-1, 2), 
    gg.show()

#------------------------------

  def test_2x1_img_easy():
    pc2x1 = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    #X,Y = pc2x1.get_seg_xy_maps_pix()
    X,Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,iX+iY)
    gg.plotImageLarge(img, amp_range=(0, 500), figsize=(8,10))
    gg.show()

#------------------------------

  def test_pix_sizes():
    w = SegGeometryCspad2x1V1()
    w.print_pixel_size_arrs()
    size_arr = w.pixel_size_array('X')
    area_arr = w.pixel_area_array()
    s = 'test_pix_sizes\n'\
      + '\n  area_arr[0:10,190:198]:\n' + str(area_arr[0:10,190:198])\
      + '\n  area_arr.shape:'           + str(area_arr.shape)\
      + '\n  size_arr[0:10,190:198]:\n' + str(size_arr[0:10,190:198])\
      + '\n  size_arr.shape:'           + str(size_arr.shape)
    logger.info(s)

#------------------------------

  def test_2x1_mask(mbits=0o377):
    pc2x1 = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    X, Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    mask = pc2x1.pixel_mask_array(mbits)
    s = '\n  mask:\n%s' % str(mask)\
      + '\n  mask.shape: %s' % str(mask.shape)
    logger.info(s)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=(8,10))
    gg.show()

#------------------------------

  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max()'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps()'
    if tname in ('0','3'): s+='\n 3 - test_img()'
    if tname in ('0','4'): s+='\n 4 - test_img_easy()'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes()'
    if tname in ('0','6'): s+='\n 6 - test_mask(mbits=1+2+4+8)'
    if tname in ('0','7'): s+='\n 7 - test_mask(mbits=16)'
    return s

#------------------------------
  
if __name__ == "__main__":

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname=='1': test_xyz_min_max()
    elif tname=='2': test_xyz_maps()
    elif tname=='3': test_2x1_img()
    elif tname=='4': test_2x1_img_easy()
    elif tname=='5': test_pix_sizes()
    elif tname=='6': test_2x1_mask(mbits=1+2+4+8)
    elif tname=='7': test_2x1_mask(mbits=16)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

#------------------------------
