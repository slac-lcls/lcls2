#!/usr/bin/env python
#------------------------------
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

    sg.print_seg_info(0377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area  = sg.pixel_area_array()
    mask = sg.pixel_mask_array(mbits=0377)
    # where mbits = +1-edges, +2-wide pixels, +4-non-bonded pixels, +8-neighbours of non-bonded

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
"""
#------------------------------

import sys
import math
import numpy as np
from time import time

from PSCalib.SegGeometry import *

#------------------------------

def matrix_pars(segname) :
    """Returns the matrix sensor parameters from its string-name, ex: MTRX:512:512:54:54
    """
    fields = segname.split(':')
    if len(fields)<5 :
        raise IOError('Matrix-sensor specification %s has less than 4 numeric fields' % segname)

    rows, cols, psize_row, psize_col = int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])
    #print 'matrix sensor %s parameters:' % (segname), rows, cols, psize_row, psize_col
    return rows, cols, psize_row, psize_col

#------------------------------

class SegGeometryMatrixV1(SegGeometry) :
    """Self-sufficient class for generation of CSPad 2x1 sensor pixel coordinate array"""

    #_rows  = 512    # Number of rows in 2x1 at rotation 0
    #_cols  = 512    # Number of cols in 2x1 at rotation 0
    #_pixs  = 75.00  # Pixel size in um (micrometer)
    #_pixd  = 400.00 # Pixel depth in um (micrometer)

#------------------------------

    def __init__(sp, rows=512, cols=512, pix_size_rows=75, pix_size_cols=75, pix_size_depth=400, pix_scale_size=75) :
        #print 'SegGeometryMatrixV1.__init__()'

        SegGeometry.__init__(sp)
        #super(SegGeometry, self).__init__()

        sp._rows = rows
        sp._cols = cols
        sp._pix_size_rows  = pix_size_rows
        sp._pix_size_cols  = pix_size_cols
        sp._pix_size_depth = pix_size_depth
        sp._pixs           = pix_scale_size

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()

#------------------------------

    def make_pixel_coord_arrs(sp) :
        """Makes maps of x, y, and z of segment pixel coordinates
        """        
        sp.x_arr_um = np.arange(sp._rows)*sp._pix_size_rows
        sp.y_arr_um = np.arange(sp._cols)*sp._pix_size_cols

        # Arguments x and y are swapped in order to get grids for "matrix" coordinate system
        # where X is directed from up to down, Y from left to right
        sp.y_pix_arr_um, sp.x_pix_arr_um = np.meshgrid(sp.y_arr_um, sp.x_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))
        
#------------------------------

    def make_pixel_size_arrs(sp) :
        """Makes maps of x, y, and z segment pixel size 
        """        
        if sp.pix_area_arr is not None : return

        x_arr_size_um = np.ones(sp._rows) * sp._pix_size_rows
        y_arr_size_um = np.ones(sp._cols) * sp._pix_size_cols

        sp.y_pix_size_um, sp.x_pix_size_um = np.meshgrid(y_arr_size_um, x_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp._pix_size_depth
 
        sp.pix_area_arr = np.ones((sp._rows,sp._cols))

#------------------------------

    def print_member_data(sp) :
        print 'SegGeometryMatrixV1.print_member_data()'
        print '    _rows : %d'     % sp._rows
        print '    _cols : %d'     % sp._cols
        print '    _pixs  : %7.2f' % sp._pixs 
        print '    _pix_size_rows  : %7.2f' % sp._pix_size_rows 
        print '    _pix_size_cols  : %7.2f' % sp._pix_size_cols 
        print '    _pix_size_depth : %7.2f' % sp._pix_size_depth 

#------------------------------

    def print_pixel_size_arrs(sp) :
        print 'SegGeometryMatrixV1.print_pixel_size_arrs()'
        sp.make_pixel_size_arrs()
        print 'sp.x_pix_size_um[0:10,190:198]:\n', sp.x_pix_size_um[0:10,190:198]
        print 'sp.x_pix_size_um.shape = ',         sp.x_pix_size_um.shape
        print 'sp.y_pix_size_um:\n',               sp.y_pix_size_um
        print 'sp.y_pix_size_um.shape = ',         sp.y_pix_size_um.shape
        print 'sp.z_pix_size_um:\n',               sp.z_pix_size_um
        print 'sp.z_pix_size_um.shape = ',         sp.z_pix_size_um.shape
        sp.make_pixel_coord_arrs()
        print 'sp.pix_area_arr[0:10,190:198]:\n',  sp.pix_area_arr[0:10,190:198]
        print 'sp.pix_area_arr.shape  = ',         sp.pix_area_arr.shape

#------------------------------

    def print_maps_seg_um(sp) :
        print 'SegGeometryMatrixV1.print_maps_seg_um()'
        print 'x_pix_arr_um =\n',      sp.x_pix_arr_um
        print 'x_pix_arr_um.shape = ', sp.x_pix_arr_um.shape
        print 'y_pix_arr_um =\n',      sp.y_pix_arr_um
        print 'y_pix_arr_um.shape = ', sp.y_pix_arr_um.shape
        print 'z_pix_arr_um =\n',      sp.z_pix_arr_um
        print 'z_pix_arr_um.shape = ', sp.z_pix_arr_um.shape

#------------------------------

    def print_xy_1darr_um(sp) :
        print 'SegGeometryMatrixV1.print_xy_1darr_um()'
        print 'x_arr_um:\n',       sp.x_arr_um
        print 'x_arr_um.shape = ', sp.x_arr_um.shape
        print 'y_arr_um:\n',       sp.y_arr_um
        print 'y_arr_um.shape = ', sp.y_arr_um.shape

#------------------------------

    def print_xyz_min_max_um(sp) :
        print 'SegGeometryMatrixV1.print_xyz_min_max_um()'
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        print 'In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f' \
              % (xmin, xmax, ymin, ymax, zmin, zmax)

#------------------------------

    def get_xyz_min_um(sp) : 
        return sp.x_arr_um[0], sp.y_arr_um[0], 0

    def get_xyz_max_um(sp) : 
        return sp.x_arr_um[-1], sp.y_arr_um[-1], 0

    def get_seg_xy_maps_um(sp) : 
        return sp.x_pix_arr_um, sp.y_pix_arr_um

    def get_seg_xyz_maps_um(sp) : 
        return sp.x_pix_arr_um, sp.y_pix_arr_um, sp.z_pix_arr_um

    def get_seg_xy_maps_um_with_offset(sp) : 
        if  sp.x_pix_arr_um_offset is None :
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset

    def get_seg_xyz_maps_um_with_offset(sp) : 
        if  sp.x_pix_arr_um_offset is None :
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
            sp.z_pix_arr_um_offset = sp.z_pix_arr_um - z_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset, sp.z_pix_arr_um_offset

    def get_pix_size_um(sp) : 
        return sp._pixs

    def get_pixel_size_arrs_um(sp) :
        sp.make_pixel_size_arrs()
        return sp.x_pix_size_um, sp.y_pix_size_um, sp.z_pix_size_um

    def get_pixel_area_arr(sp) :
        sp.make_pixel_size_arrs()
        return sp.pix_area_arr

    def get_seg_xy_maps_pix(sp) :
        sp.x_pix_arr_pix = sp.x_pix_arr_um/sp._pixs
        sp.y_pix_arr_pix = sp.y_pix_arr_um/sp._pixs
        return sp.x_pix_arr_pix, sp.y_pix_arr_pix

    def get_seg_xy_maps_pix_with_offset(sp) :
        X, Y = sp.get_seg_xy_maps_pix()
        xmin, ymin = X.min(), Y.min()
        return X-xmin, Y-ymin

#------------------------------
# INTERFACE METHODS
#------------------------------

    def print_seg_info(sp, pbits=0) :
        """ Prints segment info for selected bits
        pbits=0 - nothing,
        +1 - member data,
        +2 - coordinate maps in um,
        +4 - min, max coordinates in um,
        +8 - x, y 1-d pixel coordinate arrays in um.
        """
        if pbits & 1 : sp.print_member_data()
        if pbits & 2 : sp.print_maps_seg_um()
        if pbits & 4 : sp.print_xyz_min_max_um()
        if pbits & 8 : sp.print_xy_1darr_um()


    def size(sp) :
        """ Returns number of pixels in segment
        """
        return sp._rows*sp._cols


    def rows(sp) :
        """ Returns number of rows in segment
        """
        return sp._rows


    def cols(sp) :
        """ Returns number of cols in segment
        """
        return sp._cols


    def shape(sp) :
        """ Returns shape of the segment (rows, cols)
        """
        return (sp._rows, sp._cols)


    def pixel_scale_size(sp) :
        """ Returns pixel size in um for indexing
        """
        return sp._pixs


    def pixel_area_array(sp) :
        """ Returns pixel area array of shape=(rows, cols)
        """
        return sp.get_pixel_area_arr()


    def pixel_size_array(sp, axis=None) :
        """ Returns numpy array of pixel sizes in um for AXIS
        """
        return sp.return_switch(sp.get_pixel_size_arrs_um, axis)


    def pixel_coord_array(sp, axis=None) :
        """ Returns numpy array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_seg_xyz_maps_um, axis)


    def pixel_coord_min(sp, axis=None) :
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_min_um, axis)


    def pixel_coord_max(sp, axis=None) :
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_max_um, axis)


    def pixel_mask_array(sp, mbits=0377) :
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,
        mbits=1 - mask edges,
        +2 - mask two central columns, 
        +4 - mask non-bonded pixels,
        +8 - mask nearest neighbours of nonbonded pixels.
        """
        zero_col = np.zeros(sp._rows,dtype=np.uint8)
        zero_row = np.zeros(sp._cols,dtype=np.uint8)
        mask     = np.ones((sp._rows,sp._cols),dtype=np.uint8)

        if mbits & 1 : 
        # mask edges
            mask[0, :] = zero_row # mask top    edge
            mask[-1,:] = zero_row # mask bottom edge
            mask[:, 0] = zero_col # mask left   edge
            mask[:,-1] = zero_col # mask right  edge

        return mask

  
#------------------------------
#------------------------------

segment_one = SegGeometryMatrixV1()
#seg_andor3d = SegGeometryMatrixV1(rows=2048, cols=2048, pix_size_rows=13.5, pix_size_cols=13.5, pix_size_depth=50, pix_scale_size=13.5)

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import pyimgalgos.GlobalGraphics as gg # For test purpose in main only


def test_xyz_min_max() :
    w = SegGeometryMatrixV1()
    w.print_xyz_min_max_um() 
    print 'Ymin = ', w.pixel_coord_min('Y')
    print 'Ymax = ', w.pixel_coord_max('Y')

#------------------------------

def test_xyz_maps() :

    w = SegGeometryMatrixV1()
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]) :
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ) :
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i])
        gg.move(200*i,100*i)

    gg.show()

#------------------------------

def test_img() :

    t0_sec = time()
    w = SegGeometryMatrixV1()
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    X,Y = w.get_seg_xy_maps_pix()

    w.print_seg_info(0377)

    #print 'X(pix) :\n', X
    print 'X.shape =', X.shape

    xmin, ymin, zmin = w.get_xyz_min_um()
    xmax, ymax, zmax = w.get_xyz_max_um()
    xmin /= w.pixel_scale_size()
    xmax /= w.pixel_scale_size()
    ymin /= w.pixel_scale_size()
    ymax /= w.pixel_scale_size()

    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    print 'xsize =', xsize # 391.0 
    print 'ysize =', ysize # 185.0

    H, Xedges, Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[xmin, xmax], [ymin, ymax]], normed=False, weights=X.flatten()+Y.flatten()) 

    print 'Xedges:', Xedges
    print 'Yedges:', Yedges
    print 'H.shape:', H.shape

    gg.plotImageLarge(H, amp_range=(0, 1100), figsize=(8,10)) # range=(-1, 2), 
    gg.show()

#------------------------------

def test_img_easy() :
    pc2x1 = SegGeometryMatrixV1()
    #X,Y = pc2x1.get_seg_xy_maps_pix()
    X,Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,iX+iY)
    gg.plotImageLarge(img, amp_range=(0, 1100), figsize=(8,10))
    gg.show()

#------------------------------

def test_pix_sizes() :
    w = SegGeometryMatrixV1()
    w.print_pixel_size_arrs()
    size_arr = w.pixel_size_array('X')
    area_arr = w.pixel_area_array()
    print 'area_arr[0:10,190:198]:\n',  area_arr[0:10,190:198]
    print 'area_arr.shape :',           area_arr.shape
    print 'size_arr[0:10,190:198]:\n',  size_arr[0:10,190:198]
    print 'size_arr.shape :',           size_arr.shape

#------------------------------

def test_mask(mbits=0377) :
    pc2x1 = SegGeometryMatrixV1()
    X, Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    mask = pc2x1.pixel_mask_array(mbits)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=(8,10))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :

    if len(sys.argv)==1   : print 'For other test(s) use command: python', sys.argv[0], '<test-number=0-5>'
    elif sys.argv[1]=='0' : test_xyz_min_max()
    elif sys.argv[1]=='1' : test_xyz_maps()
    elif sys.argv[1]=='2' : test_img()
    elif sys.argv[1]=='3' : test_img_easy()
    elif sys.argv[1]=='4' : test_pix_sizes()
    elif sys.argv[1]=='5' : test_mask(mbits=1+2+4+8)
    else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit( 'End of test.' )

#------------------------------
