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

    from PSCalib.SegGeometryJungfrauV1 import jungfrau_one as sg

    sg.print_seg_info(0377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask_array(mbits=0377, width=1)
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
"""
#------------------------------

import sys
import math
import numpy as np
from time import time

from PSCalib.SegGeometry import *

#------------------------------

class SegGeometryJungfrauV1(SegGeometry) :
    """Self-sufficient class for generation of Jungfrau 2x4 sensor pixel coordinate array"""

    _rasic =  256    # Number of rows in ASIC
    _casic =  256    # Number of cols in ASIC
    _rows  =  512    # Number of rows in 2x4
    _cols  = 1024    # Number of cols in 2x4
    _pixs  =   75    # Pixel size in um (micrometer)
    _pixd  = 400.00  # Pixel depth in um (micrometer)

#------------------------------

    def __init__(sp) :
        #print 'SegGeometryJungfrauV1.__init__()'

        SegGeometry.__init__(sp)
        #super(SegGeometry, self).__init__()

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()

#------------------------------

    def make_pixel_coord_arrs(sp) :
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

    def make_pixel_size_arrs(sp) :
        """Makes [512,1024] maps of x, y, and z 2x2 pixel size 
        """        
        if sp.pix_area_arr is None :
           sh = (sp._rows, sp._cols)

           sp.x_pix_size_um = np.ones(sh)*sp._pixs
           sp.y_pix_size_um = np.ones(sh)*sp._pixs
           sp.z_pix_size_um = np.ones(sh)*sp._pixd
           sp.pix_area_arr  = np.ones(sh)
 
#------------------------------

    def print_member_data(sp) :
        print 'SegGeometryJungfrauV1.print_member_data()'
        print '    _rasic : %d'    % sp._rasic
        print '    _casic : %d'    % sp._casic
        print '    _rows  : %d'    % sp._rows
        print '    _cols  : %d'    % sp._cols
        print '    _pixs  : %7.2f' % sp._pixs 
        print '    _pixd  : %7.2f' % sp._pixd 

#------------------------------

    def print_pixel_size_arrs(sp) :
        print 'SegGeometryJungfrauV1.print_pixel_size_arrs()'
        sp.make_pixel_size_arrs()
        print 'sp.x_pix_size_um.shape = ',            sp.x_pix_size_um.shape
        print 'sp.y_pix_size_um:\n',                  sp.y_pix_size_um
        print 'sp.y_pix_size_um.shape = ',            sp.y_pix_size_um.shape
        print 'sp.z_pix_size_um:\n',                  sp.z_pix_size_um
        print 'sp.z_pix_size_um.shape = ',            sp.z_pix_size_um.shape
        print 'sp.pix_area_arr.shape  = ',            sp.pix_area_arr.shape

#------------------------------

    def print_maps_seg_um(sp) :
        print 'SegGeometryJungfrauV1.print_maps_seg_um()'
        print 'x_pix_arr_um =\n',      sp.x_pix_arr_um
        print 'x_pix_arr_um.shape = ', sp.x_pix_arr_um.shape
        print 'y_pix_arr_um =\n',      sp.y_pix_arr_um
        print 'y_pix_arr_um.shape = ', sp.y_pix_arr_um.shape
        print 'z_pix_arr_um =\n',      sp.z_pix_arr_um
        print 'z_pix_arr_um.shape = ', sp.z_pix_arr_um.shape

#------------------------------

    def print_xy_1darr_um(sp) :
        print 'SegGeometryJungfrauV1.print_xy_1darr_um()'
        print 'x_arr_um:\n',       sp.x_arr_um
        print 'x_arr_um.shape = ', sp.x_arr_um.shape
        print 'y_arr_um:\n',       sp.y_arr_um
        print 'y_arr_um.shape = ', sp.y_arr_um.shape

#------------------------------

    def print_xyz_min_max_um(sp) :
        print 'SegGeometryJungfrauV1.print_xyz_min_max_um()'
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        print 'In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f' \
              % (xmin, xmax, ymin, ymax, zmin, zmax)

#------------------------------

    def get_xyz_min_um(sp) : 
        return sp.x_arr_um[0], sp.y_arr_um[-1], 0

    def get_xyz_max_um(sp) : 
        return sp.x_arr_um[-1], sp.y_arr_um[0], 0

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

    def get_pix_depth_um(sp) : 
        return sp._pixd

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
        pbits = 0 - nothing,
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


    def pixel_mask_array(sp, mbits=0377, width=1) :
        """ Returns numpy array of pixel mask: 1/0 = ok/masked,

        Parameters

        mbits: 
            +1 - mask edges,
            +2 - mask central columns 

        width (uint) - width in pixels of masked edge
        """
        w = width
        zero_col = np.zeros((sp._rows,w),dtype=np.uint8)
        zero_row = np.zeros((w,sp._cols),dtype=np.uint8)
        mask     = np.ones((sp._rows,sp._cols),dtype=np.uint8)

        if mbits & 1 : 
        # mask edges
            mask[0:w,:] = zero_row # mask top    edge
            mask[-w:,:] = zero_row # mask bottom edge
            mask[:,0:w] = zero_col # mask left   edge
            mask[:,-w:] = zero_col # mask right  edge

        if mbits & 2 : 
        # mask central rows and colums - gaps edges
            for i in range(1,4) :
                g = sp._casic*i
                mask[:,g-w:g] = zero_col # mask central-left  column
                mask[:,g:g+w] = zero_col # mask central-right column

            g = sp._rasic
            mask[g-w:g,:]     = zero_row # mask central-low   row
            mask[g:g+w,:]     = zero_row # mask central-high  row

        return mask

#------------------------------
#------------------------------

jungfrau_one = SegGeometryJungfrauV1()

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
    w = SegGeometryJungfrauV1()
    w.print_xyz_min_max_um() 
    print 'Ymin = ', w.pixel_coord_min('Y')
    print 'Ymax = ', w.pixel_coord_max('Y')

#------------------------------

def test_xyz_maps() :

    w = SegGeometryJungfrauV1()
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]) :
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ) :
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i])
        gg.move(200*i,100*i)

    gg.show()

#------------------------------

def test_jungfrau_img() :

    t0_sec = time()
    w = SegGeometryJungfrauV1()
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
    print 'xsize =', xsize
    print 'ysize =', ysize

#    H, Xedges, Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[xmin, xmax], [ymin, ymax]], normed=False, weights=X.flatten()+Y.flatten()) 

#    print 'Xedges:', Xedges
#    print 'Yedges:', Yedges
#    print 'H.shape:', H.shape

#    gg.plotImageLarge(H, amp_range=(-800, 800), figsize=(8,10)) # range=(-1, 2), 
#    gg.show()

#------------------------------

def test_jungfrau_img_easy() :
    o = SegGeometryJungfrauV1()
    X,Y = o.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,iX+iY)
    gg.plotImageLarge(img, amp_range=(0, 1500), figsize=(14,6))
    gg.show()

#------------------------------

def test_pix_sizes() :
    w = SegGeometryJungfrauV1()
    w.print_pixel_size_arrs()
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    print 'area_arr[348:358,378:388]:\n',   area_arr[348:358,378:388]
    print 'area_arr.shape :',               area_arr.shape
    print 'size_arrX[348:358,378:388]:\n',  size_arrX[348:358,378:388]
    print 'size_arrX.shape :',              size_arrX.shape
    print 'size_arrY[348:358,378:388]:\n',  size_arrY[348:358,378:388]
    print 'size_arrY.shape :',              size_arrY.shape

#------------------------------

def test_jungfrau_mask(mbits=0377, width=1) :
    o = SegGeometryJungfrauV1()
    X, Y = o.get_seg_xy_maps_pix_with_offset()
    mask = o.pixel_mask_array(mbits, width)
    mask[mask==0]=4
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,mask)
    gg.plotImageLarge(img, amp_range=(-1, 4), figsize=(14,6))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :

    if len(sys.argv)==1   : print 'For test(s) use command: python', sys.argv[0], '<test-number=0-5>'
    elif sys.argv[1]=='0' : test_xyz_min_max()
    elif sys.argv[1]=='1' : test_xyz_maps()
    elif sys.argv[1]=='2' : test_jungfrau_img()
    elif sys.argv[1]=='3' : test_jungfrau_img_easy()
    elif sys.argv[1]=='4' : test_pix_sizes()
    elif sys.argv[1]=='5' : test_jungfrau_mask(mbits=1+2)
    elif sys.argv[1]=='6' : test_jungfrau_mask(mbits=1+2, width=10)
    else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit( 'End of test.' )

#------------------------------
