#!/usr/bin/env python
#------------------------------
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
#------------------------------

from psana.pscalib.geometry.SegGeometryJungfrauV1 import * # SegGeometryJungfrauV1, logging, np
logger = logging.getLogger(__name__)

#------------------------------

class SegGeometryJungfrauV2(SegGeometryJungfrauV1):
    """Self-sufficient class for generation of Jungfrau 2x4 sensor pixel coordinate array"""

    _name = 'SegGeometryJungfrauV2'

#------------------------------

    def __init__(sp, **kwa):
        logger.debug('SegGeometryJungfrauV2.__init__()')
        SegGeometryJungfrauV1.__init__(sp)

#------------------------------

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

#------------------------------

    def get_xyz_min_um(sp):
        return sp.x_arr_um[0], sp.y_arr_um[0], 0

    def get_xyz_max_um(sp):
        return sp.x_arr_um[-1], sp.y_arr_um[-1], 0
         
#------------------------------

jungfrau_front = SegGeometryJungfrauV2()

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__":
  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg # For test purpose in main only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_xyz_min_max():
    w = jungfrau_front
    w.print_xyz_min_max_um() 
    logger.info('\nYmin = ' + str(w.pixel_coord_min('Y'))\
              + '\nYmax = ' + str(w.pixel_coord_max('Y')))

#------------------------------

  def test_xyz_maps():

    w = jungfrau_front
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
    w = jungfrau_front
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
    o = jungfrau_front
    X, Y = o.get_seg_xy_maps_pix()
    xmin, xmax, ymin, ymax  = X.min(), X.max(), Y.min(), Y.max()
    Xoff, Yoff = X-xmin, ymax-Y
    iX, iY = (Xoff+0.25).astype(int), (Yoff+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,X+3*Y)
    gg.plotImageLarge(img, amp_range=(xmin+3*ymin, xmax+3*ymax), figsize=(14,6))
    gg.show()

#------------------------------

  def test_pix_sizes():
    w = jungfrau_front
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
    o = jungfrau_front
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
