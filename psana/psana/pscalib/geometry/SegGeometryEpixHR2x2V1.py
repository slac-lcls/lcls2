#!/usr/bin/env python
"""
Class :py:class:`SegGeometryEpixHR2x2V1` describes the EpixHR2x2V1 sensor geometry
===================================================================================

In this class we use natural matrix notations like in data array
\n We assume that
\n * sensor consists of 2x2 ASICs has 288 rows and 384 columns,
\n * Epix10ka has a pixel size 100x100um, wide pixel size 100x250um
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
   (287,0)          |           (287,383)
   (Xmin,Ymin)                  (Xmax,Ymin)


Usage::

    from SegGeometryEpixHR2x2V1 import epix10ka_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask_array(mbits=0o377, width=5, wcentral=5)
    # where mbits = +1-edges, +2-wide pixels

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
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryEpix10kaV1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-09-24 by Mikhail Dubrovin
"""
from psana.pscalib.geometry.SegGeometryEpix10kaV1 import *# SegGeometryEpix10kaV1
#from psana.pscalib.geometry.SegGeometry import *
logger = logging.getLogger(__name__)

DTYPE_MASK = np.uint8


class SegGeometryEpixHR2x2V1(SegGeometryEpix10kaV1):
    """Self-sufficient class for generation of Epix10ka sensor (2x2 ASICs) pixel coordinate array"""

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryEpixHR2x2V1'
        logger.debug('%s.__init__()'%sp._name)

        sp._rows  = 288     # Number of rows in 2x2
        sp._cols  = 384     # Number of cols in 2x2
        sp._pixs  = 100     # Pixel size in um (micrometer)
        sp._pixw  = 250     # Wide pixel size in um (micrometer)
        sp._pixd  = 400.00  # Pixel depth in um (micrometer)

        sp._colsh = sp._cols//2
        sp._rowsh = sp._rows//2
        sp._pixsh = sp._pixs/2
        sp._pixwh = sp._pixw/2

        sp._arows = sp._rowsh
        sp._acols = sp._colsh

        sp._nasics_in_rows = 2 # Number of ASICs in row direction
        sp._nasics_in_cols = 2 # Number of ASICs in column direction

        sp._asic0indices = ((0, 0), (0, sp._colsh), (sp._rowsh, 0), (sp._rowsh, sp._colsh))

        SegGeometryEpix10kaV1.__init__(sp, **kwa)


epixhr2x2_one = SegGeometryEpixHR2x2V1(use_wide_pix_center=False)
epixhr2x2_wpc = SegGeometryEpixHR2x2V1(use_wide_pix_center=True)

#----------- TEST -------------

if __name__ == "__main__":

  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg # For test purpose in main only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_xyz_min_max():
    w = SegGeometryEpixHR2x2V1()
    w.print_xyz_min_max_um()
    logger.info('\n  Ymin = %f' % w.pixel_coord_min('Y')\
              + '\n  Ymax = %f' % w.pixel_coord_max('Y'))


  def test_xyz_maps():

    w = SegGeometryEpixHR2x2V1()
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]):
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,8), title=titles[i])
        gg.move(200*i,100*i)

    gg.show()


  def test_2x2_img():

    t0_sec = time()
    w = SegGeometryEpixHR2x2V1(use_wide_pix_center=False)
    #w = SegGeometryEpixHR2x2V1(use_wide_pix_center=True)
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
    logger.info('xsize =' + str(xsize))
    logger.info('ysize =' + str(ysize))


  def test_2x2_img_easy():
    w = SegGeometryEpixHR2x2V1(use_wide_pix_center=False)
    X,Y = w.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,iX+2*iY)
    gg.plotImageLarge(img, amp_range=(0, 750), figsize=(10,8))
    gg.show()


  def test_pix_sizes():
    w = SegGeometryEpixHR2x2V1()
    w.print_pixel_size_arrs(rowslice=slice(140,148))
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    s='\n  area_arr[140:148,187:197]:\n'  + str(area_arr[140:148,187:197])\
    + '\n  area_arr.shape:'               + str(area_arr.shape)\
    + '\n  size_arrX[140:148,187:197]:\n' + str(size_arrX[140:148,187:197])\
    + '\n  size_arrX.shape:'              + str(size_arrX.shape)\
    + '\n  size_arrY[140:148,187:197]:\n' + str(size_arrY[140:148,187:197])\
    + '\n  size_arrY.shape:'              + str(size_arrY.shape)
    logger.info(s)


  def test_2x2_mask(mbits=0o377):
    pc2x2 = SegGeometryEpixHR2x2V1(use_wide_pix_center=False)
    X, Y = pc2x2.get_seg_xy_maps_pix_with_offset()
    mask = pc2x2.pixel_mask_array(mbits, width=5, wcentral=5)
    mask[mask==0]=3
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=(10,10))
    gg.show()


  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max()'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps()'
    if tname in ('0','3'): s+='\n 3 - test_2x2_img()'
    if tname in ('0','4'): s+='\n 4 - test_2x2_img_easy()'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes()'
    if tname in ('0','6'): s+='\n 6 - test_2x2_mask(mbits=1+2)'
    return s


if __name__ == "__main__":
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname in ('1',): test_xyz_min_max()
    elif tname in ('2',): test_xyz_maps()
    elif tname in ('3',): test_2x2_img()
    elif tname in ('4',): test_2x2_img_easy()
    elif tname in ('5',): test_pix_sizes()
    elif tname in ('6',): test_2x2_mask(mbits=1+2)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

# EOF
