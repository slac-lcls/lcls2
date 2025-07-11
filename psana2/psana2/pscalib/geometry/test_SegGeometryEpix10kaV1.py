#!/usr/bin/env python

if __name__ == "__main__":

  from psana2.pscalib.geometry.SegGeometryEpix10kaV1 import *
  import sys
  from time import time
import psana2.pyalgos.generic.Graphics as gg # For test purpose in main only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_xyz_min_max():
    w = SegGeometryEpix10kaV1()
    w.print_xyz_min_max_um()
    logger.info('\n  Ymin = %f' % w.pixel_coord_min('Y')\
              + '\n  Ymax = %f' % w.pixel_coord_max('Y'))


  def test_xyz_maps():

    w = SegGeometryEpix10kaV1()
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]):
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(11,10), title=titles[i])
        gg.move(200*i,100*i)

    gg.show()


  def test_2x2_img():

    t0_sec = time()
    w = SegGeometryEpix10kaV1(use_wide_pix_center=False)
    #w = SegGeometryEpix10kaV1(use_wide_pix_center=True)
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
    w = SegGeometryEpix10kaV1(use_wide_pix_center=False)
    X,Y = w.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,iX+2*iY)
    gg.plotImageLarge(img, amp_range=(0, 750), figsize=(11,10))
    gg.show()


  def test_pix_sizes():
    w = SegGeometryEpix10kaV1()
    w.print_pixel_size_arrs()
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    s='\n  area_arr[171:181,187:197]:\n'  + str(area_arr[171:181,187:197])\
    + '\n  area_arr.shape:'               + str(area_arr.shape)\
    + '\n  size_arrX[171:181,187:197]:\n' + str(size_arrX[171:181,187:197])\
    + '\n  size_arrX.shape:'              + str(size_arrX.shape)\
    + '\n  size_arrY[171:181,187:197]:\n' + str(size_arrY[171:181,187:197])\
    + '\n  size_arrY.shape:'              + str(size_arrY.shape)
    logger.info(s)


  def test_2x2_mask(width=4, wcenter=2, edge_rows=3, edge_cols=6, center_rows=2, center_cols=4):
    pc2x2 = SegGeometryEpix10kaV1(use_wide_pix_center=False)
    X, Y = pc2x2.get_seg_xy_maps_pix_with_offset()
    mask = 1+pc2x2.pixel_mask_array(width, wcenter, edge_rows, edge_cols, center_rows, center_cols)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    gg.plotImageLarge(img, amp_range=(0, 2), figsize=(11,10))
    gg.show()


  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max()'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps()'
    if tname in ('0','3'): s+='\n 3 - test_2x2_img()'
    if tname in ('0','4'): s+='\n 4 - test_2x2_img_easy()'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes()'
    if tname in ('0','6'): s+='\n 6 - test_2x2_mask(width=4, wcenter=6)'
    if tname in ('0','7'): s+='\n 7 - test_2x2_mask(width=0, wcenter=0, edge_rows=3, edge_cols=6, center_rows=2, center_cols=4)'
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
    elif tname in ('6',): test_2x2_mask(width=4, wcenter=6)
    elif tname in ('7',): test_2x2_mask(width=0, wcenter=0, edge_rows=10, edge_cols=6, center_rows=2, center_cols=4)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

# EOF
