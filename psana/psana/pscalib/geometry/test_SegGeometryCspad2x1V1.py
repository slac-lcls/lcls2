#!/usr/bin/env python

from psana.pscalib.geometry.SegGeometryCspad2x1V1 import *

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


  def test_xyz_maps():
    w = SegGeometryCspad2x1V1()
    w.print_maps_seg_um()
    titles = ['X map','Y map']
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,6), title=titles[i])
        gg.move(200*i,100*i)
    gg.show()


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

    gg.plotImageLarge(H, amp_range=(-250, 250), figsize=(7,10)) # range=(-1, 2),
    gg.show()


  def test_2x1_img_easy():
    pc2x1 = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    #X,Y = pc2x1.get_seg_xy_maps_pix()
    X,Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,iX+iY)
    gg.plotImageLarge(img, amp_range=(0, 500), figsize=(7,10))
    gg.show()


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


  def test_2x1_mask(mbits=0o7, width=0, wcenter=0, edge_rows=10, edge_cols=6, center_cols=4):
    pc2x1 = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    X, Y = pc2x1.get_seg_xy_maps_pix_with_offset()
    mask = 1 + pc2x1.pixel_mask_array(mbits, width, wcenter, edge_rows, edge_cols, center_cols)
    s = '\n  mask:\n%s' % str(mask)\
      + '\n  mask.shape: %s' % str(mask.shape)
    logger.info(s)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask)
    gg.plotImageLarge(img, amp_range=(0, 2), figsize=(7,10))
    gg.show()


  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max()'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps()'
    if tname in ('0','3'): s+='\n 3 - test_img()'
    if tname in ('0','4'): s+='\n 4 - test_img_easy()'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes()'
    if tname in ('0','6'): s+='\n 6 - test_mask(mbits=4+8, width=4, wcenter=6, mbits=4+8)'
    if tname in ('0','7'): s+='\n 7 - test_mask(width=0, wcenter=0, edge_rows=10, center_cols=8, mbits=4)'
    return s


if __name__ == "__main__":
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname=='1': test_xyz_min_max()
    elif tname=='2': test_xyz_maps()
    elif tname=='3': test_2x1_img()
    elif tname=='4': test_2x1_img_easy()
    elif tname=='5': test_pix_sizes()
    elif tname=='6': test_2x1_mask(width=8, wcenter=4, mbits=4+8)
    elif tname=='7': test_2x1_mask(width=0, wcenter=0, edge_rows=10, edge_cols=5, center_cols=8, mbits=4)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

# EOF
