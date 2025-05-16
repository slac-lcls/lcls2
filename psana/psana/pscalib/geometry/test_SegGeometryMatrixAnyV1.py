#!/usr/bin/env python

if __name__ == "__main__":

  from psana.pscalib.geometry.SegGeometryMatrixAnyV1 import *
  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg
  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(filename)s %(message)s', level=logging.DEBUG)

  FIGSIZE_INCH = (14,6)

  def test_xyz_min_max():
    w = SegGeometryMatrixAnyV1()
    w.init_matrix_parameters(shape=(512,1024), pix_size_rcsd_um=(75,75,75,400))
    w.print_xyz_min_max_um()

    s = 'test_xyz_min_max [um]'\
      + '\n  Xmin = %.1f' % w.pixel_coord_min('X')\
      + '    Xmax = %.1f' % w.pixel_coord_max('X')\
      + '\n  Ymin = %.1f' % w.pixel_coord_min('Y')\
      + '    Ymax = %.1f' % w.pixel_coord_max('Y')\
      + '\n  Zmin = %.1f' % w.pixel_coord_min('Z')\
      + '    Zmax = %.1f' % w.pixel_coord_max('Z')
    logger.info(s)


  def test_xyz_maps():

    w = SegGeometryMatrixAnyV1()
    w.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    for i,arr2d in enumerate(w.get_seg_xy_maps_pix()):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=FIGSIZE_INCH, title=titles[i])
        gg.move(200*i,100*i)

    gg.show()

  def test_img():

    w = SegGeometryMatrixAnyV1()
    w.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))

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

    gg.plotImageLarge(H, amp_range=(0, 1100), figsize=(11,10)) # range=(-1, 2),
    gg.move(20,20)
    gg.show()


  def test_img_easy():
    o = SegGeometryMatrixAnyV1()
    o.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))
    X, Y = o.get_seg_xy_maps_pix()
    xmin, xmax, ymin, ymax  = X.min(), X.max(), Y.min(), Y.max()
    Xoff, Yoff = X-xmin, Y-ymin
    iX, iY = (Xoff+0.25).astype(int), (Yoff+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,X+2*Y)
    gg.plotImageLarge(img, amp_range=(xmin+2*ymin, xmax+2*ymax), figsize=FIGSIZE_INCH)
    #gg.move(20,20)
    gg.show()


  def test_pix_sizes():
    w = SegGeometryMatrixAnyV1()
    w.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))
    w.print_pixel_size_arrs()
    size_arr = w.pixel_size_array('X')
    area_arr = w.pixel_area_array()
    s = 'test_pix_sizes\n'\
      + '\n  area_arr[0:10,190:198]:\n' + str(area_arr[0:10,190:198])\
      + '\n  area_arr.shape:'           + str(area_arr.shape)\
      + '\n  size_arr[0:10,190:198]:\n' + str(size_arr[0:10,190:198])\
      + '\n  size_arr.shape:'           + str(size_arr.shape)
    logger.info(s)


  def test_mask(width=0, edge_rows=5, edge_cols=5):
    o = SegGeometryMatrixAnyV1()
    o.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))
    X, Y = o.get_seg_xy_maps_pix_with_offset()
    mask = o.pixel_mask_array(width=width, edge_rows=edge_rows, edge_cols=edge_cols)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,mask)
    gg.plotImageLarge(img, amp_range=(-1, 2), figsize=FIGSIZE_INCH)
    gg.show()


  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0','1'): s+='\n 1 - test_xyz_min_max'
    if tname in ('0','2'): s+='\n 2 - test_xyz_maps'
    if tname in ('0','3'): s+='\n 3 - test_img'
    if tname in ('0','4'): s+='\n 4 - test_img_easy'
    if tname in ('0','5'): s+='\n 5 - test_pix_sizes'
    if tname in ('0','6'): s+='\n 6 - test_mask(width=5)'
    if tname in ('0','7'): s+='\n 7 - test_mask(edge_rows=5, edge_cols=10)'
    return s


if __name__ == "__main__":

    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if   tname=='1': test_xyz_min_max()
    elif tname=='2': test_xyz_maps()
    elif tname=='3': test_img()
    elif tname=='4': test_img_easy()
    elif tname=='5': test_pix_sizes()
    elif tname=='6': test_mask(width=5)
    elif tname=='7': test_mask(width=0, edge_rows=5, edge_cols=10)
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

# EOF
