#!/usr/bin/env python

if __name__ == "__main__":

  import logging
  logger = logging.getLogger(__name__)
  #from psana.pscalib.geometry.SegGeometryEpixUHR3x2V1 import *
  from psana.pscalib.geometry.SegGeometryStore import sgs

  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gg # For test purpose in main only

  logging.basicConfig(format='[%(levelname).1s] %(filename)s L%(lineno)04d: %(message)s', level=logging.DEBUG)


  SEGNAME = 'EPIXUHR3X2:V1'
  WINDOW = (0.03, 0.06, 0.97, 0.92)

  def test_xyz_min_max():
    #w = SegGeometryEpixM320V1()
    #w = sgs.Create(segname='MTRX:V2:168,192:100:100') # EPIXM ASIC
    w = sgs.Create(segname='EPIXUHR3X2:V1')
    w.print_xyz_min_max_um()
    logger.info('\n  Ymin = %f' % w.pixel_coord_min('Y')\
              + '\n  Ymax = %f' % w.pixel_coord_max('Y'))


  def test_xyz_maps():
    w = sgs.Create(segname=SEGNAME)
    w.print_maps_seg_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]):
    for i,arr2d in enumerate( w.get_seg_xy_maps_pix() ):
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i], window=WINDOW)
        gg.move(200*i,100*i)

    gg.show()


  def test_seg_img():

    t0_sec = time()
    w = sgs.Create(segname=SEGNAME)
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


  def test_seg_img_easy():
    w = sgs.Create(segname=SEGNAME)
    X,Y = w.get_seg_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iY,iX,iX+2*iY)
    gg.plotImageLarge(img, amp_range=(0, 750), figsize=(10,5), window=WINDOW)
    gg.show()


  def test_pix_sizes():
    w = sgs.Create(segname=SEGNAME)
    w.print_pixel_size_arrs() #rowslice=slice(68,76))
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    s='\n  area_arr[68:76,180:183]:\n'  + str(area_arr[68:76,180:183])\
    + '\n  area_arr.shape:'               + str(area_arr.shape)\
    + '\n  size_arrX[68:76,180:183]:\n' + str(size_arrX[68:76,180:183])\
    + '\n  size_arrX.shape:'              + str(size_arrX.shape)\
    + '\n  size_arrY[68:76,180:183]:\n' + str(size_arrY[68:76,180:183])\
    + '\n  size_arrY.shape:'              + str(size_arrY.shape)
    logger.info(s)


  def test_seg_mask(width=4, wcenter=4, edge_rows=3, edge_cols=6, center_rows=2, center_cols=4):
    #pcseg = SegGeometryEpixM320V1(use_wide_pix_center=False)
    #X, Y = pcseg.get_seg_xy_maps_pix_with_offset()
    #mask = 1+pcseg.pixel_mask_array(width, wcenter, edge_rows, edge_cols, center_rows, center_cols)#, dtype=DTYPE_MASK, **kwa)
    w = sgs.Create(segname=SEGNAME)
    X, Y = w.get_seg_xy_maps_pix_with_offset()
    mask = 1+w.pixel_mask_array(width=width, wcenter=wcenter, edge_rows=edge_rows, edge_cols=edge_cols,\
                                center_rows=center_rows, center_cols=center_cols)#, dtype=DTYPE_MASK, **kwa)
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,mask).T
    gg.plotImageLarge(img, amp_range=(0, 2), figsize=(10,5), window=WINDOW)
    gg.show()


if __name__ == "__main__":
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    SCRNAME = sys.argv[0].rsplit('/')[-1]

    def selector():
        tname = sys.argv[1] if len(sys.argv) > 1 else '0'
        if len(sys.argv)==1: logger.info(usage())
        elif tname in ('1',): test_xyz_min_max()
        elif tname in ('2',): test_xyz_maps()
        elif tname in ('3',): test_seg_img()
        elif tname in ('4',): test_seg_img_easy()
        elif tname in ('5',): test_pix_sizes()
        elif tname in ('6',): test_seg_mask(width=4, wcenter=6)
        elif tname in ('7',): test_seg_mask(width=0, wcenter=0, edge_rows=10, edge_cols=6) #, center_rows=2, center_cols=4)
        else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, USAGE()))
        if len(sys.argv)>1: print(USAGE())
        sys.exit('END OF TEST')


    def USAGE():
        import inspect
        return '\n  %s <tname>\n' % sys.argv[0].split('/')[-1]\
             + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname in" in s])\
             + '\n\n  HELP:\n    list of parameters: ./%s -h\n    list of tests:      ./%s' % (SCRNAME, SCRNAME)

if __name__ == "__main__":
    if len(sys.argv)==1:
        print(USAGE())
        exit('\nMISSING ARGUMENTS -> EXIT')
    selector()

# EOF
