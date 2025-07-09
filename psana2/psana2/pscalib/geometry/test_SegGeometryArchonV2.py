#!/usr/bin/env python
#import logging
#logger = logging.getLogger(__name__)

from psana2.pscalib.geometry.SegGeometryArchonV2 import *
logger = logging.getLogger(__name__)

class detector_simulator():
    def __init__(self, shape=(150,4800), dtype=np.uint8):
        self._calibconst={'pedestals': [np.empty(shape, dtype=dtype), 'metadata for pedestals'],}

if __name__ == "__main__":

  import sys
  from time import time
import psana2.pyalgos.generic.Graphics as gg # For test purpose in main only
  logging.basicConfig(format='[%(levelname).1s] %(filename)s L%(lineno)04d: %(message)s', level=logging.DEBUG)

  WINDOW = (0.03, 0.06, 0.97, 0.92)

  def test_xyz_min_max():
    w = SegGeometryArchonV2(detector=detector_simulator(shape=(300,4800)))
    w.print_seg_info()
    #w.print_xyz_min_max_um()
    logger.info('\n  Xmin = %8.1f' % w.pixel_coord_min('X')\
              + '    Xmax = %8.1f' % w.pixel_coord_max('X')\
              + '\n  Ymin = %8.1f' % w.pixel_coord_min('Y')\
              + '    Ymax = %8.1f' % w.pixel_coord_max('Y')\
              + '\n  Zmin = %8.1f' % w.pixel_coord_min('Z')\
              + '    Zmax = %8.1f' % w.pixel_coord_max('Z'))


  def test_xyz_maps():
    w = SegGeometryArchonV2(detector=detector_simulator(shape=(150,4800)))
    #w.print_seg_info()
    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]):
    maskf = w.mask_fake(dtype=np.uint8)
    #xy_maps = w.x_pix_arr_um, w.y_pix_arr_um
    xy_maps = maskf * w.x_pix_arr_um, maskf * w.y_pix_arr_um
    amp_range = [(w.pixel_coord_min(axis), w.pixel_coord_max(axis)) for axis in ('X', 'Y')]
    for i,arr2d in enumerate(xy_maps):
        gg.plotImageLarge(arr2d, amp_range=amp_range[i], figsize=(20,4), title=titles[i], window=WINDOW)
        gg.move(200*i,100*i)
        #gg.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
        gg.save_plt(fname='img_map_%s' % ('x','y')[i], verb=True)
    gg.show()


  def test_archon_img():

    t0_sec = time()
    w = SegGeometryArchonV2(detector=detector_simulator())
    #w = SegGeometryArchonV2(use_wide_pix_center=True)
    logger.info('Consumed time for coordinate arrays (sec) = %.3f' % (time()-t0_sec))

    X,Y = w.get_seg_xy_maps_pix()

    w.print_seg_info()

    logger.info('X.shape =' + str(X.shape))

    xmin, ymin, zmin = w.get_xyz_min_um()
    xmax, ymax, zmax = w.get_xyz_max_um()
    xmin /= w._pixsc # w.pixel_scale_size()
    xmax /= w._pixsc
    ymin /= w._pixsr
    ymax /= w._pixsr

    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    logger.info('xsize =' + str(xsize))
    logger.info('ysize =' + str(ysize))


  def test_archon_img_easy():
    from psana2.detector.NDArrUtils import info_ndarr
    w = SegGeometryArchonV2(detector=detector_simulator())

    iX, iY = w.get_seg_xy_maps_pix_with_offset() # contains nans for fake pixels
    cond = w.mask_fake().ravel() > 0
    iX = np.compress(cond, iX.ravel())
    iY = np.compress(cond, iY.ravel())
    print(info_ndarr(iX, 'iX first:'))
    print(info_ndarr(iX, 'iX  last:', first=iX.size-5, last=iX.size))
    print(info_ndarr(iY, 'iY first:'))
    print(info_ndarr(iY, 'iY  last:', first=iY.size-5, last=iY.size))
    img = gg.getImageFromIndexArrays(iY,iX,iX+10*iY)
    print(info_ndarr(img, 'img:'))
    gg.plotImageLarge(img, amp_range=(0, 5000), figsize=(20,4), window=WINDOW)
    gg.show()


  def test_pix_sizes():
    w = SegGeometryArchonV2(detector=detector_simulator())
    #w.print_pixel_size_arrs(rowslice=slice(68,76))
    size_arrX = w.pixel_size_array('X')
    size_arrY = w.pixel_size_array('Y')
    area_arr  = w.pixel_area_array()
    s='\n  area_arr[68:76,380:388]:\n'  + str(area_arr[68:76,380:388])\
    + '\n  area_arr.shape:'               + str(area_arr.shape)\
    + '\n  size_arrX[68:76,380:388]:\n' + str(size_arrX[68:76,380:388])\
    + '\n  size_arrX.shape:'              + str(size_arrX.shape)\
    + '\n  size_arrY[68:76,380:388]:\n' + str(size_arrY[68:76,380:388])\
    + '\n  size_arrY.shape:'              + str(size_arrY.shape)
    logger.info(s)


  def test_archon_mask(width=4, wcenter=2, edge_rows=3, edge_cols=6, center_rows=2, center_cols=4):
    w = SegGeometryArchonV2(detector=detector_simulator())
    iX, iY = w.get_seg_xy_maps_pix_with_offset()
    mask = w.pixel_mask_array() #, dtype=DTYPE_MASK, **kwa)
    #img = gg.getImageFromIndexArrays(iX, iY, weights=mask+1, mask_arr=mask).T
    img = mask
    gg.plotImageLarge(img, amp_range=(0, 2), figsize=(20,4), window=WINDOW)
    gg.show()


def usage():
    import inspect
    return '\n Usage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname in" in s])


def selector():

    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname in ('1',): test_xyz_min_max()
    elif tname in ('2',): test_xyz_maps()
    elif tname in ('3',): test_archon_img()
    elif tname in ('4',): test_archon_img_easy()
    elif tname in ('5',): test_pix_sizes()
    elif tname in ('6',): test_archon_mask()
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage())
    sys.exit('END OF TEST')

if __name__ == "__main__":
    selector()

# EOF
