#!/usr/bin/env python

if __name__ == "__main__":

  import sys
  from time import time
  from psana.pscalib.geometry.UtilsPSF import *
  from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays, GeometryAccess
  from psana.detector.NDArrUtils import info_ndarr

  logging.basicConfig(format='[%(levelname).1s] %(filename)s L%(lineno)04d: %(message)s', level=logging.DEBUG)

  scrname = sys.argv[0].rsplit('/')[-1]
  tname = sys.argv[1] if len(sys.argv)>1 else '1'

  fn_geo_cspad_cxi      = '/sdf/group/lcls/ds/ana/detector/data2_test//geometry/geo-cspad-cxi.data'
  fn_geo_cspad_xpp      = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-cspad-xpp.data'
  fn_geo_epix10ka2m     = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-mfxc00118-r0183.data'
  fn_geo_epix10ka2m_def = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-default.data'
  fn_geo_jungfrau_8     = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-jungfrau-8-segment-cxilv9518.data'
  fn_geo_pnccd_amo      = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-pnccd-amo.data'
  fn_geo_rayonix_1920   = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-rayonix-nbins1920-1920.data'

  fn_data_epix10ka2m   = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-mfxc00118-r0184-epix10ka2m-silver-behenate-max.txt'
  fn_data_cspad        = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-mfx11116-r0624-e005365-MfxEndstation-0-Cspad-0-max.txt'
  fn_data_jungfrau     = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-cxilv9518-r0008-jungfrau-lysozyme-max.npy'
  fn_data_pnccd        = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-amo86615-r0159-Camp-0-pnCCD-1-evt-w-rings.txt'
  fn_data_rayonix_1920 = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-mfxlw7519-r0070-rayonix-max.npy'

  def load_data_from_file(fname):
    assert isinstance(fname, str), 'file name is not a str'
    if fname.split('.')[-1]=='npy': return np.load(fname)
    else:
        from psana.pscalib.calib.NDArrIO import load_txt
        return load_txt(fname)


  def test_geo_from_file(fname_geo):
    geo = GeometryAccess(fname_geo)
    X, Y, Z = geo.get_pixel_coords()
    logger.info(info_ndarr(X,'X:'))
    logger.info(info_ndarr(Y,'Y:'))
    logger.info(info_ndarr(Z,'Z:'))


  def test_psf_from_file(fname_geo):
    psf, sego, geo = psf_from_file(fname_geo)
    logger.info(type(sego))
    logger.info(info_psf(psf, title='info_psf: psf.shape: %s \npsf vectors:' % (str(np.array(psf).shape))))
    savetext_psf(psf, fname='psf-test.txt')
    save_psf(psf, fname='psf-test.npy')


  def test_load_psf(fname='psf-test.npy'):
    psf = load_psf(fname)
    logger.info(info_psf(psf, title='info_psf: psf.shape: %s \npsf vectors:' % (str(np.array(psf).shape))))


  def test_load_data_from_file(fname=fn_data_epix10ka2m):
    data = load_data_from_file(fname)
    logger.info(info_ndarr(data, name='data', first=0, last=10))


  def test_psf_methods(fname_geo, fname_data):
    psf, sego, geo = psf_from_file(fname_geo)
    logger.info(type(sego))
    #logger.info(info_seg_geo(sego))
    logger.info(info_psf(psf, title='info_psf: psf.shape: %s \npsf vectors:' % (str(np.array(psf).shape))))

    data = load_data_from_file(fname_data)
    logger.info(info_ndarr(data, name='data', first=0, last=10))

    t0_sec = time()
    datapsf = data_psf(sego, data)
    logger.info(info_ndarr(datapsf,'data_psf consumed time=%.6fs:' % (time()-t0_sec)))

    t0_sec = time()
    resp = pixel_coords_psf(psf, sego.asic_rows_cols())
    logger.info(info_ndarr(resp, 'pixel_coords_psf consumed time=%.6fs:' % (time()-t0_sec)))
    x,y,z = resp
    logger.info(info_ndarr(x, 'pixel_coords_psf x', first=0, last=4))
    logger.info(info_ndarr(y, 'pixel_coords_psf y', first=0, last=4))
    logger.info(info_ndarr(z, 'pixel_coords_psf z', first=0, last=4))


  def test_psana_image(geo, data):

    logger.info('======== test_psana_image ========')

    #data = load_data_from_file(fname_data)
    logger.info(info_ndarr(data, name='data', first=0, last=4))

    #geo = GeometryAccess(fname_geo)
    x,y,z = geo.get_pixel_coords(cframe=0)

    ix = indices(x.ravel(), geo.get_pixel_scale_size(), offset=None)
    iy = indices(y.ravel(), geo.get_pixel_scale_size(), offset=None)
    arr = data.ravel()
    img = img_from_pixel_arrays(ix, iy, W=arr)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from psana.detector.UtilsGraphics import gr, fleximagespec, fleximage, flexhist #, axis_plot
    flims = fleximagespec(img, arr=arr, bins=100, w_in=16, h_in=12, fraclo=0.01, frachi=0.99)
    flims.move(500,5)
    flims.axtitle('psana image')
    gr.show()
    flims.save('fig-psana.png')


  def ascending_nda(shape, amin=0, dtype=np.float32):
    assert len(shape)==3
    nsegs, rows, cols = shape
    ash = (rows, cols)
    norm = 1
    arows = np.arange(ash[0], dtype=dtype) * norm
    acols = np.arange(ash[1], dtype=dtype) * norm
    grid = np.meshgrid(acols,arows)
    arr2d = 0.3*grid[0] + 0.7*grid[1] + amin # cols change color faster
    return np.array([arr2d for ns in range(nsegs)])


  def test_psf_graph(fname_geo, fname_data):
    psf, sego, geo = psf_from_file(fname_geo)

    #data = ascending_nda((32, 185, 194*2))
    data = load_data_from_file(fname_data)
    logger.info(info_ndarr(data, name='data', first=0, last=4))

    t0_sec = time()
    datapsf = data_psf(sego, data)
    #datapsf = ascending_nda((64, 185, 194))
    logger.info(info_ndarr(datapsf,'data_psf consumed time=%.6fs:' % (time()-t0_sec)))

    t0_sec = time()
    resp = pixel_coords_psf(psf, sego.asic_rows_cols())
    logger.debug(info_ndarr(resp,'pixel_coords_psf, evaluation time=%.6fs:' % (time()-t0_sec), first=0, last=4))
    x,y,z = resp
    logger.info(info_ndarr(x, 'pixel_coords_psf x', first=0, last=4))
    logger.info(info_ndarr(y, 'pixel_coords_psf y', first=0, last=4))
    logger.info(info_ndarr(z, 'pixel_coords_psf z', first=0, last=4))

    ix = indices(x.ravel(), sego.pixel_scale_size(), offset=None)
    iy = indices(y.ravel(), sego.pixel_scale_size(), offset=None)
    arr = datapsf.ravel()
    img = img_from_pixel_arrays(ix, iy, W=arr)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    #from Detector.UtilsGraphics import gr, fleximagespec, fleximage, flexhist, axis_plot
    from psana.detector.UtilsGraphics import gr, fleximagespec, fleximage, flexhist #, axis_plot
    flims = fleximagespec(img, arr=arr, bins=100, w_in=16, h_in=12, fraclo=0.01, frachi=0.99)
    flims.move(5,5)
    flims.axtitle('psf image')
    gr.show(mode=1)
    flims.save('fig-psf.png')

    test_psana_image(geo, data)


def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if 'TNAME ==' in s])


def selector():
  TNAME = sys.argv[1] if len(sys.argv)>1 else 'USAGE'

  if   TNAME == '0': test_geo_from_file(fn_geo_cspad_cxi)                             # test_geo_from_file(fn_geo_cspad_cxi)
  elif TNAME == '1': test_psf_from_file(fn_geo_cspad_cxi)                             # test_psf_from_file(fn_geo_cspad_cxi)
  elif TNAME == '2': test_psf_from_file(fn_geo_cspad_xpp)                             # test_psf_from_file(fn_geo_cspad_xpp)
  elif TNAME == '3': test_psf_from_file(fn_geo_epix10ka2m)                            # test_psf_from_file(fn_geo_epix10ka2m)
  elif TNAME == '4': test_psf_from_file(fn_geo_jungfrau_8)                            # test_psf_from_file(fn_geo_jungfrau_8)
  elif TNAME == '5': test_psf_from_file(fn_geo_pnccd_amo)                             # test_psf_from_file(fn_geo_pnccd_amo)
  elif TNAME == '6': test_psf_from_file(fn_geo_rayonix_1920)                          # test_psf_from_file(fn_geo_rayonix_1920)
  elif TNAME == '10': test_load_psf()                                                 # test_load_psf() - uses file psf-test.npy generated after an

  elif TNAME == '12': test_load_data_from_file(fn_data_cspad)                         # test_load_data_from_file(fn_data_cspad)
  elif TNAME == '13': test_load_data_from_file(fn_data_epix10ka2m)                    # test_load_data_from_file(fn_data_epix10ka2m)
  elif TNAME == '14': test_load_data_from_file(fn_data_jungfrau)                      # test_load_data_from_file(fn_data_jungfrau)
  elif TNAME == '15': test_load_data_from_file(fn_data_pnccd)                         # test_load_data_from_file(fn_data_pnccd)
  elif TNAME == '16': test_load_data_from_file(fn_data_rayonix_1920)                  # test_load_data_from_file(fn_data_rayonix_1920)

  elif TNAME == '22': test_psf_methods(fn_geo_cspad_xpp,    fn_data_cspad)            # test_psf_methods(fn_geo_cspad_xpp,    fn_data_cspad)
  elif TNAME == '23': test_psf_methods(fn_geo_epix10ka2m,   fn_data_epix10ka2m)       # test_psf_methods(fn_geo_epix10ka2m,   fn_data_epix10ka2m)
  elif TNAME == '24': test_psf_methods(fn_geo_jungfrau_8,   fn_data_jungfrau)         # test_psf_methods(fn_geo_jungfrau_8,   fn_data_jungfrau)
  elif TNAME == '25': test_psf_methods(fn_geo_pnccd_amo,    fn_data_pnccd)            # test_psf_methods(fn_geo_pnccd_amo,    fn_data_pnccd)
  elif TNAME == '26': test_psf_methods(fn_geo_rayonix_1920, fn_data_rayonix_1920)     # test_psf_methods(fn_geo_rayonix_1920, fn_data_rayonix_1920)

  elif TNAME == '32': test_psf_graph(fn_geo_cspad_xpp,    fn_data_cspad)              # test_psf_graph(fn_geo_cspad_xpp,    fn_data_cspad)
  elif TNAME == '33': test_psf_graph(fn_geo_epix10ka2m,   fn_data_epix10ka2m)         # test_psf_graph(fn_geo_epix10ka2m,   fn_data_epix10ka2m)
  elif TNAME == '34': test_psf_graph(fn_geo_jungfrau_8,   fn_data_jungfrau)           # test_psf_graph(fn_geo_jungfrau_8,   fn_data_jungfrau)
  elif TNAME == '35': test_psf_graph(fn_geo_pnccd_amo,    fn_data_pnccd)              # test_psf_graph(fn_geo_pnccd_amo,    fn_data_pnccd)
  elif TNAME == '36': test_psf_graph(fn_geo_rayonix_1920, fn_data_rayonix_1920)       # test_psf_graph(fn_geo_rayonix_1920, fn_data_rayonix_1920)

  else:
    print(USAGE())
    print('NON-IMPLEMENTED TEST: %s' % TNAME)

if __name__ == "__main__":
    selector()
    print(sys.exit('End of test %s' % scrname))
# EOF
