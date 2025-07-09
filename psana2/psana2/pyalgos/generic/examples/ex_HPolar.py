#!/usr/bin/env python

from psana2.pyalgos.generic.HPolar import * #HPolar

if __name__ == '__main__':

  def data_geo(ntest):
    """Method for tests: returns test data numpy array and geometry object
    """
    from time import time
    from psana2.pscalib.calib.NDArrIO import save_txt, load_txt
    from psana2.pscalib.geometry.GeometryAccess import GeometryAccess

    dir       = '/sdf/group/lcls/ds/ana/detector/alignment/cspad/calib-cxi-camera2-2016-02-05'
    #fname_nda = '%s/nda-water-ring-cxij4716-r0022-e000001-CxiDs2-0-Cspad-0-ave.txt' % dir
    #fname_nda = '%s/nda-water-ring-cxij4716-r0022-e014636-CxiDs2-0-Cspad-0-ave.txt' % dir
    #fname_nda = '%s/nda-lysozyme-cxi02416-r0010-e052421-CxiDs2-0-Cspad-0-max.txt' % dir
    fname_nda = '%s/nda-lysozyme-cxi01516-r0026-e093480-CxiDs2-0-Cspad-0-max.txt'%dir if ntest in (21,28,29,30)\
                else '%s/nda-water-ring-cxij4716-r0022-e014636-CxiDs2-0-Cspad-0-ave.txt'%dir
    fname_geo = '%s/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/geometry/geo-cxi01516-2016-02-18-Ag-behenate-tuned.data' % dir
    #fname_geo = '%s/geo-cxi02416-r0010-2016-03-11.txt' % dir
    fname_gain = '%s/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/pixel_gain/cxi01516-r0016-2016-02-18-FeKalpha.data' % dir

    # load n-d array with averaged water ring
    arr = load_txt(fname_nda)
    #arr *= load_txt(fname_gain)
    #print_ndarr(arr,'water ring')
    arr.shape = (arr.size,) # (32*185*388,)

    # retrieve geometry
    t0_sec = time()
    geo = GeometryAccess(fname_geo)
    geo.move_geo('CSPAD:V1', 0, 1600, 0, 0)
    geo.move_geo('QUAD:V1', 2, -100, 0, 0)
    #geo.get_geo('QUAD:V1', 3).print_geo()
    print('Time to load geometry %.3f sec from file\n%s' % (time()-t0_sec, fname_geo))

    return arr, geo


  def usage(ntest=None):
    s = ''
    if ntest is None     : s+='\n Tests for radial 1-d binning of entire image'
    if ntest in (None, 1): s+='\n  1 - averaged data'
    if ntest in (None, 2): s+='\n  2 - pixel radius value'
    if ntest in (None, 3): s+='\n  3 - pixel phi value'
    if ntest in (None, 4): s+='\n  4 - pixel radial bin index'
    if ntest in (None, 5): s+='\n  5 - pixel phi bin index'
    if ntest in (None, 6): s+='\n  6 - pixel sequential (rad and phi) bin index'
    if ntest in (None, 7): s+='\n  7 - mask'
    if ntest in (None, 8): s+='\n  8 - averaged radial-phi intensity'
    if ntest in (None, 9): s+='\n  9 - interpolated radial intensity'

    if ntest is None     : s+='\n Test for 2-d (default) binning of the rad-phi range of entire image'
    if ntest in (None,21): s+='\n 21 - averaged data'
    if ntest in (None,24): s+='\n 24 - pixel radial bin index'
    if ntest in (None,25): s+='\n 25 - pixel phi bin index'
    if ntest in (None,26): s+='\n 26 - pixel sequential (rad and phi) bin index'
    if ntest in (None,28): s+='\n 28 - averaged radial-phi intensity'
    if ntest in (None,29): s+='\n 29 - averaged radial-phi interpolated intensity'
    if ntest in (None,30): s+='\n 30 - r-phi'

    if ntest is None     : s+='\n Test for 2-d binning of the restricted rad-phi range of entire image'
    if ntest in (None,41): s+='\n 41 - averaged data'
    if ntest in (None,44): s+='\n 44 - pixel radial bin index'
    if ntest in (None,45): s+='\n 45 - pixel phi bin index'
    if ntest in (None,46): s+='\n 46 - pixel sequential (rad and phi) bin index'
    if ntest in (None,48): s+='\n 48 - averaged radial-phi intensity'
    if ntest in (None,49): s+='\n 49 - averaged radial-phi interpolated intensity'
    if ntest in (None,50): s+='\n 50 - r-phi'

    return s


  def test01(ntest, prefix='fig-v01'):
    """Test for radial 1-d binning of entire image.
    """
    from time import time
import psana2.pyalgos.generic.Graphics as gg
    from psana2.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    t0_sec = time()
    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten()
    print('Time to retrieve geometry %.3f sec' % (time()-t0_sec))

    t0_sec = time()
    hp = HPolar(X, Y, mask, nradbins=500, nphibins=1) # v1
    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 1: nda, title = arr,                   'averaged data'
    elif ntest == 2: nda, title = hp.pixel_rad(),        'pixel radius value'
    elif ntest == 3: nda, title = hp.pixel_phi(),        'pixel phi value'
    elif ntest == 4: nda, title = hp.pixel_irad() + 2,   'pixel radial bin index'
    elif ntest == 5: nda, title = hp.pixel_iphi() + 1,   'pixel phi bin index'
    elif ntest == 6: nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    elif ntest == 7: nda, title = mask,                  'mask'
    elif ntest == 8: nda, title = hp.pixel_avrg(nda),    'averaged radial intensity'
    elif ntest == 9: nda, title = hp.pixel_avrg_interpol(arr) * mask , 'interpolated radial intensity'
    else:
        print('Test %d is not implemented' % ntest)
        return

    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (21,) else nda[100:300,:]

    da, ds = None, None
    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter'
    if ntest in (2,3,4,5,6,7):
        da = ds = (nda.min()-1., nda.max()+1.)

    else:
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print('End of test for %s' % title)


  def test02(ntest, prefix='fig-v01'):
    """Test for 2-d (default) binning of the rad-phi range of entire image
    """
    #from Detector.GlobalUtils import print_ndarr
    from time import time
import psana2.pyalgos.generic.Graphics as gg
    from psana2.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten()

    t0_sec = time()
    #hp = HPolar(X, Y, mask) # v0
    hp = HPolar(X, Y, mask, nradbins=500) # , nphibins=8, phiedges=(-20, 240), radedges=(10000,80000))
    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 21: nda, title = arr,                   'averaged data'
    elif ntest == 24: nda, title = hp.pixel_irad() + 2,   'pixel radial bin index'
    elif ntest == 25: nda, title = hp.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 26: nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    #elif ntest == 27: nda, title = mask,                  'mask'
    elif ntest == 28: nda, title = hp.pixel_avrg(nda),    'averaged radial intensity'
    elif ntest == 29: nda, title = hp.pixel_avrg_interpol(nda) * mask, 'averaged radial interpolated intensity'
    elif ntest == 30: nda, title = hp.bin_avrg_rad_phi(nda),'r-phi'
    else:
        print('Test %d is not implemented' % ntest)
        return

    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (30,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (21,28,29,30):
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print('End of test for %s' % title)


  def test03(ntest, prefix='fig-v01'):
    """Test for 2-d binning of the restricted rad-phi range of entire image
    """
    from time import time
import psana2.pyalgos.generic.Graphics as gg
    from psana2.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten()

    t0_sec = time()

    #hp = HPolar(X, Y, mask, nradbins=5, nphibins=8, phiedges=(-20, 240), radedges=(10000,80000))
    hp = HPolar(X, Y, mask, nradbins=3, nphibins=8, phiedges=(240, -20), radedges=(80000,10000)) # v3

    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    #print('bin_number_of_pixels:',   hp.bin_number_of_pixels())
    #print('bin_intensity:', hp.bin_intensity(arr))
    #print('bin_avrg:',   hp.bin_avrg(arr))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 41: nda, title = arr,                   'averaged data'
    elif ntest == 44: nda, title = hp.pixel_irad() + 2,   'pixel radial bin index'
    elif ntest == 45: nda, title = hp.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 46: nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    #elif ntest == 47: nda, title = mask,                  'mask'
    elif ntest == 48: nda, title = hp.pixel_avrg(nda, subs_value=180), 'averaged radial intensity'
    elif ntest == 49: nda, title = hp.pixel_avrg_interpol(nda, verb=True) * mask, 'averaged radial interpolated intensity'
    elif ntest == 50: nda, title = hp.bin_avrg_rad_phi(nda),'r-phi'
    else:
        print('Test %d is not implemented' % ntest)
        return

    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (50,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (41,48,49,50):
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print('End of test for %s' % title)


if __name__ == '__main__':
    import sys

    #if len(sys.argv) == 1: print(usage())
    print(usage())

    ntest = int(sys.argv[1]) if len(sys.argv)>1 else 1
    print('Test # %d: %s' % (ntest, usage(ntest)))

    prefix = 'fig-v01-cspad-HPolar'

    if   ntest<20: test01(ntest, prefix)
    elif ntest<40: test02(ntest, prefix)
    elif ntest<60: test03(ntest, prefix)
    else: print('Test %d is not implemented' % ntest)
    #sys.exit('End of test')

# EOF
