#!/usr/bin/env python

if __name__ == "__main__":
  from psana2.pscalib.geometry.GeometryAccess import *

  from time import time # for test purpose only
import psana2.pyalgos.generic.Graphics as gg # for test purpose
  from psana2.pyalgos.generic.NDArrGenerators import cspad_ndarr # for test purpose only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)
  #logger.setLevel(logging.DEBUG)
  #logger.getEffectiveLevel()
  #logger.root.level


  def test_access(geometry):
    """ Tests geometry acess methods of the class GeometryAccess.
    """
    geometry.print_list_of_geos()
    geometry.print_list_of_geos_children()

    logger.info('TOP GEO:')
    top_geo = geometry.get_top_geo()
    top_geo.print_geo_children()

    logger.info('INTERMEDIATE GEO (QUAD):')
    geo = geometry.get_geo('QUAD:V1', 0)
    #geo = geometry.get_top_geo()
    geo.print_geo_children()

    t0_sec = time()
    X,Y,Z = geo.get_pixel_coords(do_tilt=True)
    #X,Y = geo.get_2d_pixel_coords()
    s = 'X: %s' % str(X)
    s+= '\n  Consumed time to get 3d pixel coordinates = %7.3f sec' % (time()-t0_sec)
    s+= '\n  Geometry object: %s:%d X.shape:%s' % (geo.oname, geo.oindex, str(X.shape))
    logger.info(s)

    logger.info('Test of print_pixel_coords() for quad:')
    geometry.print_pixel_coords('QUAD:V1', 1)
    logger.info('Test of print_pixel_coords() for CSPAD:')
    geometry.print_pixel_coords()

    s = 'Test of get_pixel_areas() for QUAD:'
    A = geo.get_pixel_areas()
    s+= '\n  Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    s+= '\n  A[0,0:5,190:198]:\n' + str(A[0,0:5,190:198])
    logger.info(s)

    s = 'Test of get_pixel_areas() for CSPAD:'
    A = top_geo.get_pixel_areas()
    s+= '\n  Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    s+= '\n  A[0,0,0:5,190:198]:\n' + str(A[0,0,0:5,190:198])
    logger.info(s)

    s = 'Test of get_size_geo_array()'
    s+= '\n  for QUAD: %d' % geo.get_size_geo_array()
    s+= '\n  for CSPAD: %d' % top_geo.get_size_geo_array()
    logger.info(s)

    s = 'Test of get_pixel_scale_size()'
    s+= '\n  for QUAD    : %8.2f' % geo.get_pixel_scale_size()
    s+= '\n  for CSPAD   : %8.2f' % top_geo.get_pixel_scale_size()
    s+= '\n  for geometry: %8.2f' % geometry.get_pixel_scale_size()
    logger.info(s)

    s = 'Test of get_dict_of_comments():'
    d = geometry.get_dict_of_comments()
    s+= '\n  d[0] = %s' % str(d[0])
    logger.info(s)


  def test_plot_quad(geometry):
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad.
    """
    ## get index arrays
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True)

    # get intensity array
    arr = cspad_ndarr(n2x1=rows.shape[0])
    arr.shape = (8,185,388)
    amp_range = (0,185+388)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


  def test_mask_quad(geometry, mbits):
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad.
    """
    ## get index arrays
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True)

    # get intensity array
    arr = geometry.get_pixel_mask('QUAD:V1', 1, mbits)
    arr.shape = (8,185,388)
    amp_range = (-1,2)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows, cols, W=arr, vbase=0.5)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


  def test_plot_cspad(geometry, fname_data, amp_range=(0,0.5)):
    """ The same test as previous, but use get_pixel_coord_indexes(...) method.
    """
    #rad1 =  93
    #rad2 = 146
    rad1 = 655
    rad2 = 670

    # get pixel coordinate index arrays:
    xyc = xc, yc = 500, 500# None

    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=None)
    rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc, do_tilt=True)

    ixo, iyo = geometry.point_coord_indexes(xy0_off_pix=xyc, do_tilt=True)
    logger.info('Detector origin indexes ixo:%d iyo:%d' % (ixo, iyo))

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float32)
    arr.shape= (4,8,185,388)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))

    arr.shape = rows.shape
    img = img_from_pixel_arrays(rows, cols, W=arr)

    rcc_ring = (iyo, ixo)
    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.drawCircle(axim, rcc_ring, rad1, linewidth=1, color='w', fill=False)
    gg.drawCircle(axim, rcc_ring, rad2, linewidth=1, color='w', fill=False)
    gg.drawCenter(axim, rcc_ring, rad1, linewidth=1, color='w')
    gg.move(500,10)
    gg.show()


  def test_img_default():
    """ Test default image.
    """
    axim = gg.plotImageLarge(img_default())
    gg.move(500,10)
    gg.show()


  def test_init_is_silent():
    logger.info('Init GeometryAccess is silentin INFO level? (see below)')
    logger.setLevel(logging.INFO)
    ga0 = GeometryAccess(fname_geometry)


  def test_save_pars_in_file(geometry):
    """ Test default image.
    """
    # change one line of parameters
    x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x = -3500, 5800, 0, 0.123, 0.123, 0.123, 1, 2, 3
    geometry.set_geo_pars('QUAD:V1', 1, x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)

    geometry.set_print_bits(32)
    fname = './test.txt'
    geometry.save_pars_in_file(fname)
    logger.info('saved file %s' % fname)


  def test_load_pars_from_file(geometry):
    """ Test default image.
    """
    geometry.set_print_bits(32+64)
    geometry.load_pars_from_file('./test.txt')
    geometry.print_list_of_geos()


  def test_cspad2x2():
    """ Test cspad2x2 geometry table.
    """
    basedir = '/reg/g/psdm/detector/alignment/cspad2x2/calib-cspad2x2-01-2013-02-13/'
    fname_geometry = basedir + 'calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data'
    fname_data     = basedir + 'cspad2x2.1-ndarr-ave-meca6113-r0028.dat'

    geometry = GeometryAccess(fname_geometry, pbits=0o377, use_wide_pix_center=False)
    amp_range = (0,15000)

    # get pixel coordinate index arrays:
    #xyc = xc, yc = 1000, 1000
    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)

    rows, cols = geometry.get_pixel_coord_indexes(do_tilt=True)

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float32)
    arr.shape= (185,388,2)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


  def test_epix100a():
    """ Test test_epix100a geometry table.
    """
    basedir = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.txt'

    #basedir = '/reg/neh/home1/dubrovin/LCLS/GeometryCalib/calib-xpp-Epix100a-2014-11-05/'
    #fname_geometry = basedir + 'calib/Epix100a::CalibV1/NoDetector.0:Epix100a.0/geometry/0-end.data'
    #fname_data     = basedir + 'epix100a-ndarr-ave-clb-xppi0614-r0073.dat'

    geometry = GeometryAccess(fname_geometry, pbits=0o377)
    amp_range = (-4,10)

    rows, cols = geometry.get_pixel_coord_indexes()

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float32)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


  def test_cspad_xy_at_z():
    """ Test cspad geometry table.
    """
    ## 'CxiDs1.0:Cspad.0)' or 'DscCsPad'
    basedir = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-camera1-2014-09-24/'
    fname_geometry = basedir + '2016-06-03-geometry-cxi06216-r25-camera1-z175mm.txt'
    fname_data     = basedir + '2016-06-03-chun-cxi06216-0025-DscCsPad-max.txt'

    geometry = GeometryAccess(fname_geometry, pbits=0o377)

    # get pixel coordinate index arrays:
    xyc = xc, yc = 1000, 1000
    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)
    #rows, cols = geometry.get_pixel_coord_indexes(do_tilt=True)
    #rows, cols = geometry.get_pixel_xy_inds_at_z(zplane=None, xy0_off_pix=xyc)
    rows, cols = geometry.get_pixel_xy_inds_at_z(zplane=150000)

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float32)

    #logger.info('arr.shape=', arr.shape
    arr.shape= (32,185,388)

    #ave, rms = arr.mean(), arr.std()
    #amp_range = (ave-rms, ave+3*rms)
    amp_range = (0, 1000)
    logger.info('amp_range:' + str(amp_range))

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0', '1'): s+='\n 1 - test_access(geometry)'
    if tname in ('0', '2'): s+='\n 2 - test_plot_quad(geometry)'
    if tname in ('0', '3'): s+='\n 3 - test_plot_cspad(geometry, fname_data, amp_range)'
    if tname in ('0', '4'): s+='\n 4 - test_img_default()'
    if tname in ('0', '5'): s+='\n 5 - test_init_is_silent()'
    if tname in ('0', '6'): s+='\n 6 - ga0377 = GeometryAccess(fname_geometry, pbits=0o377)'
    if tname in ('0', '7'): s+='\n 7 - test_save_pars_in_file(geometry)'
    if tname in ('0', '8'): s+='\n 8 - test_load_pars_from_file(geometry)'
    if tname in ('0', '9'): s+='\n 9 - test_mask_quad(geometry, 1+2+8)'
    if tname in ('0','10'): s+='\n10 - geometry.print_psf()'
    if tname in ('0','11'): s+='\n11 - test_cspad2x2()'
    if tname in ('0','12'): s+='\n12 - test_epix100a()'
    if tname in ('0','13'): s+='\n13 - geometry.print_comments_from_dict()'
    if tname in ('0','14'): s+='\n14 - test_cspad_xy_at_z()'
    return s


if __name__ == "__main__":

    import sys
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    ##fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    #amp_range = (0,0.5)

    # CXI
    basedir = '/sdf/group/lcls/ds/ana/detector/alignment/cspad/calib-cxi-ds1-2014-03-19/'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #fname_geometry = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    amp_range = (0,500)

    ## XPP
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp-2013-01-29/'
    #fname_data     = basedir + 'cspad-xpptut13-r1437-nda.dat'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/XppGon.0:Cspad.0/geometry/0-end.data'
    #amp_range = (1500,2500)


    #basedir = '/home/pcds/LCLS/calib/geometry/'
    #fname_geometry = basedir + '0-end.data'
    #fname_geometry = basedir + '2-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    #fname_geometry = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #amp_range = (0,500)

    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    #fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.npy'
    #amp_range = (0,500)

    logger.info('%s\nfname_geometry: %s\nfname_data: %s' %(120*'_', fname_geometry, fname_geometry))

    geometry = GeometryAccess(fname_geometry)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname=='1': test_access(geometry)
    elif tname=='2': test_plot_quad(geometry)
    elif tname=='3': test_plot_cspad(geometry, fname_data, amp_range)
    elif tname=='4': test_img_default()
    elif tname=='5': test_init_is_silent()
    elif tname=='6': ga0377 = GeometryAccess(fname_geometry, pbits=0o377)
    elif tname=='7': test_save_pars_in_file(geometry)
    elif tname=='8': test_load_pars_from_file(geometry)
    elif tname=='9': test_mask_quad(geometry, 1+2+8)
    elif tname=='10': geometry.print_psf()
    elif tname=='11': test_cspad2x2()
    elif tname=='12': test_epix100a()
    elif tname=='13': geometry.print_comments_from_dict()
    elif tname=='14': test_cspad_xy_at_z()
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

# EOF
