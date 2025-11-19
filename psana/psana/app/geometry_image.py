#!/usr/bin/env python

import os
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.INFO)
SCRNAME = sys.argv[0].rsplit('/')[-1]

test_geo = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-test.txt'
test_nda = '/sdf/group/lcls/ds/ana/detector/data2_test/npy/nda-mfxc00118-r0183-silver-behenate-max.txt'
usage = '\n  Example:'\
      + '\n    cp %s .' % (test_geo)\
      + '\n    cp %s .' % (test_nda)\
      + '\n    %s -g %s -a %s' % (SCRNAME, os.path.basename(test_geo), os.path.basename(test_nda))\
      + '\n'

def argument_parser():

    import argparse

    d_tname    = '0'
    d_geofname = 'geo.txt'
    d_ndafname = 'nda.txt'
    d_prefix   = 'fig'
    d_suffix   = ''
    d_segzero  = None
    d_imgwidth = 11 #9.6
    d_imgheight= 10 #8.9
    d_nrings   = 100
    d_cframe   = 0
    d_zplane   = None
    d_amin     = None
    d_amax     = None
    d_nrmspos  = None
    d_nrmsneg  = None
    d_fraclo   = 0.02
    d_frachi   = 0.98
    d_rmin     = 0
    d_rmax     = None
    d_show     = 'i'
    d_slice    = None
    d_radpsize = 150

    parser = argparse.ArgumentParser(usage=usage, description='Plots image from numpy array shaped as raw in DAQ using specified geometry file')
    parser.add_argument('-t', '--tname',     type=str,   default=d_tname,     help='test name: 1,2,3,..., default: %s'%d_tname)
    parser.add_argument('-g', '--geofname',  type=str,   default=d_geofname,  help='geometry constants file name, default: %s'%d_geofname)
    parser.add_argument('-a', '--ndafname',  type=str,   default=d_ndafname,  help='text ndarray file name, default: %s'%d_ndafname)
    parser.add_argument('-o', '--prefix',    type=str,   default=d_prefix,    help='file name prefix for output image, default: %s'%d_prefix)
    parser.add_argument('-s', '--suffix',    type=str,   default=d_suffix,    help='file name suffix for output image, default: %s'%d_suffix)
    parser.add_argument('-p', '--nrmspos',   type=float, default=d_nrmspos,   help='number of rms positive for intensity range, default: %s'%d_nrmspos)
    parser.add_argument('-n', '--nrmsneg',   type=float, default=d_nrmsneg,   help='number of rms negative for intensity range, default: %s'%d_nrmsneg)
    parser.add_argument('-i', '--segzero',   type=int,   default=d_segzero,   help='segment index to redefine intensity, default: %s'%d_segzero)
    parser.add_argument('-W', '--imgwidth',  type=float, default=d_imgwidth,  help='image width on screen [inch], default: %s'%d_imgwidth)
    parser.add_argument('-H', '--imgheight', type=float, default=d_imgheight, help='image height on screen [inch], default: %s'%d_imgheight)
    parser.add_argument('-R', '--nrings',    type=int,   default=d_nrings,    help='number of rings, default: %d'%d_nrings)
    parser.add_argument('-C', '--cframe',    type=int,   default=d_cframe,    help='coordinate frame 0-psana, 1-LAB, default: %d'%d_cframe)
    parser.add_argument('-Z', '--zplane',    type=float, default=d_zplane,    help='z[um] of the plane to project image, None - z is not used, default: %s'%d_zplane)
    parser.add_argument('--amin',            type=float, default=d_amin,      help='minimal intensity on image, default: %s'%d_amin)
    parser.add_argument('--amax',            type=float, default=d_amax,      help='maximal intensity on image, default: %s'%d_amax)
    parser.add_argument('--fraclo',          type=float, default=d_fraclo,    help='fraction of pixel intensities below amin, default: %s'%d_fraclo)
    parser.add_argument('--frachi',          type=float, default=d_frachi,    help='fraction of pixel intensities below amin, default: %s'%d_frachi)
    parser.add_argument('--rmin',            type=float, default=d_rmin,      help='minimal radius of circles [um] on image, default: %s'%d_rmin)
    parser.add_argument('--rmax',            type=float, default=d_rmax,      help='maximal radius of circles [um] on image, default: %s'%d_rmax)
    parser.add_argument('-S', '--show',      type=str,   default=d_show,      help='show select: i/x/y/p for image/x/y/polar projections, default: %s'%d_show)
    parser.add_argument('-L', '--slice',     type=str,   default=d_slice,     help='image slice to show, ex. 0:,0:, default: %s'%d_slice)
    parser.add_argument('-r', '--radpsize',  type=int,   default=d_radpsize,  help='radial image size in number of pixels relative point (0,0)um, default: %d'%d_radpsize)

    args = parser.parse_args()
    print('Arguments: %s\n' % str(args))
    for k,v in vars(args).items() : print('  %12s : %s' % (k, str(v)))

    return args


def fname_geo_and_nda_for_tname(tname):
    """Returns hardwired geo and nda file names for tname"""
    fname_geo, fname_nda = None, None
    if tname == '1':
      fname_geo = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-cspad-test.data'
      fname_nda = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-mfx11116-r0624-e005365-MfxEndstation-0-Cspad-0-max.txt'
      #shape = (32,185,388)
    elif tname == '2':
      fname_geo = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-jungfrau-8-segment-cxilv9518.data'
      fname_nda = '/sdf/group/lcls/ds/ana/detector/data_test/npy/nda-cxilv9518-r0008-jungfrau-lysozyme-max.npy'
      #shape = (2,512,1024)
    elif tname == '3':
      fname_geo = test_geo
      fname_nda = test_nda
      #shape = (16,352,384)
    else:
      msg = 'Not-recognized test name: %s' % tname
      sys.exit('End of test %s' % tname)
    return fname_geo, fname_nda


def geometry_image():

    args = argument_parser()

    from time import time
    import numpy as np

    global info_ndarr
    from psana.pyalgos.generic.NDArrUtils import info_ndarr, print_ndarr, divide_protected, shape_as_3d, shape_as_3d
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays

    global gr
    import psana.pyalgos.generic.Graphics as gr
    drawCenter, drawCircle = gr.drawCenter, gr.drawCircle
    #tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    tname = args.tname
    logger.info('%s\nTest %s:' % (50*'_',tname))

    fname_geo = args.geofname
    fname_nda = args.ndafname
    prefix    = args.prefix
    suffix    = args.suffix
    segzero   = args.segzero
    imgheight = args.imgheight
    imgwidth  = args.imgwidth
    nrings    = args.nrings
    cframe    = args.cframe
    zplane    = args.zplane
    amin      = args.amin
    amax      = args.amax
    fraclo    = args.fraclo
    frachi    = args.frachi
    nrmsneg   = args.nrmsneg
    nrmspos   = args.nrmspos
    rmin      = args.rmin
    rmax      = args.rmax
    show      = args.show.lower()
    radpsize  = args.radpsize

    segname, segind, dx, dy = None, 0, 0, 0

    if tname != '0': fname_geo, fname_nda = fname_geo_and_nda_for_tname(tname)

    logger.info('fname_geo: %s' % fname_geo)
    logger.info('fname_nda: %s' % fname_nda)

    t0_sec = time()
    geo = GeometryAccess(fname_geo, 0o377)

    if segname is not None:
        geo.move_geo(segname, segind, dx, dy, 0) # (hor, vert, z)

    #Z = None
    #if zplane is None: X, Y, Z = geo.get_pixel_coords(cframe=cframe) # oname=None, oindex=0, do_tilt=True)
    #else: X, Y = geo.get_pixel_xy_at_z(zplane=zplane) #None, oname=None, oindex=0, do_tilt=True)

    X, Y, Z = geo.get_pixel_coords(cframe=cframe) if zplane is None else\
              geo.get_pixel_xy_at_z(zplane=zplane) + (None,)

    ir0, ic0 = geo.point_coord_indexes(p_um=(0,0), cframe=cframe)
    logger.info('point (0,0)um indices:(%d, %d)' % (ir0, ic0))

    logger.info('GeometryAccess time = %.6f sec' % (time()-t0_sec))

    logger.info(info_ndarr(X, 'X', last=5, vfmt='%0.1f'))
    logger.info(info_ndarr(Y, 'Y', last=5, vfmt='%0.1f'))
    logger.info(info_ndarr(Z, 'Z', last=5, vfmt='%0.1f'))

    nda = X if 'x' in show else\
          Y if 'y' in show else\
          np.load(fname_nda) if '.npy' in fname_nda else\
          np.loadtxt(fname_nda) #if show=='image'

    rows, cols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=cframe) if zplane is None else\
                 geo.get_pixel_xy_inds_at_z(zplane=zplane) #, oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=cframe)

    logger.info(info_ndarr(rows, 'rows'))
    logger.info(info_ndarr(cols, 'cols'))
    logger.info(info_ndarr(nda,  'nda', vfmt='%0.1f'))

    shape = shape_as_3d(rows.shape)
    if nda.size == 2162688: shape = (16,352,384) # epix10ka2m
    if nda.size == 4194304: shape = (8,512,1024) # jungfrau4m
    logger.info('shape: %s' % str(shape))

    rows.shape = shape
    cols.shape = shape
    nda.shape  = shape

    #ave, rms = nda.mean(), nda.std()
    vmin, vmax = nda.min(), nda.max()
    med = np.median(nda)
    spr = np.median(np.abs(nda-med))
    logger.info('median: %.3f spread: %.3f' % (med, spr))

    alo = np.quantile(nda, fraclo)
    ahi = np.quantile(nda, frachi)
    logger.info('\n  fraction %.3f alow =%.1f\n  fraction %.3f ahigh=%.1f' % (fraclo, alo, frachi, ahi))

    amin = amin if amin is not None else (med-nrmsneg*spr) if nrmsneg is not None else alo
    amax = amax if amax is not None else (med+nrmspos*spr) if nrmspos is not None else ahi

    logger.info('Image med=%.1f spr=%.1f nda.min=%.1f nda.max=%.1f amin=%.1f amax=%.1f'%\
          (med, spr, vmin, vmax, amin, amax))

    # replace one segment in data
    if segzero is not None:
       segsh = nda[segzero,:].shape
       logger.info(info_ndarr(segsh,'XXX segsh'))
       norm = (amax-amin)/max(segsh[0],segsh[1])
       arows = np.arange(segsh[0], dtype=nda.dtype) * norm
       acols = np.arange(segsh[1], dtype=nda.dtype) * norm
       grid = np.meshgrid(acols,arows)
       nda[segzero,:] = 0.3*grid[0] + 0.7*grid[1] + amin # cols change color faster

    img = img_from_pixel_arrays(rows,cols,W=nda)
    logger.info(info_ndarr(img, 'img'))

    shimg = img.shape
    sl = (ir0-radpsize, ir0+radpsize, ic0-radpsize, ic0+radpsize) # slice limits
    slp = max(sl[0], 0), min(sl[1], shimg[0]), max(sl[2], 0), min(sl[3], shimg[1]) # slice oversize protection

    if True:
      pixsize = geo.get_pixel_scale_size()
      logger.info('pixsize: %.0f' % pixsize)
      logger.info('ir0: %d, ic0: %d' % (ir0, ic0))
      logger.info('slp: %s' % str(slp))
      ymin = (slp[0]-ir0) * pixsize
      ymax = (slp[1]-ir0) * pixsize
      xmin = (slp[2]-ic0) * pixsize
      xmax = (slp[3]-ic0) * pixsize

    else: # OLD
      xmin = X.min()
      xmax = X.max()
      ymin = Y.min()
      ymax = Y.max()

    logger.info('Image xmin=%.1f xmax=%.1f ymin=%.1f ymax=%.1f'% (xmin, xmax, ymin, ymax))

    if sl != slp:
        logger.warning('protected image central slice: %s differs from center+-: %s IMAGE IS OUT OF SCALE' % (str(slp), str(sl)))

    sl = args.slice if args.slice is not None else\
         '%d:%d,%d:%d' % slp
    _slice = eval('np.s_[%s]' % sl)

    x0,y0 = xy0 = (0,0) # um
    logger.info('image center: %.1f, %.1f' % xy0)

    kwa1 = {'figsize':(imgwidth,imgheight),\
            'extent':(ymin,ymax, xmax,xmin), 'interpolation':'nearest', 'aspect':'equal', 'origin':'upper', 'cmap':'inferno',\
            'vmin':amin, 'vmax':amax, 'figsize':(12,11)}
    #'extent':(left, right, bottom, top) in matplotlib.pyplot.imshow

    _img = img[_slice]
    fig, axim, axcb, imsh, cbar = fig_img_cbar(_img, **kwa1)
    gr.add_title_labels_to_axes(axim, title='image', xlabel='geo y, $\mu$m', ylabel='geo x, $\mu$m')#, fslab=14, fstit=20, color='k')

    rmax = rmax if rmax is not None else\
           1.4*max(abs(xmax-x0), abs(x0-xmin), abs(ymax-y0), abs(y0-ymin))

    drawCenter(axim, xy0, s=(xmax-xmin)/40, linewidth=1, color='w')

    for radius in np.linspace(rmin, rmax, nrings, endpoint=True):
        drawCircle(axim, xy0, radius, linewidth=1, color='w', fill=False)

    gr.show(mode='go')


    if 'p' in show:
       #from pyimgalgos.HPolar import HPolar
       from psana.pyalgos.generic.HPolar import HPolar

       um_to_units = 0.001 # um -> mm
       extent = phimin, phimax, radmin, radmax = (0, 360, 1, 100)

       logger.info(info_ndarr(X, 'X'))
       logger.info(info_ndarr(Y, 'Y'))
       hp = HPolar(um_to_units*X, um_to_units*Y, mask=None, radedges=(radmin, radmax), nradbins=500, phiedges=(phimin, phimax), nphibins=360)

       ##arr2 = nda
       #arr2 =  hp.pixel_irad() # hp.pixel_iphi() # Y # X
       #cmin, cmax = arr2.min(), arr2.max() # None, None #18,36  #amin, amax # 0,30
       #img2 = img_from_pixel_arrays(rows,cols,W=arr2)

       img2 = hp.bin_avrg_rad_phi(nda.ravel()) #, do_transp=False)
       cmin, cmax = 10, np.quantile(img2, frachi)

       #img2 = img_from_pixel_arrays(rows,cols,W=hp.pixel_iphi())
       #img2 = img
       #img2 = img_from_pixel_arrays(rows,cols,W=nda)

       logger.info(info_ndarr(img2, 'img2'))

       kwa2 = {'figsize':(10,imgheight),\
               'extent':extent, 'interpolation':'nearest', 'aspect':'auto', 'origin':'lower', 'cmap':'inferno', 'vmin':cmin, 'vmax':cmax}
       fig2, axim2, axcb2, imsh2, cbar2 = fig_img_proj_cbar(img2, **kwa2)

    gr.show(mode=None)
    if 'i' in show: gr.save_fig(fig,  fname=prefix + '-img-' + suffix + '.png', verb=True)
    if 'p' in show: gr.save_fig(fig2, fname=prefix + '-rphi-' + suffix + '.png', verb=True)


def fig_img_cbar(img, **kwa):
    fig = gr.figure(figsize=kwa.pop('figsize', (12,11)))
    gr.move_fig(fig,100,10)
    axim, axcb = gr.fig_axes(fig, windows=((0.06, 0.03, 0.87, 0.93), (0.923,0.03, 0.02, 0.93)))
    imsh = axim.imshow(img, **kwa)
    imsh.set_clim(kwa.get('vmin', None), kwa.get('vmax', None))
    cbar = fig.colorbar(imsh, cax=axcb, orientation='vertical')
    return fig, axim, axcb, imsh, cbar


def fig_img_proj_cbar(img, **kwa):
    fig = gr.figure(figsize=kwa.pop('figsize', (6,12)))
    gr.move_fig(fig,700,10)
    fymin, fymax = 0.050, 0.90
    winds =((0.07,  fymin, 0.685, fymax),\
            (0.76,  fymin, 0.15, fymax),\
            (0.915, fymin, 0.01, fymax))

    axim, axhi, axcb = gr.fig_axes(fig, windows=winds)
    imsh = axim.imshow(img, **kwa)
    imsh.set_clim(kwa.get('vmin', None), kwa.get('vmax', None))
    cbar = fig.colorbar(imsh, cax=axcb, orientation='vertical')
    #????? axim.grid(b=None, which='both', axis='both')#, **kwargs)'major'

    sh = img.shape
    w = gr.np.sum(img, axis=1)
    phimin, phimax, radmin, radmax = kwa.get('extent', (0, 360, 1, 100))
    hbins = gr.np.linspace(radmin, radmax, num=sh[0], endpoint=False)

    #print(info_ndarr(img,'XXX r-phi img'))
    #print(info_ndarr(w,'XXX r-phi weights'))
    #print(info_ndarr(hbins,'XXX hbins'))

    kwh={'bins'       : kwa.get('bins', img.shape[0]),\
         'range'      : kwa.get('range', (radmin, radmax)),\
         'weights'    : kwa.get('weights', w),\
         'color'      : kwa.get('color', 'lightgreen'),\
         'log'        : kwa.get('log',False),\
         'bottom'     : kwa.get('bottom', 0),\
         'align'      : kwa.get('align', 'mid'),\
         'histtype'   : kwa.get('histtype',u'bar'),\
         'label'      : kwa.get('label', ''),\
         'orientation': kwa.get('orientation',u'horizontal'),\
        }

    axhi.set_ylim((radmin, radmax))
    axhi.set_yticklabels([]) # removes axes labels, not ticks
    axhi.tick_params(axis='y', direction='in')

    wei, bins, patches = his = gr.pp_hist(axhi, hbins, **kwh)
    gr.add_stat_text(axhi, wei, bins)

    gr.add_title_labels_to_axes(axim, title='r vs $\phi$', xlabel='$\phi$, deg', ylabel='r, mm')#, fslab=14, fstit=20, color='k')
    gr.draw_fig(fig)
    return fig, axim, axcb, imsh, cbar

def do_main():
    if len(sys.argv)<2:
        print(usage)
        sys.exit('EXIT due to missing parameters\nTry > %s -h' % SCRNAME)
    geometry_image()
    return 0

if __name__ == "__main__":
    do_main()
    sys.exit('End of %s' % SCRNAME)

# EOF
