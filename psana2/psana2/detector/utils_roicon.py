""" Utils for psana/detector/app/roicon.py
import psana2.detector.utils_roicon as urc
"""
import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
from psana2.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
from psana2.detector.UtilsGraphics import gr
import psana2.detector.UtilsMask as um  # DTYPE_MASK = np.uint8
reshape_to_3d, info_ndarr = um.reshape_to_3d, um.info_ndarr

def mask_test_ring(shape):
    """returns one-like mask of requested shape with a couple of zero-rings"""
import psana2.pyalgos.generic.NDArrGenerators as ag
    rows, cols = shape
    rc, cc = int(0.48*rows), int(0.51*cols)
    r1, r2 = int(0.42*rows), int(0.20*rows)
    s1, s2 = int(0.04*r1), int(0.03*r2)
    a = np.ones(shape)
    ag.add_ring(a, amp=10, row=rc, col=cc, rad=r1, sigma=s1)
    ag.add_ring(a, amp=10, row=rc, col=cc, rad=r2, sigma=s2)
    return np.select([a>5,], [0,], default=1).astype(um.DTYPE_MASK)


def image_of_sensors(gfname, afname=None, ofname='mask.txt', mbits=0xffff, dotest=False, figprefix=None, **kwargs):
    """ Makes and plot an image or mask of sensors for geometry file
    """
    logger.info('Geometry file: %s' % gfname)

    geo = GeometryAccess(gfname, 0)
    ir, ic = geo.get_pixel_coord_indexes(**kwargs)
    reshape_to_3d(ir)
    reshape_to_3d(ic)

    afext = '' if afname is None else os.path.splitext(afname)[1]
    ofext = '' if ofname is None else os.path.splitext(ofname)[1]

    arr = np.ones(ir.shape, dtype=np.uint16) if afname is None else \
          np.load(afname) if afext == '.npy' else \
          np.loadtxt(afname) #, dtype=np.uint16)

    arr.shape = ir.shape

    amp_range=[-1,2]
    if afname is not None:
        mean = np.mean(arr)
        std  = np.std(arr)
        logger.info('Input array mean=%f   std=%f' % (mean, std))
        amp_range=[mean-2*std, mean+2*std]

    mask = geo.get_pixel_mask(mbits=mbits)
    mask.shape = ir.shape

    if mbits: arr *= mask
    logger.info('shape ir: %s  ic: %s  W: %s' % (str(ir.shape), str(ic.shape), str(arr.shape)))

    img = img_from_pixel_arrays(ir, ic, W=arr)

    if dotest:
      mask_rings = mask_test_ring(img.shape)
      img *= mask_rings

    axim = gr.plotImageLarge(img, amp_range=amp_range)
    do_save = (figprefix is not None)
    fname = '%s-img-test-mask2d.png' % figprefix if do_save else ''
    gr.save(fname, do_save=do_save)
    gr.move(500,10)
    gr.show()

    if ofext == '.npy': np.save(ofname, img)
    else              : np.savetxt(ofname, img, fmt='%d', delimiter=' ')

    logger.info('Image or mask of sensors is saved in the file %s' % ofname)


def roi_mask_editor(ifname='image.txt', mfname='mask', mbits=0xffff):
    """ Launch the mask editor, command "med" with parameters"""
    import sys
    sys.exit('\nWARNING: Mask Editor is not implemetted yet for lcls2.')

    #from subprocess import getoutput
    #cmd = 'med -i %s -m %s' % (ifname, mfname)
    #logger.info('Start process with mask editor by the command: %s' % cmd)
    #output = getoutput(cmd)
    #logger.info('%s' % output)


def roi_mask_to_ndarray(gfname, ifname='roi-mask.txt', ofname='mask-nda.txt', mbits=0xffff, figprefix=None, **kwargs):
    """ Makes and plot the mask of sensors for image generated from geometry file
        Mask ndarray is created by the list of comprehension
        [mask_roi[r,c] for r,c in zip(ir, ic)]
        The same timing gives mapping: map(value_of_mask, ir, ic)
    """
    ifext = os.path.splitext(ifname)[1]
    ofext = os.path.splitext(ofname)[1]

    logger.info('1. Load ROI mask from file: %s' % ifname)
    mask_roi = np.load(ifname) if ifext == '.npy' else np.loadtxt(ifname, dtype=np.uint16)

    logger.info('2. Define geometry from file: %s' % gfname)
    geo = GeometryAccess(gfname, 0)
    ir, ic = geo.get_pixel_coord_indexes(**kwargs)
    reshape_to_3d(ir)
    reshape_to_3d(ic)
    logger.info('3. Check shapes of pixel image-index arrays ir: %s ic: %s' %  (str(ir.shape), str(ic.shape)))

    logger.info('4. Plot image of the mask %s' % info_ndarr(mask_roi, 'mask_roi'))
    axim = gr.plotImageLarge(mask_roi,amp_range=[0,1], title='Image of the mask')
    do_save = (figprefix is not None)
    fname = '%s-img-mask2d.png' % figprefix if do_save else ''
    gr.save(fname, do_save=do_save)
    gr.move(400,10)
    gr.show()

    logger.info('5. Evaluate ndarray with mask')
    mask_nda = um.convert_mask2d_to_ndarray_using_pixel_coord_indexes(mask_roi, ir, ic)
    #mask_nda = um.convert_mask2d_to_ndarray_using_geo(mask_roi, geo, **kwargs)  # for test only
    #mask_nda = um.convert_mask2d_to_ndarray_using_geometry_file(mask_roi, gfname, **kwargs)  # for test only
    #mask_nda = np.array([mask_roi[r,c] for r,c in zip(ir, ic)], dtype=um.DTYPE_MASK)  # original algorithm

    if mbits:
        mask_geo = geo.get_pixel_mask(mbits=mbits)
        reshape_to_3d(mask_geo)
        mask_nda *= mask_geo

    logger.info('6. Cross-checks: shape of mask_nda: %s, mask_nda.size=%d, ir.size=%d ' % \
          (mask_nda.shape, mask_nda.size, ir.size))

    logger.info('7. Save mask for ndarray in the file %s' % ofname)
    if ofext == '.npy': np.save(ofname, mask_nda)
    else              :
        mask_nda.shape = [ir.size//ir.shape[-1],ir.shape[-1]]
        logger.info('7a. Re-shape for saving in txt to 2-d: %s' % str(mask_nda.shape))
        np.savetxt(ofname, mask_nda, fmt='%d', delimiter=' ')

    logger.info('8. Test new mask-ndarray to generate image (CLOSE image to continue)')
    logger.info(info_ndarr(mask_nda, 'mask_nda'))
    #mask_nda.shape = [ir.size//ir.shape[-1], ir.shape[-1]]

    from psana2.pyalgos.generic.PSUtils import table_nxn_epix10ka_from_ndarr
    mask_table = table_nxn_epix10ka_from_ndarr(mask_nda)
    logger.info(info_ndarr(mask_table, 'convert mask_nda to table of segments for viewable image'))
    axim = gr.plotImageLarge(mask_table, amp_range=[0,1], title='mask as ndarray')  # figsize=(13,12)
    fname = '%s-img-mask-ndarr.png' % figprefix if do_save else ''
    gr.save(fname, do_save=do_save)
    gr.move(400,10)

    mask_nda.shape = ir.shape
    img = img_from_pixel_arrays(ir, ic, W=mask_nda)
    axim = gr.plotImageLarge(img, amp_range=[0,1], title='mask re-generated from ndarray for cross-check')
    fname = '%s-img-mask-from-ndarr.png' % figprefix if do_save else ''
    gr.save(fname, do_save=do_save)
    gr.move(500,50)
    gr.show()


def do_main(parser):
    nspace = parser.parse_args()
    proc = nspace.args # [0]

    figprefix = ('%s/%s' % (parser.repoman.dir_in_repo('figs'), nspace.figprefix)) if nspace.figprefix != 'None' else None

    if   proc == '1': image_of_sensors   (nspace.gfname, nspace.afname, nspace.ifname, nspace.cbits, nspace.dotest, figprefix, **eval(nspace.kwargs))
    elif proc == '2': roi_mask_editor    (nspace.ifname, nspace.mfname)
    elif proc == '3': roi_mask_to_ndarray(nspace.gfname, nspace.mfname, nspace.nfname, nspace.cbits, figprefix, **eval(nspace.kwargs))
    else: logger.info('Non-recognized process number "%s"; implemented options: 1, 2, or 3' % proc)

# EOF
