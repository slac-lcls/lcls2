""" import psana.detector.utils_roicon as urc
"""
#from future import standard_library
#standard_library.install_aliases()

import os
import sys
from time import time
import numpy as np
# from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays
# import pyimgalgos.GlobalGraphics as gg # for test purpose
from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
from psana.detector.UtilsGraphics import gr
from psana.detector.NDArrUtils import reshape_to_3d, info_ndarr  # divide_protected, reshape_to_2d, reshape_to_3d, save_ndarray_in_textfile

def mask_test_ring(shape):
    """returns one-like mask of requested shape with a couple of zero-rings"""
    import psana.pyalgos.generic.NDArrGenerators as ag
    rows, cols = shape
    rc, cc = int(0.48*rows), int(0.51*cols)
    r1, r2 = int(0.42*rows), int(0.20*rows)
    s1, s2 = int(0.04*r1), int(0.06*r2)
    a = np.ones(shape)
    ag.add_ring(a, amp=10, row=rc, col=cc, rad=r1, sigma=s1)
    ag.add_ring(a, amp=10, row=rc, col=cc, rad=r2, sigma=s2)
    return np.select([a>5,], [0,], default=1)


def image_of_sensors(gfname, afname=None, ofname='mask.txt', mbits=0xffff, dotest=False):
    """ Makes and plot an image or mask of sensors for geometry file
    """
    print('Geometry file: %s' % gfname)

    geometry = GeometryAccess(gfname, 0)
    iX, iY = geometry.get_pixel_coord_indexes()
    reshape_to_3d(iX)
    reshape_to_3d(iY)

    afext = '' if afname is None else os.path.splitext(afname)[1]
    ofext = '' if ofname is None else os.path.splitext(ofname)[1]

    arr = np.ones(iX.shape, dtype=np.uint16) if afname is None else \
          np.load(afname) if afext == '.npy' else \
          np.loadtxt(afname) #, dtype=np.uint16)

    arr.shape = iX.shape

    amp_range=[-1,2]
    if afname is not None:
        mean = np.mean(arr)
        std  = np.std(arr)
        print('Input array mean=%f   std=%f' % (mean, std))
        amp_range=[mean-2*std, mean+2*std]

    mask = geometry.get_pixel_mask(mbits=mbits)
    mask.shape = iX.shape

    if mbits: arr *= mask
    print('iX, iY, W shape:', iX.shape, iY.shape, arr.shape)

    img = img_from_pixel_arrays(iX, iY, W=arr)

    if dotest:
      mask_rings = mask_test_ring(img.shape)
      img *= mask_rings

    axim = gr.plotImageLarge(img,amp_range=amp_range)
    gr.move(500,10)
    gr.show()

    if ofext == '.npy': np.save(ofname, img)
    else              : np.savetxt(ofname, img, fmt='%d', delimiter=' ')

    print('Image or mask of sensors is saved in the file %s' % ofname)


def roi_mask_editor(ifname='image.txt', mfname='mask', mbits=0xffff):
    """ Launch the mask editor, command "med" with parameters"""
    sys.exit('\nWARNING: Mask Editor is not implemetted yet for lcls2.')

    #from subprocess import getoutput
    #cmd = 'med -i %s -m %s' % (ifname, mfname)
    #print('Start process with mask editor by the command: %s' % cmd)
    #output = getoutput(cmd)
    #print('%s' % output)


def roi_mask_to_ndarray(gfname, ifname='roi-mask.txt', ofname='mask-nda.txt', mbits=0xffff):
    """ Makes and plot the mask of sensors for image generated from geometry file
        Mask ndarray is created by the list of comprehension
        [mask_roi[r,c] for r,c in zip(iX, iY)]
        The same timing gives mapping: map(value_of_mask, iX, iY)
    """
    ifext = os.path.splitext(ifname)[1]
    ofext = os.path.splitext(ofname)[1]

    print('1. Load ROI mask from file: %s' % ifname)
    mask_roi = np.load(ifname) if ifext == '.npy' else np.loadtxt(ifname, dtype=np.uint16)

    print('2. Define geometry from file: %s' % gfname)
    geometry = GeometryAccess(gfname, 0)
    iX, iY = geometry.get_pixel_coord_indexes()
    reshape_to_3d(iX)
    reshape_to_3d(iY)
    #arr = np.ones(iX.shape, dtype=np.uint16)
    print('3. Check shapes of pixel image-index arrays iX, iY:', iX.shape, iY.shape)

    print('4. Plot image of the mask')
    axim = gr.plotImageLarge(mask_roi,amp_range=[0,1], title='Image of the mask')
    gr.move(400,10)
    gr.show()

    print('5. Evaluate ndarray with mask')

    #t0_sec = time()
    #def value_of_mask(r,c): return mask_roi[r,c]
    #mask_nda = np.array( map(value_of_mask, iX, iY) ) # 155 msec
    #print '   Consumed time alg.2 to evaluate mask = %7.3f sec' % (time()-t0_sec)

    mask_nda = np.array([mask_roi[r,c] for r,c in zip(iX, iY)]) # 155 msec

    if mbits:
        mask_geo = geometry.get_pixel_mask(mbits=mbits)
        reshape_to_3d(mask_geo)
        mask_nda *= mask_geo

    print('6. Cross-checks: shape of mask_nda: %s, mask_nda.size=%d, iX.size=%d ' % \
          (mask_nda.shape, mask_nda.size, iX.size))

    print('7. Save mask for ndarray in the file %s' % ofname)
    if ofext == '.npy': np.save(ofname, mask_nda)
    else              :
        mask_nda.shape = [iX.size//iX.shape[-1],iX.shape[-1]]
        print('7a. Re-shape for saving in txt to 2-d:', mask_nda.shape)
        np.savetxt(ofname, mask_nda, fmt='%d', delimiter=' ')

    print('8. Test new mask-ndarray to generate image (CLOSE image to continue)')
    print(info_ndarr(mask_nda, 'mask_nda'))
    mask_nda.shape = [iX.size//iX.shape[-1], iX.shape[-1]]
    print(info_ndarr(mask_nda, 'reshape for 2-d image of panels'))
    axim = gr.plotImageLarge(mask_nda, amp_range=[0,1], figsize=(6,12), title='mask as ndarray')
    gr.move(400,10)

    mask_nda.shape = iX.shape
    img = img_from_pixel_arrays(iX, iY, W=mask_nda)
    axim = gr.plotImageLarge(img, amp_range=[0,1], title='mask generated from ndarray')
    gr.move(500,50)
    gr.show()


def do_main(parser):
    #sys.exit('TEST EXIT in do_main')

    nspace = parser.parse_args()
    proc = nspace.args # [0]

    if   proc == '1': image_of_sensors   (nspace.gfname, nspace.afname, nspace.ifname, nspace.cbits, nspace.dotest)
    elif proc == '2': roi_mask_editor    (nspace.ifname, nspace.mfname)
    elif proc == '3': roi_mask_to_ndarray(nspace.gfname, nspace.mfname, nspace.nfname, nspace.cbits)
    else: print('Non-recognized process number "%s"; implemented options: 1, 2, or 3' % proc)

# EOF
