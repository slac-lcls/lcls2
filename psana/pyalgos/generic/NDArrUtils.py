#------------------------------
"""
:py:class:`NDArrUtils` - a set of utilities working on numpy arrays
=================================================================

Usage::

    # assuming that $PYTHONPATH=.../lcls2/psana
    # Import
    import pyalgos.generic.NDArrUtils as gu

    # Methods
    #resp = gu.<method(pars)>

    shp2d = gu.shape_nda_as_2d(nda)
    shp3d = gu.shape_nda_as_3d(nda)
    arr2d = gu.reshape_to_2d(nda)
    arr3d = gu.reshape_to_3d(nda)
    mmask = gu.merge_masks(mask1=None, mask2=None, dtype=np.uint8)
    mask  = gu.mask_neighbors(mask_in, allnbrs=True, dtype=np.uint8)
    mask  = gu.mask_edges(mask, mrows=1, mcols=1, dtype=np.uint8)

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`NDArrGenerators`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
"""
#------------------------------

#import os
#from time import localtime, strftime, time

import numpy as np

#------------------------------

import logging
log = logging.getLogger('NDArrUtils')

#------------------------------

def print_ndarr(nda, name='', first=0, last=5) :
    if nda is None : print('%s: %s', name, nda)
    elif isinstance(nda, tuple) : print_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list)  : print_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray) :
                     print('%s: %s' % (name, type(nda)))
    else           : print('%s:  shape:%s  size:%d  dtype:%s %s...'%\
                           (name, str(nda.shape), nda.size, nda.dtype, nda.flatten()[first:last]))

#------------------------------

def shape_nda_as_2d(arr) :
    """Return shape of np.array to reshape to 2-d
    """
    sh = arr.shape
    if len(sh)<3 : return sh
    return (int(arr.size/sh[-1]), sh[-1])

#------------------------------

def shape_nda_as_3d(arr) :
    """Return shape of np.array to reshape to 3-d
    """
    sh = arr.shape
    if len(sh)<4 : return sh
    return (int(arr.size/sh[-1]/sh[-2]), sh[-2], sh[-1])

#------------------------------

def reshape_to_2d(arr) :
    """Reshape np.array to 2-d
    """
    sh = arr.shape
    if len(sh)<3 : return arr
    arr.shape = (int(arr.size/sh[-1]), sh[-1])
    return arr

#------------------------------

def reshape_to_3d(arr) :
    """Reshape np.array to 3-d
    """
    sh = arr.shape
    if len(sh)<4 : return arr
    arr.shape = (int(arr.size/sh[-1]/sh[-2]), sh[-2], sh[-1])
    return arr

#------------------------------

def merge_masks(mask1=None, mask2=None, dtype=np.uint8) :
    """Merging masks using np.logical_and rule: (0,1,0,1)^(0,0,1,1) = (0,0,0,1) 
    """
    if mask1 is None : return mask2
    if mask2 is None : return mask1

    shape1 = mask1.shape
    shape2 = mask2.shape

    if shape1 != shape2 :
        if len(shape1) > len(shape2) : mask2.shape = shape1
        else                         : mask1.shape = shape2

    mask = np.logical_and(mask1, mask2)
    return mask if dtype==np.bool else np.asarray(mask, dtype)

#------------------------------

def mask_neighbors(mask, allnbrs=True, dtype=np.uint8) :
    """Return mask with masked eight neighbor pixels around each 0-bad pixel in input mask.

       mask    : int - n-dimensional (n>1) array with input mask
       allnbrs : bool - False/True - masks 4/8 neighbor pixels.
    """
    shape_in = mask.shape
    if len(shape_in) < 2 :
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(shape_in))

    mask_out = np.asarray(mask, dtype)

    if len(shape_in) == 2 :
        # mask nearest neighbors
        mask_out[0:-1,:] = np.logical_and(mask_out[0:-1,:], mask[1:,  :])
        mask_out[1:,  :] = np.logical_and(mask_out[1:,  :], mask[0:-1,:])
        mask_out[:,0:-1] = np.logical_and(mask_out[:,0:-1], mask[:,1:  ])
        mask_out[:,1:  ] = np.logical_and(mask_out[:,1:  ], mask[:,0:-1])
        if allnbrs :
          # mask diagonal neighbors
          mask_out[0:-1,0:-1] = np.logical_and(mask_out[0:-1,0:-1], mask[1:  ,1:  ])
          mask_out[1:  ,0:-1] = np.logical_and(mask_out[1:  ,0:-1], mask[0:-1,1:  ])
          mask_out[0:-1,1:  ] = np.logical_and(mask_out[0:-1,1:  ], mask[1:  ,0:-1])
          mask_out[1:  ,1:  ] = np.logical_and(mask_out[1:  ,1:  ], mask[0:-1,0:-1])

    else : # shape>2

        mask_out.shape = mask.shape = shape_nda_as_3d(mask)       

        # mask nearest neighbors
        mask_out[:, 0:-1,:] = np.logical_and(mask_out[:, 0:-1,:], mask[:, 1:,  :])
        mask_out[:, 1:,  :] = np.logical_and(mask_out[:, 1:,  :], mask[:, 0:-1,:])
        mask_out[:, :,0:-1] = np.logical_and(mask_out[:, :,0:-1], mask[:, :,1:  ])
        mask_out[:, :,1:  ] = np.logical_and(mask_out[:, :,1:  ], mask[:, :,0:-1])
        if allnbrs :
          # mask diagonal neighbors
          mask_out[:, 0:-1,0:-1] = np.logical_and(mask_out[:, 0:-1,0:-1], mask[:, 1:  ,1:  ])
          mask_out[:, 1:  ,0:-1] = np.logical_and(mask_out[:, 1:  ,0:-1], mask[:, 0:-1,1:  ])
          mask_out[:, 0:-1,1:  ] = np.logical_and(mask_out[:, 0:-1,1:  ], mask[:, 1:  ,0:-1])
          mask_out[:, 1:  ,1:  ] = np.logical_and(mask_out[:, 1:  ,1:  ], mask[:, 0:-1,0:-1])

        mask_out.shape = mask.shape = shape_in

    return mask_out

#------------------------------

def mask_edges(mask, mrows=1, mcols=1, dtype=np.uint8) :
    """Return mask with a requested number of row and column pixels masked - set to 0.
       mask  : int - n-dimensional (n>1) array with input mask
       mrows : int - number of edge rows to mask
       mcols : int - number of edge columns to mask
    """
    sh = mask.shape
    if len(sh) < 2 :
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(sh))

    mask_out = np.asarray(mask, dtype)

    # print 'shape:', sh

    if len(sh) == 2 :
        rows, cols = sh

        if mrows > rows : 
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols : 
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0 :
          # mask edge rows
          mask_rows = np.zeros((mrows,cols), dtype=mask.dtype)
          mask_out[:mrows ,:] = mask_rows
          mask_out[-mrows:,:] = mask_rows

        if mcols>0 :
          # mask edge colss
          mask_cols = np.zeros((rows,mcols), dtype=mask.dtype)
          mask_out[:,:mcols ] = mask_cols
          mask_out[:,-mcols:] = mask_cols

    else : # shape>2
        mask_out.shape = shape_nda_as_3d(mask)       

        segs, rows, cols = mask_out.shape

        if mrows > rows : 
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols : 
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0 :
          # mask edge rows
          mask_rows = np.zeros((segs,mrows,cols), dtype=mask.dtype)
          mask_out[:, :mrows ,:] = mask_rows
          mask_out[:, -mrows:,:] = mask_rows

        if mcols>0 :
          # mask edge colss
          mask_cols = np.zeros((segs,rows,mcols), dtype=mask.dtype)
          mask_out[:, :,:mcols ] = mask_cols
          mask_out[:, :,-mcols:] = mask_cols

        mask_out.shape = sh

    return mask_out

#------------------------------
#----------- TEST -------------
#------------------------------

def test_mask_neighbors_2d(allnbrs=True) :

    randexp = random_exponential(shape=(40,60), a0=1)
    fig  = gr.figure(figsize=(16,6), title='Random 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)
    img1 = mask # mask # randexp
    img2 = mask_nbrs # mask # randexp
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2,  amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    
#------------------------------

def test_mask_neighbors_3d(allnbrs=True) :

    #randexp = random_exponential(shape=(2,2,30,80), a0=1)
    randexp = random_exponential(shape=(2,30,80), a0=1)

    fig  = gr.figure(figsize=(16,6), title='Random > 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)

    img1 = reshape_to_2d(mask)
    img2 = reshape_to_2d(mask_nbrs)
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    
#------------------------------

def test_mask_edges_2d(mrows=1, mcols=1) :

    fig  = gr.figure(figsize=(8,6), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    mask = np.ones((20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = mask_out
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    
#------------------------------

def test_mask_edges_3d(mrows=1, mcols=1) :

    fig  = gr.figure(figsize=(8,6), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    #mask = np.ones((2,2,20,30))
    mask = np.ones((2,20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = reshape_to_2d(mask_out)
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    
#-----------------------------

def do_test() :

    from pyalgos.generic.NDArrGenerators import random_exponential; global random_exponential 
    import pyalgos.generic.Graphics as gr; global gr

    print(80*'_')
    tname = sys.argv[1] if len(sys.argv)>1 else '1'
    if tname == '1' : test_mask_neighbors_2d(allnbrs = False)
    if tname == '2' : test_mask_neighbors_2d(allnbrs = True)
    if tname == '3' : test_mask_neighbors_3d(allnbrs = False)
    if tname == '4' : test_mask_neighbors_3d(allnbrs = True)
    if tname == '5' : test_mask_edges_2d(mrows=5, mcols=1)
    if tname == '6' : test_mask_edges_2d(mrows=0, mcols=5)
    if tname == '7' : test_mask_edges_3d(mrows=1, mcols=2)
    if tname == '8' : test_mask_edges_3d(mrows=5, mcols=0)
    else : sys.exit ('Not recognized test name: "%s"' % tname)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    do_test()
    sys.exit('\nEnd of test')

#------------------------------
