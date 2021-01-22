"""
Usage::

  from psana.detector.Utils *

  statmgd = merge_status(status, grinds=(0,1,2,3,4), dtype=np.uint64)
         # merges status.shape=(7, 16, 352, 384) to statmgd.shape=(16, 352, 384) of dtype
         # grinds list stands for gain ranges for 'FH','FM','FL','AHL-H','AML-M'

"""

import sys
import numpy as np
#from psana.pyalgos.generic.NDArrUtils import info_ndarr

def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (nrec<500 and not nrec%100)\
       or (not nrec%1000)


def info_command_line_parameters(parser) :
    """Prints input arguments and optional parameters
       from optparse import OptionParser
       parser = OptionParser(description=usage(1), usage = usage())
    """
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items() :
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


def info_command_line_arguments(parser) :
    """Prints input arguments and optional parameters
       from argparse import ArgumentParser
       parser = ArgumentParser(description=usage(1))
    """
    args = parser.parse_args()
    opts = vars(args)
    defs = vars(parser.parse_args([])) # defaults only

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items() :
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


def merge_status(stnda, grinds=(0,1,2,3,4), dtype=np.uint64): # indexes stand gain ranges for 'FH','FM','FL','AHL-H','AML-M'
    """Merges status bits over gain range index.
       Originaly intended for epix10ka(quad/2m) status array stnda.shape=(7, 16, 352, 384) merging to (16, 352, 384)
       Also can be used with Jungfrau status array stnda.shape=(7, 8, 512, 512) merging to (8, 512, 512)
       option "indexes" contains a list of stnda[i,:] indexes to combine status
    """
    if stnda.ndim < 2: return stnda # ignore 1-d arrays
    _stnda = stnda.astype(dtype)
    st1 = np.copy(_stnda[grinds[0],:])
    for i in grinds[1:]: # range(1,stnda.shape[0]) :
        if i<stnda.shape[0]: # boundary check for index
            np.bitwise_or(st1, _stnda[i,:], out=st1)
    return st1
    #print(info_ndarr(st1,    'XXX st1   '))
    #print(info_ndarr(_stnda, 'XXX stnda '))


def mask_neighbors(mask, allnbrs=True, dtype=np.uint8):
    """Return mask with masked eight neighbor pixels around each 0-bad pixel in input mask.
       mask   : int - n-dimensional (n>1) array with input mask
       allnbrs: bool - False/True - masks 4/8 neighbor pixels.
    """
    shape_in = mask.shape
    if mask.ndim < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(shape_in))

    mask_out = np.copy(mask, dtype) # np.asarray(mask, dtype)

    if mask.ndim == 2:
        # mask nearest neighbors
        mask_out[0:-1,:] = np.logical_and(mask_out[0:-1,:], mask[1:,  :])
        mask_out[1:,  :] = np.logical_and(mask_out[1:,  :], mask[0:-1,:])
        mask_out[:,0:-1] = np.logical_and(mask_out[:,0:-1], mask[:,1:  ])
        mask_out[:,1:  ] = np.logical_and(mask_out[:,1:  ], mask[:,0:-1])
        if allnbrs:
          # mask diagonal neighbors
          mask_out[0:-1,0:-1] = np.logical_and(mask_out[0:-1,0:-1], mask[1:  ,1:  ])
          mask_out[1:  ,0:-1] = np.logical_and(mask_out[1:  ,0:-1], mask[0:-1,1:  ])
          mask_out[0:-1,1:  ] = np.logical_and(mask_out[0:-1,1:  ], mask[1:  ,0:-1])
          mask_out[1:  ,1:  ] = np.logical_and(mask_out[1:  ,1:  ], mask[0:-1,0:-1])

    else: # mask.ndim > 2

        mask_out.shape = mask.shape = shape_nda_to_3d(mask)

        # mask nearest neighbors
        mask_out[:, 0:-1,:] = np.logical_and(mask_out[:, 0:-1,:], mask[:, 1:,  :])
        mask_out[:, 1:,  :] = np.logical_and(mask_out[:, 1:,  :], mask[:, 0:-1,:])
        mask_out[:, :,0:-1] = np.logical_and(mask_out[:, :,0:-1], mask[:, :,1:  ])
        mask_out[:, :,1:  ] = np.logical_and(mask_out[:, :,1:  ], mask[:, :,0:-1])
        if allnbrs:
          # mask diagonal neighbors
          mask_out[:, 0:-1,0:-1] = np.logical_and(mask_out[:, 0:-1,0:-1], mask[:, 1:  ,1:  ])
          mask_out[:, 1:  ,0:-1] = np.logical_and(mask_out[:, 1:  ,0:-1], mask[:, 0:-1,1:  ])
          mask_out[:, 0:-1,1:  ] = np.logical_and(mask_out[:, 0:-1,1:  ], mask[:, 1:  ,0:-1])
          mask_out[:, 1:  ,1:  ] = np.logical_and(mask_out[:, 1:  ,1:  ], mask[:, 0:-1,0:-1])

        mask_out.shape = mask.shape = shape_in

    return mask_out


def mask_edges(mask, mrows=1, mcols=1, dtype=np.uint8):
    """Return mask with a requested number of row and column pixels masked - set to 0.
       mask : int - n-dimensional (n>1) array with input mask
       mrows: int - number of edge rows to mask
       mcols: int - number of edge columns to mask
    """
    sh = mask.shape
    if mask.ndim < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(sh))

    mask_out = np.asarray(mask, dtype)

    # print 'shape:', sh

    if mask.ndim == 2:
        rows, cols = sh

        if mrows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0:
          # mask edge rows
          mask_rows = np.zeros((mrows,cols), dtype=mask.dtype)
          mask_out[:mrows ,:] = mask_rows
          mask_out[-mrows:,:] = mask_rows

        if mcols>0:
          # mask edge colss
          mask_cols = np.zeros((rows,mcols), dtype=mask.dtype)
          mask_out[:,:mcols ] = mask_cols
          mask_out[:,-mcols:] = mask_cols

    else: # shape > 2
        mask_out.shape = shape_nda_to_3d(mask)

        segs, rows, cols = mask_out.shape

        if mrows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0:
          # mask edge rows
          mask_rows = np.zeros((segs,mrows,cols), dtype=mask.dtype)
          mask_out[:, :mrows ,:] = mask_rows
          mask_out[:, -mrows:,:] = mask_rows

        if mcols>0:
          # mask edge colss
          mask_cols = np.zeros((segs,rows,mcols), dtype=mask.dtype)
          mask_out[:, :,:mcols ] = mask_cols
          mask_out[:, :,-mcols:] = mask_cols

        mask_out.shape = sh

    return mask_out


def merge_masks(mask1=None, mask2=None, dtype=np.uint8):
    """Merging masks using np.logical_and rule: (0,1,0,1)^(0,0,1,1) = (0,0,0,1)
    """
    assert mask1.size == mask2.size, 'Mask sizes should be equal'

    if mask1 is None: return mask2
    if mask2 is None: return mask1

    if shape1 != shape2:
        if mask1.ndim > mask2.ndim: mask2.shape = mask1.shape
        else                      : mask1.shape = mask2.shape

    mask = np.logical_and(mask1, mask2)
    return mask if dtype==np.bool else np.asarray(mask, dtype)

# EOF
