#!/usr/bin/env python

"""
:py:class:`UtilsPSF` - module for geometry conversion from psana to psf format
==============================================================================
PSF stands for asic (0,0) pixel Position, Slow, and Fast orthogonal vectors along rows and columns, respectively.
All vectors (list-of-tuples) are presented in the same units, micrometers [um] for psana.
Slow and fast vector module is equal to the pixel size in row and column directions, respectively.

USAGE::

    # User's useful methods only...

    import psana.pscalib.geometry.UtilsPSF as ups
    # or from psana.pscalib.geometry.UtilsPSF import *
    from psana.pscalib.geometry.UtilsPSF import CFRAME_PSANA, CFRAME_LAB

    psf,sego,geo = ups.psf_from_geo(geo) # geo (GeometryAccess) - detector (or its part) geometry description object.

    psf,sego,geo = ups.psf_from_file(fname_geometry) # fname_geometry (str) - psana geometry file name.

    print(ups.info_psf(psf)) # psf (list-of-tuples) shapesd as (<number-of-asics>, 3(vectors vp, vs, vf), 3(vector components x,y,z)).

    ups.savetext_psf(psf, fname='psf.txt',\
                 fmtp='\n%12.3f %12.3f %12.3f',\
                 fmts='  %8.3f %8.3f %8.3f',\
                 fmtf='  %8.3f %8.3f %8.3f', title='') # save psf as formatted text file.

    ups.save_psf(psf, fname='psf.npy') # save psf as numpy file.

    psf = ups.load_psf(fname) # loads psf vectors as list-of-tuples from *.npy file.

    datapsf = ups.data_psf(sego, data) # converts psana data to psf data. sego (SegGeometry) - segment geometry description object.

    arrx, arry, arrz = ups.pixel_coords_psf(psf, ashape) # retrieves pixel coordinate arrays. ashape (tuple of len==2) - ASIC 2-d shape.

    inds = ups.indices(values, bin_size, offset=None) # converts numpy array of values to indices is (int) (values-vmin)/bin_size.


2021-12-06 created
2025-04-07 adopted to lcls2

@author Mikhail Dubrovin
"""
#from time import time

import numpy as np
import logging
logger = logging.getLogger(__name__)

CFRAME_PSANA, CFRAME_LAB = 0, 1

def info_geo(geo):
    """Returns (str) name and index info of the GeometryObject."""
    return 'segment name: %s index: %d' % (geo.oname, geo.oindex)


def info_seg_geo(sego):
    """Returns (str) info of the SegGeometry object parameters."""
    return 'per-ASIC info'\
      + '\n  SegGeometry implementation class: %s' % sego.name()\
      + '\n  asic0ind: %s' % str(sego.asic0indices())\
      + '\n  arows: %d acols: %d' % sego.asic_rows_cols()\
      + '\n  ssize: %d' % sego.size()\
      + '\n  sego.shape(): %s' % str(sego.shape())\
      + '\n  pix_size, um: %f' % sego.pixel_scale_size()\
      + '\n  nasics_in_rows: %d nasics_in_cols: %d' % sego.number_of_asics_in_rows_cols()


def info_psf(psf,\
      fmtp='\np=(%12.2f, %12.2f, %12.2f)',\
      fmts='  s=(%8.2f, %8.2f, %8.2f)',\
      fmtf='  f=(%8.2f, %8.2f, %8.2f)', title=''):
    """Returns (str) content of the psf vectors."""
    s = title
    fmt = fmtp + fmts + fmtf
    for (px,py,pz), (sx,xy,xz), (fx,fy,fz) in psf:
        s += fmt % (px,py,pz,  sx,xy,xz,  fx,fy,fz)
    return s


def panel_psf(sego, x, y, z):
    """Returns psf (list-of-tuples) vectors for ASICs of a single segment.
       Parameters:
       - sego (SegGeometry) - segment description geometry object
       - x, y, z (float) - segment pixel coordimane arrays (in the detector coordinate frame)
    """
    return [((x[r0,c0], y[r0,c0], z[r0,c0]),\
            (x[r0+1,c0]-x[r0,c0],\
             y[r0+1,c0]-y[r0,c0],\
             z[r0+1,c0]-z[r0,c0]),\
            (x[r0,c0+1]-x[r0,c0],\
             y[r0,c0+1]-y[r0,c0],\
             z[r0,c0+1]-z[r0,c0])) for (r0,c0) in sego.asic0indices()]


def psf_from_file(fname, cframe=CFRAME_LAB):
    """
    Parameters:
    -----------
       - fname (str) - psana detector geometry file name.
       - cframe (int) - 0/1 = CFRAME_PSANA/CFRAME_LAB - psana/LAB coordinate frame.
    """
    logger.info('load geometry from file %s' % fname)
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

    geo = GeometryAccess(fname, 0, use_wide_pix_center=False)
    return psf_from_geo(geo, cframe)


def psf_from_geo(geo, cframe=CFRAME_LAB):
    """
    Parameters:
    -----------
       - geo (GeometryAccess) - psana geometry description object.
       - cframe (int) - 0/1 = CFRAME_PSANA/CFRAME_LAB - psana/LAB coordinate frame.
    """
    logger.info('psf_from_geo - converts geometry constants from psana to psf format')

    geo1 = geo.get_seg_geo() #sgs.Create(segname=segname, pbits=0)
    sego = geo1.algo
    srows, scols = sego.shape()

    logger.debug('%s\n%s' % (info_geo(geo1), info_seg_geo(sego)))

    x, y, z = geo.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=cframe)
    #logger.debug(info_ndarr(x, name='psana x', first=0, last=4))
    #logger.debug(info_ndarr(y, name='psana y', first=0, last=4))
    #logger.debug(info_ndarr(z, name='psana z', first=0, last=4))

    nsegs = int(x.size/sego.size())
    shape = (nsegs, srows, scols)
    logger.debug('geo shape: %s' % str(shape))
    x.shape = y.shape = z.shape = shape

    lst = None
    for n in range(nsegs):
        incr = panel_psf(sego, x[n,:], y[n,:], z[n,:])
        if lst is None: lst = incr
        else: lst += incr

    return lst, sego, geo


def savetext_psf(psf, fname='psf.txt',\
                 fmtp='\n%12.3f %12.3f %12.3f',\
                 fmts='  %8.3f %8.3f %8.3f',\
                 fmtf='  %8.3f %8.3f %8.3f', title=''):
    """Save psf vectors as text, each line has 3 vecotrs for ASIC: position, slow, fast."""
    if fname is None:
       logger.info('savetext_psf file name is None, file is not saved')
       return

    f = open(fname,'w')
    f.write(info_psf(psf, fmtp, fmts, fmtf, title))
    f.close()
    logger.info('geometry constants in psf format saved as text in: %s' % fname)


def save_psf(psf, fname='psf.npy'):
    """Saves psf vectors as numpy array."""
    nda = np.array(psf)
    np.save(fname, nda)
    logger.info('geometry constants in psf format saved as numpy array in: %s' % fname)


def load_psf(fname):
    """Loads psf vectors from *.npy file. and returns it as a list"""
    assert isinstance(fname, str) and fname.split('.')[-1]=='npy', 'file name is not a str object or not *.npy'
    return list(np.load(fname))


def list_of_panel_asic_data(sego, segdata):
    """Returns list of 2-d ASIC data arrays in a single segment."""
    arows, acols = sego.asic_rows_cols()
    return [segdata[r0:r0+arows,c0:c0+acols] for (r0,c0) in sego.asic0indices()]


def data_psf(sego, data):
    """Returns (numpy.array) of data shaped per-ASIC, shape=(<number-of-asics>, <asic-rows>, <asic-cols>).
       Parameters:
       - sego [SegmentGeometry] - psana segment geometry description object.
       - data [np.array] - psana data shaped per-segment, shape=(<number-of-segments>, <segment-rows>, <segment-cols>).
    """
    #logger.debug('data_psf - conversion of psana per-segmment data to psf per-asic data\n%s' % info_seg_geo(sego))
    shape0 = data.shape
    srows, scols = sego.shape()
    nsegs = int(data.size/sego.size())
    shape = (nsegs, srows, scols)
    logger.debug('nsegs in data: %d data shape: %s per-segment shape: %s' % (nsegs, str(shape0), str(shape)))
    data.shape = shape

    #t0_sec = time()
    list_asic_data = [] # list of per ASIC 2-d arrays of the detector data
    for n in range(nsegs):
        list_asic_data += list_of_panel_asic_data(sego, data[n,:]) #90us
    #logger.debug(info_ndarr(list_asic_data,'data_psf.list_asic_data consumed time=%.6fs:' % (time()-t0_sec)))

    data.shape = shape0
    return np.array(list_asic_data)


def psf_vectors(psf):
    """Converts input psf list-of-tuples to list-of-np.arrays (for vector operations)."""
    return [(np.array(vp), np.array(vs), np.array(vf)) for vp,vs,vf in psf]


def pixel_coords_psf_direct(psf, shape_asic):
    """It works, but... takes ~10sec per cspad"""
    vpsf = psf_vectors(psf)
    arows, acols = shape_asic
    coords = np.array([vp + r*vs + c*vf for vp,vs,vf in vpsf for r in range(arows) for c in range(acols)])
    return coords[:,0], coords[:,1], coords[:,2]


def coords_1d(cp, cs, cf, shape_asic):
    """evaluation of ASIC pixel coordinates for 1-d specified by vector commponents cp, cs, cf"""
    ccols = np.arange(shape_asic[1])*cf
    crows = np.arange(shape_asic[0])*cs
    grid = np.meshgrid(ccols,crows)
    return grid[0] + grid[1] + cp


def pixel_coords_psf(psf, shape_asic):
    """returns pixel coordinate arrays for x, y and z. It takes ~70ms for cspad"""
    return [np.array([coords_1d(vp[i], vs[i], vf[i], shape_asic) for vp,vs,vf in psf]) for i in range(3)]


def indices(values, bin_size, offset=None):
    """returns (numpy.array) array of indices from (numpy.array) of values.
       values (numpy.array) - array of values
       bin_size (float) - in the same units as coordinates
       offset (float) - in the same units as coordinates
    """
    vmin = values.min() if offset is None else offset
    return np.array((values-vmin)/bin_size, dtype=np.uint)


if __name__ == "__main__":
  import sys
  sys.exit('Tests moved to psana.pscalib.geometry.test_UtilsPSF.py'\
           '\nUsed in psana.pscalib.geometry.GeometryAccess.py')

# EOF
