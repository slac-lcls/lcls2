#!/usr/bin/env python
#------------------------------
"""
:py:class:`NDArrIO` - i/o methods to read/write numpy array in the text file
============================================================================

Usage::

    # Import
    from PSCalib.NDArrIO import save_txt, load_txt, list_of_comments

    # Save n-dimensional numpy array in the text file.
    save_txt(fname, arr, cmts=(), fmt='%.1f')

    # Load 1-, 2-, n-dimensional array (if metadata available) from file .
    arr = load_txt(fname)    # this version unpacks data directly in this script
    # or
    arr = load_txt_v2(fname) # v2 uses numpy.loadtxt(...) to load data (~30% slower then the load_txt) 

    # Get list of str objects - comment records with '#' in 1st position from file.
    cmts = list_of_comments(fname)

    #------------------------------
    # Example of the file header:
    #------------------------------
    # TITLE      File to load ndarray of calibration parameters
    # 
    # EXPERIMENT amo12345
    # DETECTOR   Camp.0:pnCCD.1
    # CALIB_TYPE pedestals

    # DATE_TIME  2014-05-06 15:24:10
    # AUTHOR     <user-login-name>

    # line of comment always begins with # 
    # Mandatory fields to define the ndarray<TYPE,NDIM> and its shape as unsigned shape[NDIM] = (DIM1,DIM2,DIM3)
    # DTYPE       float
    # NDIM        3
    # DIM:1       3
    # DIM:2       4
    # DIM:3       8
    #------------------------------

See: :py:class:`AreaDetector`

For more detail see `AreaDetector <https://pswww.slac.stanford.edu/swdoc/releases/ana-current/pyana-ref/html/Detector/#module-Detector.AreaDetector>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

#import os
#import sys
#import math
import numpy as np
import PSCalib.GlobalUtils as gu

  
def save_txt(fname='nda.txt', arr=None, cmts=(), fmt='%.1f', verbos=False, addmetad=True) :
    """Save n-dimensional numpy array to text file with metadata.
       - fname - file name for text file,
       - arr - numpy array,
       - cmts -list of comments which will be saved in the file header.
    """
    #recs = ['# %03d %s' % (i,cmt) for i, cmt in enumerate(cmts)]
    recs = ['# %s' % cmt for cmt in cmts]
    recs.append('\n# HOST        %s' % gu.get_hostname())
    recs.append('# WORK_DIR    %s' % gu.get_cwd())
    recs.append('# FILE_NAME   %s' % fname)
    recs.append('# DATE_TIME   %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'))
    recs.append('# UID         %s' % gu.get_login())
    recs.append('# SHAPE       %s' % str(arr.shape).replace(' ',''))
    recs.append('# DATATYPE    %s' % str(arr.dtype))

    if addmetad :
        recs.append('\n# DTYPE       %s' % str(arr.dtype))
        recs.append('# NDIM        %s' % len(arr.shape))

        for i in range(len(arr.shape)) :
            recs.append('# DIM:%d       %s'   % (i, arr.shape[i]))

    arr2d = gu.reshape_nda_to_2d(arr)

    # pretty formatting
    recs.append('' if len(arr.shape)>1 else '\n')
    nline = '\n' if len(arr.shape)>1 else ' '

    hdr = '\n'.join(recs)
    #print hdr

    np.savetxt(fname, arr, fmt, delimiter=' ', newline=nline, header=hdr, comments='') #, footer='\n') #, comments='# ')
    if verbos : print 'File %s is saved' % fname

#------------------------------

def _unpack_data(recs) :
    """Reconstruct data records from file to 2-d (or 1-d) list of values.
    """
    if len(recs) == 0 : return None 

    if len(recs) == 1 : 
        for rec in recs :
            fields = rec.strip('\n').split()
            return [float(v) for v in fields]
        
    arr = []
    for rec in recs :
        fields = rec.strip('\n').split()
        vals = [float(v) for v in fields]
        arr.append(vals)

    return arr

#------------------------------

def _metadata_from_comments(cmts) :
    """Returns metadata from the list of comments
    """
    str_dtype = ''
    dtype = None
    ndim = None
    shape = []

    if cmts is not None :
        for rec in cmts :
            fields = rec.split(' ', 2)
            if len(fields)<3 : continue
            if   fields[1] == 'DTYPE'    : str_dtype = fields[2].rstrip('\n').strip(' ')
            elif fields[1] == 'NDIM'     : ndim = int(fields[2])
            elif fields[1][:4] == 'DIM:' : shape.append(int(fields[2]))

    dtype = np.dtype(str_dtype) if str_dtype else np.float32

    return ndim, shape, dtype

#------------------------------

def list_of_comments(fname) :
    """Returns list of str objects - comment records from file.
       - fname - file name for text file.
    """
    #if not os.path.lexists(fname) : raise IOError('File %s is not available' % fname)

    f=open(fname,'r')

    cmts = []
    for rec in f :
        if rec.isspace() : continue # ignore empty lines
        elif rec[0] == '#' : cmts.append(rec.rstrip('\n'))
        else : break

    f.close()

    if len(cmts)==0 :
        return None

    return cmts

#------------------------------

def load_txt_v2(fname) :
    """Reads n-dimensional numpy array from text file with metadata.
       - fname - file name for text file.
    """
    cmts = list_of_comments(fname)
    ndim, shape, dtype = _metadata_from_comments(cmts)

    nparr = np.loadtxt(fname, dtype=dtype, comments='#')

    if dtype is None or ndim is None or shape==[] :
        # Retun data as is shaped in the text file for 1-d or 2-d
        return nparr

    if ndim != len(shape) :
        str_metad = 'dtype=%s ndim=%d shape=%s' % (str(dtype), ndim, str(shape))
        raise IOError('Inconsistent metadata (ndim != len(shape)) in file: %s' % str_metad)

    if ndim > 2: nparr.shape = shape

    return nparr
    
#------------------------------

def load_txt(fname) :
    """Reads n-dimensional numpy array from text file with metadata.
       - fname - file name for text file.
    """
    #if not os.path.lexists(fname) : raise IOError('File %s is not available' % fname)

    # Load all records from file
    f=open(fname,'r')
    recs = f.readlines()
    f.close()

    # Sort records for comments and data, discard empty records
    cmts = []
    data = []
    for rec in recs :
        if rec.isspace() : continue # ignore empty lines
        if rec[0] == '#' : cmts.append(rec)
        else             : data.append(rec)

    if data == [] :
        raise IOError('Data is missing in the file %s' % fname)

    # Get metadata from comments
    ndim, shape, dtype = _metadata_from_comments(cmts)

    # Unpack data records to 2-d list of values and convert it to np.array
    nparr = np.array(_unpack_data(data), dtype)

    if ndim is None or shape==[] or dtype is None :
        # Retun data as is shaped in the text file for 1-d or 2-d
        return nparr

    if ndim != len(shape) :
        str_metad = 'dtype=%s ndim=%d shape=%s' % (str(dtype), ndim, str(shape))
        raise IOError('Inconsistent metadata (NDIM != len(shape)) in file: %s' % str_metad)

    if ndim > 2: nparr.shape = shape

    return nparr

#------------------------------
#----------  TEST  ------------
#------------------------------

def test_save_txt() :

    arr3 = (((111,112,113),
             (121,122,123)),
            ((211,212,213),
             (221,222,223)))

    arr2 = ((11,12,13),
            (21,22,23))

    arr1 = (1,2,3,4,5)

    npa  = np.array(np.array(arr3), dtype=np.int)
    save_txt('nda.txt', npa, cmts=('Test of PSCalib.NDArrIO.save_txt(...)', 'save numpy array in the text file'), verbos=True)
             
#------------------------------

def test_load_txt() :

    from time import time

    fname = 'nda.txt'
    fname = '/reg/g/psdm/detector/alignment/andor3d/calib-andor3d-2016-02-09/calib/Andor3d::CalibV1/SxrEndstation.0:DualAndor.0/pedestals/7-end.data'
    #fname = '/reg/d/psdm/cxi/cxif5315/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/pedestals/1-end.data'

    print 'Test list_of_comments:'
    t0_sec = time()
    cmts = list_of_comments(fname)
    print 'Consumed time for list_of_comments = %10.6f sec' % (time()-t0_sec)
    if cmts is not None :
        for cmt in cmts: print cmt

    t0_sec = time()
    nparr = load_txt(fname)
    #nparr = load_txt_v1(fname)
    #nparr = load_txt_v2(fname)
    print 'Consumed time for load_txt = %10.6f sec' % (time()-t0_sec)

    print 'Test np.array from file:\n', nparr
    print 'nparr.shape:', nparr.shape
    print 'nparr.dtype:', nparr.dtype

#------------------------------

if __name__ == "__main__" :
    test_save_txt()
    test_load_txt()

#------------------------------
