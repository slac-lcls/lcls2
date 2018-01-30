#!/usr/bin/env python
#------------------------------
"""
:py:class:`GlobalUtils` - a set of utilities
============================================

Usage::

    # Import
    import PSCalib.GlobalUtils as gu

    # Methods
    #resp = gu.<method(pars)>

    dettype = gu.det_type_from_source(source)
    detname = gu.string_from_source(source)

    mmask = gu.merge_masks(mask1=None, mask2=None, dtype=np.uint8)
    mask  = gu.mask_neighbors(mask_in, allnbrs=True, dtype=np.uint8)
    lo,hi = gu.evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=1000, verbos=1, cmt='')

    arr2d = gu.reshape_nda_to_2d(nda)
    arr3d = gu.reshape_nda_to_3d(nda)

    # Get string with time stamp, ex: 2016-01-26T10:40:53
    ts    = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None)

    usr   = gu.get_enviroment(env='USER')
    usr   = gu.get_login()
    host  = gu.get_hostname()
    cwd   = gu.get_cwd()
    fmode = gu.file_mode(fname)
    rec   = gu.log_rec_on_start() # e.g. '2017-09-27T10:40:24 user:dubrovin@psanagpu104 cwd:/reg/neh/home4/dubrovin/LCLS/con-jungfrau ...'
    gu.add_rec_to_log(lfname, rec, verbos=False)

    exp  = gu.exp_name(env)
    cdir = gu.calib_dir(env)
    #### tsec, tnsec, fiducial, tsdate, tstime = gu.time_pars(evt) # needs in psana...

    gu.create_directory(dir, verb=False)
    gu.create_directory_with_mode(dir, mode=0777, verb=False)
    exists = gu.create_path(path, depth=6, mode=0777, verb=True)

    arr  = gu.load_textfile(path)
    gu.save_textfile(text, path, mode='w') # mode: 'w'-write, 'a'-append 

    path = gu.replace('/path/#YYYY-MM/fname.txt', '#YYYY-MM', gu.str_tstamp(fmt='%Y/%m'))

    ifname = 'fname.txt'
    ctypedir = '/some-path/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/'
    ctype = 'pedestals'
    ofname = '123-end.data'
    rec = gu.history_record(ifname, ctypedir, ctype, ofname, comment='')
    path_clb = gu.path_to_calib_file(ctypedir, ctype, ofname)
    path_his = gu.path_to_history_file(ctypedir, ctype)

    cmd = gu.command_deploy_file(ifname, path_clb)
    cmd = gu.command_add_record_to_file(rec, path_his)

    tpl = gu.calib_fname_template(exp, runnum, tsec, tnsec, fid, tsdate, tstime, src, nevts, ofname)

    gu.deploy_file(ifname, ctypedir, ctype, ofname, lfname=None, verbos=False)

See: other methods in :py:class:`CalibPars`, :py:class:`CalibParsStore`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
"""
#--------------------------------

import sys
import os
from stat import ST_MODE
import getpass
import socket
import numpy as np
from time import localtime, strftime

#------------------------------

DIR_INS = '/reg/d/psdm'

#------------------------------

# ATTENTION !!!!! ALL LISTS SHOULD BE IN THE SAME ORDER (FOR DICTIONARIES)

# Enumerated and named parameters

PEDESTALS    = 0
PIXEL_STATUS = 1
PIXEL_RMS    = 2
PIXEL_GAIN   = 3
PIXEL_MASK   = 4
PIXEL_BKGD   = 5
COMMON_MODE  = 6
GEOMETRY     = 7
PIXEL_OFFSET = 8
PIXEL_DATAST = 9

calib_types  = ( PEDESTALS,   PIXEL_STATUS,   PIXEL_RMS,   PIXEL_GAIN,   PIXEL_MASK,   PIXEL_BKGD,   COMMON_MODE,   GEOMETRY,   PIXEL_OFFSET,   PIXEL_DATAST)
calib_names  = ('pedestals', 'pixel_status', 'pixel_rms', 'pixel_gain', 'pixel_mask', 'pixel_bkgd', 'common_mode', 'geometry', 'pixel_offset', 'pixel_datast')
calib_dtypes = ( np.float32,  np.uint16,      np.float32,  np.float32,   np.uint8,     np.float32,   np.double,     str,        np.float32,     np.uint16)

dic_calib_type_to_name  = dict(zip(calib_types, calib_names))
dic_calib_name_to_type  = dict(zip(calib_names, calib_types))
dic_calib_type_to_dtype = dict(zip(calib_types, calib_dtypes))

LOADED     = 1
DEFAULT    = 2
UNREADABLE = 3
UNDEFINED  = 4
WRONGSIZE  = 5
NONFOUND   = 6
DCSTORE    = 7

calib_statvalues = ( LOADED,   DEFAULT,   UNREADABLE,   UNDEFINED,   WRONGSIZE,   NONFOUND,   DCSTORE)
calib_statnames  = ('LOADED', 'DEFAULT', 'UNREADABLE', 'UNDEFINED', 'WRONGSIZE', 'NONFOUND', 'DCSTORE')

dic_calib_status_value_to_name = dict(zip(calib_statvalues, calib_statnames))
dic_calib_status_name_to_value = dict(zip(calib_statnames,  calib_statvalues))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

UNDEFINED   = 0
CSPAD       = 1 
CSPAD2X2    = 2 
PRINCETON   = 3 
PNCCD       = 4 
TM6740      = 5 
OPAL1000    = 6 
OPAL2000    = 7 
OPAL4000    = 8 
OPAL8000    = 9 
ORCAFL40    = 10
EPIX        = 11
EPIX10K     = 12
EPIX100A    = 13
FCCD960     = 14
ANDOR       = 15
ACQIRIS     = 16
IMP         = 17
QUARTZ4A150 = 18
RAYONIX     = 19
EVR         = 20
FCCD        = 21
TIMEPIX     = 22
FLI         = 23
PIMAX       = 24
ANDOR3D     = 25
JUNGFRAU    = 26
ZYLA        = 27
EPICSCAM    = 28
EPIX10KA    = 29

#XAMPS    # N/A data
#FEXAMP   # N/A data
#PHASICS  # N/A data
#OPAL1600 # N/A data
#EPIXS    # N/A data
#GOTTHARD # N/A data
""" Enumetated detector types"""

list_of_det_type = (UNDEFINED, CSPAD, CSPAD2X2, PRINCETON, PNCCD, TM6740, \
                    OPAL1000, OPAL2000, OPAL4000, OPAL8000, \
                    ORCAFL40, EPIX, EPIX10K, EPIX100A, FCCD960, ANDOR, ACQIRIS, IMP, QUARTZ4A150, RAYONIX,
                    EVR, FCCD, TIMEPIX, FLI, PIMAX, ANDOR3D, JUNGFRAU, ZYLA, EPICSCAM, EPIX10KA)
""" List of enumetated detector types"""

list_of_det_names = ('UNDEFINED', 'Cspad', 'Cspad2x2', 'Princeton', 'pnCCD', 'Tm6740', \
                     'Opal1000', 'Opal2000', 'Opal4000', 'Opal8000', \
                     'OrcaFl40', 'Epix', 'Epix10k', 'Epix100a', 'Fccd960', 'Andor', 'Acqiris', 'Imp', 'Quartz4A150', 'Rayonix',\
                     'Evr', 'Fccd', 'Timepix', 'Fli', 'Pimax', 'Andor3d', 'Jungfrau', 'Zyla', 'ControlsCamera', 'Epix10ka')
""" List of enumetated detector names"""

list_of_calib_groups = ('UNDEFINED',
                        'CsPad::CalibV1',
                        'CsPad2x2::CalibV1',
                        'Princeton::CalibV1',
                        'PNCCD::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Epix::CalibV1',
                        'Epix10k::CalibV1',
                        'Epix100a::CalibV1',
                        'Camera::CalibV1',
                        'Andor::CalibV1',
                        'Acqiris::CalibV1',
                        'Imp::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'EvrData::CalibV1',
                        'Camera::CalibV1',
                        'Timepix::CalibV1',
                        'Fli::CalibV1',
                        'Pimax::CalibV1',
                        'Andor3d::CalibV1',
                        'Jungfrau::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Epix10ka::CalibV1',
                        )
""" List of enumetated detector calibration groups"""

dic_det_type_to_name = dict(zip(list_of_det_type, list_of_det_names))
""" Dictionary for detector type : name"""

dic_det_type_to_calib_group = dict(zip(list_of_det_type, list_of_calib_groups))
""" Dictionary for detector type : group"""

#------------------------------
bld_names = \
['EBeam',
'PhaseCavity',
'FEEGasDetEnergy',
'Nh2Sb1Ipm01',
'HxxUm6Imb01',
'HxxUm6Imb02',
'HfxDg2Imb01',
'HfxDg2Imb02',
'XcsDg3Imb03',
'XcsDg3Imb04',
'HfxDg3Imb01',
'HfxDg3Imb02',
'HxxDg1Cam',
'HfxDg2Cam',
'HfxDg3Cam',
'XcsDg3Cam',
'HfxMonCam',
'HfxMonImb01',
'HfxMonImb02',
'HfxMonImb03',
'MecLasEm01',
'MecTctrPip01',
'MecTcTrDio01',
'MecXt2Ipm02',
'MecXt2Ipm03',
'MecHxmIpm01',
'GMD',
'CxiDg1Imb01',
'CxiDg2Imb01',
'CxiDg2Imb02',
'CxiDg4Imb01',
'CxiDg1Pim',
'CxiDg2Pim',
'CxiDg4Pim',
'XppMonPim0',
'XppMonPim1',
'XppSb2Ipm',
'XppSb3Ipm',
'XppSb3Pim',
'XppSb4Pim',
'XppEndstation0',
'XppEndstation1',
'MecXt2Pim02',
'MecXt2Pim03',
'CxiDg3Spec',
'Nh2Sb1Ipm02',
'FeeSpec0',
'SxrSpec0',
'XppSpec0',
'XcsUsrIpm01',
'XcsUsrIpm02',
'XcsUsrIpm03',
'XcsUsrIpm04',
'XcsSb1Ipm01',
'XcsSb1Ipm02',
'XcsSb2Ipm01',
'XcsSb2Ipm02',
'XcsGonIpm01',
'XcsLamIpm01',
'XppAin01',
'XcsAin01',
'AmoAin01']


#------------------------------

def det_type_from_source(source) :
    """ Returns enumerated detector type for string source
    """
    str_src = str(source)
    if   ':Cspad.'          in str_src : return CSPAD
    elif ':Cspad2x2.'       in str_src : return CSPAD2X2
    elif ':pnCCD.'          in str_src : return PNCCD
    elif ':Princeton.'      in str_src : return PRINCETON
    elif ':Andor.'          in str_src : return ANDOR
    elif ':Epix100a.'       in str_src : return EPIX100A
    elif ':Epix10k.'        in str_src : return EPIX10K
    elif ':Epix.'           in str_src : return EPIX
    elif ':Opal1000.'       in str_src : return OPAL1000
    elif ':Opal2000.'       in str_src : return OPAL2000
    elif ':Opal4000.'       in str_src : return OPAL4000
    elif ':Opal8000.'       in str_src : return OPAL8000
    elif ':Tm6740.'         in str_src : return TM6740
    elif ':OrcaFl40.'       in str_src : return ORCAFL40
    elif ':Fccd960.'        in str_src : return FCCD960
    elif ':Acqiris.'        in str_src : return ACQIRIS
    elif ':Imp.'            in str_src : return IMP
    elif ':Quartz4A150.'    in str_src : return QUARTZ4A150
    elif ':Rayonix.'        in str_src : return RAYONIX
    elif ':Evr.'            in str_src : return EVR
    elif ':Fccd.'           in str_src : return FCCD
    elif ':Timepix.'        in str_src : return TIMEPIX
    elif ':Fli.'            in str_src : return FLI
    elif ':Pimax.'          in str_src : return PIMAX
    elif ':DualAndor.'      in str_src : return ANDOR3D
    elif ':Jungfrau.'       in str_src : return JUNGFRAU
    elif ':Zyla.'           in str_src : return ZYLA
    elif ':ControlsCamera.' in str_src : return EPICSCAM
    elif ':Epix10ka.'       in str_src : return EPIX10KA
    else                               : return UNDEFINED

#------------------------------
##-----------------------------
#------------------------------

def string_from_source(source) :
  """Returns string like "CxiDs2.0:Cspad.0" from "Source('DetInfo(CxiDs2.0:Cspad.0)')" or "Source('DsaCsPad')"
  """
  str_in_quots = str(source).split('"')[1]
  str_split = str_in_quots.split('(') 
  return str_split[1].rstrip(')') if len(str_split)>1 else str_in_quots

##-----------------------------

def shape_nda_to_2d(arr) :
    """Return shape of np.array to reshape to 2-d
    """
    sh = arr.shape
    if len(sh)<3 : return sh
    return (arr.size/sh[-1], sh[-1])

##-----------------------------

def shape_nda_to_3d(arr) :
    """Return shape of np.array to reshape to 3-d
    """
    sh = arr.shape
    if len(sh)<4 : return sh
    return (arr.size/sh[-1]/sh[-2], sh[-2], sh[-1])

##-----------------------------

def reshape_nda_to_2d(arr) :
    """Reshape np.array to 2-d
    """
    sh = arr.shape
    if len(sh)<3 : return arr
    arr.shape = (arr.size/sh[-1], sh[-1])
    return arr

##-----------------------------

def reshape_nda_to_3d(arr) :
    """Reshape np.array to 3-d
    """
    sh = arr.shape
    if len(sh)<4 : return arr
    arr.shape = (arr.size/sh[-1]/sh[-2], sh[-2], sh[-1])
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

        mask_out.shape = mask.shape = shape_nda_to_3d(mask)       

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
        mask_out.shape = shape_nda_to_3d(mask)       

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

##-----------------------------

def evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=1000, verbos=1, cmt='') :
    """Evaluates low and high limit of the array, which are used to find bad pixels.
    """
    ave, std = (arr.mean(), arr.std()) # if (nneg>0 or npos>0) else (-1,-1)
    lo = ave-nneg*std if nneg>0 else lim_lo
    hi = ave+npos*std if npos>0 else lim_hi
    lo, hi = max(lo, lim_lo), min(hi, lim_hi)

    if verbos & 1 :
        print '  %s: %s ave, std = %.3f, %.3f  low, high limits = %.3f, %.3f'%\
              (sys._getframe().f_code.co_name, cmt, ave, std, lo, hi)

    return lo, hi

##-----------------------------

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None) :
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    return strftime(fmt, localtime(time_sec))

#------------------------------

def get_enviroment(env='USER') :
    """Returns the value of specified by string name environment variable
    """
    return os.environ[env]

#------------------------------

def get_login() :
    """Returns login name
    """
    #return os.getlogin()
    return getpass.getuser()

#------------------------------

def get_hostname() :
    """Returns login name
    """
    #return os.uname()[1]
    return socket.gethostname()

#------------------------------

def get_cwd() :
    """Returns current working directory
    """
    return os.getcwd()

#------------------------------

def file_mode(fname) :
    """Returns file mode 
    """
    return os.stat(fname)[ST_MODE]

#------------------------------

def create_directory(dir, verb=False) : 
    if os.path.exists(dir) :
        pass
        #if verb : print 'Directory exists: %s' % dir
    else :
        os.makedirs(dir)
        if verb : print 'Directory created: %s' % dir

#------------------------------

def create_directory_with_mode(dir, mode=0777, verb=False) :
    """Creates directory and sets its mode"""

    if os.path.exists(dir) :
        pass
        #if verb : print 'Directory exists: %s' % dir
    else :
        os.makedirs(dir)
        os.chmod(dir, mode)
        if verb : print 'Directory created: %s, mode(oct)=%s' % (dir, oct(mode))

#------------------------------

def create_path(path, depth=6, mode=0777, verb=False) : 
    """Creates missing path of specified depth from the beginning
       e.g. for '/reg/g/psdm/logs/calibman/2016/07/log-file-name.txt'
       or '/reg/d/psdm/cxi/cxi11216/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/9-end.data'

       Returns True if path to file exists, False othervise
    """
    if verb : print 'create_path: %s' % path

    #subdirs = path.strip('/').split('/')
    subdirs = path.split('/')
    cpath = subdirs[0]
    for i,sd in enumerate(subdirs[:-1]) :
        if i>0 : cpath += '/%s'% sd 
        if i<depth : continue
        if cpath=='' : continue
        create_directory_with_mode(cpath, mode, verb)

    return os.path.exists(cpath)

#------------------------------

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    f=open(path, mode)
    f.write(text)
    f.close() 

#------------------------------

def load_textfile(path) :
    """Returns text file as a str object
    """
    f=open(path, 'r')
    recs = f.read() # f.readlines()
    f.close() 
    return recs

#------------------------------

def calib_dir_for_exp(exp) :
    if not isinstance(exp, str) : raise IOError('Experiment name "%s" is expected as str object', str(exp))
    if len(exp) < 8 : raise IOError('Experiment name "%s" has <8 letters', str(exp))
    if len(exp) > 9 : raise IOError('Experiment name "%s" has >9 letters', str(exp))
    return '%s/%s/%s/calib' % (DIR_INS, exp[:3].upper(), exp)

#------------------------------

def calib_dir(env) :
    cdir = env.calibDir()
    #if cdir == '/reg/d/psdm///calib' :
    #    return None
    if os.path.exists(cdir) :
        return cdir

#------------------------------

def exp_name(env) :
    exp = env.experiment()
    if exp=='' : return None
    return exp

#------------------------------

def log_rec_on_start() :
    """Returns (str) record containing timestamp, login, host, cwd, and command line
    """
    return '\n%s user:%s@%s cwd:%s\n  command:%s'%\
           (str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'), get_login(), get_hostname(), get_cwd(), ' '.join(sys.argv))

#------------------------------

def add_rec_to_log(lfname, rec, verbos=False) :
    """Adds record rec to the log file with path lfname. If path does not exist, it is created beginning from depth=5.
    """
    path = replace(lfname, '#YYYY-MM', str_tstamp(fmt='%Y/%m'))

    #print 'XXX:add_rec_to_log, path=%s' % path

    if create_path(path, depth=6, mode=0777, verb=verbos) :
        cmd = 'echo "%s" >> %s' % (rec, path)
        if verbos : print 'command: %s' % cmd
        os.system(cmd)
        mode_log = 0666
        if (file_mode(path) & 0777) == mode_log : return 
        os.chmod(path, mode_log)

#------------------------------

def alias_for_src_name(env) :
    ckeys = env.configStore().keys()
    srcs  = [k.src()   for k in ckeys]
    alias = [k.alias() for k in ckeys]
    d = dict(zip(srcs, alias))
    for s,a in d.items() : print 'src: %40s   alias: %s' % (s, a)
    #print keysalias

#------------------------------

def replace(template, pattern, subst) :
    """If pattern in the template replaces it with subst.
       Returns str object template with replaced patterns. 
    """
    fields = template.split(pattern, 1) 
    if len(fields) > 1 :
        return '%s%s%s' % (fields[0], subst, fields[1])
    else :
        return template

#------------------------------

def calib_fname_template(exp, runnum, tsec, tnsec, fid, tsdate, tstime, src, nevts, ofname):
    """Replaces parts of the file name ofname specified as
       #src, #exp, #run, #evts, #type, #date, #time, #fid, #sec, #nsec
       with actual values.

       RETURNS

       - template (str) - file name template, e.g.: nda-cxi11216-r0009-CxiEndstation.0:Jungfrau.0-e000010-%s.txt,
                        where %s stands for supplied late type.
    """
    template = replace(ofname,   '#src',  src)
    template = replace(template, '#exp',  exp)
    template = replace(template, '#run',  'r%04d'%runnum)
    template = replace(template, '#type', '%s')
    template = replace(template, '#date', tsdate)
    template = replace(template, '#time', tstime)
    template = replace(template, '#fid',  '%06d'%fid)
    template = replace(template, '#sec',  '%d' % tsec)
    template = replace(template, '#nsec', '%09d' % tnsec)
    template = replace(template, '#evts', 'e%06d' % nevts)
    if not '%s' in template : template += '-%s'
    return template

#------------------------------

def history_record(ifname, ctypedir, ctype, ofname, comment='') :
    """Returns (str) history record about deployed constants.

    Parameters

    - ifname : str - input file name, e.g.: 'fname.txt'
    - ctypedir : str - path to calibtype directory, e.g. '/some-path/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/'
    - ctype : str - calibration type, e.g.: 'pedestals'
    - ofname : str - output file name, e.g.: '123-end.data'
    - comment : str - any comment
    - verbos : bool - verbosity
    """
    return 'file:%s copy_of:%s ctype:%s user:%s host:%s cptime:%s cwd:%s cmt:%s' % \
           (ofname.ljust(14),
           ifname,
           ctype,
           get_login(),
           get_hostname(),
           str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'),
           get_cwd(),
           comment)

#------------------------------

def path_to_history_file(ctypedir, ctype) :
    """Returns path to HISTORY file in the calib store.
       e.g. /some-path/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/HYSTORY
       See parameters description in :py:meth:`history_record`.
    """
    return '%s/%s/HISTORY' % (ctypedir, ctype)

#------------------------------

def path_to_calib_file(ctypedir, ctype, ofname) :
    """Returns path to file wirh calibration constants in the calib store.
       e.g. /some-path/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/9-end.data
       See parameters description in :py:meth:`history_record`.
    """
    return '%s/%s/%s' % (ctypedir, ctype, ofname)

#------------------------------

def command_deploy_file(ifname, ofname) :
    """Returns command to deploys file with calibration constants in the calib store.
    """
    return 'cat %s > %s' % (ifname, ofname) # > stands for copy

#------------------------------

def command_add_record_to_file(rec, fname) :
    """Returns command to add record to file.
    """
    return 'echo "%s" >> %s' % (rec, fname) # >> stands for append

#------------------------------

def deploy_file(ifname, ctypedir, ctype, ofname, lfname=None, verbos=False) :
    """Deploys file with calibration constants in the calib store, adds history record in file and in logfile.

    Parameters

    - ifname : str - input file name, e.g.: 'fname.txt'
    - ctypedir : str - path to calibtype directory, e.g. '/some-path/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/'
    - ctype : str - calibration type, e.g.: 'pedestals'
    - ofname : str - output file name, e.g.: '123-end.data'
    - lfname : str - log file path or None, to add history record
    - verbos : bool - verbosity
    """
    path_clb = path_to_calib_file(ctypedir, ctype, ofname)
    path_his = path_to_history_file(ctypedir, ctype)

    dep = 6 if '/reg/d/psdm/' in path_clb else 0

    if create_path(path_clb, depth=dep, mode=02770, verb=verbos) : # mode=02770 makes drwxrws---+

        cmd = command_deploy_file(ifname, path_clb)
        print 'cmd: %s' % cmd
        os.system(cmd)

        rec = history_record(ifname, ctypedir, ctype, ofname, comment='')
        cmd = command_add_record_to_file(rec, path_his)
        if verbos : print 'cmd: %s' % cmd
        os.system(cmd)
        if lfname is not None : add_rec_to_log(lfname, '  %s' % rec, verbos)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def test_mask_neighbors_2d(allnbrs=True) :
    from pyimgalgos.NDArrGenerators import random_exponential
    import pyimgalgos.Graphics as gr

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
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2,  amin=0, amax=10, orientation='vertical')
    gr.show(mode=None)
    
#------------------------------

def test_mask_neighbors_3d(allnbrs=True) :
    from pyimgalgos.NDArrGenerators import random_exponential
    import pyimgalgos.Graphics as gr

    #randexp = random_exponential(shape=(2,2,30,80), a0=1)
    randexp = random_exponential(shape=(2,30,80), a0=1)

    fig  = gr.figure(figsize=(16,6), title='Random > 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)

    img1 = reshape_nda_to_2d(mask)
    img2 = reshape_nda_to_2d(mask_nbrs)
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2, amin=0, amax=10, orientation='vertical')
    gr.show(mode=None)
    
#------------------------------

def test_mask_edges_2d(mrows=1, mcols=1) :
    from pyimgalgos.NDArrGenerators import random_exponential
    import pyimgalgos.Graphics as gr

    fig  = gr.figure(figsize=(8,6), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    mask = np.ones((20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = mask_out
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical')
    gr.show(mode=None)
    
#------------------------------

def test_mask_edges_3d(mrows=1, mcols=1) :
    from pyimgalgos.NDArrGenerators import random_exponential
    import pyimgalgos.Graphics as gr

    fig  = gr.figure(figsize=(8,6), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    #mask = np.ones((2,2,20,30))
    mask = np.ones((2,20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = reshape_nda_to_2d(mask_out)
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical')
    gr.show(mode=None)
    
#------------------------------

#def src_name_from_alias(env, alias='') :
#    amap = env.aliasMap()
#    for s in amap.srcs() :
#        str_s = str(s) # 'MfxEndstation.0:Rayonix.0'
#        print s, amap.src(str_s), ' alias ="%s"' % amap.alias(amap.src(str_s))
#
#    #psasrc = amap.src(str_src)
#    #source  = src if amap.alias(psasrc) == '' else amap.src(str_src)

#------------------------------
#------------------------------

def do_test() :

    print 'get_enviroment(USER) : %s' % get_enviroment()
    print 'get_login()          : %s' % get_login()
    print 'get_hostname()       : %s' % get_hostname()
    print 'get_cwd()            : %s' % get_cwd()
    #print ': %s' % 

    if len(sys.argv) > 1 :

      if sys.argv[1] == '1' : test_mask_neighbors_2d(allnbrs = False)
      if sys.argv[1] == '2' : test_mask_neighbors_2d(allnbrs = True)
      if sys.argv[1] == '3' : test_mask_neighbors_3d(allnbrs = False)
      if sys.argv[1] == '4' : test_mask_neighbors_3d(allnbrs = True)
      if sys.argv[1] == '5' : test_mask_edges_2d(mrows=5, mcols=1)
      if sys.argv[1] == '6' : test_mask_edges_2d(mrows=0, mcols=5)
      if sys.argv[1] == '7' : test_mask_edges_3d(mrows=1, mcols=2)
      if sys.argv[1] == '8' : test_mask_edges_3d(mrows=5, mcols=0)

#------------------------------

if __name__ == "__main__" :
    do_test()

#------------------------------
