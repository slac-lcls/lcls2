"""
:py:class:`UtilsAreaCalib` dark processing algorithms for generic area detector 
===============================================================================

Usage::

    from psana.detector.UtilsAreaCalib import *
    #OR
    import psana.detector.UtilsAreaCalib as uac

    uac.

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-02-10 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys
import numpy as np
from psana.pyalgos.generic.Utils import create_directory, log_rec_on_start, str_tstamp, time_sec_from_stamp
from psana.detector.Utils import info_dict, info_namespace, info_command_line
from psana.detector.UtilsEpix10kaCalib import proc_dark_block
from psana.detector.utils_psana import seconds, datasource_kwargs, info_run #, timestamp_run
from psana.pyalgos.generic.NDArrUtils import info_ndarr, save_2darray_in_textfile#, save_ndarray_in_textfile
from psana import DataSource

#from psana.pscalib.calib.NDArrIO import save_txt; global save_txt
from psana.pscalib.calib.MDBUtils import data_from_file
import psana.pscalib.calib.MDBWebUtils as wu

cc = wu.cc # import psana.pscalib.calib.CalibConstants as cc
#cc.TSFORMAT_SHORT = '%Y%m%d%H%M%S'

def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (not nrec%100)
       #or (nrec<500 and not nrec%100)\
       #or (not nrec%1000)


def info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):', sep='\n '):
    return cmt + sep.join(det.raw._uniqueid.split('_'))


def info_detector(det, cmt='detector info:', sep='\n    '):
    return cmt\
        +  'det.raw._det_name   : %s' % (det.raw._det_name)\
        +'%sdet.raw._dettype    : %s' % (sep, det.raw._dettype)\
        +'%s_sorted_segment_ids : %s' % (sep, str(det.raw._sorted_segment_ids))\
        +'%sdet.raw._uniqueid   : %s' % (sep, det.raw._uniqueid)\
        +'%s%s' % (sep, info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):%s     '%sep, sep=sep+'     '))\
        +'%sdet methods vbisible: %s' % (sep, ' '.join([v for v in dir(det) if v[0]!='_']))\
        +'%s            hidden  : %s' % (sep, ' '.join([v for v in dir(det) if (v[0]=='_' and v[1]!='_')]))\
        +'%sdet.raw._calibconst.keys(): %s' % (sep, ', '.join(det.raw._calibconst.keys()))


def fname_prefix(detname, tstamp, exp, runnum, dirrepo):
    return '%s/%s-%s-%s-r%04d' % (dirrepo, detname, tstamp, exp, runnum)


def deploy_constants(dic_consts, **kwa):

    CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}

    expname  = kwa.get('exp',None)
    detname  = kwa.get('det',None)
    deploy   = kwa.get('deploy', False)
    dirrepo  = kwa.get('dirrepo', './work')
    dirmode  = kwa.get('dirmode',  0o774)
    filemode = kwa.get('filemode', 0o664)
    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('run_orig',None)

    fmt_peds   = kwa.get('fmt_peds', '%.3f')
    fmt_rms    = kwa.get('fmt_rms',  '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')

    CTYPE_FMT = {'pedestals'   : fmt_peds,
                 'pixel_rms'   : fmt_rms,
                 'pixel_status': fmt_status}

    create_directory(dirrepo, dirmode)

    fprefix = fname_prefix(detname, tsshort, expname, runnum, dirrepo)

    for ctype, nda in dic_consts.items():
        fname = '%s-%s.txt' % (fprefix, ctype)
        fmt = CTYPE_FMT.get(ctype,'%.5f')
        #logger.info(info_ndarr(nda, 'constants for %s ' % ctype))
        #logger.info(info_ndarr(nda, 'constants'))
        #save_ndarray_in_textfile(nda, fname, filemode, fmt)
        save_2darray_in_textfile(nda, fname, filemode, fmt)

        dtype = 'ndarray'
        kwa['ctype'] = ctype
        kwa['dtype'] = dtype
        kwa['extpars'] = {'content':'extended parameters dict->json->str',}
        #kwa['extpars'] = {'content':'other script parameters', 'script_parameters':kwa}
        _ = kwa.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
        _ = kwa.pop('det',None)

        logger.info('DEPLOY metadata: %s' % info_dict(kwa, fmt='%12s : %s', sep='\n  '))

        data = data_from_file(fname, ctype, dtype, True)
        logger.info(info_ndarr(data, 'constants loaded from file'))

        if deploy:
            detname = kwa['longname']
            id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
              wu.add_data_and_two_docs(data, expname, detname, **kwa) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
        else:
            logger.warning('TO DEPLOY CONSTANTS ADD OPTION -D')


def add_metadata_kwargs(orun, odet, **kwa):

    trun_sec = seconds(orun.timestamp) # 1607569818.532117 sec

    # check opt "-t" if constants need to be deployed with diffiernt time stamp or run number
    tstamp = kwa.get('tstamp', None)
    use_external_run = tstamp is not None and tstamp<10000
    use_external_ts  = tstamp is not None and tstamp>9999
    tvalid_sec = time_sec_from_stamp(fmt=cc.TSFORMAT_SHORT, time_stamp=str(tstamp))\
                  if use_external_ts else trun_sec
    ivalid_run = tstamp if use_external_run else orun.runnum\
                  if not use_external_ts else 0

    kwa['experiment'] = kwa.get('exp', orun.expt)
    kwa['detector']   = kwa.get('det', odet.raw._det_name)
    kwa['dettype']    = odet.raw._dettype
    kwa['longname']   = odet.raw._uniqueid # kwa.get('longname', odet.raw._uniqueid)
    kwa['time_sec']   = tvalid_sec
    kwa['time_stamp'] = str_tstamp(fmt=cc.TSFORMAT, time_sec=int(tvalid_sec))
    kwa['tsshort']    = str_tstamp(fmt=cc.TSFORMAT_SHORT, time_sec=int(tvalid_sec))
    kwa['tstamp_orig']= str_tstamp(fmt=cc.TSFORMAT, time_sec=int(trun_sec))
    kwa['run']        = ivalid_run
    kwa['run_end']    = kwa.get('run_end', 'end')
    kwa['run_orig']   = orun.runnum
    kwa['version']    = kwa.get('version', 'N/A')
    kwa['comment']    = kwa.get('comment', 'no comment')
    kwa['dettype']    = odet.raw._dettype
    return kwa



def pedestals_calibration(**kwa):

  print('log_rec_on_start: %s' % log_rec_on_start()) # tsfmt='%Y-%m-%dT%H:%M:%S%z'

  print('command line: %s' % info_command_line())
  #print('input parameters:\n%s' % info_namespace(pars)) #, fmt='%s: %s', sep=', '))
  print('input parameters: %s' % info_dict(kwa, fmt='%s: %s', sep=' '))

  #ds = DataSource(**datasource_arguments(pars_namespace))
  ds = DataSource(**datasource_kwargs(**kwa))
  #ds = DataSource(exp='tmoc00118', run=123, max_events=100)
  #ds = DataSource(exp=pars.expname, run=pars.runs, max_events=pars.evtmax)

  detname = kwa.get('det',None)
  expname = kwa.get('exp',None)
  nrecs   = kwa.get('nrecs',100)

  block = None
  nrecs2 = nrecs-2
  iblrec = -1
  break_loop = False
  kwa_depl = None

  for irun,orun in enumerate(ds.runs()):
    print('\n==== %02d run: %d exp: %s' % (irun, orun.runnum, orun.expt))
    print(info_run(orun, cmt='run info:\n    ', sep='\n    ', verb=3))

    odet = orun.Detector(detname)
    print('\n  created %s detector object' % detname)
    print(info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

    if kwa_depl is None: kwa_depl = add_metadata_kwargs(orun, odet, **kwa)

    for istep,step in enumerate(orun.steps()):
      print('\nStep %1d' % istep)

      for ievt,evt in enumerate(step.events()):
        print('Event %04d' % ievt, end='')

        raw  = odet.raw.raw(evt)
        if raw is None:
            logger.info('raw is None')
            continue

        rows, cols = raw.shape
        if block is None:
           segs = odet.raw.segments(evt)
           print(info_ndarr(segs, '\n det.raw.segments(evt) '))
           block=np.zeros((nrecs, rows, cols),dtype=raw.dtype)
           print(info_ndarr(block,' Createsd array for accumulation of raw data block[nrecs, nrows, ncols]\n '))
           print(end=10*' ')

        iblrec += 1
        block[iblrec,:] = raw
        print(info_ndarr(raw,  ' record %04d   raw ' % iblrec), end='\r')
        if selected_record(ievt): print() # new line

        if iblrec > nrecs2:
            print('\nNumber of records limit is reached, iblrec=%d' % iblrec)
            break_loop = True
            break          # break evt  loop
      if break_loop: break # break step loop
    if break_loop: break   # break run  loop

  print('Run/step/event loop is completed')
  blk = block[:iblrec+1,:]
  print(info_ndarr(blk,'Begin processing of the data block '))

  arr_av1, arr_rms, arr_sta = proc_dark_block(block, **{\
  'exp': expname,\
  'det': detname,\
  })

  print(info_ndarr(arr_av1, 'arr_av1 '))
  print(info_ndarr(arr_rms, 'arr_rms '))
  print(info_ndarr(arr_sta, 'arr_sta ', last=10))

  dic_consts = {
    'pedestals'    : arr_av1,\
    'pixel_rms'    : arr_rms,\
    'pixel_status' : arr_sta\
  }

  deploy_constants(dic_consts, **kwa_depl)



if __name__ == "__main__":

    print(80*'_')  
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.INFO)
    #logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.DEBUG)
  
    SCRNAME = sys.argv[0].rsplit('/')[-1]
  
    kwa = {\
        'fname'   : None,\
        'exp'     : 'tmoc00118',\
        'runs'    : '123',\
        'det'     : 'tmoopal',\
        'nrecs'   : 100,\
        #'evtmax'  : 200,\
    }

    pedestals_calibration(**kwa)
  
    sys.exit('End of %s' % sys.argv[0])

# EOF
