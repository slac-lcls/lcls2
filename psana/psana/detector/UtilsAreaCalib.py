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
from psana.pyalgos.generic.Utils import create_directory # log_rec_on_start, str_tstamp, save_textfile, set_file_access_mode, time_sec_from_stamp
from psana.detector.Utils import info_dict, info_namespace, info_command_line
from psana.detector.UtilsEpix10kaCalib import proc_dark_block
from psana.detector.utils_psana import seconds, timestamp_run, datasource_kwargs, info_run
from psana.pyalgos.generic.NDArrUtils import info_ndarr, save_2darray_in_textfile#, save_ndarray_in_textfile
from psana import DataSource

#from psana.pscalib.calib.NDArrIO import save_txt; global save_txt
#import psana.pscalib.calib.MDBUtils as mu
import psana.pscalib.calib.MDBWebUtils as wu

cc = wu.cc # import psana.pscalib.calib.CalibConstants as cc


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

    expname  = kwa.get('expname',None)
    detname  = kwa.get('detname',None)
    do_deploy= kwa.get('do_deploy', False)
    dirrepo  = kwa.get('dirrepo', './work')
    dirmode  = kwa.get('dirmode',  0o774)
    filemode = kwa.get('filemode', 0o664)
    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('runnum',None)

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

        if False: # deploy:

          dtype = 'ndarray'

          _ivalid_run = irun
          _tvalid_sec = cpdic.get('trun_sec', None) 
          if tstamp is not None:
            if tstamp>9999:
              str_ts = str(tstamp)
              _tvalid_sec = time_sec_from_stamp(fmt='%Y%m%d%H%M%S', time_stamp=str_ts)
              _ivalid_run = 0
            else: 
              _ivalid_run = tstamp

          _tvalid_stamp = str_tstamp(fmt=cc.TSFORMAT, time_sec=_tvalid_sec)
          _longname = cpdic.get('longname', detname)

          dic_extpars = {
            'content':'extended parameters dict->json->str',
          }

          kwa = {
            'experiment': exp,
            'ctype': octype,
            'dtype': dtype,
            'detector': detname,
            'longname': _longname,
            'time_sec':_tvalid_sec,
            'time_stamp': _tvalid_stamp,
            'tstamp_orig': cpdic.get('tsrun_dark', None),
            'run': _ivalid_run,
            'run_end': run_end,
            'run_orig': irun,
            'version': version,
            'comment': comment,
            'extpars': dic_extpars,
          }

          logger.debug('DEPLOY metadata: %s' % str(kwa))

          _detname = _longname # cpdic.get('longname', detname)

          data = mu.data_from_file(fmerge, octype, dtype, True)

          logger.info(info_ndarr(data, 'merged constants loaded from file'))

          if do_deploy:
            id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
              wu.add_data_and_two_docs(data, exp, _detname, **kwa) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
          else:
            logger.warning('TO DEPLOY CONSTANTS ADD OPTION -D True')




def deployment_kwargs(run, det, **kwa):
    detname  = kwa.get('detname', det.raw._det_name)
    longname = kwa.get('longname', det.raw._uniqueid)
    expname  = kwa.get('expname', None)
    runnum   = kwa.get('runnum', None)
    tstamp   = kwa.get('tstamp', None)
    tsshort  = kwa.get('tsshort', None)
    version  = kwa.get('version', 'N/A')
    dettype  = kwa.get('dettype', det.raw._dettype)

    runtstamp = run.timestamp    # 4193682596073796843 relative to 1990-01-01
    trun_sec = seconds(runtstamp) # 1607569818.532117 sec
    #tstamp = str_tstamp(time_sec=int(trun_sec)) #fmt='%Y-%m-%dT%H:%M:%S%z'

    if runnum  is None: kwa['runnum']  = run.runnum # 1-st file in case of list, non-defined
    if expname is None: kwa['expname'] = run.expt # owerride expname in case of input from xtc2 file
    if tstamp  is None: kwa['tstamp']  = timestamp_run(run, fmt=cc.TSFORMAT) # '%Y-%m-%dT%H:%M:%S'
    if tsshort is None: kwa['tsshort'] = timestamp_run(run, fmt='%Y%m%d%H%M%S')


    print('deployment_kwargs: %s' % info_dict(kwa, fmt='%s: %s', sep='\n  '))

    return kwa



def pedestals_calibration(**kwa):

  print('command line: %s' % info_command_line())
  #print('input parameters:\n%s' % info_namespace(pars)) #, fmt='%s: %s', sep=', '))
  print('input parameters: %s' % info_dict(kwa, fmt='%s: %s', sep=' '))

  #ds = DataSource(**datasource_arguments(pars_namespace))
  ds = DataSource(**datasource_kwargs(**kwa))
  #ds = DataSource(exp='tmoc00118', run=123, max_events=100)
  #ds = DataSource(exp=pars.expname, run=pars.runs, max_events=pars.evtmax)

  detname = kwa.get('detname',None)
  expname = kwa.get('expname',None)
  nrecs   = kwa.get('nrecs',100)

  block = None
  nrecs2 = nrecs-2
  iblrec = -1
  break_loop = False
  kwa_depl = None

  for irun,run in enumerate(ds.runs()):
    print('\n==== %02d run: %d exp: %s' % (irun, run.runnum, run.expt))
    print(info_run(run, cmt='run info:\n    ', sep='\n    ', verb=3))

    det = run.Detector(detname)
    print('\n  created %s detector object' % detname)
    print(info_detector(det, cmt='  detector info:\n      ', sep='\n      '))

    if kwa_depl is None: kwa_depl = deployment_kwargs(run, det, **kwa)

    for istep,step in enumerate(run.steps()):
      print('\nStep %1d' % istep)

      for ievt,evt in enumerate(step.events()):
        print('Event %04d' % ievt, end='')

        raw  = det.raw.raw(evt)
        if raw is None:
            logger.info('raw is None')
            continue

        rows, cols = raw.shape
        if block is None:
           segs = det.raw.segments(evt)
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
  print(info_ndarr(arr_sta, 'arr_sta ', last=20))

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
        'expname' : 'tmoc00118',\
        'runs'    : '123',\
        'detname' : 'tmoopal',\
        'evtmax'  : 200,\
        'nrecs'   : 100,\
    }

    pedestals_calibration(**kwa)
  
    sys.exit('End of %s' % sys.argv[0])

# EOF
