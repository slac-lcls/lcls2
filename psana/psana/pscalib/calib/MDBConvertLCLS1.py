
#------------------------------
"""
:py:class:`CalibUtils` - generic utilities for Calib project
===================================================================

Usage ::

    # python lcls2/psana/pscalib/calib/CalibUtils.py

    from psana.pscalib.calib.MDBConvertLCLS1 import *

See:
 * :py:class:`CalibBase`
 * :py:class:`CalibConstant`

For more detail see `Calibration Store <https://confluence.slac.stanford.edu/display/PCDS/MongoDB+evaluation+for+calibration+store>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-03-05 by Mikhail Dubrovin
"""
#------------------------------
import os
#import numpy as np
from requests import get as requests_get
#from psana.pyalgos.generic.PSConstants import INSTRUMENTS, DIR_INS, DIR_FFB # , DIR_LOG
from psana.pyalgos.generic.PSNameManager import nm
import psana.pyalgos.generic.Utils as gu
import psana.pscalib.calib.CalibConstants as cc # list_calib_names

from psana.pscalib.calib.NDArrIO import load_txt
import psana.pyalgos.generic.NDArrUtils as ndu

import psana.pscalib.calib.MDBUtils as dbu # insert_constants, time_and_timestamp

from psana.pscalib.calib.CalibUtils import history_dict_for_file, history_list_of_dicts, parse_calib_file_name

from psana.pscalib.calib.XtcavConstants import Load #, Save

from psana.pscalib.calib.MDBConversionMap import DETECTOR_NAME_CONVERSION_DICT as dic_det_name_conv

import logging
logger = logging.getLogger(__name__)

#------------------------------

def run_begin_end_time(exp:str, runnum:int) :
    # returns a list of dicts per run with 'begin_time', 'end_time', 'run_num', 'run_type'
    if runnum>0 :
        resp = requests_get('https://pswww.slac.stanford.edu/prevlgbk/lgbk/%s/ws/runs' % exp).json()
        for d in resp :
            if d['run_num'] == runnum :
                return int(d['begin_time']), int(d['end_time'])
    logger.debug('begin and end time info is not found in mysql for run=%d. Uses default times.' %runnum)
    return 1000000000, 5000000000

#------------------------------

def load_xtcav_calib_file(fpath) :
    """Returns object retrieved from hdf5 file by XtcavConstants.Load method.

       Xtcav constants are wrapped in python object and accessible through attributes with arbitrary names.
       list of attribute names can be retrieved as dir(o) 
       and filtred from system names like '__*__' by the comprehension list [p for p in dir(o) if p[:2] != '__']
       access to attributes can be done through the python built-in method getattr(o, name, None), etc...
    """
    logger.info('Load xtcav calib object from file: %s'%fpath)
    return Load(fpath)


def is_xtcav(calibvers, cftype) : 
    return ('Xtcav' in calibvers) and (cftype in ('lasingoffreference', 'pedestals'))

#------------------------------

def add_calib_file_to_cdb(exp, dircalib, calibvers, detname, cftype, fname, cfdir, listdicts, **kwargs) :
    """
    """
    d = history_dict_for_file(listdicts, fname)

    resp = parse_calib_file_name(fname)
    begin, end, ext = resp if resp is not None else (None, None, None)
    if begin is not None : begin=int(begin)

    if None in (begin, end, ext) : return

    fpath = '%s/%s' % (cfdir, fname)

    verbose = kwargs.get('verbose', False)

    data = gu.load_textfile(fpath, verbose) if cftype in ('geometry','code_geometry') else\
           load_xtcav_calib_file(fpath) if is_xtcav(calibvers, cftype) else\
           load_txt(fpath) # using NDArrIO

    begin_time, end_time = run_begin_end_time(exp, int(begin))

    if verbose :
        ndu.print_ndarr(data, 'scan calib: data')
        msg = 'scan calib: %s %s %s %s %s %s %s %s %s' % (exp, cfdir, fname, begin, end, ext, calibvers, detname, cftype)
        logger.info(msg)
        logger.info('begin_time: %s end_time: %s' % (begin_time, end_time))

    if data is None :
        msg = 'data is None, conversion is dropped for for file: %s' % fpath
        logger.warning(msg)        
        return

    kwargs['run']        = begin
    kwargs['run_end']    = end
    kwargs['detector']   = detname
    kwargs['ctype']      = cftype
    kwargs['time_sec']   = begin_time
    kwargs['end_time']   = end_time
    kwargs['time_stamp'] = dbu._timestamp(begin_time)
    kwargs['extpars']    = d # just in case save entire history dict
    #kwargs['comment']    = 'HISTORY: %s' % d.get('comment', '')

    dbu.insert_calib_data(data, **kwargs)
    #dbu.insert_constants(data, d['experiment'], d['detector'], d['ctype'], d['run'], d['time_sec'], d['version'], **kwargs)

#------------------------------

def detname_conversion(detname='XcsEndstation.0:Epix100a.1') :
    """Converts LCLS1 detector name like 'XcsEndstation.0:Epix100a.1' to 'xcsendstation_0_epix100a_1'
    """
    name2 = dic_det_name_conv.get(detname, None)
    return name2 if name2 is not None else ('%s-unknown-in-dict'%detname.replace(":","_").replace(".","_"))

#    fields = detname.split(':')
#    if len(fields) == 2 : 
#        dettype = fields[1].split('.')
#        if len(dettype) == 2 :
#            return '%s_detnum1234' % dettype[0].lower() # returns 'epix100a_detnum1234'
#    return detname.replace(":","_").replace(".","_").lower()

#------------------------------

def scan_calib_for_experiment(exp='cxix25615', **kwargs) :

    host    = kwargs.get('host', None)
    port    = kwargs.get('port', None)
    verbose = kwargs.get('verbose', False)

    client = dbu.connect_to_server(host, port)
    dbname = dbu.db_prefixed_name(exp)
    if dbu.database_exists(client, dbname) :
        msg = 'Experiment %s already has a database. Consider to delete it from the list:\n%s'%\
              (exp, str(dbu.database_names(client)))+\
              '\nBefore adding consider to delete existing DB using command: cdb deldb --dbname %s -C' % dbname
        logger.warning(msg)
        return

    dircalib = nm.dir_calib(exp)
    #if verbose : 
    logger.info('Scan: %s' % dircalib)

    for dir0 in gu.get_list_of_files_in_dir_for_part_fname(dircalib, pattern='::') :
        if not os.path.isdir(dir0) : continue
        calibvers = os.path.basename(dir0)
        logger.debug('  %s ' % calibvers)

        for dir1 in gu.get_list_of_files_in_dir_for_part_fname(dir0, pattern=':') :
            if not os.path.isdir(dir1) : continue
            detname = os.path.basename(dir1)
            detname_m = detname_conversion(detname)
            logger.debug('    %s' % detname_m)        

            for cftype in gu.get_list_of_files_in_dir(dir1) :
                if not(cftype in cc.list_calib_names) : continue
                dir2 = '%s/%s' % (dir1, cftype)
                if not os.path.isdir(dir2) : continue
                logger.debug('      %s' % cftype)

                cfdir = '%s/%s/%s/%s' % (dircalib, calibvers, detname, cftype)
                listdicts = history_list_of_dicts('%s/HISTORY' % cfdir, verbose)
                #logger.debug('XXX listdicts %s' % listdicts)
                count = 0
                for fname in gu.get_list_of_files_in_dir(dir2) :
                    logger.debug('        %s' % fname)
                    if fname == 'HISTORY' : continue
                    if os.path.splitext(fname)[1] != '.data' : continue
                    #logger.debug('  XXX begin adding: %s %s %s %s' % (dircalib, detname_m, cftype, fname))
                    add_calib_file_to_cdb(exp, dircalib, calibvers, detname_m, cftype, fname, cfdir, listdicts, **kwargs)
                    count += 1

                logger.info('  converted %3d files from: %s' % (count, cfdir))

#------------------------------

if __name__ == "__main__" :
  def usage() : return 'Use command: python %s <test-number>, where <test-number> = 1,2,...' % sys.argv[0]

#------------------------------
  def test_detname_conversion(tname) :
      logger.info('detname_conversion: %s' % detname_conversion('XcsEndstation.0:Epix100a.1'))

#------------------------------

  def test_dir_calib(tname) :
      logger.info('dir_calib: %s' % nm.dir_calib('cxi02117'))

#------------------------------

  def test_all(tname) :
    logger.info('\n%s\n' % usage())
    if len(sys.argv) != 2 : test_dir_calib(tname)
    elif tname == '1': test_dir_calib(tname)
    elif tname == '2': scan_calib_for_experiment('cxix25615')
    elif tname == '3': test_detname_conversion(tname)
    else : sys.exit('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s' % (50*'_', tname))
    test_all(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
