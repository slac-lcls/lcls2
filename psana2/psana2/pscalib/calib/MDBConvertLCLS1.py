
"""
:py:class:`CalibUtils` - generic utilities for Calib project
============================================================

Usage ::
    # python lcls2/psana/pscalib/calib/CalibUtils.py

    from psana2.pscalib.calib.MDBConvertLCLS1 import *

See:
 * :py:class:`CalibBase`
 * :py:class:`CalibConstant`

For more detail see `Calibration Store <https://confluence.slac.stanford.edu/display/PCDS/MongoDB+evaluation+for+calibration+store>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-03-05 by Mikhail Dubrovin
"""

import os
import sys
import numpy as np
import json

import psana2.pyalgos.generic.Utils as gu
import psana2.pyalgos.generic.NDArrUtils as ndu
from   psana.pyalgos.generic.PSNameManager import nm

import psana2.pscalib.calib.CalibConstants as cc # list_calib_names
import psana2.pscalib.calib.MDBUtils as mu # insert_constants, time_and_timestamp
from   psana.pscalib.calib.NDArrIO import load_txt
from   psana.pscalib.calib.CalibUtils import history_dict_for_file, history_list_of_dicts, parse_calib_file_name
from   psana.pscalib.calib.XtcavUtils import load_xtcav_calib_file
from   psana.pscalib.calib.MDBConvertUtils import serialize_dict, info_dict, print_dict
from   psana.pscalib.calib.MDBConversionMap import DETECTOR_NAME_CONVERSION_DICT as dic_det_name_conv

from requests import get as requests_get
from urllib.parse import urlparse
from krtc import KerberosTicket

import logging
logger = logging.getLogger(__name__)


def must_to_fix_error(msg):
    logger.error(msg)
    sys.exit('THIS ERROR MUST TO BE FIXED')

def get_for_url(url):

    krbheaders = KerberosTicket("HTTP@" + urlparse(url).hostname).getAuthHeaders()
    r = requests_get(url, headers=krbheaders) # returns {'success': True, 'value': [<old-style-responce>]}
    if not r:
        msg = '\nget_for_url try to get run times from experiment DB'\
              '\nurl: %s\nstatus_code: %d\ncontent: %s' %(url, r.status_code, r.text)
        must_to_fix_error(msg)

    jdic = r.json()
    if jdic['success']: return jdic['value']
    else:
        must_to_fix_error('\nget_for_url un-successful responce: %s\nfor url: %s' % (str(r), url))

def run_begin_end_time(exp, runnum):
    # returns a list of dicts per run with 'begin_time', 'end_time', 'run_num', 'run_type'
    if runnum>0:
        url = 'https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/runs_for_calib' % exp
        resp = get_for_url(url)
        #returns [{'begin_time': 1438751015, 'run_num': 201, 'run_type': 'DATA', 'end_time': 1438751093},...]

        for d in resp:
            if d['run_num'] == runnum:
                return int(d['begin_time']), int(d['end_time'])
    logger.debug('begin and end time info is not found in mysql for run=%d. Uses default times.' %runnum)
    return 1000000000, 5000000000

def is_xtcav(calibvers, cftype):
    return ('Xtcav' in calibvers) and (cftype in ('lasingoffreference', 'pedestals'))

def check_data_shape(data, detname, ctype):
    """Re-shape data if necessary"""
    if 'cspad' in detname:
        if data.size==32*185*388 and data.shape!=(32,185,388):
            logger.info('Reshape data for det: %s ctype: %s' % (detname, ctype))
            data.shape = (32,185,388)

def add_calib_file_to_cdb(exp, dircalib, calibvers, detname, cftype, fname, cfdir, listdicts, **kwargs):

    d = history_dict_for_file(listdicts, fname)

    resp = parse_calib_file_name(fname)
    begin, end, ext = resp if resp is not None else (None, None, None)
    if begin is not None: begin=int(begin)

    if None in (begin, end, ext): return

    fpath = '%s/%s' % (cfdir, fname)

    verbose = kwargs.get('verbose', False)

    data = gu.load_textfile(fpath, verbose) if cftype in ('geometry','code_geometry') else\
           load_xtcav_calib_file(fpath) if is_xtcav(calibvers, cftype) else\
           load_txt(fpath) # using NDArrIO

    if isinstance(data, dict):
        serialize_dict(data)
        logger.debug(info_dict(data))
        #print_dict(data)
        #data = json.dumps(data) # (data,ensure_ascii=True) json.dumps converts dict -> str # .replace("'", '"') for json
        data = str(data)

    if isinstance(data, np.ndarray):
        check_data_shape(data, detname, cftype)

    begin_time, end_time = run_begin_end_time(exp, int(begin))

    if verbose:
        ndu.print_ndarr(data, 'scan calib: data')
        msg = 'scan calib: %s %s %s %s %s %s %s %s %s' % (exp, cfdir, fname, begin, end, ext, calibvers, detname, cftype)
        logger.info(msg)
        logger.info('begin_time: %s end_time: %s' % (begin_time, end_time))

    if data is None:
        msg = 'data is None, conversion is dropped for for file: %s' % fpath
        logger.warning(msg)
        return

    kwargs['run']        = begin
    kwargs['run_end']    = end
    kwargs['detector']   = detname
    kwargs['ctype']      = cftype
    kwargs['time_sec']   = begin_time
    kwargs['end_time']   = end_time
    kwargs['time_stamp'] = mu._timestamp(begin_time)
    kwargs['extpars']    = d if d is not None else {} # just in case save entire history dict

    mu.insert_calib_data(data, **kwargs)

def detname_conversion(detname='XcsEndstation.0:Epix100a.1'):
    """Converts LCLS1 detector name like 'XcsEndstation.0:Epix100a.1' to 'xcsendstation_0_epix100a_1'"""
    name2 = dic_det_name_conv.get(detname, None)
    return name2 if name2 is not None else ('%s-unknown-in-dict'%detname.replace(":","_").replace(".","_"))

def scan_calib_for_experiment(exp='cxix25615', **kwargs):

    host    = kwargs.get('host', None)
    port    = kwargs.get('port', None)
    user    = kwargs.get('user', None)
    upwd    = kwargs.get('upwd', None)
    verbose = kwargs.get('verbose', False)

    client = mu.connect_to_server(host, port, user, upwd)
    dbname = mu.db_prefixed_name(exp)
    if mu.database_exists(client, dbname):
        msg = 'Experiment %s already has a database. Consider to delete it from the list:\n%s'%\
              (exp, str(mu.database_names(client)))+\
              '\nBefore adding consider to delete existing DB using command: cdb deldb --dbname %s -C -u <username> -p <password>' % dbname
        logger.warning(msg)
        return

    dircalib = nm.dir_calib(exp)
    logger.info('Scan: %s' % dircalib)

    for dir0 in gu.get_list_of_files_in_dir_for_part_fname(dircalib, pattern='::'):
        if not os.path.isdir(dir0): continue
        calibvers = os.path.basename(dir0)
        logger.debug('  %s ' % calibvers)

        for dir1 in gu.get_list_of_files_in_dir_for_part_fname(dir0, pattern=':'):
            if not os.path.isdir(dir1): continue
            detname = os.path.basename(dir1)
            detname_m = detname_conversion(detname)
            logger.debug('    %s' % detname_m)

            for cftype in gu.get_list_of_files_in_dir(dir1):
                if not(cftype in cc.list_calib_names): continue
                dir2 = '%s/%s' % (dir1, cftype)
                if not os.path.isdir(dir2): continue
                logger.debug('      %s' % cftype)

                cfdir = '%s/%s/%s/%s' % (dircalib, calibvers, detname, cftype)
                listdicts = history_list_of_dicts('%s/HISTORY' % cfdir, verbose)
                #logger.debug('XXX listdicts %s' % listdicts)
                count = 0
                for fname in gu.get_list_of_files_in_dir(dir2):
                    logger.debug('        %s' % fname)
                    if fname == 'HISTORY': continue
                    if os.path.splitext(fname)[1] != '.data': continue
                    logger.debug('  XXX begin adding: %s %s %s %s' % (dircalib, detname_m, cftype, fname))
                    add_calib_file_to_cdb(exp, dircalib, calibvers, detname_m, cftype, fname, cfdir, listdicts, **kwargs)
                    count += 1

                logger.info('  converted %3d files from: %s' % (count, cfdir))


if __name__ == "__main__":

  def usage(): return 'Use command: python %s <test-number>, where <test-number> = 1,2,...' % sys.argv[0]

  def test_detname_conversion(tname):
      detnames = ('XcsEndstation.0:Epix100a.1', 'FeeHxSpectrometer.0:Opal1000.1', 'MfxEndstation.0:Rayonix.0')
      for detname in detnames:
          print('detname_conversion(%s) --> %s' % (detname, detname_conversion(detname)))

  def test_dir_calib(tname):
      logger.info('dir_calib: %s' % nm.dir_calib('cxi02117'))

  def test_get_for_url():
      #url = 'https://pswww.slac.stanford.edu/prevlgbk/lgbk/amo86615/ws/runs'
      url = 'https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/amo86615/ws/runs_for_calib'
      resp = get_for_url(url)
      print('url : %s\nresp: %s' % (url, resp))

  def test_run_begin_end_time():
      exp, runnum = 'amo86615', 23
      print('test_run_begin_end_time: %s' % str(run_begin_end_time(exp, runnum)))

  def test_all(tname):
    logger.info('\n%s\n' % usage())
    kwa = {'host':cc.HOST,\
           'port':cc.PORT,\
           'user':gu.get_login()}
    if len(sys.argv) != 2: test_dir_calib(tname)
    elif tname == '1': test_dir_calib(tname)
    elif tname == '2': scan_calib_for_experiment('cxix25615', **kwa)
    elif tname == '3': test_detname_conversion(tname)
    elif tname == '4': scan_calib_for_experiment('amox23616', **kwa)
    elif tname == '5': test_get_for_url()
    elif tname == '6': test_run_begin_end_time()
    # /reg/d/psdm/AMO/amox23616/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/104-end.data
    else: sys.exit('Test number parameter is not recognized.\n%s' % usage())


if __name__ == "__main__":
    import sys; global sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s' % (50*'_', tname))
    test_all(tname)
    sys.exit('End of Test %s' % tname)

# EOF
