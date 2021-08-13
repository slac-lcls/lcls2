#!/usr/bin/env python

"""
Usage ::

    import psana.graphqt.UtilsWebServ as uws

Created on 2021-08-09 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import sys
import requests
import json
import kerberos
from krtc import KerberosTicket
logger = logging.getLogger(__name__)


def kerberos_headers(serv='HTTP@pswww.slac.stanford.edu'):
    try:
      krbheaders = KerberosTicket(serv).getAuthHeaders()
    except kerberos.GSSError as err:
      logger.warning(str(err))
      sys.exit('\nEXIT: CHECK VALIDITI OF KERBEROS TICKET (commands klist, kinit)')
    except err:
      logger.warning(str(err))
      sys.exit('\nUNSPECIFIED ERROR at getting KerberosTicket')
    return krbheaders


def json_pro(r):
    if r: return r.json()
    logger.warning('unexpected responce from requests.get:\n%s' % str(r))
    return None


def value_from_json(jo):
    if jo is None: return None
    if jo['success']:
        return jo['value']
    else:
        logger.warning('unsuccessful responce from requests.get:\n%s' % json.dumps(jo, indent=2))
        return None


def value_from_responce(r):
    return value_from_json(json_pro(r))


# curl -s "https://pswww.slac.stanford.edu/ws/lgbk/lgbk/xpptut15/ws/files_for_live_mode_at_location?location=SLAC"
def json_runs(expname, location='SLAC'):
    assert isinstance(expname, str)
    assert len(expname) in (8,9)
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/files_for_live_mode_at_location?location=%s" % (expname, location)
    krbheaders = kerberos_headers()
    r = requests.get(ws_url, headers=krbheaders)
    return value_from_responce(r)


def run_numbers(expname, location='SLAC'):
    joruns = json_runs(expname, location='SLAC')
    if joruns is None: return None
    return sorted([d['run_num'] for d in joruns])


def run_info(expname, location='SLAC'):
    joruns = json_runs(expname, location='SLAC')
    if joruns is None: return None
    return {d['run_num']: (d['begin_time'], d['end_time'], d['is_closed'], d['all_present']) for d in joruns}


def list_exp_tags(expname):
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/get_elog_tags" % expname
    krbheaders = kerberos_headers()
    r = requests.get(ws_url, headers=krbheaders)
    return value_from_responce(r)


def runnums_with_tag(expname, tag='DARK'):
    ws_url = "https://pswww.slac.stanford.edu/ws/lgbk/lgbk/%s/ws/get_runs_with_tag?tag=%s" % (expname, tag)
    r = requests.get(ws_url)
    return value_from_responce(r)


def run_table_data(expname):
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/run_table_data" % expname
    krbheaders = kerberos_headers()
    r = requests.get(ws_url, headers=krbheaders, params={"tableName": "Scan Table"})
    return value_from_responce(r)
    #return json_pro(r)


def run_parameters(expname, runnum):
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/runs/%d" % (expname, runnum)
    krbheaders = kerberos_headers()
    r = requests.get(ws_url, headers=krbheaders)
    return value_from_responce(r)


if __name__ == "__main__":

  def test_run_parameters(expname, runnum):
    jo = run_parameters(expname, runnum)
    if jo: print(json.dumps(jo, indent=2))


  def test_all_runs_with_par_value():
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{experiment_name}/ws/get_runs_matching_params".format(experiment_name="xcsdaq13")
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    r = requests.post(ws_url, headers=krbheaders, json={"USEG:UND1:3250:KACT": "3.47697", "XCS:R42:EVR:01:TRIG7:BW_TDES": "896924"})
    if r: print(r.json())


  def test_jsinfo_runs(expname, location='SLAC'):
    jo = json_runs(expname, location)
    print(json.dumps(jo, indent=2) if jo else str(jo))


  def test_run_numbers(expname, location='SLAC'):
    print('run numbers:', run_numbers(expname, location))


  def test_list_exp_tags(expname):
    print(list_exp_tags(expname))


  def test_runnums_with_tag(expname, tag='DARK'):
    print(runnums_with_tag(expname, tag))


  def test_run_table_data(expname):
    jo = run_table_data(expname)
    if jo: print(json.dumps(jo, indent=2))


if __name__ == "__main__":
    
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_run_parameters('xcsdaq13', 200)
    elif tname == '1': test_all_runs_with_par_value()
    elif tname == '2': test_run_numbers('xpptut15', location='SLAC')
    elif tname == '3': test_jsinfo_runs('xpptut15', location='SLAC')
    elif tname == '4': test_list_exp_tags('xpptut15') #xcsdaq13') #cxi78513')
    elif tname == '5': test_runnums_with_tag('xcsdaq13', tag='SCREENSHOT') # DARK')
    elif tname == '6': test_run_table_data(expname='xcsdaq13')
    elif tname == '7': test_run_table_data(expname='xpplw3319')
    else: print('test %s is not implemented' % tname)

    sys.exit('End of Test %s' % tname)

# EOF
