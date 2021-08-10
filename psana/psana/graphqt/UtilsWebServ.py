#!/usr/bin/env python

"""
Usage ::

    import psana.graphqt.UtilsWebServ as uws

Created on 2021-08-09 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import requests
import json
from krtc import KerberosTicket
logger = logging.getLogger(__name__)

def json_runs(expname, location='SLAC'):
    assert isinstance(expname, str)
    assert len(expname) in (8,9)
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/%s/ws/files_for_live_mode_at_location?location=%s" % (expname, location)
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    r = requests.get(ws_url, headers=krbheaders)
    return r.json() if r else r


def list_runs(expname, location='SLAC'):
    jo = json_runs(expname, location='SLAC')
    if not jo:
        logger.warning('json object is not available, resp: %s' % str(jo))
        return None
    if not jo['success']:
        logger.warning('json object is not successful, resp: %s' % str(jo))
        return None
    return jo['value']


def run_numbers(expname, location='SLAC'):
    lruns = list_runs(expname, location='SLAC')
    if lruns is None: return None
    return sorted([d['run_num'] for d in lruns])


def run_info(expname, location='SLAC'):
    lruns = list_runs(expname, location='SLAC')
    if lruns is None: return None
    dinfo = {d['run_num'] : (d['begin_time'], d['end_time'], d['is_closed'], d['all_present']) for d in lruns}
    return dinfo


if __name__ == "__main__":
  def test_all_run_parameters():
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{experiment_name}/ws/runs/{run_num}".format(experiment_name="xcsdaq13", run_num=200)
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    r = requests.get(ws_url, headers=krbheaders)
    if r: print(r.json())


  def test_all_runs_with_par_value():
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{experiment_name}/ws/get_runs_matching_params".format(experiment_name="xcsdaq13")
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    #r = requests.post(ws_url, headers=krbheaders, json={"USEG:UND1:3250:KACT": "3.47697", "XCS:R42:EVR:01:TRIG7:BW_TDES": "896924"})
    r = requests.post(ws_url, headers=krbheaders, json={'type': 'DATA'})
    if r: print(r.json())


  # curl -s "https://pswww.slac.stanford.edu/ws/lgbk/lgbk/xpptut15/ws/files_for_live_mode_at_location?location=SLAC"
  def test_all_runs():
    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/xpptut15/ws/files_for_live_mode_at_location?location=SLAC"
    #ws_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/xpptut15/ws/files_for_live_mode_at_location?location=NERSK"
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    r = requests.get(ws_url, headers=krbheaders)
    s = json.dumps(r.json(), indent=2) if r else 'resp: %s' % str(r)
    print(s)


  def test_jsinfo_runs(expname, location='SLAC'):
    jo = json_runs(expname, location)
    s = json.dumps(jo, indent=2) if jo else str(jo)
    print(s)
    print('run numbers:', run_numbers(expname, location))

if __name__ == "__main__":
    
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    import sys 
    tname = sys.argv[1] if len(sys.argv) > 1 else '3'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_all_run_parameters()
    elif tname == '1': test_all_runs_with_par_value()
    elif tname == '2': test_all_runs()
    elif tname == '3': test_jsinfo_runs('xpptut15', location='SLAC')
    else: print('test %s is not implemented' % tname)

    sys.exit('End of Test %s' % tname)

# EOF
