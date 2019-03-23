#--------------------
"""
:py:class:`CGJsonUtils` - json parsing methods for specific requests
=====================================================================

Usage::

    # Import
    from psdaq.control_gui.CGJsonUtils import get_platform

    # Methods - see :py:class:`CGWMainPartition`

See:
    - :py:class:`CGJsonUtils`
    - :py:class:`CGWMainPartition`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-08 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

import json
from psdaq.control_gui.CGDaqControl import daq_control #, DaqControl #, worker_set_state

#from psana.pyalgos.generic.Utils import load_textfile
from psdaq.control_gui.Utils import load_textfile

#--------------------

def _display_name(pname, v) :
    pinfo = v['proc_info']
    host  = pinfo.get('host', 'non-def')
    pid   = pinfo.get('pid',  'non-def')
    alias = pinfo.get('alias','non-def')
    return '%s/%s/%s %s' % (pname, pid, host, alias)

#--------------------

def get_platform():
    """returns list of processes after control.getPlatform() request
    """
    dict_procs = {}

    try:
        dict_platf = daq_control().getPlatform() # returns dict

    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed on request control.getPlatform()')

    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt')            
        return {}, dict_procs

    sj = json.dumps(dict_platf, indent=2, sort_keys=False)
    logger.debug('control.getPlatform() json:\n%s' % str(sj))

    try:
        for pname in dict_platf:
            #print("proc_name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v)
                #print(display)
                dict_procs[display] = (v['active'] == 1)

    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))

    return dict_platf, dict_procs

#--------------------

def set_platform(dict_platf, dict_procs):
    """ Sets processes active/inactive
    """
    try:
        for pname in dict_platf:
            #print("proc_name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v)
                int_active = {True:1, False:0}[dict_procs[display]]
                #print(display, 'int_active: %d' % int_active)
                dict_platf[pname][k]['active'] = int_active
        
        sj = json.dumps(dict_platf, indent=2, sort_keys=False)
        logger.debug('control.setPlatform() json:\n%s' % str(sj))
        
        daq_control().selectPlatform(dict_platf)

    except Exception as ex:
        logger.error('Exception: %s' % ex)

#--------------------

def load_json_from_file(fname) :
    s = load_textfile(fname)
    #ucode = s.decode('utf8').replace("\'t", ' not').replace("'", '"')
    #ss = s.replace(' ', '').replace("'", '"')
    return json.loads(s)

#--------------------

def json_from_str(s) :
    return json.loads(s)

#--------------------

def str_json(jo, indent=4, sort_keys=False, separators=(', ', ': ')) :
    """ returns formatted str with json kwargs: indent=2, sort_keys=False
    """
    return json.dumps(jo, indent=indent, sort_keys=sort_keys, separators=separators)

#--------------------
