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

#--------------------

def get_platform():
    """returns list of processes after control.getPlatform() request
    """
    list_procs = []

    try:
        jo = daq_control().getPlatform()

    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed on request control.getPlatform()')

    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt')            
        return list_procs

    sj = json.dumps(jo, indent=2, sort_keys=False)
    logger.debug('control.getPlatform() json:\n%s' % str(sj))

    try:
        for pname in jo:
            #print("proc_name: %s" % str(pname))
            for k,v in jo[pname].items() :
                host = v['proc_info']['host']
                if v['active'] == 1: host += ' *'
                pid = v['proc_info']['pid']
                display = '%s/%s/%-16s' % (pname, pid, host)
                #print(display)
                if pname == 'control' : list_procs.insert(0, display) # show control level first
                else:                   list_procs.append(display)
    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))

    return list_procs

#--------------------

def load_json_from_file(fname) :
    from psana.pyalgos.generic.Utils import load_textfile
    s = load_textfile(fname)
    #ucode = s.decode('utf8').replace("\'t", ' not').replace("'", '"')
    #ss = s.replace(' ', '').replace("'", '"')
    return json.loads(s)

#--------------------

def str_json(jo, indent=4, sort_keys=False, separators=(', ', ': ')) :
    """ returns formatted str with json kwargs: indent=2, sort_keys=False
    """
    return json.dumps(jo, indent=indent, sort_keys=sort_keys, separators=separators)

#--------------------
