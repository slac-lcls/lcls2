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
    """ returns [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]
        #returns [[[True,'test/19670/daq-tst-dev02'], 'testClient2b'], ...]
        after control.getPlatform() request
    """
    list2d = []

    try:
        dict_platf = daq_control().getPlatform() # returns dict

    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed on request control.getPlatform()')

    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt')           
        return {}, list2d

    sj = json.dumps(dict_platf, indent=2, sort_keys=False)
    logger.debug('control.getPlatform() json(type:%s):\n%s' % (type(dict_platf), str(sj)))

    try:
        for pname in dict_platf: # iterate over top key-wards
            #logger.debug("json top key name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v)
                #print(display)
                flds = display.split(' ')
                #list2d.append([[v['active']==1, flds[0]], flds[1] if len(flds)==2 else ' '])
                list2d.append([[v['active'] == 1, ''], flds[0], flds[1] if len(flds) == 2 else ''])

    except Exception as ex:
        logger.error('Exception: %s' % ex)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))

    return dict_platf, list2d

#--------------------

def set_platform(dict_platf, list2d):
    """ Sets processes active/inactive
    """
    #print('dict_platf: ',dict_platf)
    #print('list2d:',list2d)

    try:
        s = ''
        i = -1
        for pname in dict_platf:
            #print("proc_name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v)
                #status = display.split(' ')[0][0] # bool
                i += 1
                status = list2d[i][0][0] # bool
                int_active = {True:1, False:0}[status]
                dict_platf[pname][k]['active'] = int_active
                s += '%s   int_active: %d\n' % (display,int_active)
        
        sj = json.dumps(dict_platf, indent=2, sort_keys=False)
        logger.debug('control.setPlatform() json:\n%s' % str(sj))
        logger.debug('summary:\n%s' % s)        
        daq_control().selectPlatform(dict_platf)

    except Exception as ex:
        logger.error('Exception: %s' % ex)

#--------------------

def list_active_procs(lst):
    """returns a subset list2d of active/selected only from list2d of all processes.
    """
    assert(isinstance(lst, list))
    return [r for r in lst if r[0][0]]

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
