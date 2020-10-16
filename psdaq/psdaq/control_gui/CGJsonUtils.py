#--------------------
"""
:py:class:`CGJsonUtils` - json parsing methods for specific requests
=====================================================================

Usage::

    # Import
    from psdaq.control_gui.CGJsonUtils import get_platform

    # Methods - see :py:class:`CGWMainPartition`

    # global methods
    pname, pid, host, alias = _display_name(pname, dic)
    v = dict_platform() # returns control.getPlatform() or cp.s_platform or None

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
from psdaq.control_gui.CGConfigParameters import cp

#from psana.pyalgos.generic.Utils import load_textfile
from psdaq.control_gui.Utils import load_textfile

#--------------------

def _display_name(pname, v) :
    """returns (str) like 'pname/pid/host alias' """
    pinfo = v['proc_info']
    host  = pinfo.get('host', 'non-def')
    pid   = pinfo.get('pid',  'non-def')
    alias = pinfo.get('alias','non-def')
    return '%s/%s/%s %s' % (pname, pid, host, alias)

#--------------------

def dict_platform():
    """ returns control.getPlatform() or cp.s_platform or None
    """
    if cp.s_platform is not None : return cp.s_platform

    dict_platf = None
    try:
        dict_platf = daq_control().getPlatform() # returns dict

    except Exception as ex:
        logger.error('Exception in daq_control.getPlatform(): %s' % ex)
        print('failed on request control.getPlatform()')

    sj = json.dumps(dict_platf, indent=2, sort_keys=False)
    logger.debug('control.getPlatform() json(type:%s):\n%s' % (type(dict_platf), str(sj)))

    return dict_platf

#--------------------

def get_status(header=['drp','teb','meb']):
    """ returns 2-d list for CGWMainCollection.py table from dict_platf = daq_control().getPlatform()
    """
    ncols = len(header)

    dict_platf = dict_platform()
    sj = json.dumps(dict_platf, indent=2, sort_keys=False)
    #logger.debug('get_status for json(type:%s):\n%s' % (type(dict_platf), str(sj)))

    row_counter={v:0 for v in header}

    # find number of rows
    try:
        for pname in dict_platf:
            in_header = pname in header
            #logger.debug('XXX: proc grp %s %s found in header'%\
            #      (pname,{True:'is', False:'is not'}[in_header]))
            for k,v in dict_platf[pname].items() :
                if pname in header : row_counter[pname] += 1

    except Exception as ex:
        logger.error('Exception in parsing dict_platform (1): %s' % ex)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))
        return []

    nrows = max(row_counter.values())

    #logger.debug('fill out table for nrows: %d ncols: %d' % (nrows,ncols))

    # fill out table
    row_counter={v:0 for v in header}
    list2d = [['' for c in range(ncols)] for r in range(nrows)]
    try:
        for pname in dict_platf:
            #logger.debug("json top key name: %s" % str(pname))
            if not(pname in header) : continue
            col = header.index(pname)
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v) # 'pname/pid/host alias'
                flds = display.split(' ')
                alias = flds[1] if len(flds)==2 else ''
                name = alias if alias else flds[0]

                if not (v['active']==1) : continue

                row = row_counter[pname]

                #if pname=='drp' : list2d[row][0] = str(v['readout'])

                #print('XXX fill field row:%d col:%d name:%s' % (row,col,name))
                list2d[row][col] = name
                row_counter[pname] += 1

    except Exception as ex:
        logger.error('Exception in parsing dict_platf (2): %s' % ex)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))

    return list2d

    #return [['1', 'drp1','teb1','meb1'],\
    #        ['2', 'drp2','teb2','meb2'],\
    #        ['3', 'drp3','teb3','meb3']]

#--------------------

def get_platform():
    """ used in CGWMainPartition.py -> 
        returns dict_platf and 2-d list of fields: [['just-text', [True,''], [True,'cbx-descr', <int-flag>, "<validator reg.exp.>"]], ...],
        - 'just-text' (str) - field of text
        - [True,''] (list)  - field of check-box
        - [True,'', <int-flag>] (list)  - field of check-box with flags
        - [True,'cbx-descr', <int-flag>, "<validator reg.exp.>"] (list) - field of check-box and text with flags and validator
        after control.getPlatform() request.
        <int-flag> = 1/2/4/8 : BIT_CHECKABLE/ENABLED/EDITABLE/SELECTABLE (see QWTableOfCheckBoxes.py)
    """
    list2d = []
    dict_platf = dict_platform()

    try:
        for pname in dict_platf: # iterate over top key-wards
            #logger.debug("json top key name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                if v['hidden'] == 1 : continue    # skip hidden entry
                display = _display_name(pname, v) # 'pname/pid/host alias'
                flds = display.split(' ')
                alias = flds[1] if len(flds)==2 else ''

                is_drp = pname=='drp'
                readgr = v['det_info']['readout'] if is_drp else ''

                flag = 6 if is_drp else 2
                list2d.append([[v['active']==1, ''],\
                               [False, str(readgr), flag, "^([0-9]|1[0-5])$"],\
                               [False, flds[0], 2],\
                               [False, alias, 2]])

    except Exception as ex:
        logger.error('Exception in parsing dict_platf (3): %s' % ex)
        sj = json.dumps(dict_platf, indent=2, sort_keys=False)
        print('failed to parse json after control.getPlatform() request:\n%s' % str(sj))

    return dict_platf, list2d

#--------------------

def set_platform(dict_platf, list2d):
    """ Converts list2d (table presentation) to dict_platf; sets processes active/inactive etc.
    """
    #print('dict_platf: ',dict_platf)
    #print('list2d:',list2d)

    try:
        s = ''
        for pname in dict_platf:
            #print("proc_name: %s" % str(pname))
            for k,v in dict_platf[pname].items() :
                display = _display_name(pname, v) # 'pname/pid/host alias'
                proc_pid_host = display.split(' ')[0]
                # find record/row in the table
                rec = None
                for rec in list2d :
                    if rec[2][1] == proc_pid_host : break
                if rec is None :
                    logger.error('proc/pid/host "%s" is not found in the table' % proc_pid_host)
                    continue

                status = rec[0][0] # bool
                int_active = {True:1, False:0}[status]
                dict_platf[pname][k]['active'] = int_active
                if pname=='drp' :
                    val = rec[1][1]
                    if isinstance(val, str) and val.isdigit() :
                        dict_platf[pname][k]['det_info']['readout'] = int(val)
                    else :
                        logger.error('set_platform WRONG drp "group readout" value "%s" for "%s"'%\
                                     (str(val), display))
 
                s += '%s   int_active: %d\n' % (display,int_active)
        
        sj = json.dumps(dict_platf, indent=2, sort_keys=False)
        logger.debug('control.setPlatform() json:\n%s' % str(sj))
        logger.debug('summary:\n%s' % s)        
        daq_control().selectPlatform(dict_platf)

    except Exception as ex:
        logger.error('Exception in parsing dict_platf (4): %s' % ex)

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
