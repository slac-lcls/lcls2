from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def epics_names_values(pvtable,cfg,d):
    for k, v1 in pvtable.items():
        v2 = cfg[k]
        if isinstance(v1, dict):
            epics_names_values(v1,v2,d)
        else:
            # Not sure whether configuration should be a structure of arrays or array of structures
            # For now, just convert each array to its first element
            if isinstance(v2, list):
                d[v1] = v2[0]    
            else:
                d[v1] = v2

def hsd_config(connect_str,epics_prefix,cfgtype,detname,group):

    cfg = get_config(connect_str,cfgtype,detname)

    # this structure of epics variable names must mirror
    # the configdb.  alternatively, we could consider
    # putting these in the configdb, perhaps as readonly fields.
    pvtable = {'enable':'enable',
               'raw' : {'start'    : 'raw_start',
                        'gate'     : 'raw_gate',
                        'prescale' : 'raw_prescale'},
               'fex' : {'start'    : 'fex_start',
                        'gate'     : 'fex_gate',
                        'prescale' : 'fex_prescale',
                        'ymin'     : 'fex_ymin',
                        'ymax'     : 'fex_ymax',
                        'xpre'     : 'fex_xpre',
                        'xpost'    : 'fex_xpost'},
               'expert' : {'datamode'   : 'test_pattern',
                           'fullthresh' : 'full_event',
                           'fullsize'   : 'full_size',
                           'fsrange'    : 'fs_range_vpp',    # FMC134 only
                           'trigshift'  : 'trig_shift',      # FMC126 only
                           'synce'      : 'sync_ph_even',    # FMC126 only
                           'synco'      : 'sync_ph_odd'},    # FMC126 only
    }

    # fetch the current configuration for defaults not specified in the configuration
    ctxt = Context('pva')
    values = ctxt.get(epics_prefix+':CONFIG')

    # look in the cfg dictionary for values that match the epics
    # variables in the pvtable
    epics_names_values(pvtable,cfg,values)
    values['readoutGroup'] = group

    # program the values
    print(epics_prefix)
    ctxt.put(epics_prefix+':READY',0,wait=True)

    print(values)
    ctxt.put(epics_prefix+':CONFIG',values,wait=True)

    # the completion of the "put" guarantees that all of the above
    # have completed (although in no particular order)
    complete = False
    for i in range(100):
        complete = ctxt.get(epics_prefix+':READY')!=0
        if complete: break
        print('hsd config wait for complete',i)
        time.sleep(0.1)
    if complete:
        print('hsd config complete')
    else:
        raise Exception('timed out waiting for hsd configure')

    ctxt.close()

    return json.dumps(cfg)

def hsd_unconfig(epics_prefix):

    ctxt = Context('pva')
    values = ctxt.get(epics_prefix+':CONFIG')
    values['enable'] = 0
    print(values)
    print(epics_prefix)
    ctxt.put(epics_prefix+':CONFIG',values,wait=True)

    #  This handshake seems to be necessary, or at least the .get()
    complete = False
    for i in range(100):
        complete = ctxt.get(epics_prefix+':READY')!=0
        if complete: break
        print('hsd_unconfig wait for complete',i)
        time.sleep(0.1)
    if complete:
        print('hsd config complete')
    else:
        raise Exception('timed out waiting for hsd_unconfig')

    ctxt.close()

    return None;
