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
            d[v1] = v2[0]    

def hsd_config(connect_str,epics_prefix,cfgtype,detname):

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
    }

    # look in the cfg dictionary for values that match the epics
    # variables in the pvtable
    values = {}
    epics_names_values(pvtable,cfg,values)

    # program the values
    ctxt = Context('pva')
    ctxt.put(epics_prefix+':READY',0)

    print(epics_prefix+':CONFIG')
    print(values)
    ctxt.put(epics_prefix+':CONFIG',values)

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
