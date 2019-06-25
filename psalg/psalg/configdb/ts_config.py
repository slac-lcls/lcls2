from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def epics_names_values(group,pvtable,cfg,d):
    for k, v1 in pvtable.items():
        v2 = cfg[k]
        if isinstance(v1, dict):
            epics_names_values(group,v1,v2,d)
        else:
            # Not sure whether configuration should be a structure of arrays or array of structures
            # For now, just convert each array to its first element
            d[str(group)+':'+v1] = v2

def ts_config(connect_json,cfgtype,detname):

    cfg = get_config(connect_json,cfgtype,detname)
    connect_info = json.loads(connect_json)

    # get the list of readout groups that the user has selected
    # so we only configure those
    readout_groups = []
    connect_info = json.loads(connect_json)
    for nodes in connect_info['body']['drp'].values():
        readout_groups.append(nodes['det_info']['readout'])
    readout_groups = set(readout_groups)

    control_info = connect_info['body']['control']['0']['control_info']
    pv_prefix = control_info['pv_base']+':PART:'

    # this structure of epics variable names must mirror
    # the configdb.  alternatively, we could consider
    # putting these in the configdb, perhaps as readonly fields.

    # only do a few of these for now, since Matt is switching
    # to rogue
    pvtable = {}
    mydict = {}
    for group in readout_groups:
        grp_prefix = 'group'+str(group)
        pvtable[grp_prefix] = {'trigMode':'L0Select',
                               'fixed' : {'rate'    : 'L0Select_FixedRate'}
                           }
        epics_names_values(group,pvtable,cfg,mydict)

    names = list(mydict.keys())
    values = list(mydict.values())
    names = [pv_prefix+n for n in names]
    print('TS config names and values:',names,values)

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)
    ctxt.close()

    return json.dumps(cfg)
