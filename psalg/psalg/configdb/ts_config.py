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
    xpm_master = control_info['xpm_master']
    pv_prefix = control_info['pv_base']+':XPM:'+str(xpm_master)+':PART:'

    # this structure of epics variable names must mirror
    # the configdb.  alternatively, we could consider
    # putting these in the configdb, perhaps as readonly fields.

    # only do a few of these for now, since Matt is switching
    # to rogue
    pvtable = {}
    mydict = {}
    for group in readout_groups:
        grp_prefix = 'group'+str(group)
        grp = cfg[grp_prefix]

        pvtable[grp_prefix] = {'trigMode':'L0Select',
                               'delay':'L0Delay',
                               'fixed' : {'rate'    : 'L0Select_FixedRate'},
                               'ac' : {'rate'    : 'L0Select_ACRate'},
                               'seq' : {'mode'   : 'L0Select_Sequence'},
                               'destination' : {'select'   : 'DstSelect'},
                  }

        epics_names_values(group,pvtable,cfg,mydict)

        # handle special cases that don't work in the "pvtable" paradigm

        # convert ac.ts0 through ac.ts5 to L0Select_ACTimeslot bitmask
        tsmask = 0
        for tsnum in range(6):
            tsval = grp['ac']['ts'+str(tsnum)]
            tsmask |= 1<<tsval
        mydict[str(group)+':L0Select_ACTimeslot'] = tsmask

        # L0Select_SeqBit is one var used by all of seq.(burst/fixed/local)
        if grp['seq']['mode']==15: # burst
            seqbit = grp['seq']['burst']['mode']
        elif grp['seq']['mode']==16: # fixed rate
            seqbit = grp['seq']['fixed']['rate']
        elif grp['seq']['mode']==17: # local
            seqbit = grp['seq']['local']['rate']
        else:
            raise ValueError('Illegal value for trigger sequence mode')
        mydict[str(group)+':L0Select_SeqBit'] = seqbit

        # DstSelect_Mask should come from destination.dest0 through dest15
        dstmask = 0
        for dstnum in range(16):
            dstval = grp['destination']['dest'+str(dstnum)]
            if dstval:
                dstmask |= 1<<dstnum
        mydict[str(group)+':DstSelect_Mask'] = dstmask

        # 4 InhEnable/InhInterval/InhLimit
        for inhnum in range(4):
            mydict[str(group)+':InhInterval'+str(inhnum)] = grp['inhibit'+str(inhnum)]['interval']
            mydict[str(group)+':InhLimit'+str(inhnum)] = grp['inhibit'+str(inhnum)]['limit']
            mydict[str(group)+':InhEnable'+str(inhnum)] = grp['inhibit'+str(inhnum)]['enable']

    names = list(mydict.keys())
    values = list(mydict.values())
    names = [pv_prefix+n for n in names]
    print('TS config names and values:',names,values)

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)
    ctxt.close()

    return json.dumps(cfg)
