from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from p4p.client.thread import Context
import json
import time

ocfg = None
pv_prefix = None
readout_groups = None

def ts_config(connect_json,cfgtype,detname,detsegm):
    global ocfg
    global pv_prefix
    global readout_groups

    cfg = get_config(connect_json,cfgtype,detname,detsegm)
    ocfg = cfg
    connect_info = json.loads(connect_json)

    # get the list of readout groups that the user has selected
    # so we only configure those
    readout_groups = []
    connect_info = json.loads(connect_json)
    for nodes in connect_info['body']['drp'].values():
        readout_groups.append(nodes['det_info']['readout'])
    readout_groups = set(readout_groups)

    control_info = connect_info['body']['control']['0']['control_info']
    xpm_master   = control_info['xpm_master']
    pv_prefix    = control_info['pv_base']+':XPM:'+str(xpm_master)+':'

    return apply_config(cfg)

def apply_config(cfg):
    global pv_prefix
    rcfg = {}
    rcfg = cfg.copy()
    rcfg['user'] = {}
    rcfg['expert'] = {}

    linacMode = cfg['user']['LINAC']
    rcfg['user']['LINAC'] = linacMode
    rcfg['user']['Cu' if linacMode==0 else 'SC'] = {}

    pvdict  = {}  # dictionary of epics pv name : value
    for group in readout_groups:
        if linacMode == 0:   # Cu
            grp_prefix = 'group'+str(group)+'_eventcode'
            eventcode  = cfg['user']['Cu'][grp_prefix]
            rcfg['user']['Cu'][grp_prefix] = eventcode
            pvdict[str(group)+':L0Select'          ] = 2  # eventCode
            pvdict[str(group)+':L0Select_EventCode'] = eventcode
            pvdict[str(group)+':DstSelect'         ] = 1  # DontCare
        else:                # SC
            grp_prefix = 'group'+str(group)
            grp = cfg['user']['SC'][grp_prefix]
            rcfg['user']['SC'][grp_prefix] = grp
            pvdict[str(group)+':L0Select'          ] = grp['trigMode']
            pvdict[str(group)+':L0Select_FixedRate'] = grp['fixed']['rate']
            pvdict[str(group)+':L0Select_ACRate'   ] = grp['ac']['rate']
            pvdict[str(group)+':L0Select_EventCode'] = 0  # not an option
            pvdict[str(group)+':L0Select_Sequence' ] = grp['seq']['mode']
            pvdict[str(group)+':DstSelect'         ] = grp['destination']['select']

            # convert ac.ts0 through ac.ts5 to L0Select_ACTimeslot bitmask
            tsmask = 0
            for tsnum in range(6):
                tsval = grp['ac']['ts'+str(tsnum)]
                tsmask |= 1<<tsval
            pvdict[str(group)+':L0Select_ACTimeslot'] = tsmask

            # L0Select_SeqBit is one var used by all of seq.(burst/fixed/local)
            if grp['seq']['mode']==15: # burst
                seqbit = grp['seq']['burst']['mode']
            elif grp['seq']['mode']==16: # fixed rate
                seqbit = grp['seq']['fixed']['rate']
            elif grp['seq']['mode']==17: # local
                seqbit = grp['seq']['local']['rate']
            else:
                raise ValueError('Illegal value for trigger sequence mode')
            pvdict[str(group)+':L0Select_SeqBit'] = seqbit

            # DstSelect_Mask should come from destination.dest0 through dest15
            dstmask = 0
            for dstnum in range(16):
                dstval = grp['destination']['dest'+str(dstnum)]
                if dstval:
                    dstmask |= 1<<dstnum
            pvdict[str(group)+':DstSelect_Mask'] = dstmask

        grp_prefix = 'group'+str(group)
        grp = cfg['expert'][grp_prefix]
        rcfg['expert'][grp_prefix] = grp
        # 4 InhEnable/InhInterval/InhLimit
        for inhnum in range(4):
            pvdict[str(group)+':InhInterval'+str(inhnum)] = grp['inhibit'+str(inhnum)]['interval']
            pvdict[str(group)+':InhLimit'+str(inhnum)] = grp['inhibit'+str(inhnum)]['limit']
            pvdict[str(group)+':InhEnable'+str(inhnum)] = grp['inhibit'+str(inhnum)]['enable']

    names  = list(pvdict.keys())
    values = list(pvdict.values())
    names = [pv_prefix+'PART:'+n for n in names]

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)

    #  Capture firmware version for persistence in xtc
    #rcfg['firmwareVersion'] = ctxt.get(pv_prefix+'FwVersion').raw.value
    rcfg['firmwareBuild'  ] = ctxt.get(pv_prefix+'FwBuild').raw.value
    ctxt.close()

    return json.dumps(rcfg)

def apply_update(cfg):
    global pv_prefix

    rcfg = {}
    pvdict  = {}  # dictionary of epics pv name : value

    for key in cfg:
        if key == 'user':
            rcfg['user'] = {}

            linacMode = ocfg['user']['LINAC']  # this won't scan
            if full:
                rcfg['user']['LINAC'] = linacMode
            rcfg['user']['Cu' if linacMode==0 else 'SC'] = {}

            for group in readout_groups:
                if linacMode == 0:   # Cu
                    try:
                        grp_prefix = 'group'+str(group)+'_eventcode'
                        eventcode  = cfg['user']['Cu'][grp_prefix]
                        rcfg['user']['Cu'][grp_prefix] = eventcode
                        pvdict[str(group)+':L0Select'          ] = 2  # eventCode
                        pvdict[str(group)+':L0Select_EventCode'] = eventcode
                        pvdict[str(group)+':DstSelect'         ] = 1  # DontCare
                    except KeyError:
                        pass
                else:                # SC
                    pass   # nothing here to scan (too complicated to implement)

        if key == 'expert':
            rcfg['expert'] = {}
            for group in readout_groups:
                grp_prefix = 'group'+str(group)
                if grp_prefix in cfg['expert']:
                    grp = cfg['expert'][grp_prefix]
                    rcfg['expert'][grp_prefix] = {}
                    # 4 InhEnable/InhInterval/InhLimit
                    for inhnum in range(4):
                        inhkey = 'inhibit'+str(inhnum)
                        if inhkey in grp:
                            inhgrp = grp[inhkey]
                            rcfg['expert'][grp_prefix][inhkey] = inhgrp
                            rgrp = rcfg['expert'][grp_prefix][inhkey]
                            if 'interval' in inhgrp:
                                pvdict[str(group)+':InhInterval'+str(inhnum)] = inhgrp['interval']
                            if 'limit' in inhgrp:
                                pvdict[str(group)+':InhLimit'+   str(inhnum)] = inhgrp['limit']
                            if 'enable' in inhgrp:
                                pvdict[str(group)+':InhEnable'+  str(inhnum)] = inhgrp['enable']

        else:
            rcfg[key] = cfg[key]

    names  = list(pvdict.keys())
    values = list(pvdict.values())
    names = [pv_prefix+'PART:'+n for n in names]

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)
    ctxt.close()

    return json.dumps(rcfg)

def ts_scan_keys(update):
    global ocfg
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg,ocfg, json.loads(update))

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def ts_update(update):
    global ocfg
    #  extract updates
    cfg = {}
    update_config_entry(cfg,ocfg, json.loads(update))

    #  Apply config
    apply_update(cfg)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def ts_unconfig():
    pass
