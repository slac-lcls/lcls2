from psdaq.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def ts_config(connect_json,cfgtype,detname,detsegm):

    cfg = get_config(connect_json,cfgtype,detname,detsegm)
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
    pv_prefix    = control_info['pv_base']+':XPM:'+str(xpm_master)+':PART:'

    rcfg = cfg.copy()
    rcfg['user'  ] = {}
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
    names = [pv_prefix+n for n in names]

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)

    #  Capture firmware version for persistence in xtc
    pv_prefix = control_info['pv_base']+':XPM:'+str(xpm_master)+':'
    #rcfg['firmwareVersion'] = ctxt.get(pv_prefix+'FwVersion').raw.value
    rcfg['firmwareBuild'  ] = ctxt.get(pv_prefix+'FwBuild').raw.value
    ctxt.close()

    return json.dumps(rcfg)
