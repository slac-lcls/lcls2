from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.ts_connect import ts_connector
from psdaq.seq.globals import *
from p4p.client.thread import Context
import json
import logging
import time

ocfg = None
pv_prefix = None
readout_groups = None
connector = None

DEST_INCLUDE  = 0
DEST_DONTCARE = 1
DEST_BSY = 2
DEST_HXR = 3
DEST_SXR = 4

#  XpmApp's l0RawUpdate register is 20 bits (pyxpm/xpm/_XpmApp.py)
L0RAWUPDATE_MAX = (1 << 20) - 1

def _rawUpdateDivisor(rawRate, where):
    """Convert keepRawRate in Hz to the XPM's fiducial divisor, or fail loudly.

    The config field is a RATE; the PV is a DIVISOR, and they differ by ~10^6.
    Entering a divisor here used to truncate to 0 silently, which the XPM takes
    as 'never', so a run would record no raw data at all and look configured.
    """
    if rawRate <= 0:
        raise RuntimeError(f'keepRawRate in {where} must be positive, not {rawRate}')
    divisor = int(TPGSEC/rawRate)
    if divisor < 1:
        raise RuntimeError(f'keepRawRate in {where} is {rawRate} Hz, which needs a '
                           f'divisor below 1 and would be programmed as 0 (never).  '
                           f'This field is a RATE IN HZ, not the divisor the '
                           f'L0RawUpdate PV takes: the maximum is {TPGSEC} Hz.')
    if divisor > L0RAWUPDATE_MAX:
        raise RuntimeError(f'keepRawRate in {where} is {rawRate} Hz, needing divisor '
                           f'{divisor}, which does not fit L0RawUpdate\'s 20 bits.  '
                           f'The minimum rate is {TPGSEC/L0RAWUPDATE_MAX:.4f} Hz.')
    return divisor

def ts_connect(json_connect_info):
    global connector
    connector = ts_connector(json_connect_info)
    return json.dumps({})

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
    # the tpr's also send out their desired readout groups
    if 'tpr' in connect_info['body']:
        for nodes in connect_info['body']['tpr'].values():
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
        #  Cu stores this flat as groupN_keepRawRate, SC nested under groupN
        if linacMode == 0:   # Cu
            grp_prefix = 'group'+str(group)+'_eventcode'
            eventcode  = cfg['user']['Cu'][grp_prefix]
            rcfg['user']['Cu'][grp_prefix] = eventcode
            pvdict[str(group)+':L0Select'          ] = 2  # eventCode
            pvdict[str(group)+':L0Select_EventCode'] = eventcode
            pvdict[str(group)+':DstSelect'         ] = 1  # DontCare

            rawKey  = 'group'+str(group)+'_keepRawRate'
            rawRate = cfg['user']['Cu'].get(rawKey)
            rawWhere = f'user.Cu.{rawKey}'
            if rawRate is not None:
                rcfg['user']['Cu'][rawKey] = rawRate
        else:                # SC
            grp_prefix = 'group'+str(group)
            grp = cfg['user']['SC'][grp_prefix]
            rcfg['user']['SC'][grp_prefix] = grp
            pvdict[str(group)+':L0Select'          ] = grp['trigMode']
            pvdict[str(group)+':L0Select_FixedRate'] = grp['fixed']['rate']
            pvdict[str(group)+':L0Select_EventCode'] = grp['eventcode']

            rawRate  = grp.get('keepRawRate')
            rawWhere = f'user.SC.{grp_prefix}.keepRawRate'

            if 'ac' in grp:
                pvdict[str(group)+':L0Select_ACRate'   ] = grp['ac']['rate']
                # convert ac.ts0 through ac.ts5 to L0Select_ACTimeslot bitmask
                tsmask = 0
                for tsnum in range(6):
                    tsval = grp['ac']['ts'+str(tsnum)]
                    if tsval:
                        tsmask |= 1<<tsnum
                pvdict[str(group)+':L0Select_ACTimeslot'] = tsmask

            # DstSelect_Mask should come from destination.dest0 through dest15
            dstmask = 0
            if 'select' in grp['destination']:   # old
                pvdict[str(group)+':DstSelect'] = grp['destination']['select']
                for dstnum in range(16):
                    dstval = grp['destination']['dest'+str(dstnum)]
                    if dstval:
                        dstmask |= 1<<dstnum
            else:  # new
                dstmask |= (1<<DEST_BSY) if grp['destination']['BsyDump' ] else 0
                dstmask |= (1<<DEST_HXR) if grp['destination']['HardXRay'] else 0
                dstmask |= (1<<DEST_SXR) if grp['destination']['SoftXRay'] else 0
                pvdict[str(group)+':DstSelect_Mask'] = dstmask
                pvdict[str(group)+':DstSelect'     ] = DEST_INCLUDE if dstmask else DEST_DONTCARE

        #  Both LINAC modes program the raw-data insert period: the XPM counts
        #  fiducials, so a rate in Hz becomes a divisor.  Cu never did this, so
        #  keepRawRate was ignored there and every event came back marked.
        if rawRate is not None:
            pvdict[str(group)+':L0RawUpdate'   ] = _rawUpdateDivisor(rawRate, rawWhere)
        elif linacMode == 0:
            #  Cu has never programmed this, so a config predating the entry is
            #  no worse off than before: warn rather than refuse to configure
            logging.warning(f'No keepRawRate entry in {rawWhere}; every event will be marked.  Run ts_config_update.py')
        else:
            #  until we update all timing configurations
            raise RuntimeError(f'No keepRawRate entry in {rawWhere}.  Run ts_config_update.py')

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

    #  Check the errors
    connector.check_errors('config')

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
    #  Check the errors
    connector.check_errors('unconfig')
