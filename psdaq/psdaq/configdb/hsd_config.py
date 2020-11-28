from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from p4p.client.thread import Context
import json
import time

ocfg = None
partitionDelay = None
epics_prefix = None

def hsd_connect(prefix):
    global epics_prefix
    epics_prefix = prefix

    # Retrieve connection information from EPICS
    # May need to wait for other processes here {PVA Server, hsdioc}, so poll
    ctxt = Context('pva')
    for i in range(50):
        values = ctxt.get(epics_prefix+':PADDR_U')
        if values!=0:
            break
        print('{:} is zero, retry'.format(epics_prefix+':PADDR_U'))
        time.sleep(0.1)

    ctxt.close()

    d = {}
    d['paddr'] = values
    return d

def hsd_config(connect_str,prefix,cfgtype,detname,detsegm,group):
    global partitionDelay
    global epics_prefix
    global ocfg
    epics_prefix = prefix

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    # fetch the current configuration for defaults not specified in the configuration
    ctxt = Context('pva')
    values = ctxt.get(epics_prefix+':CONFIG')

    # fetch the xpm delay
    partitionDelay = ctxt.get(epics_prefix+':MONTIMING').msgdelayset
    print('partitionDelay {:}'.format(partitionDelay))

    #
    #  Validate user raw values
    #
    raw            = cfg['user']['raw']
    raw_start      = (raw['start_ns']*1300/7000 - partitionDelay*200)*160/200 # in "160MHz"(*13/14) clks
    # raw_start register is 14 bits
    if raw_start < 0:
        print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.format(partitionDelay,raw['start_ns'],raw_start))
        raise ValueError('raw_start is too small by {:} ns'.format(-raw_start/0.16*14./13))
    if raw_start > 0x3fff:
        print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.format(partitionDelay,raw['start_ns'],raw_start))
        raise ValueError('start_ns is too large by {:} ns'.format((raw_start-0x3fff)/0.16*14./13))

    raw_gate     = int(raw['gate_ns']*0.160*13/14) # in "160" MHz clks
    raw_nsamples = raw_gate*40
    # raw_gate register is 14 bits
    if raw_gate < 0:
        raise ValueError('raw_gate computes to < 0')

    if raw_gate > 4000:
        raise ValueError('raw_gate computes to > 4000; raw_nsamples > 160000')

    #
    #  Validate user fex values
    #
    fex            = cfg['user']['fex']
    fex_start      = int((fex['start_ns']*1300/7000 - partitionDelay*200)*160/200) # in "160MHz"(*13/14) clks
    if fex_start < 0:
        print('partitionDelay {:}  fex_start_ns {:}  fex_start {:}'.format(partitionDelay,fex['start_ns'],fex_start))
        raise ValueError('fex_start computes to < 0')

    fex_gate     = int(fex['gate_ns']*0.160*13/14) # in "160" MHz clks
    fex_nsamples = fex_gate*40
    if fex_gate < 0:
        raise ValueError('fex_gate computes to < 0')
    # Place no constraint on upper bound.  Assumes sparsification will reduce to < 160000 recorded samples

    # hsd_thr_ilv_native_fine firmware expects xpre,xpost in # of super samples (4 samples) 
    fex_xpre       = int((fex['xpre' ]+3)/4)
    fex_xpost      = int((fex['xpost']+3)/4)

    # overwrite expert fields from user input
    expert = cfg['expert']
    expert['readoutGroup'] = group
    expert['enable'   ] = 1
    expert['raw_start'] = raw_start
    expert['raw_gate' ] = raw_gate
    expert['raw_prescale'] = raw['prescale']
    expert['fex_start'] = fex_start
    expert['fex_gate' ] = fex_gate
    expert['fex_xpre' ] = fex_xpre
    expert['fex_xpost'] = fex_xpost
    expert['fex_prescale'] = fex['prescale']

    # program the values
    apply_config(ctxt,cfg)

    fwver = ctxt.get(epics_prefix+':FWVERSION').value
    fwbld = ctxt.get(epics_prefix+':FWBUILD'  ).value
    cfg['firmwareVersion'] = fwver
    cfg['firmwareBuild'  ] = fwbld
    print(f'fwver: {fwver}')
    print(f'fwbld: {fwbld}')
    
    ctxt.close()

    ocfg = cfg
    return json.dumps(cfg)

def hsd_unconfig(prefix):
    global epics_prefix
    epics_prefix = prefix

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
        print('hsd unconfig complete')
    else:
        raise Exception('timed out waiting for hsd_unconfig')

    ctxt.close()

    return None;

def user_to_expert(cfg, full=False):
    global group
    global ocfg

    d = {}
    hasUser = 'user' in cfg
    if hasUser:
        hasRaw = 'raw' in cfg['user']
        raw = cfg['user']['raw']
        if (hasRaw and 'start_ns' in raw):
            raw_start      = (raw['start_ns']*1300/7000 - partitionDelay*200)*160/200

            if raw_start < 0:
                print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.
                      format(partitionDelay,raw['start_ns'],raw_start))
                raise ValueError('raw_start is too small by {:} ns'.
                                 format(-raw_start/0.16*14./13))
            if raw_start > 0x3fff:
                print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.
                      format(partitionDelay,raw['start_ns'],raw_start))
                raise ValueError('start_ns is too large by {:} ns'.
                                 format((raw_start-0x3fff)/0.16*14./13))

            d['expert.raw_start'] = raw_start

        if (hasRaw and 'gate_ns' in raw):
            raw_gate     = int(raw['gate_ns']*0.160*13/14) # in "160" MHz clks
            raw_nsamples = raw_gate*40
            # raw_gate register is 14 bits
            if raw_gate < 0:
                raise ValueError('raw_gate computes to < 0')
            if raw_gate > 4000:
                raise ValueError('raw_gate computes to > 4000; raw_nsamples > 160000')

            d['expert.raw_gate'] = raw_gate

        hasFex = 'fex' in cfg['user']
        fex = cfg['user']['fex']
        if (hasFex and 'start_ns' in fex):
            fex_start      = (fex['start_ns']*1300/7000 - partitionDelay*200)*160/200

            if fex_start < 0:
                print('partitionDelay {:}  fex_start_ns {:}  fex_start {:}'.
                      format(partitionDelay,fex['start_ns'],fex_start))
                raise ValueError('fex_start is too small by {:} ns'.
                                 format(-fex_start/0.16*14./13))
            if fex_start > 0x3fff:
                print('partitionDelay {:}  fex_start_ns {:}  fex_start {:}'.
                      format(partitionDelay,fex['start_ns'],fex_start))
                raise ValueError('start_ns is too large by {:} ns'.
                                 format((fex_start-0x3fff)/0.16*14./13))

            d['expert.fex_start'] = fex_start

        if (hasFex and 'gate_ns' in fex):
            fex_gate     = int(fex['gate_ns']*0.160*13/14) # in "160" MHz clks
            fex_nsamples = fex_gate*40
            # fex_gate register is 14 bits
            if fex_gate < 0:
                raise ValueError('fex_gate computes to < 0')
            if fex_gate > 4000:
                raise ValueError('fex_gate computes to > 4000; fex_nsamples > 160000')

            d['expert.fex_gate'] = fex_gate

    update_config_entry(cfg,ocfg,d)

def apply_config(ctxt,cfg):
    global epics_prefix

    # program the values
    print(epics_prefix)
    ctxt.put(epics_prefix+':READY',0,wait=True)
    if 'adccal' in cfg:
        values = ctxt.get(epics_prefix+':ADCCAL')
        for k,v in cfg['adccal'].items():
            values[k] = v
        ctxt.put(epics_prefix+':ADCCAL',values,wait=True)
    values = ctxt.get(epics_prefix+':CONFIG')
    if 'expert' in cfg:
        for k,v in cfg['expert'].items():
            values[k] = v
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
        time.sleep(2)
        print('hsd config returning')
    else:
        raise Exception('timed out waiting for hsd configure')


def hsd_scan_keys(update):
    global ocfg
    print('hsd_scan_keys update {}'.format(update))
    print('hsd_scan_keys ocfg {}'.format(ocfg))
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg, ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def hsd_update(update):
    global ocfg
    #  extract updates
    cfg = {}
    update_config_entry(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cfg,full=False)
    #  Apply config
    ctxt = Context('pva')
    apply_config(ctxt,cfg)
    ctxt.close()

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

