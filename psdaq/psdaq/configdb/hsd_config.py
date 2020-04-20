from psdaq.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def hsd_config(connect_str,epics_prefix,cfgtype,detname,detsegm,group):

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
    if raw_start < 0:
        print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.format(partitionDelay,raw['start_ns'],raw_start))
        raise ValueError('raw_start computes to < 0')

    raw_gate     = int(raw['gate_ns']*0.160*13/14) # in "160" MHz clks
    raw_nsamples = raw_gate*40
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
    print(epics_prefix)
    ctxt.put(epics_prefix+':READY',0,wait=True)
    ctxt.put(epics_prefix+':CONFIG',expert,wait=True)

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

    cfg['firmwareVersion'] = ctxt.get(epics_prefix+':FWVERSION').raw.value
    cfg['firmwareBuild'  ] = ctxt.get(epics_prefix+':FWBUILD'  ).raw.value
    
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
