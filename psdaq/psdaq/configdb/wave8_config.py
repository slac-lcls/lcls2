from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from p4p.client.thread import Context

import json
import time
import pprint

ocfg = None
group = None
lane = 0

def epics_put(cfg,epics_prefix,names,values):
    global lane

    # translate legal Python names to Rogue names
    rogue_translate = {'TriggerEventBuffer':'TriggerEventBuffer[%d]'%lane,
                       'AdcReadout0'       :'AdcReadout[0]',
                       'AdcReadout1'       :'AdcReadout[1]',
                       'AdcReadout2'       :'AdcReadout[2]',
                       'AdcReadout3'       :'AdcReadout[3]',
                       'AdcConfig0'        :'AdcConfig[0]',
                       'AdcConfig1'        :'AdcConfig[1]',
                       'AdcConfig2'        :'AdcConfig[2]',
                       'AdcConfig3'        :'AdcConfig[3]'}

    rogue_not_arrays = ['CorrCoefficientFloat64','BuffEn','DelayAdcALane','DelayAdcBLane']

    for key,val in cfg.items():
        if key in rogue_translate:
            key = rogue_translate[key]
        if isinstance(val,dict):
            epics_put(val,epics_prefix+key+':',names,values)
        else:
            if key in rogue_not_arrays:
                for i,v in enumerate(val):
                    names.append(epics_prefix+key+'[%d]'%i)
                    values.append(v)
            names.append(epics_prefix+key)
            values.append(val)
        
def wave8_init(epics_prefix, xpmpv=None):
    return epics_prefix

def wave8_init_feb(slane=None,schan=None):
    global lane
    if slane is not None:
        lane = int(slane)

def wave8_connect(epics_prefix):

    # Retrieve connection information from EPICS
    # May need to wait for other processes here, so poll
    ctxt = Context('pva')
    for i in range(50):
        values = ctxt.get(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId').raw.value
        if values!=0:
            break
        print('{:} is zero, retry'.format(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId'))
        time.sleep(0.1)

    ctxt.close()

    d = {}
    d['paddr'] = values
    return json.dumps(d)

def user_to_expert(ctxt, prefix, cfg, full=False):
    global group
    global ocfg

    d = {}
    try:
        partitionDelay = ctxt.get(prefix+'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]'%group)
        raw            = cfg['user']['raw']
        rawStart       = raw['start_ns']
        triggerDelay   = rawStart*1300/7000 - partitionDelay*200
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            raise ValueError('triggerDelay computes to < 0')
        d['expert.TriggerEventManager.TriggerEventBuffer.TriggerDelay'] = triggerDelay
    except KeyError:
        pass

    try:
        raw         = cfg['user']['raw']
        rawNsamples = int(raw['gate_ns']*0.25)
        if rawNsamples>256:
            raise ValueError('raw.gate_ns > 1020')
        d['raw.nsamples'] = rawNsamples
        d['expert.Top.RawBuffers.BuffLen'] = rawNsamples-1
    except KeyError:
        pass

    try:
        fex           = cfg['user']['fex']
        intStart      = fex['start_ns']
        rawStart      = ocfg['raw']['start_ns']
        if intStart < rawStart:
            print('fex.start_ns {:}  raw.start_ns {:}'.format(intStart,rawStart))
            raise ValueError('fex.start_ns < raw.start_ns')

        fexTrigDelay  = int((intStart-rawStart)*250/1000)
        if fexTrigDelay > 255:
            raise ValueError('fex.start_ns > raw.start_ns + 1020')
        d['expert.Integrators.TrigDelay'] = fexTrigDelay
    except KeyError:
        pass

    try:
        fex           = cfg['user']['fex']
        fexNsamples = int(fex['gate_ns']*0.25)
        if fexNsamples>255:
            raise ValueError('fex.gate_ns > 1020')
        d['fex.nsamples'] = fexNsamples
        d['expert.Integrators.IntegralSize'] = fexNsamples-1
    except KeyError:
        pass

    try:
        raw         = cfg['user']['raw'] 
        # Firmware needs a value one less
        prescale = raw['prescale']
        if prescale>0:  
            prescale -= 1
        d['expert.RawBuffers.TrigPrescale'] = prescale
    except KeyError:
        pass

    if full:
        raw = cfg['user']['raw'] 
        fex = cfg['user']['fex']
        d['expert.RawBuffers.BuffEn'] = raw['enable']
        d['expert.TriggerEventManager.TriggerEventBuffer.Partition'] = group
        d['expert.Integrators.BaselineSize'] = fex['baseline']
        d['expert.Integrators.CorrCoefficientFloat64'] = fex['coeff']

    update_config_entry(cfg,ocfg, d)

def config_expert(ctxt, prefix, cfg):
    names  = []
    values = []

    epics_put(cfg,prefix+':',names,values)
    print('names {:}'.format(names))
    ctxt.put(names,values)


def wave8_config(prefix,connect_str,cfgtype,detname,detsegm,grp):
    global ctxt
    global lane
    global group

    group = grp
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    ctxt = Context('pva')

    epics_prefix = prefix + ':Top:'
    user_to_expert(ctxt, epics_prefix, cfg, full=True) 

    #  Assert clears
    names_clr = [epics_prefix+'BatcherEventBuilder:Blowoff',
                 epics_prefix+'TimingFrameRx:RxCountReset',
                 epics_prefix+'RawBuffers:CntRst',
                 epics_prefix+'Integrators:CntRst']
    values = [1]*len(names_clr)
    print('names {:}'.format(names_clr))
    ctxt.put(names_clr,values)

    config_expert(ctxt, epics_prefix, cfg['expert'])

    ctxt.put(epics_prefix+'TriggerEventManager:TriggerEventBuffer[%d]:MasterEnable'%lane, 1, wait=True)

    time.sleep(0.2)

    #  Deassert clears
    values = [0]*len(names_clr)
    print('names {:}'.format(names_clr))
    ctxt.put(names_clr,values)
    ctxt.put(epics_prefix+'BatcherEventBuilder:Blowoff', 0, wait=True)

    cfg['firmwareVersion'] = ctxt.get(epics_prefix+'AxiVersion:FpgaVersion').raw.value
    cfg['firmwareBuild'  ] = ctxt.get(epics_prefix+'AxiVersion:BuildStamp').raw.value
    
    ctxt.close()

    v = json.dumps(cfg)
    return v

def wave8_scan_keys(prefix,update):
    global ocfg
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg,ocfg, json.loads(update))
    #  Apply group
    ctxt = Context('pva')
    user_to_expert(ctxt, prefix+':Top:',cfg,full=False)
    ctxt.close()
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def wave8_update(prefix,update):
    global ocfg
    #  extract updates
    cfg = {}
    epics_prefix = prefix+':Top:'
    update_config_entry(cfg,ocfg, json.loads(update))
    #  Apply group
    ctxt = Context('pva')
    user_to_expert(ctxt, epics_prefix, cfg, full=False)
    #  Apply config
    config_expert(ctxt, epics_prefix, cfg['expert'])
    ctxt.close()
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)


def wave8_unconfig(epics_prefix):

    return None;

    ctxt = Context('pva')
    ctxt.put(epics_prefix+':TriggerEventManager:TriggerEventBuffer[%d]:MasterEnable'%lane, 0, wait=True)
    ctxt.close()

    return None;
