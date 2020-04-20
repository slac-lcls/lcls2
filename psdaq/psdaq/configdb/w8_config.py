from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def epics_put(cfg,epics_prefix,names,values):

    for key,val in cfg.items():
        if isinstance(val,dict):
            epics_put(val,epics_prefix+key+':',names,values)
        else:
            names.append(epics_prefix+key)
            values.append(val)
        
def w8_config(connect_str,prefix,cfgtype,detname,detsegm,group):
    global ctxt

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    ctxt = Context('pva')

    #  | timing fiducial
    #  PartitionDelay  | TriggerEventManager.TriggerEventBuffer receives xpm trigger
    #                    TriggerDelay |  TriggerEventManager.triggerBus asserts trigger
    #                                    IntStart | Integrators.intStart (baseline latched)
    #                                               IntLen  | Intregrators.intEnd
    #                                 |  RawDataBuffer start
    #                                   RawBuffLen             |  RawDataBuffer End

    epics_prefix = prefix + ':Top:'
    partitionDelay = ctxt.get(epics_prefix+'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]'%group)
    raw            = cfg['user']['raw']
    rawStart       = raw['start_ns']
    triggerDelay   = rawStart*1300/7000 - partitionDelay*200
    if triggerDelay < 0:
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        raise ValueError('triggerDelay computes to < 0')

    rawNsamples = int(raw['gate_ns']*0.25)
    if rawNsamples>256:
        raise ValueError('raw.gate_ns > 1020')
    raw['nsamples'] = rawNsamples

    fex           = cfg['user']['fex']
    intStart      = fex['start_ns']
    if intStart < rawStart:
        print('fex.start_ns {:}  raw.start_ns {:}'.format(intStart,rawStart))
        raise ValueError('fex.start_ns < raw.start_ns')

    fexTrigDelay  = int((intStart-rawStart)*250/1000)
    if fexTrigDelay > 255:
        raise ValueError('fex.start_ns > raw.start_ns + 1020')

    fexNsamples = int(fex['gate_ns']*0.25)
    if fexNsamples>255:
        raise ValueError('fex.gate_ns > 1020')
    fex['nsamples'] = rawNsamples

    #  Assert clears
    names_clr = [epics_prefix+'BatcherEventBuilder:Blowoff',
                 epics_prefix+'TimingFrameRx:RxCountReset',
                 epics_prefix+'RawBuffers:CntRst',
                 epics_prefix+'Integrators:CntRst']
    values = [1]*len(names_clr)
    print('names {:}'.format(names_clr))
    ctxt.put(names_clr,values)

    expert = cfg['expert']['Top']
    expert['TriggerEventManager']['TriggerEventBuffer[0]']['TriggerDelay'] = triggerDelay
    for i in range(8):
        expert['RawBuffers']['BuffEn[%d]'%i] = raw['enable[%d]'%i]

    # Firmware needs a value one less        
    expert['RawBuffers']['BuffLen'] = rawNsamples-1
    # Firmware needs a value one less
    prescale = raw['prescale']
    if prescale>0:  
        prescale -= 1
    expert['RawBuffers']['TrigPrescale'] = prescale

    expert['Integrators']['TrigDelay'] = fexTrigDelay
    # Firmware needs a value one less        
    expert['Integrators']['IntegralSize'] = fexNsamples-1
    expert['Integrators']['BaselineSize'] = fex['baseline']
    
    for i in range(4):
        expert['Integrators']['CorrCoefficientFloat64[%d]'%i] = fex['coeff[%d]'%i]

    expert['TriggerEventManager']['TriggerEventBuffer[0]']['Partition'] = group

    names  = []
    values = []
    epics_put(cfg['expert'],prefix+':',names,values)
    ctxt.put(names,values)

    ctxt.put(epics_prefix+'TriggerEventManager:TriggerEventBuffer[0]:MasterEnable', 1, wait=True)

    time.sleep(0.2)

    #  Deassert clears
    values = [0]*len(names_clr)
    ctxt.put(names_clr,values)
    ctxt.put(epics_prefix+'BatcherEventBuilder:Blowoff', 0, wait=True)

    cfg['firmwareVersion'] = ctxt.get(epics_prefix+'AxiVersion:FpgaVersion').raw.value
    cfg['firmwareBuild'  ] = ctxt.get(epics_prefix+'AxiVersion:BuildStamp').raw.value
    
    ctxt.close()

    v = json.dumps(cfg)
    return v

def w8_unconfig(epics_prefix):

    return None;

    ctxt = Context('pva')
    ctxt.put(epics_prefix+':TriggerEventManager:TriggerEventBuffer[0]:MasterEnable', 0, wait=True)
    ctxt.close()

    return None;
