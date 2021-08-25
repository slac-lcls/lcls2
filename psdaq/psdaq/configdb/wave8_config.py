from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import *
import epics

import json
import time
import pprint

prefix = None
ocfg   = None
group  = None
lane   = 0

#
#  Change this script to do minimal configuration:
#     1) switch to LCLS2 timing with one relative timing parameter
#        LCLS1: TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].Delay
#        LCLS2: TriggerEventManager.TriggerEventBuffer[0].TrigDelay
#     2) configure the DAQ triggering
#     3) record the full configuration (maintained by controls)
#
def ctxt_get(names):
    v = None
    if isinstance(names,str):
        v = epics.PV(names).get()
    else:
        if isinstance(names,list):
            v = []
            for i,n in enumerate(names):
                v.append(epics.PV(n).get())
    return v

def ctxt_put(names, values):
    if isinstance(names,str):
        print(f'Put {names} {values}')
        epics.PV(names).put(values)
        print(f'Put {names} complete')
    else:
        if isinstance(names,list):
            for i,n in enumerate(names):
                print(f'Put {n} {values[i]}')
                epics.PV(n).put(values[i])
                print(f'Put {n} complete')

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
            if ':RO' in key:
                continue
            if key in rogue_not_arrays:
                for i,v in enumerate(val):
                    names.append(epics_prefix+key+'[%d]'%i)
                    values.append(v)
            names.append(epics_prefix+key)
            values.append(val)

#  Create a dictionary of config key to PV name
def epics_get(d):
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

    out = {}
    for key,val in d.items():
        #  Skip these that have no PVs yet
        if ('AdcPatternTester' in key or 
            'CorrCoefficient' in key):
            continue
        
        pvname = rogue_translate[key] if key in rogue_translate else key
        if isinstance(val,dict):
            r = epics_get(val)
            for k,v in r.items():
                out[key+'.'+k] = pvname+':'+v
        else:
            if key in rogue_not_arrays:
                for i,v in enumerate(val):
                    out[key+f'.[{i}]'] = pvname+'[%d]'%i
            else:
                out[key] = pvname
    return out

def config_timing(epics_prefix, lcls2=False):
    names = [epics_prefix+':Top:SystemRegs:timingUseMiniTpg',
             epics_prefix+':Top:TimingFrameRx:ModeSelEn',
             epics_prefix+':Top:TimingFrameRx:ClkSel',
             epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0, 0, 1, 1] if lcls2 else [0, 0, 0, 1]
    ctxt_put(names,values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0]
    ctxt_put(names,values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxDown']
    values = [0]
    ctxt_put(names,values)
        
def wave8_init(epics_prefix, dev='/dev/datadev_0', lanemask=1, xpmpv=None, timebase="186M"):
    global prefix
    prefix = epics_prefix
    return epics_prefix

def wave8_init_feb(slane=None,schan=None):
    global lane
    if slane is not None:
        lane = int(slane)

def wave8_connect(epics_prefix):

    #  Switch to LCLS2 Timing
    #    Need this to properly receive RxId
    #    Controls is no longer in-control
    config_timing(epics_prefix,lcls2=True)

    # Retrieve connection information from EPICS
    # May need to wait for other processes here, so poll
    for i in range(50):
        values = ctxt_get(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId')
        if values!=0:
            break
        print('{:} is zero, retry'.format(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId'))
        time.sleep(0.1)

    d = {}
    d['paddr'] = values
    print(f'wave8_connect returning {d}')
    return d

def user_to_expert(prefix, cfg, full=False):
    global group
    global ocfg

    d = {}
    try:
#        lcls1Delay     = ctxt_get(prefix+'TriggerEventManager:EvrV2CoreTriggers.EvrV2TriggerReg[0]:Delay')
        lcls1Delay = 0.9e-3*119e6
        partitionDelay = ctxt_get(prefix+'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]'%group)
        delta          = cfg['user']['delta_ns']
        triggerDelay   = int(lcls1Delay*1300/(7*119) + delta*1300/7000 - partitionDelay*200)
        print('lcls1Delay {:}  partitionDelay {:}  delta_ns {:}  triggerDelay {:}'.format(lcls1Delay,partitionDelay,delta,triggerDelay))
        if triggerDelay < 0:
            raise ValueError('triggerDelay computes to < 0')
        
        ctxt_put(prefix+'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay', triggerDelay)

    except KeyError:
        pass

#    try:
#        prescale = cfg['user']['raw_prescale'] 
#        # Firmware needs a value one less
#        if prescale>0:  
#           prescale -= 1
#        d['expert.RawBuffers.TrigPrescale'] = prescale
#    except KeyError:
#        pass


def wave8_config(prefix,connect_str,cfgtype,detname,detsegm,grp):
    global lane
    global group
    global ocfg

    group = grp

    #  Read the configdb
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #  Apply the user configs
    epics_prefix = prefix + ':Top:'
    user_to_expert(epics_prefix, cfg, full=True) 

    #  Assert clears
    names_clr = [epics_prefix+'BatcherEventBuilder:Blowoff',
#                 epics_prefix+'TimingFrameRx:RxCountReset',
                 epics_prefix+'RawBuffers:CntRst',
                 epics_prefix+'Integrators:CntRst']
    values = [1]*len(names_clr)
    ctxt_put(names_clr,values)

    names_cfg = [epics_prefix+'TriggerEventManager:TriggerEventBuffer[0]:Partition',
                 epics_prefix+'TriggerEventManager:TriggerEventBuffer[0]:PauseThreshold',
                 epics_prefix+'TriggerEventManager:TriggerEventBuffer[0]:MasterEnable']
    values = [group,16,1]
    ctxt_put(names_cfg, values)

    time.sleep(0.2)

    #  Deassert clears
    values = [0]*len(names_clr)
    ctxt_put(names_clr,values)

    #
    #  Now construct the configuration we will record
    #
    top = cdict()
    top.setAlg('config', [2,0,0])
    detname = cfg['detName:RO'].rsplit('_',1)
    top.setInfo(detType='wave8', detName=detname[0], detSegm=int(detname[1]), detId=cfg['detId:RO'], doc='No comment')

    top.define_enum('baselineEnum', {'_%d_samples'%(2**key):key for key in range(1,8)})
    top.define_enum('quadrantEnum', {'Even':0, 'Odd':1})

    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    top.set("expert.SystemRegs.AvccEn0"         ,  1,'UINT8')
    top.set("expert.SystemRegs.AvccEn1"         ,  1,'UINT8')
    top.set("expert.SystemRegs.Ap5V5En"         ,  1,'UINT8')
    top.set("expert.SystemRegs.Ap5V0En"         ,  1,'UINT8')
    top.set("expert.SystemRegs.A0p3V3En"        ,  1,'UINT8')
    top.set("expert.SystemRegs.A1p3V3En"        ,  1,'UINT8')
    top.set("expert.SystemRegs.Ap1V8En"         ,  1,'UINT8')
    top.set("expert.SystemRegs.FpgaTmpCritLatch",  0,'UINT8')
    top.set("expert.SystemRegs.AdcCtrl1"        ,  0,'UINT8')
    top.set("expert.SystemRegs.AdcCtrl2"        ,  0,'UINT8')
    top.set("expert.SystemRegs.TrigEn"          ,  0,'UINT8')
    top.set("expert.SystemRegs.timingRxUserRst" ,  0,'UINT8')
    top.set("expert.SystemRegs.timingTxUserRst" ,  0,'UINT8')
    top.set("expert.SystemRegs.timingUseMiniTpg",  0,'UINT8')
    top.set("expert.SystemRegs.TrigSrcSel"      ,  1,'UINT8')

    top.set("expert.Integrators.TrigDelay"             ,    0,'UINT32')            # user config
    top.set("expert.Integrators.IntegralSize"          ,    0,'UINT32')            # user config
    top.set("expert.Integrators.BaselineSize"          ,    0,'UINT8')             # user config
    top.set("expert.Integrators.QuadrantSel"           ,    0,'quadrantEnum')      # user config
    top.set("expert.Integrators.CorrCoefficientFloat64", [1.0]*4, 'DOUBLE')  # user config
    top.set("expert.Integrators.ProcFifoPauseThreshold",  255,'UINT32')
    top.set("expert.Integrators.IntFifoPauseThreshold" ,  255,'UINT32')

    top.set("expert.RawBuffers.BuffEn"            ,[0]*8,'UINT32')  # user config
    top.set("expert.RawBuffers.BuffLen"           , 100,'UINT32')  # user config
    top.set("expert.RawBuffers.FifoPauseThreshold", 100,'UINT32')
    top.set("expert.RawBuffers.TrigPrescale"      , 0,'UINT32')    # user config

    top.set("expert.BatcherEventBuilder.Bypass" , 0,'UINT8')
    top.set("expert.BatcherEventBuilder.Timeout", 0,'UINT32')
    top.set("expert.BatcherEventBuilder.Blowoff", 0,'UINT8')

    top.set("expert.TriggerEventManager.TriggerEventBuffer.TriggerDelay"  , 0,'UINT32')  # user config

    dlyAlane = [ [ 0x0c,0x0b,0x0e,0x0e,0x10,0x10,0x12,0x0b ],
                 [ 0x0a,0x08,0x0c,0x0b,0x0d,0x0c,0x0b,0x0c ],
                 [ 0x12,0x13,0x13,0x13,0x13,0x13,0x13,0x13 ],
                 [ 0x0d,0x0c,0x0d,0x0b,0x0a,0x12,0x12,0x13 ] ]
    dlyBlane = [ [ 0x11,0x11,0x12,0x12,0x10,0x11,0x0b,0x0b ],
                 [ 0x0a,0x0a,0x0c,0x0c,0x0c,0x0b,0x0b,0x0a ],
                 [ 0x14,0x14,0x14,0x14,0x14,0x12,0x10,0x11 ],
                 [ 0x13,0x12,0x13,0x12,0x12,0x11,0x12,0x11 ] ]

    for iadc in range(4):
        base = 'AdcReadout%d'%iadc
        top.set('expert.'+base+'.DelayAdcALane', dlyAlane[iadc], 'UINT8')
        top.set('expert.'+base+'.DelayAdcBLane', dlyBlane[iadc], 'UINT8')
        top.set('expert.'+base+'.DMode'  , 3, 'UINT8')
        top.set('expert.'+base+'.Invert' , 0, 'UINT8')
        top.set('expert.'+base+'.Convert', 3, 'UINT8')

    for iadc in range(4):
        base = 'AdcConfig%d'%iadc
        zeroregs = [7,8,0xb,0xc,0xf,0x10,0x11,0x12,0x12,0x13,0x14,0x16,0x17,0x18,0x20]
        for r in zeroregs:
            top.set('expert.'+base+'.AdcReg_0x%04X'%r,    0, 'UINT8')
        top.set('expert.'+base+'.AdcReg_0x0006'  , 0x80, 'UINT8')
        top.set('expert.'+base+'.AdcReg_0x000D'  , 0x6c, 'UINT8')
        top.set('expert.'+base+'.AdcReg_0x0015'  ,    1, 'UINT8')
        top.set('expert.'+base+'.AdcReg_0x001F'  , 0xff, 'UINT8')

    top.set('expert.AdcPatternTester.Channel', 0, 'UINT8' )
    top.set('expert.AdcPatternTester.Mask'   , 0, 'UINT8' )
    top.set('expert.AdcPatternTester.Pattern', 0, 'UINT8' )
    top.set('expert.AdcPatternTester.Samples', 0, 'UINT32' )
    top.set('expert.AdcPatternTester.Request', 0, 'UINT8' )

    scfg = top.typed_json()

    #
    #  Retrieve full configuration for recording
    #
    d = epics_get(scfg['expert'])
    keys  = [key for key,v in d.items()]
    names = [epics_prefix+v for key,v in d.items()]
    values = ctxt_get(names)
    for i,v in enumerate(values):
        k = keys[i].split('.')
        c = scfg['expert']
        while len(k)>1:
            c = c[k[0]]
            del k[0]
        if k[0][0]=='[':
            elem = int(k[0][1:-1])
            c[elem] = v
        else:
            c[k[0]] = v

    scfg['firmwareVersion:RO'] = ctxt_get(prefix+':AxiVersion:FpgaVersion')
    #scfg['firmwareBuild:RO'  ] = ctxt_get(prefix+':AxiVersion:BuildStamp')

    pprint.pprint(scfg)
    v = json.dumps(scfg)
    return v

def wave8_scan_keys(update):
    global prefix
    global ocfg
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(prefix+':Top:',cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def wave8_update(update):
    global prefix
    global ocfg
    #  extract updates
    cfg = {}
    epics_prefix = prefix+':Top:'
    update_config_entry(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(epics_prefix, cfg, full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)


#  This is really shutdown/disconnect
def wave8_unconfig(epics_prefix):

    ctxt_put(epics_prefix+':Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable', 0)
    #config_timing(epics_prefix, lcls2=False)

    return None;
