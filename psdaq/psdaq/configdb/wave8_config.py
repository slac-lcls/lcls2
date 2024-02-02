from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import *
from psdaq.cas.xpm_utils import timTxId
import lcls2_pgp_pcie_apps
import epics

import json
import time
import pprint
import logging

prefix = None
ocfg   = None
group  = None
lane   = 0
timebase = '186M'
base = {'timebase':'186M', 'prefix':None, lane:0}

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

    r = []
    print(f'ctxt_put [{names}] [{values}]')
    if isinstance(names,str):
        r.append(epics.PV(names).put(values))
    else:
        if isinstance(names,list):
            for i,n in enumerate(names):
                r.append(epics.PV(n).put(values[i]))
    print(f'returned {r}')

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

def config_timing(epics_prefix, timebase='186M'):
    # cpo found on 01/30/23 that when we toggle between LCLS1/LCLS2 timing
    # using ModeSel that we generate junk into the KCU giving these errors:
    # PGPReader: Jump in complete l1Count 0 -> 2 | difference 2, tid ClearReadout
    # We used to go to LCLS1 timing on disconnect (which called
    # this routine) to be friendly to the controls group.  Since we're now
    # in the LCLS2 era, keep this always hardwired to ModeSel=1 (i.e. LCLS2 timing).
    names = [epics_prefix+':Top:SystemRegs:timingUseMiniTpg',
             epics_prefix+':Top:TimingFrameRx:ModeSelEn',
             epics_prefix+':Top:TimingFrameRx:ModeSel',
             epics_prefix+':Top:TimingFrameRx:ClkSel',
             epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0, 1, 1, 1 if timebase=='186M' else 0, 1]
    ctxt_put(names,values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0]
    ctxt_put(names,values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxDown',
             epics_prefix+':Timing:TriggerSource']  # 0=XPM/DAQ, 1=EVR
    values = [0,0]
    ctxt_put(names,values)

def wave8_init(epics_prefix, dev='/dev/datadev_0', lanemask=1, xpmpv=None, timebase="186M", verbosity=0):
    global prefix
    global lane
    logging.getLogger().setLevel(40-10*verbosity)
    prefix = epics_prefix
    base['prefix'] = epics_prefix
    base['timebase'] = timebase
    lm=lanemask
    lane = (lm&-lm).bit_length()-1
    assert(lm==(1<<lane)) # check that lanemask only has 1 bit for wave8

    print(f'--- lanemask {lanemask:x}  lane {lane}  timebase {timebase} ---')

    if timebase=="119M":  # UED
        print('Configure for UED')
        #  We need pcie control to configure event batcher (which normally doesnt exist)
        #  Concerned if we will interfere with other pcie operation (for epixquad)
        pbase = lcls2_pgp_pcie_apps.DevRoot(dev           =dev,
                                            enLclsI       =False,
                                            enLclsII      =True,
                                            yamlFileLclsI =None,
                                            yamlFileLclsII=None,
                                            startupMode   =True,
                                            standAloneMode=False,
                                            pgp4          =True,
                                            dataVc        =0,
                                            pollEn        =False,
                                            initRead      =False)
        pbase.__enter__()
        base['pci'] = pbase
        eventBuilder = getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').EventBuilder
        eventBuilder.Blowoff.set(True)

    wave8_unconfig(base)

    return base

def wave8_init_feb(slane=None,schan=None):
    global lane
    if slane is not None:
        lane = int(slane)

def wave8_connectionInfo(base, alloc_json_str):
    epics_prefix = base['prefix']

    #  Switch to LCLS2 Timing
    #    Need this to properly receive RxId
    #    Controls is no longer in-control
    config_timing(epics_prefix,timebase=base['timebase'])

    #  This fails with the current IOC, but hopefully it will be fixed.  It works directly via pgp.
    txId = timTxId('wave8')
    ctxt_put(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:TxId', txId)
    ctxt_put(epics_prefix+':Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable',0)

    # Retrieve connection information from EPICS
    # May need to wait for other processes here, so poll
    for i in range(50):
        values = int(ctxt_get(epics_prefix+':Top:TriggerEventManager:XpmMessageAligner:RxId'))
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
    global timebase

    d = {}
    try:
        lcls1Delay     = 0.9e-3*119e6
        ctrlDelay      = ctxt_get(prefix+'TriggerEventManager:EvrV2CoreTriggers:EvrV2TriggerReg[0]:Delay')
        partitionDelay = ctxt_get(prefix+'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]'%group)
        delta          = cfg['user']['delta_ns']
        #  This is not so good; using timebase to distinguish LCLS from UED
        if timebase=='186M':
            if False:
                print('lcls1Delay {:}  partitionDelay {:}  delta_ns {:}'.format(lcls1Delay,partitionDelay,delta))
                triggerDelay   = int(lcls1Delay*1300/(7*119) + delta*1300/7000 - partitionDelay*200)
            else:
                #  LCLS2 timing.  Let controls set the delay value.
                print('ctrlDelay {:}  partitionDelay {:}  delta_ns {:}'.format(ctrlDelay,partitionDelay,delta))
                # since controls now also runs off the LCLS2 timing fiber there
                # is not reason to have a "delta".  This was put in place to
                # compensate for different lcls1/lcls2 timing fiber lengths
                # when controls used the lcls1 timing fiber = cpo 02/01/24
                #triggerDelay   = int(ctrlDelay + delta*1300/7000 - partitionDelay*200)
                triggerDelay   = int(ctrlDelay - partitionDelay*200)

            print('triggerDelay {:}'.format(triggerDelay))
            if triggerDelay < 0:
                print('Raise controls trigger delay >= {:} nanoseconds ({:} 185MHz clock ticks)'.format(-triggerDelay*7000/1300.,-triggerDelay))
                raise ValueError('triggerDelay computes to < 0')

            ctxt_put(prefix+'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay', triggerDelay)
        else:
            #  119M = UED, 238 clks per timing frame (500kHz)
            #  UED is only LCLS2 timing.  Let controls set the delay value.
            pass

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


def wave8_config(base,connect_str,cfgtype,detname,detsegm,grp):
    global lane
    global group
    global ocfg
    global timebase

    print(f'base [{base}]')
    prefix = base['prefix']
    timebase = base['timebase']
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
                 epics_prefix+'TriggerEventManager:TriggerEventBuffer[0]:MasterEnable',
                 epics_prefix+'DataPathCtrl:EnableStream', # 0x1 for Controls, 0x2 for DAQ
                 epics_prefix+'RawBuffers:FifoPauseThreshold',
                 epics_prefix+'Integrators:ProcFifoPauseThreshold',
                 epics_prefix+'Integrators:IntFifoPauseThreshold']
    values = [group,16,1,0x2,127,127,127]
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
        adc = 'AdcReadout%d'%iadc
        top.set('expert.'+adc+'.DelayAdcALane', dlyAlane[iadc], 'UINT8')
        top.set('expert.'+adc+'.DelayAdcBLane', dlyBlane[iadc], 'UINT8')
        top.set('expert.'+adc+'.DMode'  , 3, 'UINT8')
        top.set('expert.'+adc+'.Invert' , 0, 'UINT8')
        top.set('expert.'+adc+'.Convert', 3, 'UINT8')

    for iadc in range(4):
        adc = 'AdcConfig%d'%iadc
        zeroregs = [7,8,0xb,0xc,0xf,0x10,0x11,0x12,0x12,0x13,0x14,0x16,0x17,0x18,0x20]
        for r in zeroregs:
            top.set('expert.'+adc+'.AdcReg_0x%04X'%r,    0, 'UINT8')
        top.set('expert.'+adc+'.AdcReg_0x0006'  , 0x80, 'UINT8')
        top.set('expert.'+adc+'.AdcReg_0x000D'  , 0x6c, 'UINT8')
        top.set('expert.'+adc+'.AdcReg_0x0015'  ,    1, 'UINT8')
        top.set('expert.'+adc+'.AdcReg_0x001F'  , 0xff, 'UINT8')

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

    if 'pci' in base:
        #  Note that other segment levels can step on EventBuilder settings (Bypass,VcDataTap)
        pbase = base['pci']
        getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').VcDataTap.Tap.set(1)
        eventBuilder = getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').EventBuilder
        eventBuilder.Bypass.set(5)
        eventBuilder.Blowoff.set(False)
        eventBuilder.SoftRst()

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
def wave8_unconfig(base):

    epics_prefix = base['prefix']
    # cpo removed setting Partition=1 (aka readout group) here
    # because this is called in init() and writes fail before the timing
    # the timing system is initialized.  Then subsequent writes start
    # silently failing as well resulting in lost configure phase2.
    names_cfg = [epics_prefix+':Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable']
    values = [0]
    ctxt_put(names_cfg, values)

    #  Leaving DAQ control.
    if base['timebase']=='186M':
        config_timing(epics_prefix)

    return None;
