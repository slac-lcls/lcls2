from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import *
from psdaq.configdb.wave8_common import (
    ctxt_get, ctxt_put, confirm_xpm_rxid, config_timing,
    retrieve_config_from_epics,
    set_system_regs, set_raw_buffers, set_batcher_event_builder,
    set_trigger_event_manager, set_adc_readout, set_adc_config,
    set_adc_pattern_tester, set_firmware_info, define_common_enums,
)
from psdaq.cas.xpm_utils import timTxId

from psdaq.utils import enable_lcls2_pgp_pcie_apps
from psdaq.cas.pgpmonitor import PgpMonitor

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

#  Create a dictionary of config key to PV name
def epics_get(d):
    # translate legal Python names to Rogue names
    rogue_translate = {'TriggerEventBuffer':'TriggerEventBuffer[0]',
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

def wave8_init(epics_prefix, dev='/dev/datadev_0', lanemask=1, xpmpv=None, timebase="186M", verbosity=0):
    global prefix
    global lane
    global base

    logging.getLogger().setLevel(40-10*verbosity)
    prefix = epics_prefix
    base['prefix'] = epics_prefix
    base['timebase'] = timebase
    lm=lanemask
    lane = (lm&-lm).bit_length()-1
    assert(lm==(1<<lane)) # check that lanemask only has 1 bit for wave8

    print(f'--- lanemask {lanemask:x}  lane {lane}  timebase {timebase} ---')

    pcie_card = PgpMonitor(pollEn=False,
                           initRead=False,
                           dev=dev,
                           lanemask=lanemask,
                           numVc=2)
    pcie_card.__enter__()
    pcie_card.init_lanes()

    base['pcie'] = pcie_card
    
    wave8_unconfig(base)

    return base

def wave8_init_feb(slane=None,schan=None):
    global lane
    if slane is not None:
        lane = int(slane)

def wave8_connectionInfo(base, alloc_json_str):

    base['pcie'].check_lanes('connect')

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

    # Retrieve the XPM connection information from EPICS
    # to verify a direct connection (not through a fanout)
#    confirm_xpm_rxid( txId, values, alloc_json_str)

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
        ctrlDelay      = ctxt_get(prefix + 'TriggerEventManager:EvrV2CoreTriggers:EvrV2TriggerReg[0]:Delay')
        if ctrlDelay is None:
            print("Warning: Failed to retrieve control trigger delay.  Using partition delay as fallback.")
            ctrlDelay      = ctxt_get(prefix + 'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay')
            delayFlag = False
        else:
            delayFlag = True
        partitionDelay = ctxt_get(prefix + 'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]' % group)

        clksPerFid = 200 if timebase=='186M' else 238
        nsPerClk   = 7000/1300. if timebase=='186M' else 1000/119.

        if delayFlag:
            #  LCLS2 timing. Let controls set the delay value.
            print('ctrlDelay {:}  partitionDelay {:}'.format(ctrlDelay, partitionDelay))

            # Since controls now also runs off the LCLS2 timing fiber, there
            # is no reason to have a "delta". This was put in place to
            # compensate for different LCLS1/LCLS2 timing fiber lengths
            # when controls used the LCLS1 timing fiber = cpo 02/01/24
            # triggerDelay = int(ctrlDelay + delta*1300/7000 - partitionDelay*200)
            triggerDelay = int(ctrlDelay - partitionDelay * clksPerFid)

            print('triggerDelay {:}'.format(triggerDelay))
            if triggerDelay < 0:
                print('Raise controls trigger delay >= {:} nanoseconds ({:} clock ticks)'.format(
                    -triggerDelay * nsPerClk, -triggerDelay))
                raise ValueError('triggerDelay computes to < 0')

            ctxt_put(prefix + 'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay', triggerDelay)

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

    base['pcie'].check_lanes('config')

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

    define_common_enums(top)
    set_firmware_info(top)
    set_system_regs(top)

    # Wave8-specific: Integrators configuration
    top.set("expert.Integrators.TrigDelay"             ,    0,'UINT32')            # user config
    top.set("expert.Integrators.IntegralSize"          ,    0,'UINT32')            # user config
    top.set("expert.Integrators.BaselineSize"          ,    0,'UINT8')             # user config
    top.set("expert.Integrators.QuadrantSel"           ,    0,'quadrantEnum')      # user config
    top.set("expert.Integrators.CorrCoefficientFloat64", [1.0]*4, 'DOUBLE')  # user config
    top.set("expert.Integrators.ProcFifoPauseThreshold",  255,'UINT32')
    top.set("expert.Integrators.IntFifoPauseThreshold" ,  255,'UINT32')

    set_raw_buffers(top)
    set_batcher_event_builder(top)
    set_trigger_event_manager(top)
    set_adc_readout(top)
    set_adc_config(top)
    set_adc_pattern_tester(top)

    scfg = top.typed_json()

    #  Retrieve full configuration for recording
    retrieve_config_from_epics(epics_prefix, scfg, epics_get)
    version = ctxt_get(prefix+':Top:AxiVersion:FpgaVersion')
    scfg['firmwareVersion:RO'] = version if version else scfg['firmwareVersion:RO']

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

    base['pcie'].check_lanes('unconfig')

    epics_prefix = base['prefix']
    # cpo removed setting Partition=1 (aka readout group) here
    # because this is called in init() and writes fail before the timing
    # the timing system is initialized.  Then subsequent writes start
    # silently failing as well resulting in lost configure phase2.
    names_cfg = [epics_prefix+':Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable',
                 epics_prefix+':Top:DataPathCtrl:EnableStream',] # 0x1 for Controls, 0x2 for DAQ
    values = [0,0]
    ctxt_put(names_cfg, values)

    #  Leaving DAQ control.
    config_timing(epics_prefix)

    return None;
