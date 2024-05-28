from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.det_config import *
from psdaq.cas.xpm_utils import timTxId
#from .xpmmini import *
import pyrogue as pr
import rogue
import rogue.hardware.axi
import pyrogue.protocols

import epix_uhr_dev as epixUhrDev
import surf.protocols.batcher as batcher

import time
import json
import os
import numpy as np
import IPython
import datetime
import logging
import os 
from psdaq.configdb.Board import *
import json

#import epixUHR_ordering


import pprint

rogue.Version.minVersion('5.14.0')
base = None
pv = None
#lane = 0  # An element consumes all 4 lanes
chan = None
group = None
ocfg = None
segids = None
seglist = [0,1]
asics = None

#  Timing delay scans can be limited by this
EventBuilderTimeout = 4*int(1.0e-3*156.25e6)


def _dict_compare(d1,d2,path):
    for k in d1.keys():
        if k in d2.keys():
            if isinstance(d1[k],dict):
                _dict_compare(d1[k],d2[k],path+'.'+k)
            elif (d1[k] != d2[k]):
                print(f'key[{k}] d1[{d1[k]}] != d2[{d2[k]}]')
        else:
            print(f'key[{k}] not in d1')
    for k in d2.keys():
        if k not in d1.keys():
            print(f'key[{k}] not in d2')

def gain_mode_map(gain_mode):
    mapv  = (0xc,0x8,0x0,0x0)[gain_mode] # SH/SL/A/M
    trbit = (0x1,0x0,0x1,0x0)[gain_mode]
    return (mapv,trbit)

#
#  Scramble the user element pixel array into the native asic orientation
#
#
#    A1   |   A3
# --------+--------
#    A0   |   A2
#
def user_to_rogue(a):
    v = a.reshape((elemRows*2,elemCols*2))
    s = np.zeros((4,elemRows,elemCols),dtype=np.uint8)
    s[0,:elemRows] = v[elemRows:,:elemCols]
    s[2,:elemRows] = v[elemRows:,elemCols:]
    s[1,:elemRows] = v[elemRows:,elemCols:]
    s[3,:elemRows] = v[elemRows:,:elemCols]
    return s

def rogue_to_user(s):
    v = np.zeros((elemRows*2,elemCols*2),dtype=np.uint8)
    v[elemRows:,:elemCols] = s[3,:elemRows]
    v[elemRows:,elemCols:] = s[1,:elemRows]
    v[elemRows:,elemCols:] = s[2,:elemRows]
    v[elemRows:,:elemCols] = s[0,:elemRows]
    return v.reshape(elemRows*elemCols*4)

#
#  Initialize the rogue accessor
#
def epixUHR_init(arg,dev='/dev/datadev_1',lanemask=0xf,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv

#    logging.getLogger().setLevel(40-10*verbosity) # way too much from rogue
    logging.getLogger().setLevel(30)
    logging.warning('epixUHR_init')

    base = {}
    #  Connect to the camera and the PCIe card
    ###assert(lanemask.bit_length() == 4)
    cbase = Board(
        dev         = dev,
        emuMode     = True,
        pollEn      = False,
        initRead    = False,
        linkRate    = 512,
        mhzMode     = False,
        numClusters = 14,
    )
    cbase.__enter__()
    base['cam'] = cbase

        
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()
    buildDate       = cbase.Core.AxiVersion.BuildDate.get()
    gitHashShort    = cbase.Core.AxiVersion.GitHashShort.get()
    print(f'firmwareVersion [{firmwareVersion:x}]')
    print(f'buildDate       [{buildDate}]')
    print(f'gitHashShort    [{gitHashShort}]')

    # Ric: These don't exist, so need to find equivalents
    ##  Enable the environmental monitoring
    #cbase.App.SlowAdcRegisters.enable.set(1)
    #cbase.App.SlowAdcRegisters.StreamPeriod.set(100000000)  # 1Hz
    #cbase.App.SlowAdcRegisters.StreamEn.set(1)
    #cbase.App.SlowAdcRegisters.enable.set(0)

    # configure timing
    logging.warning(f'Using timebase {timebase}')
    if timebase=="119M":  # UED
        base['bypass'] = 0x0
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True
        

        logging.warning('epixUHR_unconfig')
        epixUHR_unconfig(base)

        cbase.App.TimingFrameRx.ModeSelEn.set(1) # UseModeSel
        cbase.App.TimingFrameRx.ClkSel.set(0)    # LCLS-1 Clock
        cbase.App.TimingFrameRx.RxDown.set(0)
    else:
        base['bypass'] = 0 
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        base['pcie_timing'] = False

        logging.warning('epixUHR_unconfig')
        epixUHR_unconfig(base)

        cbase.App.ConfigLclsTimingV2()

    # Delay long enough to ensure that timing configuration effects have completed
    cnt = 0
    while cnt < 10:
        time.sleep(1)
        rxId = cbase.App.TriggerEventManager.XpmMessageAligner.RxId.get()
        if rxId != 0xffffffff:  break
        del rxId                # Maybe this can help getting RxId reevaluated
        cnt += 1

    if cnt == 10:
        raise ValueError("rxId didn't become valid after configuring timing")

    # Ric: Not working yet
    ## configure internal ADC
    ##cbase.App.InitHSADC()
    base['cfg'] = None
    time.sleep(1)               # Still needed?
#    epixm320_internal_trigger(base)
    return base

#
#  Set the PGP lane
#
def epixUHR_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixUHR_connectionInfo(base, alloc_json_str):

#
#  To do:  get the IDs from the detector and not the timing link
#
    txId = timTxId('epixUHR')
    logging.info('TxId {:x}'.format(txId))

    cbase = base['cam']
    rxId = cbase.App.TriggerEventManager.XpmMessageAligner.RxId.get()
    logging.info('RxId {:x}'.format(rxId))
    cbase.App.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    epixUHRid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixUHRid

    return d

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the ocfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, full=False):
    global ocfg
    global group
    global lane

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        cbase = base['cam']
        
        rtp = cfg['user']['run_trigger_group']   # run trigger partition
        for i,p in enumerate([rtp,group]):
            partitionDelay = getattr(cbase.App.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%p).get()
            rawStart       = cfg['user']['start_ns']
            triggerDelay   = int(rawStart*base['clk_period'] - partitionDelay*base['msg_period'])
            logging.warning(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            if triggerDelay < 0:
                logging.error(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
                raise ValueError('triggerDelay computes to < 0')

            d[f'expert.App.TriggerEventManager.TriggerEventBuffer[{i}].TriggerDelay']=triggerDelay

        if full:
            d[f'expert.App.TriggerEventManager.TriggerEventBuffer[0].Partition']=rtp    # Run trigger
            print(f"########## GROUP VALUE {group}")
            d[f'expert.App.TriggerEventManager.TriggerEventBuffer[1].Partition']=group  # DAQ trigger

    pixel_map_changed = False
    a = None
    hasUser = 'user' in cfg
    #if hasUser and 'gain_mode' in cfg['user']:
    #    gain_mode = cfg['user']['gain_mode']
    #    if gain_mode==3:  # user map
    #        if 'pixel_map' in cfg['user']:
    #            a  = cfg['user']['pixel_map']
    #            logging.warning('pixel_map len {}'.format(len(a)))
    #            d['user.pixel_map'] = a
                # what about associated trbit?
    #    else:
    #        mapv, trbit = gain_mode_map(gain_mode)
            
    #        d[f'expert.App.Mv2Asic.trbit'] = trbit
    #    pixel_map_changed = True

#    update_config_entry(cfg,ocfg,d)

    return pixel_map_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writePixelMap=True, secondPass=False):
    global asics  # Need to maintain this across configuration updates

    #  Disable internal triggers during configuration
    epixUHR_external_trigger(base)

    cbase = base['cam']

    # overwrite the low-level configuration parameters with calculations from the user configuration
    if 'expert' in cfg:
        try:  # config update might not have this
            apply_dict('cbase.App.TriggerEventManager',cbase.App.TriggerEventManager,cfg['expert']['App']['TriggerEventManager'])
        except KeyError:
            pass

    app = None
    #modelist=['_35kHz', '_100kHz', '_1MHz', 'temp', 'MHzmode']
    #mode = cfg['expert']['App']['mode']
    
    if 'expert' in cfg and 'App' in cfg['expert']:
        app = cfg['expert']['App'].copy()

    base['bypass'] = 0#x3f  # Enable Timing (bit-0) and Data (bit-1)
    base['batchers'] = 0# cbase.numOfAsics * [1]  # list of active batchers
    
    
    cbase.App.BatcherEventBuilder0.Bypass.set(base['bypass'])
    
    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    
    cbase.App.BatcherEventBuilder0.Timeout.set(EventBuilderTimeout) # 400 us
    if not base['pcie_timing']:
        eventBuilder = cbase.find(typ=batcher.AxiStreamBatcherEventBuilder)
        for eb in eventBuilder:
            eb.Timeout.set(EventBuilderTimeout)
            eb.Blowoff.set(False)
    #
    #  For some unknown reason, performing this part of the configuration on BeginStep
    #  causes the readout to fail until the next Configure
    #
    # getattr(cbase.App.AsicTop, f'BatcherEventBuilder0').Bypass.set(base['bypass'])

            
    if app is not None and not secondPass:
        # Work hard to use the underlying rogue interface
        # Config data was initialized from the distribution's yaml files by epixhr_config_from_yaml.py
        # Translate config data to yaml files
        path = '/tmp/ePixUHR_'
        epixUHRTypes = cfg[':types:']['expert']['App']
        
        tmpfiles = []

        tree = ('Board','App')
        
        def toYaml(keys,name):
            #if sect == tree[-1]:
            tmpfiles.append(dictToYaml(app,epixUHRTypes,keys,cbase.App,path,name,tree,ordering))
            #else:
            #    tmpfiles.append(dictToYaml(app[sect],epixUHRTypes[sect],keys,cbase.App,path,name,(*tree,sect),ordering))
        ordering={}

        ordering['WaveformControl']       = cfg['expert']['App']['sorted']['RegisterControl'].split(',')
        ordering['TriggerRegisters']      = cfg['expert']['App']['sorted']['TriggerReg'].split(',')
        ordering['Asic']                  = cfg['expert']['App']['sorted']['SACIReg'].split(',')
        ordering['Framer']                = cfg['expert']['App']['sorted']['FramerReg'].split(',')
        ordering['General']               = ['LockOnIdleOnly','ByPass']
        
        
        toYaml(['WaveformControl'],'RegisterControl')
        toYaml(['TriggerRegisters'],'TriggerReg')
        toYaml(['Asic'],'SACIReg')
        toYaml(['Framer'],'FramerReg')
        toYaml(['SspMon','BatcherEventBuilder0'],'General')
        
        #print(json.dumps(cbase.App.__dict__, indent=4, sort_keys=True))
        
        cbase.App.fnInitAsicScript(None,None,None)

        #  Remove the yml files
        #for f in tmpfiles:
        #    os.remove(f)

        # run some triggers and exercise lanes and locks
        #frames = 5000
        #rate = 1000

        #cbase.hwTrigger(frames, rate)

        #get locked lanes
        #print('Locked lanes:')
        #cbase.getLaneLocks()

        # Disable non-locking lanes
        
        #lanes = cbase.App.SspMon.Locked.get() ^ 0xffffff;
        #print(f'Setting DigAsicStrmRegisters.DisableLane to 0x{lanes:x}')
        #cbase.App.DigAsicStrmRegisters0.DisableLane.set(lanes);

        # Enable the batchers
        
        cbase.App.BatcherEventBuilder0.enable.set(base['batchers'] == 1)

    # Ric: I think here is where the Calibration registers should conditionally be set
    # The writePixelMap variable provided the condition for the 2x2
    
    logging.info('config_expert complete')

def reset_counters(base):
    # Reset the timing counters
    base['cam'].App.TimingFrameRx.countReset()

    # Reset the trigger counters
    base['cam'].App.TriggerEventManager.TriggerEventBuffer[1].countReset()

#
#  Called on Configure
#
def epixUHR_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids
    
    
    group = rog
    
    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    
    ocfg = cfg
    
    #  Translate user settings to the expert fields
    writePixelMap=user_to_expert(base, cfg, full=True)

    if cfg==base['cfg']:
        print('### Skipping redundant configure')
        return base['result']

    if base['cfg']:
        print('--- config changed ---')
        _dict_compare(base['cfg'],cfg,'cfg')
        print('--- /config changed ---')

    #  Apply the expert settings to the device
    _stop(base)
    
    config_expert(base, cfg, writePixelMap)

    time.sleep(0.01)
    print("#### start config before ###")
    _start(base)
    print("#### start config after ###")
    #  Add some counter resets here
    reset_counters(base)

    #  Enable triggers to continue monitoring
    # epixUHR_internal_trigger(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()

    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    #trbit = cfg['expert']['App']['Mv2Asic']['trbit'] 

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    #for seg in range(1):
        #  Construct the ID
    digitalId = [0, 0]# if base['pcie_timing'] else cbase.App.RegisterControlDualClock.DigIDLow.get(),
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.DigIDHigh.get()]
    pwrCommId = [0, 0]# if base['pcie_timing'] else cbase.App.RegisterControlDualClock.PowerAndCommIDLow.get(),
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.PowerAndCommIDHigh.get()]
    carrierId = [0, 0]# if base['pcie_timing'] else cbase.App.RegisterControlDualClock.CarrierIDLow.get(),
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.CarrierIDHigh.get()]
    id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                      carrierId[0], carrierId[1],
                                                      digitalId[0], digitalId[1],
                                                      pwrCommId[0], pwrCommId[1])
    
    segids[0] = id
    top = cdict()
    top.setAlg('config', [0,0,0])
    top.setInfo(detType='epixUHR', detName='_'.join(topname[:-1]), detSegm=int(topname[-1]), detId=id, doc='No comment')
    #top.set('asicPixelConfig', pixelConfigUsr)
    #top.set('trbit'          , trbit, 'UINT8')
    scfg[1] = top.typed_json()

    # Sanitize the json for json2xtc by removing offensive characters
    def translate_config(src):
        dst = {}
        for k, v in src.items():
            if isinstance(v, dict):
                v = translate_config(v)
            dst[k.replace('[','').replace(']','').replace('(','').replace(')','')] = v
        return dst

    result = []

    result.append( json.dumps(translate_config(scfg[0])) )
    result.append( json.dumps(translate_config(scfg[1])) )

    #base['cfg']    = copy.deepcopy(cfg)
    #base['result'] = copy.deepcopy(result)
    
    return result

def epixUHR_unconfig(base):
    print('epixUHR_unconfig')
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixUHR_scan_keys(update):
    logging.warning('epixUHR_scan_keys')
    global ocfg
    global base
    global segids

    cfg = {}
    copy_reconfig_keys(cfg,ocfg,json.loads(update))
    # Apply to expert
    pixelMapChanged = user_to_expert(base,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]
    
    #if pixelMapChanged:
    #    gain_mode = cfg['user']['gain_mode']
    #    if gain_mode==3:
    #        pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8).reshape(2*elemRowsD,2*elemCols)
    #    else:
    #        mapv,trbit = gain_mode_map(gain_mode)
    #        pixelConfigUsr = np.zeros((2*elemRowsD,2*elemCols),dtype=np.uint8)+mapv

    #    cbase = base['cam']
    #    pixelConfigMap = user_to_rogue(pixelConfigUsr)
        #trbit = cfg['expert']['App'][f'Mv2Asic']['trbit']

        #for seg in range(1):
        #    id = segids[seg]
        ##    top = cdict()
        #    top.setAlg('config', [0,0,0])
        #    top.setInfo(detType='epixUHR', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
        #    top.set('asicPixelConfig', pixelConfigUsr)
        #    top.set('trbit'          , trbit                  , 'UINT8')
        #    scfg[seg+1] = top.typed_json()

    result = []
    #for i in range(len(scfg)):
    result.append( json.dumps(scfg[0]) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixUHR_update(update):
    logging.warning('epixUHR_update')
    global ocfg
    global base

    #  Queue full configuration next Configure transition
    base['cfg'] = None

    _stop(base)
    ##
    ##  Having problems with partial configuration
    ##
    # extract updates
    cfg = {}
    update_config_entry(cfg,ocfg,json.loads(update))
    #  Apply to expert
    writePixelMap = user_to_expert(base,cfg,full=False)
    print(f'Partial config writePixelMap {writePixelMap}')
    if True:
        #  Apply config
        config_expert(base, cfg, writePixelMap, secondPass=True)
    else:
        ##
        ##  Try full configuration
        ##
        ncfg = ocfg.copy()
        update_config_entry(ncfg,ocfg,json.loads(update))
        _writePixelMap = user_to_expert(base,ncfg,full=True)
        print(f'Full config writePixelMap {_writePixelMap}')
        config_expert(base, ncfg, _writePixelMap, secondPass=True)

    _start(base)

    #  Enable triggers to continue monitoring
#    epixUHR_internal_trigger(base)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    #scfg[0] = cfg

    #if writePixelMap:
        #gain_mode = cfg['user']['gain_mode']
        #if gain_mode==3:
        #    pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8).reshape(2*elemRowsD,2*elemCols)
        #else:
        #    mapv,trbit = gain_mode_map(gain_mode)
        #    pixelConfigUsr = np.zeros((2*elemRowsD,2*elemCols),dtype=np.uint8)+mapv

        #pixelConfigMap = user_to_rogue(pixelConfigUsr)
       # try:
       #     trbit = cfg['expert']['App'][f'Mv2Asic']['trbit'] 
       # except:
       #     trbit = None

       # cbase = base['cam']
       # for seg in range(1):
       #     id = segids[seg]
       #     top = cdict()
       #     top.setAlg('config', [0,0,0])
       #     top.setInfo(detType='epixUHR', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
       #     top.set('asicPixelConfig', pixelConfigUsr)
       #     if trbit is not None:
       #         top.set('trbit'      , trbit                  , 'UINT8')
       #     scfg[seg+1] = top.typed_json()

    result = []
    #for i in range(len(scfg)):
     #   result.append( json.dumps(scfg[i]) )

    #logging.warning('update complete')

    return result

def _resetSequenceCount():
    cbase = base['cam']
    cbase.App.RegisterControlDualClock.ResetCounters.set(1)
    time.sleep(1.e6)
    cbase.App.RegisterControlDualClock.ResetCounters.set(0)

def epixUHR_external_trigger(base):
    #  Switch to external triggering
    
    cbase = base['cam']
    cbase.App.TriggerRegisters.SetTimingTrigger(1)

def epixUHR_internal_trigger(base):
    #  Disable frame readout
    mask = 0x3
    print('=== internal triggering with bypass {:x} ==='.format(mask))
    cbase = base['cam']
    cbase.App.BatcherEventBuilder0.Bypass.set(mask)
    return

    #  Switch to internal triggering
    print('=== internal triggering ===')
    cbase = base['cam']
    cbase.App.TriggerRegisters.SetAutoTrigger(1)

def epixUHR_enable(base):
    print('epixUHR_enable')
    epixUHR_external_trigger(base)
    _start(base)

def epixUHR_disable(base):
    print('epixUHR_disable')
    # Prevents transitions going through: epixUHR_internal_trigger(base)

def _stop(base):
    print('_stop')
    cbase = base['cam']
    cbase.App.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    print('_start')
    cbase = base['cam']
    cbase.App.TriggerRegisters.SetTimingTrigger()
    cbase.App.StartRun()
    
    cbase.App.BatcherEventBuilder0.Bypass.set(0)
    cbase.App.BatcherEventBuilder0.Blowoff.set(0)

#
#  Test standalone
#
if __name__ == "__main__":

    _base = epixUHR_init(None,dev='/dev/datadev_1')
    epixUHR_init_feb()
    epixUHR_connectionInfo(_base, None)

    db = 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/configDB'
    d = {'body':{'control':{'0':{'control_info':{'instrument':'tst', 'cfg_dbase' :db}}}}}

    print('***** CONFIG *****')
    _connect_str = json.dumps(d)
    epixUHR_config(_base,_connect_str,'BEAM','epixUHR',0,4)
    
    print('***** SCAN_KEYS *****')
    epixUHR_scan_keys(json.dumps(["user.gain_mode"]))
    #
    for i in range(100):
        print(f'***** UPDATE {i} *****')
        epixUHR_update(json.dumps({'user.gain_mode':i%5}))
    
    
    print('***** DONE *****')

