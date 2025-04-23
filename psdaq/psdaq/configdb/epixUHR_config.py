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

from psdaq.utils import enable_epix_uhr_gtreadout_dev
import epix_uhr_gtreadout_dev as epixUhrDev
import surf.protocols.batcher as batcher

import time
import json
import os
import numpy as np
import IPython
import datetime
import logging
import copy # deepcopy
#import psdaq.configdb.EpixUHRBoard as EpixUHRBoard
import functools

rogue.Version.minVersion('6.1.0')

base = None
pv = None
#lane = 0  # An element consumes all 4 lanes
chan = None
group = None
origcfg = None

segids = None
seglist = [0,1]
asics = None

nColumns = 384

#  Timing delay scans can be limited by this
EventBuilderTimeout = 0 #4*int(1.0e-3*156.25e6)
def sorting_dict(asics):
    sortdict={}
        
    for n in asics:
        sortdict[f'Asic{n}'] = ["enable",
                                "DacVthr", 
                                "DacVthrGain", 
                                "DacVfiltGain", 
                                "DacVfilt", 
                                "DacVrefCdsGain", 
                                "DacVrefCds", 
                                "DacVprechGain", 
                                "DacVprech", 
                                "CompEnGenEn", 
                                "CompEnGenCfg",  
                                ]
        
        sortdict[f'BatcherEventBuilder{n}']= [  "enable", 
                                                "Timeout", 
                                                ]	

    sortdict['WaveformControl'] = [    "enable",
                                       "GlblRstPolarity", 
                                       "SR0Polarity", 
                                       "SR0Delay", 
                                       "SR0Width",
                                       "AcqPolarity", 
                                       "AcqDelay", 
                                       "AcqWidth", 
                                       "R0Polarity", 
                                       "R0Delay", 
                                       "R0Width",
                                       "InjPolarity", 
                                       "InjDelay", 
                                       "InjWidth", 
                                       "InjEn", 
                                       "InjSkipFrames"]
    
    sortdict['TriggerRegisters'] = [    "enable",
                                        "RunTriggerEnable",
                                        "RunTriggerDelay", 
                                        "DaqTriggerEnable", 
                                        "DaqTriggerDelay",
                                        "TimingRunTriggerEnable",
                                        "TimingDaqTriggerEnable", 
                                        "AutoRunEn", 
                                        "AutoDaqEn", 
                                        "AutoTrigPeriod", 
                                        "numberTrigger", 
                                        "PgpTrigEn"] 
    return sortdict
    
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

# Sanitize the json for json2xtc by removing offensive characters
def sanitize_config(src):
    dst = {}
    for k, v in src.items():
        if isinstance(v, dict):
            v = sanitize_config(v)
        dst[k.replace('[','').replace(']','').replace('(','').replace(')','')] = v
    return dst

def setSaci(reg,field,di):
    if field in di:
        v = di[field]
        reg.set(v)

def gain_mode_map(gain_mode):
    compTH        = ( 0, 44, 24)[gain_mode] # SoftHigh/SoftLow/Auto
    precharge_DAC = (45, 45, 45)[gain_mode]
    return (compTH, precharge_DAC)

def cbase_ASIC_init(cbase, asics):
    for asic in asics:
        
        wdet(getattr(cbase.App,f'Asic{asic}').enable,                True)			  	
        wdet(getattr(cbase.App,f'Asic{asic}').TpsDacGain,            1)						
        wdet(getattr(cbase.App,f'Asic{asic}').TpsDac,                34)						
        wdet(getattr(cbase.App,f'Asic{asic}').TpsGr,                 12)						
        wdet(getattr(cbase.App,f'Asic{asic}').TpsMux,                0)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasTpsBuffer,         5)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasTps,               4)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasTpsDac,            4)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVthr,               52)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasDac,               4)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrCtrlDacTps,         3)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrCtrlDacComp,        0)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVthrGain,           2)						
        wdet(getattr(cbase.App,f'Asic{asic}').PpbitBe,               1)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasPxlCsa,            0)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasPxlBuf,            0)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasAdcComp,           0)						
        wdet(getattr(cbase.App,f'Asic{asic}').BiasAdcRef,            0)						
        wdet(getattr(cbase.App,f'Asic{asic}').CmlRxBias,             3)						
        wdet(getattr(cbase.App,f'Asic{asic}').CmlTxBias,             3)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVfiltGain,          2)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVfilt,              28)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVrefCdsGain,        2)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVrefCds,            44)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVprechGain,         2)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacVprech,             34)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrCtrlDacFilt,        2)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrCtrlDacAdcRef,      2)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrCtrlDacPrechCds,    2)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrfCtrlDacAll,        2)						
        wdet(getattr(cbase.App,f'Asic{asic}').BgrDisable,            0)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefpGain,       3)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefp,           53)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefnGain,       0)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefn,           12)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefCmGain,      1)						
        wdet(getattr(cbase.App,f'Asic{asic}').DacAdcVrefCm,          45)						
        wdet(getattr(cbase.App,f'Asic{asic}').AdcCalibEn,            0)						
        wdet(getattr(cbase.App,f'Asic{asic}').CompEnGenEn,           1)						
        wdet(getattr(cbase.App,f'Asic{asic}').CompEnGenCfg,          5)						
        wdet(getattr(cbase.App,f'Asic{asic}').CfgAutoflush,          0)						
        wdet(getattr(cbase.App,f'Asic{asic}').ExternalFlushN,        1)						
        wdet(getattr(cbase.App,f'Asic{asic}').ClusterDvMask,         16383)					
        wdet(getattr(cbase.App,f'Asic{asic}').PixNumModeEn,          0)
        #PixNumModeEn, change this value to 1 to create a fixed pattern						
        wdet(getattr(cbase.App,f'Asic{asic}').SerializerTestEn,      0)						
        wdet(getattr(cbase.App,f'BatcherEventBuilder{asic}').enable, True)			  	
        wdet(getattr(cbase.App,f'BatcherEventBuilder{asic}').Bypass, 0)						
        wdet(getattr(cbase.App,f'BatcherEventBuilder{asic}').Timeout, 0)						
        wdet(getattr(cbase.App,f'BatcherEventBuilder{asic}').Blowoff, False)					
        wdet(getattr(cbase.App,f'FramerAsic{asic}').enable,          False)
        wdet(getattr(cbase.App,f'FramerAsic{asic}').DisableLane,     0)						
        wdet(getattr(cbase.App,f'AsicGtData{asic}').enable,          True)
        wdet(getattr(cbase.App,f'AsicGtData{asic}').gtStableRst,     False)

def cbase_init(cbase):
    wdet(cbase.App.WaveformControl.enable,               True)			  	
    wdet(cbase.App.WaveformControl.GlblRstPolarity,      True)		  	
    wdet(cbase.App.WaveformControl.SR0Polarity,          False)			  	
    wdet(cbase.App.WaveformControl.SR0Delay,             1195)	  	
    wdet(cbase.App.WaveformControl.SR0Width,             1)		  	
    wdet(cbase.App.WaveformControl.AcqPolarity,          False)			  	
    wdet(cbase.App.WaveformControl.AcqDelay,             655)	  	
    wdet(cbase.App.WaveformControl.AcqWidth,             535)	  	
    wdet(cbase.App.WaveformControl.R0Polarity,           False)			  	
    wdet(cbase.App.WaveformControl.R0Delay,              70)		  	
    wdet(cbase.App.WaveformControl.R0Width,              1125)		  	
    wdet(cbase.App.WaveformControl.InjPolarity,          False)		  	
    wdet(cbase.App.WaveformControl.InjDelay,             700)	  	
    wdet(cbase.App.WaveformControl.InjWidth,             535)	  	
    wdet(cbase.App.WaveformControl.InjEn,                False)		  	
    wdet(cbase.App.WaveformControl.InjSkipFrames,        0) 		
    wdet(cbase.App.TriggerRegisters.enable,              True)			  	
    wdet(cbase.App.TriggerRegisters.RunTriggerEnable,    False)					
    wdet(cbase.App.TriggerRegisters.RunTriggerDelay,     0)					
    wdet(cbase.App.TriggerRegisters.DaqTriggerEnable,    False)					
    wdet(cbase.App.TriggerRegisters.DaqTriggerDelay,     0)					
    wdet(cbase.App.TriggerRegisters.TimingRunTriggerEnable, False)				
    wdet(cbase.App.TriggerRegisters.TimingDaqTriggerEnable, False)			
    wdet(cbase.App.TriggerRegisters.AutoRunEn,           False)					
    wdet(cbase.App.TriggerRegisters.AutoDaqEn,           False)					
    wdet(cbase.App.TriggerRegisters.AutoTrigPeriod,      42700000)				
    wdet(cbase.App.TriggerRegisters.numberTrigger,       0)						
    wdet(cbase.App.TriggerRegisters.PgpTrigEn,           False)					
    wdet(cbase.App.GTReadoutBoardCtrl.enable,            True)
    wdet(cbase.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard, False)		 	
    wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn0,      False)
    wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn1,      False)
    wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn2,      False)
    wdet(cbase.App.AsicGtClk.enable,                     True)
    wdet(cbase.App.AsicGtClk.gtRstAll,                   False)					
    wdet(cbase.App.TimingRx.enable,                      True)
    wdet(cbase.Core.Si5345Pll.enable,                    False)
    wdet(cbase.App.VINJ_DAC.dacEn,                       False)
    wdet(cbase.App.VINJ_DAC.rampEn,                      False)
#
#  Initialize the rogue accessor
#
def epixUHR_init(arg,dev='/dev/datadev_0',lanemask=0xf,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv
    global config_completed
    global gainMapSelection
    global gainValSelection
    
    gainMapSelection=np.zeros((4, 168, 192))
    gainValSelection=np.zeros(4)
    
#    logging.getLogger().setLevel(40-10*verbosity) # way too much from rogue
    logging.getLogger().setLevel(30)
    logging.info('epixUHR_init')
    config_completed = False
    base = {}
    #  Connect to the camera and the PCIe card

    cbase = epixUhrDev.Root(
        dev          = dev,
        defaultFile  = ' ',
        emuMode      = False,
        pollEn       = True,
        initRead     = True,
        viewAsic     = 0,
        dataViewer   = False,
        numClusters  = 14,
        otherViewers = False,
        numOfAsics   = 4,
        timingMessage= False,
        justCtrl     = True,
        loadPllCsv   = False,
    )
    
    cbase.__enter__()
    
    base['cam'] = cbase

    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()
    buildDate       = cbase.Core.AxiVersion.BuildDate.get()
    gitHashShort    = cbase.Core.AxiVersion.GitHashShort.get()
    logging.info(f'firmwareVersion [{firmwareVersion:x}]')
    logging.info(f'buildDate       [{buildDate}]')
    logging.info(f'gitHashShort    [{gitHashShort}]')

    # configure timing
    logging.warning(f'Using timebase {timebase}')

    cbase_init(cbase)
    
    if timebase=="119M":  # UED
        base['bypass'] = cbase.numOfAsics * [0x3]
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True

        epixUHR_unconfig(base)
        
        wdet(cbase.App.TimingRx.TimingFrameRx.ModeSelEn, 1) # UseModeSel
        wdet(cbase.App.TimingRx.TimingFrameRx.ClkSel, 0)    # LCLS-1 Clock
        wdet(cbase.App.TimingRx.TimingFrameRx.RxDown, 0)
    else:
        base['bypass'] = cbase.numOfAsics * [0x3]
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        base['pcie_timing'] = False

        epixUHR_unconfig(base)

        cbase.App.TimingRx.ConfigLclsTimingV2()

    # Delay long enough to ensure that timing configuration effects have completed
    cnt = 0
    while cnt < 15:
        time.sleep(1)
        rxId = cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        if rxId != 0xffffffff:  break
        del rxId                # Maybe this can help getting RxId reevaluated
        cnt += 1

    if cnt == 15:
        raise ValueError("rxId didn't become valid after configuring timing")
    print(f"rxId {rxId:x} found after {cnt}s")

    # Ric: Not working yet
    ## configure internal ADC
    ##cbase.App.InitHSADC()

    #  store previously applied configuration
    base['cfg'] = None

    time.sleep(1)               # Still needed?
#    epixUHR_internal_trigger(base)
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
    rxId = cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    logging.info('RxId {:x}'.format(rxId))
    wdet(cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.TxId, txId)

    epixUHRid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixUHRid

    return d

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the origcfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, full=False):
    global origcfg
    global group
    global lane

    cbase = base['cam']
    
    deltadelay = -192
    
    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        #rtp = origcfg['user']['run_trigger_group'] # run trigger partition
        
        #for i,p in enumerate([rtp,group]):
        partitionDelay = getattr(cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']

        triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
        logging.warning(f'partitionDelay[{group}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
        if triggerDelay < 0:
            logging.error(f'partitionDelay[{group}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            logging.error('Raise start_ns >= {:}'.format(partitionDelay*base['msg_period']*base['clk_period']))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay']=triggerDelay
        
        triggerDelay=int(rawStart/base["clk_period"]) - deltadelay
        
        if triggerDelay < 0:
            logging.error(f'partitionDelay[{group+1}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            logging.error('Raise start_ns >= {:}'.format(partitionDelay*base['msg_period']*base['clk_period']))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].Delay'] = triggerDelay
        logging.warning(f'partitionDelay[{group+1}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')

        if full:
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition']= group+1    # Run trigger
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition']= group  # DAQ trigger
    
        a = None
    hasUser = 'user' in cfg
    conv = functools.partial(int, base=16)
    calibRegsChanged = False
    if hasUser and 'Gain'  in cfg['user']:
        calibRegsChanged = True

    update_config_entry(cfg,origcfg,d)

    return calibRegsChanged

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writeCalibRegs=True, secondPass=False):
    global asics  # Need to maintain this across configuration updates
    global gainMapSelection
    global gainValSelection
    
    path = '/tmp/ePixUHR_GTReadout_default_'
    pathPll = '/tmp/'

    #  Disable internal triggers during configuration
    epixUHR_external_trigger(base)

    cbase = base['cam']
    
    # overwrite the low-level configuration parameters with calculations from the user configuration
    if 'expert' in cfg:
        try:  # config update might not have this
            apply_dict('cbase.App.TimingRx.TriggerEventManager',
                       cbase.App.TimingRx.TriggerEventManager,
                       cfg['expert']['App']['TimingRx']['TriggerEventManager'])
        except KeyError:
            pass

    app = None
    if 'expert' in cfg and 'App' in cfg['expert']:
        app = copy.deepcopy(cfg['expert']['App'])

     #  Make list of enabled ASICs
    if 'user' in cfg and 'asic_enable' in cfg['user']:
        asics = []
        for i in range(cbase.numOfAsics):
            if cfg['user']['asic_enable']&(1<<i):
                asics.append(i+1)
    
    pll = cbase.Core.Si5345Pll
    
    tmpfiles = []
    if not secondPass:

        Pll_sel=[None, '_temp250', '_2_3_7', '_0_5_7', '_2_3_9', '_0_5_7_v2']
        if pll.enable.get() == False:
            pll.enable.set(True)
        
        clk = cfg['user']['PllRegistersSel'] 
        
        freq = Pll_sel[clk]
        logging.info(f"Loading PLL file: {freq}")
        
        pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
        fn = pathPll+'PllConfig'+'.csv'
        np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
        
        tmpfiles.append(fn)
        setattr(cbase,"filenamePLL", fn)
        
        pll.LoadCsvFile(pathPll+'PllConfig'+'.csv')    
        cbase_ASIC_init(cbase, asics)
               
    base['bypass']   = cbase.numOfAsics * [0x2]  # Enable Timing (bit-0) and Data (bit-1)
    base['batchers'] = cbase.numOfAsics * [1]  # list of active batchers
    
    for i in range(cbase.numOfAsics):
        if i+1 in asics: 
            base['bypass'][i] = 0
        
        wdet(getattr(cbase.App, f'BatcherEventBuilder{i+1}').Bypass, base['bypass'][i])
       
    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    
    for i in asics:
        wdet(getattr(cbase.App, f'BatcherEventBuilder{i}').Timeout, EventBuilderTimeout) # 400 us
    if not base['pcie_timing']:
        eventBuilder = cbase.find(typ=batcher.AxiStreamBatcherEventBuilder)
        for eb in eventBuilder:
            eb.Timeout.set(EventBuilderTimeout)
            eb.Blowoff.set(True)
    #
    #  For some unknown reason, performing this part of the configuration on BeginStep
    #  causes the readout to fail until the next Configure
    #
    
    if app is not None and not secondPass:
        # Work hard to use the underlying rogue interface
        # Config data was initialized from the distribution's yaml files by epixhr_config_from_yaml.py
        # Translate config data to yaml files
        
        epixMTypes = cfg[':types:']['expert']['App']
        tree = ('Root','App')
        
        def toYaml(sect,keys,name):
            
            tmpfiles.append(dictToYaml(app,epixMTypes,keys,cbase.App,path,name,tree,ordering))
            
        ordering=sorting_dict(asics)
        
        toYaml('App',['WaveformControl'],'RegisterControl')
        toYaml('App',['TriggerRegisters'],'TriggerReg')
        toYaml('App',[f'Asic{i}' for i in asics ],'SACIReg')
        toYaml('App',[f'BatcherEventBuilder{i}' for i in asics],'General')
        
        arg = [1,1,1,1,1]
        logging.info(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.App.fnInitAsicScript(None,None,arg)
        logging.info("### FINISHED YAML LOAD ###")

        # Enable the batchers for all ASICs
        for i in range(cbase.numOfAsics):
            wdet(getattr(cbase.App, f'BatcherEventBuilder{i+1}').enable, base['batchers'][i] == 1)
            
        wdet(cbase.App.GTReadoutBoardCtrl.enable, app['GTReadoutBoardCtrl']['enable']==1)
        wdet(cbase.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard, app['GTReadoutBoardCtrl']['pwrEnableAnalogBoard'])
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn0, app['GTReadoutBoardCtrl']['timingOutEn0']==1)
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn1, app['GTReadoutBoardCtrl']['timingOutEn1']==1)
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutEn2, app['GTReadoutBoardCtrl']['timingOutEn2']==1)
        
        timingOutEnum=['asicR0', 'asicACQ', 'asicSRO', 'asicInj', 'asicGlbRstN', 'timingRunTrigger', 'timingDaqTrigger', 'acqStart', 'dataSend', '_0', '_1']
        timingOutMux0_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux0'])
        timingOutMux1_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux1'])
        timingOutMux3_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux3'])
        logging.info(f'Setting timingOutMux0 to {timingOutEnum[timingOutMux0_Sel]}')
        logging.info(f'Setting timingOutMux1 to {timingOutEnum[timingOutMux1_Sel]}')
        logging.info(f'Setting timingOutMux3 to {timingOutEnum[timingOutMux3_Sel]}')
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutMux0, timingOutMux0_Sel)
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutMux1, timingOutMux1_Sel)
        wdet(cbase.App.GTReadoutBoardCtrl.timingOutMux3, timingOutMux3_Sel)
        wdet(cbase.App.AsicGtClk.enable, cfg['expert']['App']['AsicGtClk']['enable']==1)
        for i in asics:
            wdet(getattr(cbase.App,f"AsicGtData{i}").enable, cfg['expert']['App'][f'AsicGtData{i}']['enable']==1)			
            wdet(getattr(cbase.App,f"AsicGtData{i}").gtStableRst, cfg['expert']['App'][f'AsicGtData{i}']['gtStableRst']		)	
        if cbase.App.VCALIBP_DAC.enable.get() != cfg['user']['App']['VCALIBP_DAC']['enable']:
            wdet(cbase.App.VCALIBP_DAC.enable, cfg['user']['App']['VCALIBP_DAC']['enable']==1)	

        wdet(cbase.App.VCALIBP_DAC.dacSingleValue, cfg['user']['App']['VCALIBP_DAC']['dacSingleValue'])
        wdet(cbase.App.VCALIBP_DAC.resetDacRamp, cfg['user']['App']['VCALIBP_DAC']['resetDacRamp'])
               
        wdet(cbase.App.ADS1217.enable, cfg['user']['App']['ADS1217']['enable']==1)	
        wdet(cbase.App.ADS1217.adcStartEnManual, cfg['user']['App']['ADS1217']['adcStartEnManual']	)        
        
        
        
        for i in asics: 
            wdet(getattr(cbase.App,f"Asic{i}").PixNumModeEn, True)
            
        csvCfg = 0
    
    if writeCalibRegs:
        gainValue = 0
        pixelBitMapDic = ['_0_default', '_1_injection_truck', '_2_injection_corners_FHG', '_3_injection_corners_AHGLG1', '_4_extra_config', '_5_extra_config', '_6_truck2', '_7_on_the_fly', ]
        
        gainMapSelection=np.zeros((4, 168, 192))
        gainValSelection=np.zeros(4)
        
        wdet(cbase.App.EpixUhrMatrixConfig.enable, True)

        if ( cfg['user']['Gain']['SetSameGain4All']):
            logging.info("Set same Gain for all ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #same MAP for each
                logging.info("Use Pixel MAP")
                PixMapSel = int(cfg['user']['Gain']['PixelBitMapSel'])
                
                PixMapSelected= pixelBitMapDic[PixMapSel]
                
    #                csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (-1, 192))
                
                if ('on_the_fly' not in PixMapSelected):
                    fn = pathPll+'csvConfig'+'.csv'
                    csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (168, 192))
                    np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='', fmt='%d')
                    tmpfiles.append(fn)
                    
                else:
                    fn = pathPll+'onthefly.csv'    
                    csvCfg = np.loadtxt(pathPll+'onthefly.csv', dtype='uint16', delimiter=',')
                for i in asics: 
                    print(f"ASIC{i}")
                    
                    if i == 1:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic1(fn)
                    if i == 2:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic2(fn)
                    if i == 3:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic3(fn)
                    if i == 4:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic4(fn)
                    
                    logging.info(f"{PixMapSelected} CSV File Loaded")
                gainMapSelection[i-1,:,:]=csvCfg

            else:
                #same value for all
                logging.info("Use single value for all ASICS")
                gainValue=str(cfg['user']['Gain']['SetGainValue'])
                
                for i in asics: 
                    print(f"ASIC{i}")
                    gainValSelection[i-1]=gainValue
                    getattr(cbase.App,f"Asic{i}").progPixelMatrixConstantValue(gainValue)
        else:
            logging.info("Set single Gain per ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #a map per each
                logging.info("Use a Pixel MAP per each ASIC")
                for i in asics:
                    print(f"ASIC{i}")
                    PixMapSel = cfg['expert']['App'][f'Asic{i}']['PixelBitMapSel']    
                    PixMapSelected= pixelBitMapDic[PixMapSel]
                    print(PixMapSelected)
                    if ('on_the_fly' not in PixMapSelected):
                        csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (168, 192))
                        fn = pathPll+f'csvConfigAsic{i}'+'.csv'
                        np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='')
                        tmpfiles.append(fn)
                    else:
                        fn = pathPll+'onthefly.csv'
                        csvCfg = np.loadtxt(f'{pathPll}onthefly.csv', dtype='uint16', delimiter=',')
                    gainMapSelection[i-1,:,:]=csvCfg
                    if i == 1:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic1(fn)
                    if i == 2:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic2(fn)
                    if i == 3:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic3(fn)
                    if i == 4:
                        cbase.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic4(fn)
            else:
                #a value per each
                logging.info("Use a value per ASIC")
                for i in asics: 
                    gainValue=str(cfg['expert']['App'][f'Asic{i}']['SetGainValue'])
                    print(f"ASIC{i}")
                    gainValSelection[i-1]=gainValue
                    getattr(cbase.App,f"Asic{i}").progPixelMatrixConstantValue(gainValue)
                    
        for i in asics: wdet(getattr(cbase.App,f"Asic{i}").PixNumModeEn, False)
        
        
        
        if(cfg['user']['App']['VINJ_DAC']['enable']==1):
            wdet(cbase.App.WaveformControl.InjEn,  True    )
            wdet(cbase.App.VINJ_DAC.enable,        True   )
            wdet(cbase.App.VINJ_DAC.dacEn,         True   )
            
            
            if (not cfg['user']['App']['VINJ_DAC']['rampEn']==1): 
                wdet(cbase.App.VINJ_DAC.dacSingleValue, cfg['user']['App']['VINJ_DAC']['dacSingleValue'])
            else:
                wdet(cbase.App.VINJ_DAC.resetDacRamp, True)
                #wdet(cbase.App.VINJ_DAC.rampEn,        False   )	
                wdet(cbase.App.VINJ_DAC.dacStartValue, cfg['user']['App']['VINJ_DAC']['dacStartValue'])
                wdet(cbase.App.VINJ_DAC.dacStopValue,  cfg['user']['App']['VINJ_DAC']['dacStopValue'] )
                wdet(cbase.App.VINJ_DAC.dacStepValue,  cfg['user']['App']['VINJ_DAC']['dacStepValue'] )    
                wdet(cbase.App.VINJ_DAC.resetDacRamp, False)
                wdet(cbase.App.VINJ_DAC.rampEn,        True    )                
        else:            
            wdet(cbase.App.VINJ_DAC.dacEn,         False       )	
            wdet(cbase.App.VINJ_DAC.rampEn,        False       )
            wdet(cbase.App.WaveformControl.InjEn,  False       )
            wdet(cbase.App.VINJ_DAC.enable,        False       )
        
        
        # Remove the yml files
        for f in tmpfiles:
            os.remove(f)
                
    logging.info('config_expert complete')
    config_completed = True
def reset_counters(base):
    # Reset the timing counters
    base['cam'].App.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
    base['cam'].App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

#
#  Called on Configure
#
def epixUHR_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global origcfg
    global group
    global segids
    global gainMapSelection
    global gainValSelection
    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    origcfg = cfg
    
    
    #  Translate user settings to the expert fields
    writeCalibRegs=user_to_expert(base, cfg, full=True)
    
    
    if cfg==base['cfg']:
        logging.info('### Skipping redundant configure')
        return base['result']

    if base['cfg']:
        logging.info('--- config changed ---')
        _dict_compare(base['cfg'],cfg,'cfg')
        logging.info('--- /config changed ---')

    #  Apply the expert settings to the device
    _stop(base)
    
    config_expert(base, cfg, writeCalibRegs)
    
    time.sleep(0.01)
    _start(base)

    #  Add some counter resets here
    reset_counters(base)

    #  Enable triggers to continue monitoring
    epixUHR_internal_trigger(base)
    
    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()

    origcfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
   # compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(cbase.numOfAsics) ]
   # precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(cbase.numOfAsics) ]

    topname = cfg['detName:RO'].split('_')

    segcfg = {}
    segids = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]
    
    
    #gain_mode = cfg['user']['gain_mode']
    #if gain_mode==3:
    #    column_map    = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
    #else:
    #    compTH0,precharge_DAC0 = gain_mode_map(gain_mode)
    #    compTH        = [compTH0        for i in range(cbase.numOfAsics)]
    #    precharge_DAC = [precharge_DAC0 for i in range(cbase.numOfAsics)]


    #for seg in range(1):
        #  Construct the ID
    digitalId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.DigitalBoardId.get()
                 #0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.DigIDHigh.get()]
    pwrCommId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.AnalogBoardId.get()
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.PowerAndCommIDHigh.get()]
    carrierId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.CarrierBoardId.get()
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.CarrierIDHigh.get()]

        
    id = '%010d-%010d-%010d-%010d'%(firmwareVersion,
                                    carrierId,
                                    digitalId,
                                    pwrCommId)

    segids[0] = id
    top = cdict()
    top.setAlg('config', [2,0,0])
    top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=int(topname[-1]), detId=id, doc='No comment')
    
    top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
    top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')        
    
    #top.set('CompTH_ePixM',        compTH,        'UINT8')
    #top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
    #   if gain_mode==3:
    #       top.set('chgInj_column_map', column_map)
    segcfg[1] = top.typed_json()
    
    result = []
    for i in seglist:
        logging.debug('json seg {}  detname {}'.format(i, segcfg[i]['detName:RO']))
        result.append( json.dumps(sanitize_config(segcfg[i])) )
    
    base['cfg']    = copy.deepcopy(cfg)
    base['result'] = copy.deepcopy(result)
    logging.info("created gain values in XTC file")
    return result

def epixUHR_unconfig(base):
    logging.info('epixUHR_unconfig')
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixUHR_scan_keys(update):
    logging.debug('epixUHR_scan_keys')
    global origcfg
    global base
    global segids
    global gainValSelection
    global gainMapSelection
    
    cfg = {}
    copy_reconfig_keys(cfg,origcfg,json.loads(update))
    # Apply to expert
    calibRegsChanged = user_to_expert(base,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,origcfg,key)
        copy_config_entry(cfg[':types:'],origcfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    segcfg = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]
    
    
    if calibRegsChanged:
       
        cbase = base['cam']
   #     compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(1, cbase.numOfAsics+1) ]
   #     precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(1, cbase.numOfAsics+1) ]
   #     if 'chgInj_column_map' in cfg['user']:
   #         gain_mode = cfg['user']['gain_mode']
   #         if gain_mode==3:
   #             column_map = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
   #         else:
   #             column_map = np.zeros(nColumns, dtype=np.uint8)

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
            top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')
   
   #         top.set('CompTH_ePixM',        compTH,        'UINT8')
   #         top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
   #         if 'chgInj_column_map' in cfg['user']:
   #             top.set('chgInj_column_map', column_map)
            segcfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(segcfg)):
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    base['scan_keys'] = copy.deepcopy(result)
    if not check_json_keys(result, base['result']): # @todo: Too strict?
        logging.error('epixUHR_scan_keys json is inconsistent with that of epixUHR_config')

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixUHR_update(update):
    logging.debug('epixUHR_update')
    global origcfg
    global base
    global gainMapSelection
    global gainValSelection
    
    #  Queue full configuration next Configure transition
    base['cfg'] = None

    _stop(base)
    ##
    ##  Having problems with partial configuration
    ##
    # extract updates
    cfg = {}
    
    update_config_entry(cfg,origcfg,json.loads(update))
    #  Apply to expert
    
    writeCalibRegs = user_to_expert(base,cfg,full=False)
    logging.info(f'Partial config writeCalibRegs {writeCalibRegs}')
    
    config_expert(base, cfg, writeCalibRegs, secondPass=True)
    _start(base)

    #  Enable triggers to continue monitoring
#    epixUHR_internal_trigger(base)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,origcfg,key)
        copy_config_entry(cfg[':types:'],origcfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    segcfg = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    if writeCalibRegs:
        cbase = base['cam']

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            
            top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
            top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')

            segcfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(segcfg)):
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    logging.info('update complete')

    return result

def _resetSequenceCount():
    cbase = base['cam']
    wdet(cbase.App.RegisterControlDualClock.ResetCounters, 1)
    time.sleep(1.e6)
    wdet(cbase.App.RegisterControlDualClock.ResetCounters, 0)

def epixUHR_external_trigger(base):
    #  Switch to external triggering
    logging.info("external triggering")
    cbase = base['cam']
    cbase.App.TriggerRegisters.SetTimingTrigger()

def wdet(var, val):
    if (var.get() != val):
        var.set(val)
        if var.get() != val:
            logging.error(f"Failed to write to detector {var}:{val}")
        else:
            logging.debug(f"File written correctly {var}:{val}")
    else:
        logging.debug(f"Variable already set {var}:{val}")
        
def epixUHR_internal_trigger(base):
    logging.info('internal triggering')
    ##  Disable frame readout
    #mask = 0x3
    #print('=== internal triggering with bypass {:x} ==='.format(mask))
    #cbase = base['cam']
    #for i in range(cbase.numOfAsics):
    #    # This should be base['pci'].DevPcie.Application.EventBuilder.Bypass
    #    getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').Bypass.set(mask)
    #return

    #  Switch to internal triggering
    
    cbase = base['cam']
    cbase.App.TriggerRegisters.StartAutoTrigger()
    

def epixUHR_enable(base):
    logging.info('epixUHR_enable')
    epixUHR_external_trigger(base)
    _start(base)

def epixUHR_disable(base):
    logging.info('epixUHR_disable')
    # Prevents transitions going through: epixUHR_internal_trigger(base)

def _stop(base):
    logging.info('_stop')
    cbase = base['cam']
    cbase.App.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    logging.info('_start')
    cbase = base['cam']
    cbase.App.SetTimingTrigger()
    
            # Get devices
    eventBuilder = cbase.App.find(typ=batcher.AxiStreamBatcherEventBuilder)
    
    for devPtr in eventBuilder:
        devPtr.Blowoff.set(False)
    
        devPtr.SoftRst()

    # Turn on the triggering

    wdet(cbase.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable, True)
    wdet(cbase.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].MasterEnable, True)
    

    # Update the run state status variable
    wdet(cbase.App.RunState, True)  

   
#
#  Test standalone
#
if __name__ == "__main__":

    _base = epixUHR_init(None, dev='/dev/datadev_0')
    epixUHR_init_feb()
    epixUHR_connectionInfo(_base, None)

    db = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB'
    d = {'body':{'control':{'0':{'control_info':{'instrument':'tst',
                                                 'cfg_dbase' :db}}}}}

    logging.ingo('***** CONFIG *****')
    _connect_str = json.dumps(d)
    epixUHR_config(_base,_connect_str,'BEAM','tst',0,4)

    logging.info('***** SCAN_KEYS *****')
    epixUHR_scan_keys(json.dumps(["user.gain_mode"]))

    for i in range(100):
        logging.info(f'***** UPDATE {i} *****')
        epixUHR_update(json.dumps({'user.gain_mode':i%3}))

    logging.info('***** DONE *****')

