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
        
        getattr(cbase.App,f'Asic{asic}').enable.set(True)			  	
        getattr(cbase.App,f'Asic{asic}').TpsDacGain.set(1)						
        getattr(cbase.App,f'Asic{asic}').TpsDac.set(34)						
        getattr(cbase.App,f'Asic{asic}').TpsGr.set(12)						
        getattr(cbase.App,f'Asic{asic}').TpsMux.set(0)						
        getattr(cbase.App,f'Asic{asic}').BiasTpsBuffer.set(5)						
        getattr(cbase.App,f'Asic{asic}').BiasTps.set(4)						
        getattr(cbase.App,f'Asic{asic}').BiasTpsDac.set(4)						
        getattr(cbase.App,f'Asic{asic}').DacVthr.set(52)						
        getattr(cbase.App,f'Asic{asic}').BiasDac.set(4)						
        getattr(cbase.App,f'Asic{asic}').BgrCtrlDacTps.set(3)						
        getattr(cbase.App,f'Asic{asic}').BgrCtrlDacComp.set(0)						
        getattr(cbase.App,f'Asic{asic}').DacVthrGain.set(2)						
        getattr(cbase.App,f'Asic{asic}').PpbitBe.set(1)						
        getattr(cbase.App,f'Asic{asic}').BiasPxlCsa.set(0)						
        getattr(cbase.App,f'Asic{asic}').BiasPxlBuf.set(0)						
        getattr(cbase.App,f'Asic{asic}').BiasAdcComp.set(0)						
        getattr(cbase.App,f'Asic{asic}').BiasAdcRef.set(0)						
        getattr(cbase.App,f'Asic{asic}').CmlRxBias.set(3)						
        getattr(cbase.App,f'Asic{asic}').CmlTxBias.set(3)						
        getattr(cbase.App,f'Asic{asic}').DacVfiltGain.set(2)						
        getattr(cbase.App,f'Asic{asic}').DacVfilt.set(28)						
        getattr(cbase.App,f'Asic{asic}').DacVrefCdsGain.set(2)						
        getattr(cbase.App,f'Asic{asic}').DacVrefCds.set(44)						
        getattr(cbase.App,f'Asic{asic}').DacVprechGain.set(2)						
        getattr(cbase.App,f'Asic{asic}').DacVprech.set(34)						
        getattr(cbase.App,f'Asic{asic}').BgrCtrlDacFilt.set(2)						
        getattr(cbase.App,f'Asic{asic}').BgrCtrlDacAdcRef.set(2)						
        getattr(cbase.App,f'Asic{asic}').BgrCtrlDacPrechCds.set(2)						
        getattr(cbase.App,f'Asic{asic}').BgrfCtrlDacAll.set(2)						
        getattr(cbase.App,f'Asic{asic}').BgrDisable.set(0)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefpGain.set(3)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefp.set(53)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefnGain.set(0)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefn.set(12)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefCmGain.set(1)						
        getattr(cbase.App,f'Asic{asic}').DacAdcVrefCm.set(45)						
        getattr(cbase.App,f'Asic{asic}').AdcCalibEn.set(0)						
        getattr(cbase.App,f'Asic{asic}').CompEnGenEn.set(1)						
        getattr(cbase.App,f'Asic{asic}').CompEnGenCfg.set(5)						
        getattr(cbase.App,f'Asic{asic}').CfgAutoflush.set(0)						
        getattr(cbase.App,f'Asic{asic}').ExternalFlushN.set(1)						
        getattr(cbase.App,f'Asic{asic}').ClusterDvMask.set(16383)					
        getattr(cbase.App,f'Asic{asic}').PixNumModeEn.set(0)
        #PixNumModeEn, change this value to 1 to create a fixed pattern						
        getattr(cbase.App,f'Asic{asic}').SerializerTestEn.set(0)						
        getattr(cbase.App,f'BatcherEventBuilder{asic}').enable.set(True)			  	
        getattr(cbase.App,f'BatcherEventBuilder{asic}').Bypass.set(0)						
        getattr(cbase.App,f'BatcherEventBuilder{asic}').Timeout.set(0)						
        getattr(cbase.App,f'BatcherEventBuilder{asic}').Blowoff.set(False)					
        getattr(cbase.App,f'FramerAsic{asic}').enable.set(False)
        getattr(cbase.App,f'FramerAsic{asic}').DisableLane.set(0)						
        getattr(cbase.App,f'AsicGtData{asic}').enable.set(True)
        getattr(cbase.App,f'AsicGtData{asic}').gtStableRst.set(False)

def cbase_init(cbase):
    cbase.App.WaveformControl.enable.set(True)			  	
    cbase.App.WaveformControl.GlblRstPolarity.set(True)		  	
    cbase.App.WaveformControl.SR0Polarity.set(False)			  	
    cbase.App.WaveformControl.SR0Delay.set(1195)	  	
    cbase.App.WaveformControl.SR0Width.set(1)		  	
    cbase.App.WaveformControl.AcqPolarity.set(False)			  	
    cbase.App.WaveformControl.AcqDelay.set(655)	  	
    cbase.App.WaveformControl.AcqWidth.set(535)	  	
    cbase.App.WaveformControl.R0Polarity.set(False)			  	
    cbase.App.WaveformControl.R0Delay.set(70)		  	
    cbase.App.WaveformControl.R0Width.set(1125)		  	
    cbase.App.WaveformControl.InjPolarity.set(False)		  	
    cbase.App.WaveformControl.InjDelay.set(700)	  	
    cbase.App.WaveformControl.InjWidth.set(535)	  	
    cbase.App.WaveformControl.InjEn.set(False)		  	
    cbase.App.WaveformControl.InjSkipFrames.set(0) 		
    cbase.App.TriggerRegisters.enable.set(True)			  	
    cbase.App.TriggerRegisters.RunTriggerEnable.set(False)					
    cbase.App.TriggerRegisters.RunTriggerDelay.set(0)					
    cbase.App.TriggerRegisters.DaqTriggerEnable.set(False)					
    cbase.App.TriggerRegisters.DaqTriggerDelay.set(0)					
    cbase.App.TriggerRegisters.TimingRunTriggerEnable.set(False)				
    cbase.App.TriggerRegisters.TimingDaqTriggerEnable.set(False)			
    cbase.App.TriggerRegisters.AutoRunEn.set(False)					
    cbase.App.TriggerRegisters.AutoDaqEn.set(False)					
    cbase.App.TriggerRegisters.AutoTrigPeriod.set(42700000)				
    cbase.App.TriggerRegisters.numberTrigger.set(0)						
    cbase.App.TriggerRegisters.PgpTrigEn.set(False)					
    cbase.App.GTReadoutBoardCtrl.enable.set(True)
    cbase.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard.set(False)		 	
    cbase.App.GTReadoutBoardCtrl.timingOutEn0.set(False)
    cbase.App.GTReadoutBoardCtrl.timingOutEn1.set(False)
    cbase.App.GTReadoutBoardCtrl.timingOutEn2.set(False)
    cbase.App.AsicGtClk.enable.set(True)
    cbase.App.AsicGtClk.gtRstAll.set(False)					
    cbase.App.TimingRx.enable.set(True)
    cbase.Core.Si5345Pll.enable.set(False)
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
    print(f'firmwareVersion [{firmwareVersion:x}]')
    print(f'buildDate       [{buildDate}]')
    print(f'gitHashShort    [{gitHashShort}]')

    # configure timing
    logging.warning(f'Using timebase {timebase}')

    cbase_init(cbase)
    
    if timebase=="119M":  # UED
        base['bypass'] = cbase.numOfAsics * [0x3]
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True

        epixUHR_unconfig(base)
        
        cbase.App.TimingRx.TimingFrameRx.ModeSelEn.set(1) # UseModeSel
        cbase.App.TimingRx.TimingFrameRx.ClkSel.set(0)    # LCLS-1 Clock
        cbase.App.TimingRx.TimingFrameRx.RxDown.set(0)
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
    cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

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

    calibRegsChanged = False
    a = None
    hasUser = 'user' in cfg
    conv = functools.partial(int, base=16)
    
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
    #asics = []
    #for i in range(1, cbase.numOfAsics+1):
    #    if cfg['expert']['App'][f'Asic{i}']['enable']==1:
    #        asics.append(i)

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
        print(f"Loading PLL file: {freq}")
        
        pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
        fn = pathPll+'PllConfig'+'.csv'
        np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
        
        tmpfiles.append(fn)
        setattr(cbase, 'filenamePLL', fn)
        
        pll.LoadCsvFile(pathPll+'PllConfig'+'.csv')    
        cbase_ASIC_init(cbase, asics)
               
                # remove the ASIC configuration so we don't try it
            #    del app['Mv2Asic[{}]'.format(i)]
    
# Ric: Don't understand what this is doing
#    #  Set the application event builder for the set of enabled asics
#    if base['pcie_timing']:
#        m=3
#        for i in asics:
#            m = m | (4<<i)
#    else:
##        Enable batchers for all ASICs.  Data will be padded.
##        m=0
##        for i in asics:
##            m = m | (4<<int(i/2))
#        m=3<<2
#    base['bypass'] = 0x3f^m  # mask of active batcher channels
#    base['batchers'] = m>>2  # mask of active batchers
    
    base['bypass']   = cbase.numOfAsics * [0x2]  # Enable Timing (bit-0) and Data (bit-1)
    base['batchers'] = cbase.numOfAsics * [1]  # list of active batchers
    
    for i in range(cbase.numOfAsics):
        if i+1 in asics: 
            base['bypass'][i] = 0
        
        getattr(cbase.App, f'BatcherEventBuilder{i+1}').Bypass.set(base['bypass'][i])
        #getattr(cbase.App, f'Asic{i+1}').enable.set(base['bypass'][i]==0)
        #getattr(cbase.App, f'BatcherEventBuilder{i}', base['bypass']).set(True)

    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    
    for i in asics:
        getattr(cbase.App, f'BatcherEventBuilder{i}').Timeout.set(EventBuilderTimeout) # 400 us
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
            
            #if sect == tree[-1]:
            tmpfiles.append(dictToYaml(app,epixMTypes,keys,cbase.App,path,name,tree,ordering))
            #else:
            #    tmpfiles.append(dictToYaml(app[sect],epixMTypes[sect],keys,cbase,path,name,(*tree,sect),ordering))
        ordering=sorting_dict(asics)
        
        
        #clk = cfg['expert']['Pll']['Clock']
        #if clk != 4:            # 4 is the Default firmware setting
        #    freq = [None,'_250_MHz','_125_MHz','_168_MHz'][clk]
        #    pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
        #    fn = path+'PllConfig'+'.csv'
        #    np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
        #    tmpfiles.append(fn)
        #    setattr(cbase, 'filenamePLL', fn)
        
        toYaml('App',['WaveformControl'],'RegisterControl')
        toYaml('App',['TriggerRegisters'],'TriggerReg')
        toYaml('App',[f'Asic{i}' for i in asics ],'SACIReg')
        toYaml('App',[f'BatcherEventBuilder{i}' for i in asics],'General')
        
       # setattr(cbase, 'filenameASIC',4*[None]) # This one is a little different
       # for i in asics:
       #     toYaml('App',[f'Asic[{i}]'],f'ASIC_u{i+1}')
       #     cbase.filenameASIC[i] = getattr(cbase,f'filenameASIC_u{i+1}')

        arg = [1,1,1,1,1]
        logging.info(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.App.fnInitAsicScript(None,None,arg)
        print("### FINISHED YAML LOAD ###")

       #for i in range(1, cbase.numOfAsics+1):
            # Prevent disabled ASICs from participating by disabling their lanes
            # It seems like disabling their Batchers should be sufficient,
            # but that prevents transitions from going through
        #    if i not in asics:  # Override configDb's value for disabled ASICs
        #        getattr(cbase.App, f'DigAsicStrmRegisters{i}').DisableLane.set(0xffffff)

        # Adjust for intermitent lanes of enabled ASICs
        #cbase.laneDiagnostics(arg[1:5], threshold=20, loops=5, debugPrint=False)

        # Enable the batchers for all ASICs
        for i in range(cbase.numOfAsics):
            getattr(cbase.App, f'BatcherEventBuilder{i+1}').enable.set(base['batchers'][i] == 1)
            
            

   # if writeCalibRegs:
   #     hasGainMode = 'gain_mode' in cfg['user']
   #     if (hasGainMode and cfg['user']['gain_mode']==3) or not hasGainMode:
            #
            #  Write the general pixel map
            #
   #         column_map = np.array(cfg['user']['chgInj_column_map'],dtype=np.uint8)

   #         for i in asics:
                #  Don't forget about the gain_mode and charge injection
    #            asicName = f'Asic[{i}]'
    #            saci = getattr(cbase.App,asicName)
    #            saci.enable.set(True)

#### WHAT DOES THIS DO?
                #  Don't forget about charge injection
     #           if app is not None and asicName in app:
     #               di = app[asicName]
     #               setSaci(saci.CompTH_ePixM,'CompTH_epixM',di)
     #               setSaci(saci.Precharge_DAC_ePixM,'Precharge_DAC_epixM',di)

     #               cbase.App.setupChargeInjection(i, column_map, di['Pulser'])
     #           else:
     #               cbase.App.chargeInjectionCleanup(i)

      #          saci.enable.set(False)
      #  else:
      #      gain_mode = cfg['user']['gain_mode']
      #      compTH, precharge_DAC = gain_mode_map(gain_mode)
       #     print(f'Setting gain mode {gain_mode}:  compTH {compTH},  precharge_DAC {precharge_DAC}')

        #    for i in asics:
        #        saci = getattr(cbase.App,f'Mv2Asic[{i}]')
        #        saci.enable.set(True)
        #        cbase.App.chargeInjectionCleanup(i)
        #        saci.CompTH_ePixM.set(compTH)
        #        saci.Precharge_DAC_ePixM.set(precharge_DAC)
        #        saci.enable.set(False)
        
        #pixelBitMapDic = ['_FL_FM_FH', '_FL_FM_FH_InjOff', '_allConfigs', '_allPx_52', '_allPx_AutoHGLG_InjOff', '_allPx_AutoHGLG_InjOn', '_allPx_AutoMGLG_InjOff', '_allPx_AutoMGLG_InjOn', '_allPx_FixedHG_InjOff', '_allPx_FixedHG_InjOn', '_allPx_FixedLG_InjOff', '_allPx_FixedLG_InjOn', '_allPx_FixedMG_InjOff', '_allPx_FixedMG_InjOn', '_crilin', '_crilin_epixuhr100k', '_defaults', '_injection_corners', '_injection_corners_px1', '_management', '_management_epixuhr100k', '_management_inj', '_maskedCSA', '_truck', '_truck_epixuhr100k', '_xtalk_hole']
        pixelBitMapDic = ['_0_default', '_1_injection_truck', '_2_injection_corners_FHG', '_3_injection_corners_AHGLG1', '_4_extra_config', '_5_extra_config', '_6_truck2', ]
    
        #pixelBitMapDic = ['default', 'injection_truck', 'injection_corners_FHG', 'injection_corners_AHGLG1', 'extra_config_1', 'extra_config_2', 'truck2']
        for i in asics: getattr(cbase.App,f"Asic{i}").PixNumModeEn.set(True)
        csvCfg = 0
        gainValue = 0
    
        gainMapSelection=np.zeros((4, 168, 192))
        gainValSelection=np.zeros(4)
        
        cbase.App.EpixUhrkMatrixConfig.enable.set("True")
        
        if ( cfg['user']['Gain']['SetSameGain4All']):
            print("Set same Gain for all ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #same MAP for each
                print("Use Pixel MAP")
                PixMapSel = int(cfg['user']['Gain']['PixelBitMapSel'])
                
                PixMapSelected= pixelBitMapDic[PixMapSel]
                
#                csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (-1, 192))
                csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (168, 192))
                fn = pathPll+'csvConfig'+'.csv'
                
                np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='', fmt='%d')    
                tmpfiles.append(fn)

                for i in asics: 
                    print(f"ASIC{i}")
                    gainMapSelection[i-1,:,:]=csvCfg
                    if i == 1:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic1(fn)
                    if i == 2:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic2(fn)
                    if i == 3:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic3(fn)
                    if i == 4:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic4(fn)
                   # print(f"EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic{i}('{fn}')")
                   # getattr(cbase.App,f"EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic{i}('{fn}')")
                    
                    #getattr(cbase.App,f"Asic{i}").LoadCsvPixelBitmap(fn)                        
                    #getattr(cbase.App,f"Asic{i}").SetPixelBitmap(csvCfg)
                    print(f"{PixMapSelected} CSV File Loaded")
                

            else:
                #same value for all
                print("Use single value for all ASICS")
                gainValue=str(cfg['user']['Gain']['SetGainValue'])
                
                for i in asics: 
                    print(f"ASIC{i}")
                    gainValSelection[i-1]=gainValue
                    getattr(cbase.App,f"Asic{i}").progPixelMatrixConstantValue(gainValue)
        else:
            print("Set single Gain per ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #a map per each
                print("Use a Pixel MAP per each ASIC")
                for i in asics:
                    print(f"ASIC{i}")
                    PixMapSel = cfg['expert']['App'][f'Asic{i}']['PixelBitMapSel']    
                    PixMapSelected= pixelBitMapDic[PixMapSel]
                    print(PixMapSelected)
                    csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (168, 192))
                    fn = pathPll+f'csvConfigAsic{i}'+'.csv'
                    np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='')
                    tmpfiles.append(fn)
                    gainMapSelection[i-1,:,:]=csvCfg
                    if i == 1:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic1(fn)
                    if i == 2:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic2(fn)
                    if i == 3:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic3(fn)
                    if i == 4:
                        cbase.App.EpixUhrkMatrixConfig.progPixelMatrixFromCsvAsic4(fn)

#                    getattr(cbase.App,f"EpixUhrkMatrixConfig.progPixelMapFromCSVAsic{i}('{fn}')")
#                    getattr(cbase.App,f"Asic{i}").SetPixelBitmap(csvCfg)
            else:
                #a value per each
                print("Use a value per ASIC")
                for i in asics: 
                    gainValue=str(cfg['expert']['App'][f'Asic{i}']['SetGainValue'])
                    
                    print(f"ASIC{i}")
                    gainValSelection[i-1]=cfg['expert']['App'][f'Asic{i}']['SetGainValue']
                    getattr(cbase.App,f"Asic{i}").progPixelMatrixConstantValue(gainValue)
            
        
        for i in asics: getattr(cbase.App,f"Asic{i}").PixNumModeEn.set(False)
        
        cbase.App.GTReadoutBoardCtrl.enable.set(app['GTReadoutBoardCtrl']['enable'])
        cbase.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard.set(app['GTReadoutBoardCtrl']['pwrEnableAnalogBoard'])
        cbase.App.GTReadoutBoardCtrl.timingOutEn0.set(app['GTReadoutBoardCtrl']['timingOutEn0'])
        cbase.App.GTReadoutBoardCtrl.timingOutEn1.set(app['GTReadoutBoardCtrl']['timingOutEn1'])
        cbase.App.GTReadoutBoardCtrl.timingOutEn2.set(app['GTReadoutBoardCtrl']['timingOutEn2'])
        
        cbase.App.AsicGtClk.enable.set(cfg['expert']['App']['AsicGtClk']['enable'])
        #cbase.App.AsicGtClk.gtRstAll.set(cfg['expert']['App']['AsicGtClk']['gtResetAll'])
        for i in asics:
            getattr(cbase.App,f"AsicGtData{i}").enable.set(cfg['expert']['App'][f'AsicGtData{i}']['enable'])			
            getattr(cbase.App,f"AsicGtData{i}").gtStableRst.set(cfg['expert']['App'][f'AsicGtData{i}']['gtStableRst']		)	
        
        
        if cbase.App.VINJ_DAC.dacSingleValue.get() != cfg['user']['App']['VINJ_DAC']['enable']:
            cbase.App.VINJ_DAC.dacSingleValue.set(cfg['user']['App']['VINJ_DAC']['enable']			)	
 
        if cbase.App.VCALIBP_DAC.dacSingleValue.get() != cfg['user']['App']['VCALIBP_DAC']['enable']:
            cbase.App.VCALIBP_DAC.dacSingleValue.set(cfg['user']['App']['VCALIBP_DAC']['enable']			)	
        
        cbase.App.VCALIBP_DAC.dacEn.set(cfg['user']['App']['VCALIBP_DAC']['dacEn'])
        cbase.App.VCALIBP_DAC.dacSingleValue.set(cfg['user']['App']['VCALIBP_DAC']['dacSingleValue'])
        cbase.App.VCALIBP_DAC.rampEn.set(cfg['user']['App']['VCALIBP_DAC']['rampEn'])
        cbase.App.VCALIBP_DAC.dacStartValue.set(cfg['user']['App']['VCALIBP_DAC']['dacStartValue'])
        cbase.App.VCALIBP_DAC.dacStopValue.set(cfg['user']['App']['VCALIBP_DAC']['dacStopValue'])
        cbase.App.VCALIBP_DAC.dacStepValue.set(cfg['user']['App']['VCALIBP_DAC']['dacStepValue'])
        cbase.App.VCALIBP_DAC.resetDacRamp.set(cfg['user']['App']['VCALIBP_DAC']['resetDacRamp'])

        cbase.App.VINJ_DAC.dacEn.set(cfg['user']['App']['VINJ_DAC']['dacEn'])
        cbase.App.VINJ_DAC.dacSingleValue.set(cfg['user']['App']['VINJ_DAC']['dacSingleValue'])
        cbase.App.VINJ_DAC.rampEn.set(cfg['user']['App']['VINJ_DAC']['rampEn'])
        cbase.App.VINJ_DAC.dacStartValue.set(cfg['user']['App']['VINJ_DAC']['dacStartValue'])
        cbase.App.VINJ_DAC.dacStopValue.set(cfg['user']['App']['VINJ_DAC']['dacStopValue'])
        cbase.App.VINJ_DAC.dacStepValue.set(cfg['user']['App']['VINJ_DAC']['dacStepValue'])
        cbase.App.VINJ_DAC.resetDacRamp.set(cfg['user']['App']['VINJ_DAC']['resetDacRamp'])
        
        if cbase.App.ADS1217.enable.get() != 	cfg['user']['App']['ADS1217']['adcStartEnManual']:
            cbase.App.ADS1217.enable.set(cfg['user']['App']['ADS1217']['enable']						)	
        cbase.App.ADS1217.adcStartEnManual.set(cfg['user']['App']['ADS1217']['adcStartEnManual']	)
        
        #for i in asics: 
        #    print(f"ASIC{i}")
        #cbase.App.Asic1.LoadCsvPixelBitmap.set('/tmp/csvConfig.csv')            
        #print("########## ASIC1 csv test")        
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
        print('### Skipping redundant configure')
        return base['result']

    if base['cfg']:
        print('--- config changed ---')
        _dict_compare(base['cfg'],cfg,'cfg')
        print('--- /config changed ---')

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

    #print(f'gain_mode {gain_mode}  CompTH_ePixM {compTH}  Precharge_DAC_ePixM {precharge_DAC}  column_map shape {column_map.shape if gain_mode==3 else None}')

    #for seg in range(1):
        #  Construct the ID
    digitalId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.DigitalBoardId.get()
                 #0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.DigIDHigh.get()]
    pwrCommId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.AnalogBoardId.get()
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.PowerAndCommIDHigh.get()]
    carrierId =  0 if base['pcie_timing'] else cbase.App.GTReadoutBoardCtrl.CarrierBoardId.get()
        #             0 if base['pcie_timing'] else cbase.App.RegisterControlDualClock.CarrierIDHigh.get()]
        #print(f'ePixUHRk ids: f/w {firmwareVersion:x}, carrier {carrierId:x}, digital {digitalId:x}, pwrComm {pwrCommId:x}')
        
    id = '%010d-%010d-%010d-%010d'%(firmwareVersion,
                                    carrierId,
                                    digitalId,
                                    pwrCommId)

    segids[0] = id
    top = cdict()
    top.setAlg('config', [1,0,0])
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
    print("created gain values in XTC file")
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
            top.setAlg('config', [1,0,0])
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
        logging.error('epixm320_scan_keys json is inconsistent with that of epix320_config')

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
    print(f'Partial config writeCalibRegs {writeCalibRegs}')

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
#        try:
#            compTH        = [ cfg['expert']['App'][f'Asic[{i}]']['CompTH_ePixM']        for i in range(1, cbase.numOfAsics+1) ]
#        except:
#            compTH        = None; print('CompTH is None')
#        try:
#            precharge_DAC = [ cfg['expert']['App'][f'Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(1, cbase.numOfAsics+1) ]
#        except:
#            precharge_DAC = None; print('Precharge_DAC is None')
#        try:
#            column_map    = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
#        except:
#            column_map    = None; print('column_map is None')

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [1,0,0])
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            
            top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
            top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')

#            if compTH        is not None:  top.set('CompTH_ePixM',        compTH,        'UINT8')
#            if precharge_DAC is not None:  top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
#            if column_map    is not None:  top.set('chgInj_column_map',   column_map)
            segcfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(segcfg)):
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    logging.info('update complete')

    return result

def _resetSequenceCount():
    cbase = base['cam']
    cbase.App.RegisterControlDualClock.ResetCounters.set(1)
    time.sleep(1.e6)
    cbase.App.RegisterControlDualClock.ResetCounters.set(0)

def epixUHR_external_trigger(base):
    #  Switch to external triggering
    print("external triggering")
    cbase = base['cam']
    cbase.App.TriggerRegisters.SetTimingTrigger.set(1)

def epixUHR_internal_trigger(base):
    print('internal triggering')
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
    cbase.App.TriggerRegisters.SetAutoTrigger.set(1)
    

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
    cbase.App.SetTimingTrigger()
    #cbase.App.StartRun()
            # Get devices
    eventBuilder = cbase.App.find(typ=batcher.AxiStreamBatcherEventBuilder)
    #trigger      = cbase.App.find(typ=l2si.TriggerEventBuffer)

    # Reset all counters
    #cbase.App.CountReset()

    # Arm for data/trigger stream
    #for i in asics:
    #    getattr(cbase.App, f'BatcherEventBuilder{i}').Blowoff.set(False)
        
    for devPtr in eventBuilder:
        devPtr.Blowoff.set(False)
    #    devPtr.Bypass.set(0x0)
        devPtr.SoftRst()

    # Turn on the triggering
    #for devPtr in trigger:
    cbase.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)
    cbase.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].MasterEnable.set(True)
    #self.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition.set(1)

    # Update the run state status variable
    cbase.App.RunState.set(True)  

    # This is unneccessary as it is handled above and in StartRun()
    #m = base['batchers']
    #for i in range(cbase.numOfAsics):
    #    getattr(cbase.App.AsicTop,f'BatcherEventBuilder{i}').Bypass.set(0x0)
    #    getattr(cbase.App.AsicTop,f'BatcherEventBuilder{i}').Blowoff.set(m[i]==0)
    #print(f'Blowoff BatcherEventBuilders {[x^0x1 for x in m]}')

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

    print('***** CONFIG *****')
    _connect_str = json.dumps(d)
    epixUHR_config(_base,_connect_str,'BEAM','tst',0,4)

    print('***** SCAN_KEYS *****')
    epixUHR_scan_keys(json.dumps(["user.gain_mode"]))

    for i in range(100):
        print(f'***** UPDATE {i} *****')
        epixUHR_update(json.dumps({'user.gain_mode':i%3}))

    print('***** DONE *****')

