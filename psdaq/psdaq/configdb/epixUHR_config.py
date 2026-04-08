from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.det_config import *
from psdaq.cas.xpm_utils import timTxId
import pyrogue as pr
import rogue
import rogue.hardware.axi

from psdaq.utils import enable_epix_uhr_gtreadout_dev
import epix_uhr_gtreadout_dev as epixUhrDev
import surf.protocols.batcher as batcher

import time
import json
import os
import numpy as np
import logging
import copy # deepcopy

rogue.Version.minVersion('6.1.0')

base = None

chan = None
group = None
orig_cfg = None
seg_ids = None
SEGLIST = [0,1]
asics = None

DET_SIZE = (4, 168, 192)  # 4 ASICs of 168x192 pixels each
gain_map=np.zeros(DET_SIZE)

#  Timing delay scans can be limited by this
EVENT_BUILDER_TIMEOUT = 0 #4*int(1.0e-3*156.25e6)
#this is supposed to be constant for every detector
DELTA_DELAY = -192

#path were to store cvs files for gain definition
GAIN_PATH = '/tmp/ePixUHR_GTReadout_default_'
#path were to store pll files used for calibration
PLL_PATH = '/tmp/'

PLL_LABELS = [None, '_temp250', '_2_3_7', '_0_5_7', '_2_3_9', '_0_5_7_v2']
TIMING_OUT_EN_LIST=[
    'asicR0', 
    'asicACQ', 
    'asicSRO', 
    'asicInj', 
    'asicGlbRstN', 
    'timingRunTrigger', 
    'timingDaqTrigger', 
    'acqStart', 
    'dataSend', 
    '_0', 
    '_1'
    ]

GAIN_CSV_LIST = [
    '_0_default', 
    '_1_injection_truck', 
    '_2_injection_corners_FHG', 
    '_3_injection_corners_AHGLG1', 
    '_4_extra_config', 
    '_5_extra_config', 
    '_6_truck2', 
    '_7_on_the_fly', 
    ]
ALG_VERSION = [3,2,1]


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
def sanitize_config(src: dict) -> dict:
    dst = {}
    for k, v in src.items():
        if isinstance(v, dict):
            v = sanitize_config(v)
        dst[k.replace('[','').replace(']','').replace('(','').replace(')','')] = v
    return dst

#Initialization of ASICs, this happens after getting configdb data because we need to know which ASIC to init
def panel_ASIC_init(det_root: dict, asics: list):
    
    for asic in asics:				

        write_to_detector(getattr(det_root.App,f"BatcherEventBuilder{asic}").enable,True)			  	
        write_to_detector(getattr(det_root.App,f"BatcherEventBuilder{asic}").Bypass,0)						
        write_to_detector(getattr(det_root.App,f"BatcherEventBuilder{asic}").Timeout,0)						
        write_to_detector(getattr(det_root.App,f"BatcherEventBuilder{asic}").Blowoff,False)					
        write_to_detector(getattr(det_root.App,f"FramerAsic{asic}").enable,         False)
        write_to_detector(getattr(det_root.App,f"FramerAsic{asic}").DisableLane,    0)						
        write_to_detector(getattr(det_root.App,f"AsicGtData{asic}").enable,         True)
        write_to_detector(getattr(det_root.App,f"AsicGtData{asic}").gtStableRst,    False)

                            
#Initialization of the detector; this is meant to put the detector in a pre defined working state.
def panel_init(det_root: dict):
        write_to_detector(det_root.App.WaveformControl.enable,              True)			  	
        write_to_detector(det_root.App.WaveformControl.GlblRstPolarity,     True)
        write_to_detector(det_root.App.WaveformControl.AsicSroEn,           True)			  			  	
        write_to_detector(det_root.App.WaveformControl.SroPolarity,         False)			  	
        write_to_detector(det_root.App.WaveformControl.SroDelay,            1195)	  	
        write_to_detector(det_root.App.WaveformControl.SroWidth,            1)		  	
        write_to_detector(det_root.App.WaveformControl.AsicAcqEn,           True)			  	
        write_to_detector(det_root.App.WaveformControl.AcqPolarity,         False)			  	
        write_to_detector(det_root.App.WaveformControl.AcqDelay,            655)	  	
        write_to_detector(det_root.App.WaveformControl.AcqWidth,            535)	  	
        write_to_detector(det_root.App.WaveformControl.R0Polarity,          False)			  	
        write_to_detector(det_root.App.WaveformControl.R0Delay,             70)		  	
        write_to_detector(det_root.App.WaveformControl.R0Width,             1125)
        write_to_detector(det_root.App.WaveformControl.AsicR0En,            True)			  			  	
        write_to_detector(det_root.App.WaveformControl.InjPolarity,         False)		  	
        write_to_detector(det_root.App.WaveformControl.InjDelay,            700)	  	
        write_to_detector(det_root.App.WaveformControl.InjWidth,            535)	  	
        write_to_detector(det_root.App.WaveformControl.InjEn,               False)		  	
        write_to_detector(det_root.App.WaveformControl.InjSkipFrames,       0) 		
        write_to_detector(det_root.App.TriggerRegisters.enable,             True)			  	
        write_to_detector(det_root.App.TriggerRegisters.RunTriggerEnable,   False)					
        write_to_detector(det_root.App.TriggerRegisters.RunTriggerDelay,    0)					
        write_to_detector(det_root.App.TriggerRegisters.DaqTriggerEnable,   False)					
        write_to_detector(det_root.App.TriggerRegisters.DaqTriggerDelay,    0)					
        write_to_detector(det_root.App.TriggerRegisters.TimingRunTriggerEnable,False)				
        write_to_detector(det_root.App.TriggerRegisters.TimingDaqTriggerEnable,False)			
        write_to_detector(det_root.App.TriggerRegisters.AutoRunEn,          False)					
        write_to_detector(det_root.App.TriggerRegisters.AutoDaqEn,          False)					
        write_to_detector(det_root.App.TriggerRegisters.AutoTrigPeriod,     42700000)				
        write_to_detector(det_root.App.TriggerRegisters.numberTrigger,      0)						
        write_to_detector(det_root.App.TriggerRegisters.PgpTrigEn,          False)					
        write_to_detector(det_root.App.GTReadoutBoardCtrl.enable,           True)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard,False)		 	
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn0,     False)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn1,     False)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn2,     False)
        write_to_detector(det_root.App.AsicGtClk.enable,                    True)
        write_to_detector(det_root.App.AsicGtClk.gtRstAll,                  False)					
        #  Add AsicGtDataX.enable as found in InitAsics[]
#        write_to_detector(det_root.App.AsicGtData1.enable,                  True)					
#        write_to_detector(det_root.App.AsicGtData2.enable,                  True)					
#        write_to_detector(det_root.App.AsicGtData3.enable,                  True)					
#        write_to_detector(det_root.App.AsicGtData4.enable,                  True)					
        write_to_detector(det_root.App.TimingRx.enable,                     True)
        write_to_detector(det_root.Core.Si5345Pll.enable,                   False)
        write_to_detector(det_root.App.VINJ_DAC.dacEn,                      False)
        write_to_detector(det_root.App.VINJ_DAC.rampEn,                     False)


#
#  Initialize the rogue accessor
#
def epixUHR_init(arg, dev='/dev/datadev_0',lanemask=0xf,xpmpv=None,timebase="186M",verbosity=0) -> dict:
    global base
    
    logging.getLogger().setLevel(logging.WARNING)
    logging.info('epixUHR_init')
    
    base = {}
    #  Connect to the camera and the PCIe card

    det_root = epixUhrDev.Root(
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
    
    det_root.__enter__()
    
    base['cam'] = det_root

    firmwareVersion = det_root.Core.AxiVersion.FpgaVersion.get()
    buildDate       = det_root.Core.AxiVersion.BuildDate.get()
    gitHashShort    = det_root.Core.AxiVersion.GitHashShort.get()
    logging.info(f'firmwareVersion [{firmwareVersion:x}]')
    logging.info(f'buildDate       [{buildDate}]')
    logging.info(f'gitHashShort    [{gitHashShort}]')

    # configure timing
    logging.warning(f'Using timebase {timebase}')

    panel_init(det_root)
    
    if timebase=="119M":  # UED
        base['bypass'] = det_root.numOfAsics * [0x3]
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True

        epixUHR_unconfig(base)
        
        write_to_detector(det_root.App.TimingRx.TimingFrameRx.ModeSelEn, 1) # UseModeSel
        write_to_detector(det_root.App.TimingRx.TimingFrameRx.ClkSel, 0)    # LCLS-1 Clock
        write_to_detector(det_root.App.TimingRx.TimingFrameRx.RxDown, 0)
    else:
        base['bypass'] = det_root.numOfAsics * [0x3]
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        base['pcie_timing'] = False

        epixUHR_unconfig(base)

        det_root.App.TimingRx.ConfigLclsTimingV2()

    # Delay long enough to ensure that timing configuration effects have completed
    cnt = 0
    while cnt < 15:
        time.sleep(1)
        rxId = det_root.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        if rxId != 0xffffffff:  break
        del rxId                # Maybe this can help getting RxId reevaluated
        cnt += 1

    if cnt == 15:
        raise ValueError("rxId didn't become valid after configuring timing")
    print(f"rxId {rxId:x} found after {cnt}s")

    #  store previously applied configuration
    base['cfg'] = None

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
def epixUHR_connectionInfo(base, alloc_json_str) -> dict:

#
#  To do:  get the IDs from the detector and not the timing link
#
    txId = timTxId('epixUHR')
    logging.info('TxId {:x}'.format(txId))

    det_root = base['cam']
    rxId = det_root.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    logging.info('RxId {:x}'.format(rxId))
    write_to_detector(det_root.App.TimingRx.TriggerEventManager.XpmMessageAligner.TxId, txId)

    epixUHRid = '-'

    new_cfg = {}
    new_cfg['paddr'] = rxId
    new_cfg['serno'] = epixUHRid

    return new_cfg

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the orig_cfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, fullConfig=False) -> bool:
    global orig_cfg
    global group
    global lane

    det_root = base['cam']
    
    new_cfg = {}
    has_user = 'user' in cfg
    
    #there are scripts that new_cfgo not have user
    if (has_user and 'start_ns' in cfg['user']):
        #rtp = orig_cfg['user']['run_trigger_group'] # run trigger partition
        
        #for i,p in enumerate([rtp,group]):
        partition_delay = getattr(det_root.App.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        raw_start       = cfg['user']['start_ns']

        trigger_delay   = int(raw_start/base['clk_period'] - partition_delay*base['msg_period'])
        logging.warning(f'partition_delay[{group}] {partition_delay}  raw_start {raw_start}  trigger_delay {trigger_delay}')
        if trigger_delay < 0:
            logging.error(f'partition_delay[{group}] {partition_delay}  raw_start {raw_start}  trigger_delay {trigger_delay}')
            logging.error('Raise start_ns >= {:}'.format(partition_delay*base['msg_period']*base['clk_period']))
            raise ValueError('trigger_delay computes to < 0')

        new_cfg[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay'] = trigger_delay
        
        trigger_delay=int(raw_start/base["clk_period"]) - DELTA_DELAY
        
        if trigger_delay < 0:
            logging.error(f'partition_delay[{group+1}] {partition_delay}  raw_start {raw_start}  trigger_delay {trigger_delay}')
            logging.error('Raise start_ns >= {:}'.format(partition_delay*base['msg_period']*base['clk_period']))
            raise ValueError('trigger_delay computes to < 0')

        new_cfg[f'expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].Delay'] = trigger_delay
        logging.warning(f'partition_delay[{group+1}] {partition_delay}  raw_start {raw_start}  trigger_delay {trigger_delay}')

        if fullConfig:
            new_cfg[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition']= group+1    # Run trigger
            new_cfg[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition']= group  # DAQ trigger
    
    calib_regs_changed = True if has_user and 'Gain'  in cfg['user'] else False

    update_config_entry(cfg,orig_cfg,new_cfg)

    return calib_regs_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writeCalibRegs=True, second_pass=False):
    global asics  # Need to maintain this across configuration updates
    global gain_map

    #  Disable internal triggers during configuration
    epixUHR_external_trigger(base)

    det_root = base['cam']
    
    # overwrite the low-level configuration parameters with calculations from the user configuration
    if 'expert' in cfg:
        try:  # config update might not have this
            apply_dict('det_root.App.TimingRx.TriggerEventManager',
                       det_root.App.TimingRx.TriggerEventManager,
                       cfg['expert']['App']['TimingRx']['TriggerEventManager'])
        except KeyError:
            pass

    app = None
    if 'expert' in cfg and 'App' in cfg['expert']:
        app = cfg['expert']['App']

     #  Make list of enabled ASICs
    if 'user' in cfg and 'asic_enable' in cfg['user']:
        asics = []
        for asic in range(det_root.numOfAsics):
            if cfg['user']['asic_enable']&(1<<asic):
                asics.append(asic+1)
    
    pll = det_root.Core.Si5345Pll
    
    tmpfiles = []
    if not second_pass:
        #Load Pll config values
        
        if pll.enable.get() == False:
            pll.enable.set(True)
        
        clk = cfg['user']['PllRegistersSel'] 
        
        freq = PLL_LABELS[clk]
        logging.info(f"Loading PLL file: {freq}")
        
        db_Pll = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
        fn = PLL_PATH+'PllConfig'+'.csv'
        np.savetxt(fn, db_Pll, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
        
        tmpfiles.append(fn)
        setattr(det_root,"filenamePLL", fn)
        
        pll.LoadCsvFile(PLL_PATH+'PllConfig'+'.csv')

        panel_ASIC_init(det_root, asics)
               
    base['bypass']   = det_root.numOfAsics * [0x2]  # bitposition enables ['Bypass','Timeout','Blowoff',]
    base['batchers'] = det_root.numOfAsics * [1]  # list of active batchers
    
    for asic in range(det_root.numOfAsics):
        if asic+1 in asics: 
            base['bypass'][asic] = 0
        
        write_to_detector(getattr(det_root.App, f'BatcherEventBuilder{asic+1}').Bypass, base['bypass'][asic])
       
    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    
    for asic in asics:
        write_to_detector(getattr(det_root.App, f'BatcherEventBuilder{asic}').Timeout, EVENT_BUILDER_TIMEOUT) # 400 us
    if not base['pcie_timing']:
        eventBuilder = det_root.find(typ=batcher.AxiStreamBatcherEventBuilder)
        for eb in eventBuilder:
            eb.Timeout.set(EVENT_BUILDER_TIMEOUT)
            eb.Blowoff.set(True)
    
    if app is not None and not second_pass:
        # Work hard to use the underlying rogue interface
        # Config data was initialized from the distribution's yaml files by epixhr_config_from_yaml.py
        # Translate config data to yaml files
        
        epixMTypes = cfg[':types:']['expert']['App']
        tree = ('Root','App')
        
        def toYaml(sect,keys,name):
            tmpfiles.append(dictToYaml(app,epixMTypes,keys,det_root.App,GAIN_PATH,name,tree))
            
        
        #Creating yaml files to be loaded durint detector initialization
        toYaml('App',['WaveformControl'],'RegisterControl')
        toYaml('App',['TriggerRegisters'],'TriggerReg')
        toYaml('App',[f'Asic{asic}' for asic in asics ],'SACIReg')
        toYaml('App',[f'BatcherEventBuilder{asic}' for asic in asics],'General')
        
        arg = [1,1,1,1,1]
        logging.info(f'Calling fnInitAsicScript(None,None,{arg})')
        det_root.App.fnInitAsicScript(None,None,arg)
        logging.info("### FINISHED YAML LOAD ###")

        # Enable the batchers for all ASICs; removed because already in ynl file General
 #       for i in range(det_root.numOfAsics):
 #           write_to_detector(getattr(det_root.App, f'BatcherEventBuilder{i+1}').enable, base['batchers'][i] == 1)
            
        write_to_detector(det_root.App.GTReadoutBoardCtrl.enable,               app['GTReadoutBoardCtrl']['enable']==1)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard, app['GTReadoutBoardCtrl']['pwrEnableAnalogBoard'])
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn0,         app['GTReadoutBoardCtrl']['timingOutEn0']==1)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn1,         app['GTReadoutBoardCtrl']['timingOutEn1']==1)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutEn2,         app['GTReadoutBoardCtrl']['timingOutEn2']==1)
        
        # Enables the use of the oscilloscope
        
        timingOutMux0_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux0'])
        timingOutMux1_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux1'])
        timingOutMux3_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux3'])
        logging.info(f'Setting timingOutMux0 to {TIMING_OUT_EN_LIST[timingOutMux0_Sel]}')
        logging.info(f'Setting timingOutMux1 to {TIMING_OUT_EN_LIST[timingOutMux1_Sel]}')
        logging.info(f'Setting timingOutMux3 to {TIMING_OUT_EN_LIST[timingOutMux3_Sel]}')
        
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutMux0, timingOutMux0_Sel)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutMux1, timingOutMux1_Sel)
        write_to_detector(det_root.App.GTReadoutBoardCtrl.timingOutMux3, timingOutMux3_Sel)
        write_to_detector(det_root.App.AsicGtClk.enable, cfg['expert']['App']['AsicGtClk']['enable']==1)
        for asic in asics:
            write_to_detector(getattr(det_root.App,f"AsicGtData{asic}").enable, cfg['expert']['App'][f'AsicGtData{asic}']['enable']==1)			
            write_to_detector(getattr(det_root.App,f"AsicGtData{asic}").gtStableRst, cfg['expert']['App'][f'AsicGtData{asic}']['gtStableRst']		)	
        if det_root.App.VCALIBP_DAC.enable.get() != cfg['user']['App']['VCALIBP_DAC']['enable']:
            write_to_detector(det_root.App.VCALIBP_DAC.enable, cfg['user']['App']['VCALIBP_DAC']['enable']==1)	

        write_to_detector(det_root.App.VCALIBP_DAC.dacSingleValue, cfg['user']['App']['VCALIBP_DAC']['dacSingleValue'])
        write_to_detector(det_root.App.VCALIBP_DAC.resetDacRamp, cfg['user']['App']['VCALIBP_DAC']['resetDacRamp'])
               
        write_to_detector(det_root.App.ADS1217.enable, cfg['user']['App']['ADS1217']['enable']==1)	
        write_to_detector(det_root.App.ADS1217.adcStartEnManual, cfg['user']['App']['ADS1217']['adcStartEnManual']	)        
                  
    
    if writeCalibRegs:
        gain_map = np.zeros(DET_SIZE)
        
        #Need to turn PixNumModeEn true to modify Gain
        for asic in asics: 
            write_to_detector(getattr(det_root.App,f"Asic{asic}").PixNumModeEn, True)

        write_to_detector(det_root.App.EpixUhrMatrixConfig.enable, True)
        
        #Gain value can be set via single value or via CSV file, also it is possible to select the same value 
        #for every ASIC or a specific one per ASIC.
        #SetSameGain4All establishes if all 4 ASICs uses the same gain
        #UsePixalMap establishes if a CSV file is used
        #if SetSameGain4All is True then cfg['user']['Gain']['PixelBitMapSel'] or cfg['user']['Gain']['SetGainValue']
        # are used
        #if SetSameGain4All is False then cfg['expert']['App'][f'Asic{i}']['PixelBitMapSel'] or cfg['expert']['App'][f'Asic{i}']['SetGainValue']
        # are used
        if ( cfg['user']['Gain']['SetSameGain4All']):
            logging.info("Set same Gain for all ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #same MAP for each
                logging.info("Use Pixel MAP")
                
                gain_map_name = GAIN_CSV_LIST[int(cfg['user']['Gain']['PixelBitMapSel'])]
                fn = PLL_PATH+'csvConfig'+'.csv'
                print(type(cfg['expert']['pixelBitMaps'][gain_map_name]))
                print(len(cfg['expert']['pixelBitMaps'][gain_map_name]))
                db_gain_map = np.uint16(np.reshape(cfg['expert']['pixelBitMaps'][gain_map_name], (DET_SIZE[1], DET_SIZE[2])))
                np.savetxt(fn, db_gain_map, delimiter=',', newline='\n', comments='', fmt='%d')
                tmpfiles.append(fn)
                
                for asic in asics: 
                    print(f"ASIC{asic}")
                    eval(f"det_root.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic{asic}('{fn}')")
                                        
                    logging.info(f"{gain_map_name} CSV File Loaded")
                    gain_map[asic-1,:,:]=db_gain_map

            else:
                #same value for all
                logging.info("Use single value for all ASICS")
                gain_value = str(cfg['user']['Gain']['SetGainValue'])
                
                for asic in asics: 
                    print(f"ASIC{asic}")
                    gain_map[asic-1,:, :] = np.int16(np.full((DET_SIZE[1],DET_SIZE[2]), gain_value))
                    getattr(det_root.App,f"Asic{asic}").progPixelMatrixConstantValue(gain_value)
        else:
            logging.info("Set single Gain per ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #a map per each
                logging.info("Use a Pixel MAP per each ASIC")
                for asic in asics:
                    print(f"ASIC{asic}")
                    
                    gain_map_name = GAIN_CSV_LIST[cfg['user']['App'][f'Asic{asic}']['PixelBitMapSel']]
                    print(f"GAIN MAP NAME: {gain_map_name}")
                    
                    db_gain_map = np.reshape(cfg['expert']['pixelBitMaps'][gain_map_name], (DET_SIZE[1], DET_SIZE[2]))
                    fn = PLL_PATH+f'csvConfigAsic{asic}'+'.csv'
                    np.savetxt(fn, db_gain_map, delimiter=',', newline='\n', comments='')
                    tmpfiles.append(fn)
                    
                    gain_map[i-1,:,:] = db_gain_map
                    eval(f"det_root.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic{asic}('{fn}')")
                    
            else:
                #a value per each
                logging.info("Use a value per ASIC")
                for asic in asics: 
                    gain_value=str(cfg['user']['App'][f'Asic{i}']['SetGainValue'])
                    print(f"ASIC{asic}")
                    gain_map[asic-1, :, :] = np.full((DET_SIZE[1],DET_SIZE[2]), gain_value)
                    getattr(det_root.App,f"Asic{asic}").progPixelMatrixConstantValue(gain_value)
        
        # deactivate gain modification        
        for asic in asics: write_to_detector(getattr(det_root.App,f"Asic{asic}").PixNumModeEn, False)
        
        #Charge Injection definitions
        if(cfg['user']['App']['VINJ_DAC']['enable']==1):
            print("Set Charge Injection")
            write_to_detector(det_root.App.WaveformControl.InjEn,       True   )
            write_to_detector(det_root.App.WaveformControl.AsicInjEn,   True   )
            write_to_detector(det_root.App.VINJ_DAC.enable,             True   )
            write_to_detector(det_root.App.VINJ_DAC.dacEn,              True   )
            
            #If ramp is not used, set Gain single Value
            if (not cfg['user']['App']['VINJ_DAC']['rampEn']==1): 
                write_to_detector(det_root.App.VINJ_DAC.dacSingleValue, cfg['user']['App']['VINJ_DAC']['dacSingleValue'])
            else:
                #Need to reset ramp when running inj script in between gain values
                write_to_detector(det_root.App.VINJ_DAC.resetDacRamp, True)
                write_to_detector(det_root.App.VINJ_DAC.dacStartValue, cfg['user']['App']['VINJ_DAC']['dacStartValue'])
                write_to_detector(det_root.App.VINJ_DAC.dacStopValue,  cfg['user']['App']['VINJ_DAC']['dacStopValue'] )
                write_to_detector(det_root.App.VINJ_DAC.dacStepValue,  cfg['user']['App']['VINJ_DAC']['dacStepValue'] )    
                write_to_detector(det_root.App.VINJ_DAC.resetDacRamp, False)
                write_to_detector(det_root.App.VINJ_DAC.rampEn,        True    )       
                         
        else:            
            write_to_detector(det_root.App.VINJ_DAC.dacEn,              False       )	
            write_to_detector(det_root.App.VINJ_DAC.rampEn,             False       )
            write_to_detector(det_root.App.WaveformControl.InjEn,       False       )
            write_to_detector(det_root.App.WaveformControl.AsicInjEn,   False       )
            write_to_detector(det_root.App.VINJ_DAC.enable,             False       )
        
        
        # Remove the yml files
        for f in tmpfiles:
            os.remove(f)
                
    logging.info('config_expert complete')
    
def reset_counters(base: dict):
    # Reset the timing counters
    base['cam'].App.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
    base['cam'].App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

#
#  Called on Configure
#
def epixUHR_config(base,connect_str, cfgtype,detname,detsegm,rog) -> list:
    global orig_cfg
    global group
    global seg_ids
    global gain_map
    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    #cfg['expert']['pixelBitMaps']['_7_on_the_fly']=0

    orig_cfg = cfg
    
    
    #  Translate user settings to the expert fields
    writeCalibRegs=user_to_expert(base, cfg, fullConfig=True)
    
    
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
    det_root = base['cam']
    firmwareVersion = det_root.Core.AxiVersion.FpgaVersion.get()

    orig_cfg = cfg

    topname = cfg['detName:RO'].split('_')

    seg_cfg = {}
    seg_ids = {}

    #  Rename the complete config detector
    seg_cfg[0] = cfg.copy()
    seg_cfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    #for seg in range(1):
        #  Construct the ID
    digitalId =  0 if base['pcie_timing'] else det_root.App.GTReadoutBoardCtrl.DigitalBoardId.get()
                 #0 if base['pcie_timing'] else det_root.App.RegisterControlDualClock.DigIDHigh.get()]
    pwrCommId =  0 if base['pcie_timing'] else det_root.App.GTReadoutBoardCtrl.AnalogBoardId.get()
        #             0 if base['pcie_timing'] else det_root.App.RegisterControlDualClock.PowerAndCommIDHigh.get()]
    carrierId =  0 if base['pcie_timing'] else det_root.App.GTReadoutBoardCtrl.CarrierBoardId.get()
        #             0 if base['pcie_timing'] else det_root.App.RegisterControlDualClock.CarrierIDHigh.get()]

        
    id = '%010d-%010d-%010d-%010d'%(firmwareVersion,
                                    carrierId,
                                    digitalId,
                                    pwrCommId)

    seg_ids[0] = id
    top = cdict()
    top.setAlg('config', ALG_VERSION)
    top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=int(topname[-1]), detId=id, doc='No comment')
    
    #top.set(f'gainCSVAsic' , gain_map.tolist(), 'UINT8')  # only the rows which have readable pixels
    top.set(f'gainMap'    , gain_map.tolist(), 'UINT8')        
    
    seg_cfg[1] = top.typed_json()
    
    result = []
    for i in SEGLIST:
        logging.debug('json seg {}  detname {}'.format(i, seg_cfg[i]['detName:RO']))
        result.append( json.dumps(sanitize_config(seg_cfg[i])) )
    
    base['cfg']    = copy.deepcopy(cfg)
    base['result'] = copy.deepcopy(result)
    logging.info("created gain values in XTC file")
    return result

def epixUHR_unconfig(base) -> dict:
    logging.info('epixUHR_unconfig')
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixUHR_scan_keys(update) -> list:
    logging.debug('epixUHR_scan_keys')
    global orig_cfg
    global base
    global seg_ids
    
    cfg = {}
    copy_reconfig_keys(cfg,orig_cfg,json.loads(update))
    # Apply to expert
    calib_regs_changed = user_to_expert(base,cfg,fullConfig=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,orig_cfg,key)
        copy_config_entry(cfg[':types:'],orig_cfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    seg_cfg = {}

    #  Rename the complete config detector
    seg_cfg[0] = cfg.copy()
    seg_cfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]
    
    
    if calib_regs_changed:

        for seg in range(1):
            id = seg_ids[seg]
            top = cdict()
            top.setAlg('config', ALG_VERSION)
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            top.set(f'gainMap' , gain_map.tolist(), 'UINT8')  # only the rows which have readable pixels
            seg_cfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(seg_cfg)):
        result.append( json.dumps(sanitize_config(seg_cfg[i])) )

    base['scan_keys'] = copy.deepcopy(result)
    if not check_json_keys(result, base['result']): # @todo: Too strict?
        logging.error('epixUHR_scan_keys json is inconsistent with that of epixUHR_config')

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixUHR_update(update) -> list:
    logging.debug('epixUHR_update')
    global orig_cfg
    global base
    global gain_map
    #  Queue full configuration next Configure transition
    base['cfg'] = None

    _stop(base)
    ##
    ##  Having problems with partial configuration
    ##
    # extract updates
    cfg = {}
    
    update_config_entry(cfg,orig_cfg,json.loads(update))
    #  Apply to expert
    
    writeCalibRegs = user_to_expert(base,cfg,fullConfig=False)
    logging.info(f'Partial config writeCalibRegs {writeCalibRegs}')
    
    config_expert(base, cfg, writeCalibRegs, second_pass=True)
    _start(base)

    #  Enable triggers to continue monitoring
#    epixUHR_internal_trigger(base)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,orig_cfg,key)
        copy_config_entry(cfg[':types:'],orig_cfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    seg_cfg = {}

    #  Rename the complete config detector
    seg_cfg[0] = cfg.copy()
    seg_cfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    if writeCalibRegs:
        for seg in range(1):
            id = seg_ids[seg]
            top = cdict()
            top.setAlg('config', ALG_VERSION)
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            
            top.set(f'gainMap' , gain_map.tolist(), 'UINT8')  

            seg_cfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(seg_cfg)):
        result.append( json.dumps(sanitize_config(seg_cfg[i])) )

    logging.info('update complete')

    return result

def epixUHR_external_trigger(base):
    #  Switch to external triggering
    logging.info("external triggering")
    det_root = base['cam']
    det_root.App.TriggerRegisters.SetTimingTrigger()

#checks if value is already set, then sets it, and check if it has been set
#reading is faster than writing, therefor if it is already set initialization is faster
#Considering introducing the function directly in Root
def write_to_detector(var, val):
#   Try writing all the time
#    var.set(val)
#    return
#
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

    #  Switch to internal triggering
    det_root = base['cam']
    det_root.App.TriggerRegisters.StartAutoTrigger()
    
def epixUHR_enable(base):
    logging.info('epixUHR_enable')
    epixUHR_external_trigger(base)
    _start(base)

def epixUHR_disable(base):
    logging.info('epixUHR_disable')
    # Prevents transitions going through: epixUHR_internal_trigger(base)

def _stop(base):
    logging.info('_stop')
    det_root = base['cam']
    det_root.App.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    logging.info('_start')
    det_root = base['cam']
    det_root.App.SetTimingTrigger()
    
            # Get devices
    event_builder = det_root.App.find(typ=batcher.AxiStreamBatcherEventBuilder)
    
    for devPtr in event_builder:
        devPtr.Blowoff.set(False)
    
        devPtr.SoftRst()

    # Turn on the triggering
    write_to_detector(det_root.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable, True)
    write_to_detector(det_root.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].MasterEnable, True)
    

    # Update the run state status variable
    write_to_detector(det_root.App.RunState, True)  

   
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

