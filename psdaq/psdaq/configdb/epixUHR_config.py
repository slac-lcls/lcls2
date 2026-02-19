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
pv = None

chan = None
group = None
origcfg = None

segids = None
seglist = [0,1]
asics = None

DET_SIZE = (4, 168, 192)  # 4 ASICs of 168x192 pixels each

#  Timing delay scans can be limited by this
EVENT_BUILDER_TIMEOUT = 0 #4*int(1.0e-3*156.25e6)
#this is supposed to be constant for every detector
DELTA_DELAY = -192

#path were to store cvs files for gain definition
GAIN_PATH = '/tmp/ePixUHR_GTReadout_default_'
#path were to store pll files used for calibration
PLL_PATH = '/tmp/'

ALG_VERSION = [3,2,0]


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

#Initialization of ASICs, this happens after getting configdb data because we need to know which ASIC to init
def panel_ASIC_init(det_root, asics):
    
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
def panel_init(det_root):
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
def epixUHR_init(arg,dev='/dev/datadev_0',lanemask=0xf,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv
    
    global gainMapSelection
    global gainValSelection
    
    #used to store gain configuration
    gainMapSelection=np.zeros(DET_SIZE)
    gainValSelection=np.zeros(4)
    
    logging.getLogger().setLevel(logging.WARNING)
#    logging.getLogger().setLevel(logging.INFO)
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
def epixUHR_connectionInfo(base, alloc_json_str):

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

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixUHRid

    return d

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the origcfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, fullConfig=False):
    global origcfg
    global group
    global lane

    det_root = base['cam']
    
    d = {}
    hasUser = 'user' in cfg
    
    #there are scripts that do not have user
    if (hasUser and 'start_ns' in cfg['user']):
        #rtp = origcfg['user']['run_trigger_group'] # run trigger partition
        
        #for i,p in enumerate([rtp,group]):
        partitionDelay = getattr(det_root.App.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']

        #
        #  The EPIX DAQ trigger source is the DAQ readout group
        #  The EPIX DAQ trigger delay is user.start_ns - L0Delay (readout group)
        #
        triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
        logging.warning(f'partitionDelay[{group}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
        if triggerDelay < 0:
            logging.error(f'partitionDelay[{group}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            logging.error('Raise start_ns >= {:}'.format(partitionDelay*base['msg_period']*base['clk_period']))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay']=triggerDelay
        
        triggerDelay=int(rawStart/base["clk_period"]) - DELTA_DELAY
        
        if triggerDelay < 0:
            logging.error(f'partitionDelay[{group+1}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            logging.error('Raise start_ns >= {:}'.format(partitionDelay*base['msg_period']*base['clk_period']))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].Delay'] = triggerDelay
        logging.warning(f'partitionDelay[{group+1}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')

        if fullConfig:
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition']= group+1    # Run trigger
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition']= group  # DAQ trigger
    
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
    if not secondPass:
        #Load Pll config values
        Pll_sel=[None, '_temp250', '_2_3_7', '_0_5_7', '_2_3_9', '_0_5_7_v2']
        if pll.enable.get() == False:
            pll.enable.set(True)
        
        clk = cfg['user']['PllRegistersSel'] 
        
        freq = Pll_sel[clk]
        logging.info(f"Loading PLL file: {freq}")
        
        pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
        fn = PLL_PATH+'PllConfig'+'.csv'
        np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
        
        tmpfiles.append(fn)
        setattr(det_root,"filenamePLL", fn)
        
        pll.LoadCsvFile(PLL_PATH+'PllConfig'+'.csv')

        panel_ASIC_init(det_root, asics)
               
    base['bypass']   = det_root.numOfAsics * [0x2]  # bitposition enables ['Bypass','Timeout','Blowoff',]
    base['batchers'] = det_root.numOfAsics * [1]  # list of active batchers
    
    for asic in range(det_root.numOfAsics):
        if asic+1 in asics: 
            base['bypass'][asic] = 0
        
        write_to_detector(getattr(det_root.App, f'BatcherEventBuilder{i+1}').Bypass, base['bypass'][i])
       
    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    
    for asic in asics:
        write_to_detector(getattr(det_root.App, f'BatcherEventBuilder{asic}').Timeout, EVENT_BUILDER_TIMEOUT) # 400 us
    if not base['pcie_timing']:
        eventBuilder = det_root.find(typ=batcher.AxiStreamBatcherEventBuilder)
        for eb in eventBuilder:
            eb.Timeout.set(EVENT_BUILDER_TIMEOUT)
            eb.Blowoff.set(True)
    
    if app is not None and not secondPass:
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
        timingOutEnum=['asicR0', 'asicACQ', 'asicSRO', 'asicInj', 'asicGlbRstN', 'timingRunTrigger', 'timingDaqTrigger', 'acqStart', 'dataSend', '_0', '_1']
        timingOutMux0_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux0'])
        timingOutMux1_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux1'])
        timingOutMux3_Sel=int(app['GTReadoutBoardCtrl']['TimingOutMux3'])
        logging.info(f'Setting timingOutMux0 to {timingOutEnum[timingOutMux0_Sel]}')
        logging.info(f'Setting timingOutMux1 to {timingOutEnum[timingOutMux1_Sel]}')
        logging.info(f'Setting timingOutMux3 to {timingOutEnum[timingOutMux3_Sel]}')
        
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
                    
        csvCfg = 0
    
    if writeCalibRegs:
        gainValue = 0
        pixelBitMapDic = ['_0_default', '_1_injection_truck', '_2_injection_corners_FHG', '_3_injection_corners_AHGLG1', '_4_extra_config', '_5_extra_config', '_6_truck2', '_7_on_the_fly', ]
        
        gainMapSelection=np.zeros(DET_SIZE)
        gainValSelection=np.zeros(4)
        
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
                PixMapSel = int(cfg['user']['Gain']['PixelBitMapSel'])
                
                PixMapSelected= pixelBitMapDic[PixMapSel]
                
                if ('on_the_fly' not in PixMapSelected):
                    fn = PLL_PATH+'csvConfig'+'.csv'
                    csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (DET_SIZE[1], DET_SIZE[2]))
                    np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='', fmt='%d')
                    tmpfiles.append(fn)
                    
                else:
                    fn = PLL_PATH+'onthefly.csv'    
                    csvCfg = np.loadtxt(PLL_PATH+'onthefly.csv', dtype='uint16', delimiter=',')
                for asic in asics: 
                    print(f"ASIC{asic}")
                    eval(f"det_root.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic{asic}('{fn}')")
                                        
                    logging.info(f"{PixMapSelected} CSV File Loaded")
                    gainMapSelection[asic-1,:,:]=csvCfg

            else:
                #same value for all
                logging.info("Use single value for all ASICS")
                gainValue=str(cfg['user']['Gain']['SetGainValue'])
                
                for asic in asics: 
                    print(f"ASIC{asic}")
                    gainValSelection[asic-1]=gainValue
                    getattr(det_root.App,f"Asic{asic}").progPixelMatrixConstantValue(gainValue)
        else:
            logging.info("Set single Gain per ASIC")
            if ( cfg['user']['Gain']['UsePixelMap']):
                #a map per each
                logging.info("Use a Pixel MAP per each ASIC")
                for asic in asics:
                    print(f"ASIC{asic}")
                    PixMapSel = cfg['user']['App'][f'Asic{asic}']['PixelBitMapSel']    
                    PixMapSelected= pixelBitMapDic[PixMapSel]
                    print(PixMapSelected)
                    if ('on_the_fly' not in PixMapSelected):
                        csvCfg = np.reshape(cfg['expert']['pixelBitMaps'][PixMapSelected], (DET_SIZE[1], DET_SIZE[2]))
                        fn = PLL_PATH+f'csvConfigAsic{asic}'+'.csv'
                        np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='')
                        tmpfiles.append(fn)
                    else:
                        fn = PLL_PATH+'onthefly.csv'
                        csvCfg = np.loadtxt(f'{PLL_PATH}onthefly.csv', dtype='uint16', delimiter=',')
                    gainMapSelection[i-1,:,:]=csvCfg
                    eval(f"det_root.App.EpixUhrMatrixConfig.progPixelMatrixFromCsvAsic{asic}('{fn}')")
                    
            else:
                #a value per each
                logging.info("Use a value per ASIC")
                for asic in asics: 
                    gainValue=str(cfg['user']['App'][f'Asic{i}']['SetGainValue'])
                    print(f"ASIC{asic}")
                    gainValSelection[asic-1]=gainValue
                    getattr(det_root.App,f"Asic{asic}").progPixelMatrixConstantValue(gainValue)
        
        # deactivate gain modification            
        for asic in asics: write_to_detector(getattr(det_root.App,f"Asic{i}").PixNumModeEn, False)
        
        #Charge Injection definitions
        if(cfg['user']['App']['VINJ_DAC']['enable']==1):
            write_to_detector(det_root.App.WaveformControl.InjEn,  True   )
            write_to_detector(det_root.App.VINJ_DAC.enable,        True   )
            write_to_detector(det_root.App.VINJ_DAC.dacEn,         True   )
            
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
            write_to_detector(det_root.App.VINJ_DAC.dacEn,         False       )	
            write_to_detector(det_root.App.VINJ_DAC.rampEn,        False       )
            write_to_detector(det_root.App.WaveformControl.InjEn,  False       )
            write_to_detector(det_root.App.VINJ_DAC.enable,        False       )
        
        
        # Remove the yml files
        for f in tmpfiles:
            os.remove(f)
                
    logging.info('config_expert complete')
    
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

    origcfg = cfg

    topname = cfg['detName:RO'].split('_')

    segcfg = {}
    segids = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

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

    segids[0] = id
    top = cdict()
    top.setAlg('config', ALG_VERSION)
    top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=int(topname[-1]), detId=id, doc='No comment')
    
    top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
    top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')        
    
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
    calibRegsChanged = user_to_expert(base,cfg,fullConfig=False)
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

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', ALG_VERSION)
            top.setInfo(detType='epixuhr', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            top.set(f'gainCSVAsic' , gainMapSelection.tolist(), 'UINT8')  # only the rows which have readable pixels
            top.set(f'gainAsic'    , gainValSelection.tolist(), 'UINT8')
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
    
    writeCalibRegs = user_to_expert(base,cfg,fullConfig=False)
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
        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', ALG_VERSION)
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
    det_root = base['cam']
    write_to_detector(det_root.App.RegisterControlDualClock.ResetCounters, 1)
    time.sleep(1.e6)
    write_to_detector(det_root.App.RegisterControlDualClock.ResetCounters, 0)

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
    eventBuilder = det_root.App.find(typ=batcher.AxiStreamBatcherEventBuilder)
    
    for devPtr in eventBuilder:
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

