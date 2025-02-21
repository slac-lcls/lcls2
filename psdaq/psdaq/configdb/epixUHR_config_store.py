'''python epixUHR_config_store.py --user tstopr --inst tst --alias BEAM --name epixuhr --segm 0 --id epixuhr_serial1234 --prod --update'''


from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import update_config
import numpy as np
import sys
import IPython
import argparse
import functools
import yaml
import pprint

numAsics = 4
elemRows = 168
elemCols = 192

import os


def recursive_list(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_list(value)
        else:
            yield (key, value)


def epixUHR_cdict():

    top = cdict()
    top.setAlg('config', [1,1,0])
    top.define_enum('boolEnum', {'False':0, 'True':1})
    top.set("expert.Core.Si5345Pll.enable",						        1   ,	'boolEnum')
    
    for n in range(1, 5):
        top.set(f"expert.App.Asic{n}.enable",							1   ,	"boolEnum")
        top.set(f"expert.App.Asic{n}.DacVthr",							53	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVthrGain",						3	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVfiltGain",						2	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVfilt",							30	,	'UINT8'   )	
        top.set(f"expert.App.Asic{n}.DacVrefCdsGain",					2	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVrefCds",						44	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVprechGain",				    2	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.DacVprech",						34	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.CompEnGenEn",						1	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.CompEnGenCfg",						5	,	'UINT8'   )
        top.set(f"expert.App.Asic{n}.BiasPxlCsa",                       0   ,   "UINT8"   )						
        top.set(f"expert.App.Asic{n}.BiasPxlBuf",                       0   ,   "UINT8"   )						
        top.set(f"expert.App.Asic{n}.BiasAdcComp",                      0   ,   "UINT8"   )						
        top.set(f"expert.App.Asic{n}.BiasAdcRef",                       0   ,   "UINT8"   )
        top.set(f"expert.App.BatcherEventBuilder{n}.enable",			1   ,	"boolEnum")
        top.set(f"expert.App.BatcherEventBuilder{n}.Timeout",			0	,   'UINT8'   )
        
    conv = functools.partial(int, base=16)
    pathPll='/cds/home/m/melchior/git/EVERYTHING_EPIX_UHR/epix-uhr-gtreadout-dev/software/config/pll/'

    base = 'expert.Pll.'
    conv = functools.partial(int, base=16)
    
    top.set(base+'_temp250',    np.loadtxt(pathPll+'PLLConfig_Si5345_temp_250.csv',                  dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_2_3_7',      np.loadtxt(pathPll+'Si5345-B-156MHZ-out2-3-7-Registers.csv',         dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_0_5_7',      np.loadtxt(pathPll+'Si5345-B-156MHZ-out-0-5-and-7-Registers.csv',    dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_2_3_9',      np.loadtxt(pathPll+'Si5345-B-156MHZ-out2-3-9.csv',                   dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_0_5_7_v2',   np.loadtxt(pathPll+'Si5345-B-156MHZ-out-0-5-and-7-v2-Registers.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
            
    pathpix='/cds/home/m/melchior/git/EVERYTHING_EPIX_UHR/epix-uhr-gtreadout-dev/software/config/pixelBitMaps_prod/'
    #pixelBitMapDic = {'_FL_FM_FH':0, '_FL_FM_FH_InjOff':1, '_allConfigs':2, '_allPx_52':3, '_allPx_AutoHGLG_InjOff':4, '_allPx_AutoHGLG_InjOn':5, '_allPx_AutoMGLG_InjOff':6, '_allPx_AutoMGLG_InjOn':7, '_allPx_FixedHG_InjOff':8, '_allPx_FixedHG_InjOn':9, '_allPx_FixedLG_InjOff':10, '_allPx_FixedLG_InjOn':11, '_allPx_FixedMG_InjOff':12, '_allPx_FixedMG_InjOn':13, '_crilin':14, '_crilin_epixuhr100k':15, '_defaults':16, '_injection_corners':17, '_injection_corners_px1':18, '_management':19, '_management_epixuhr100k':20, '_management_inj':21, '_maskedCSA':22, '_truck':23, '_truck_epixuhr100k':24, '_xtalk_hole':25}
    pixelBitMapDic = {'_0_default':0, '_1_injection_truck':1, '_2_injection_corners_FHG':2, '_3_injection_corners_AHGLG1':3, '_4_extra_config':4, '_5_extra_config':5, '_6_truck2':6, }
    top.define_enum('pixelMapEnum', pixelBitMapDic)
    
    base = 'expert.pixelBitMaps.'
    for pixelmap in pixelBitMapDic:
        top.set(base+pixelmap, np.loadtxt(f'{pathpix}{pixelmap[1:]}.csv', dtype='uint16', delimiter=','))
    for n in range(1, 5):
        base = f'expert.App.Asic{n}.'
        top.set(base+'PixelBitMapSel', 5, 'pixelMapEnum')
        top.set(base+"SetGainValue",							    48              ,'UINT8'   )    
        
    top.set("expert.App.WaveformControl.enable",					1			  	,'boolEnum')
    top.set("expert.App.WaveformControl.GlblRstPolarity",			1			  	,'boolEnum')
    top.set("expert.App.WaveformControl.SR0Polarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.SR0Delay",				    1195			,'UINT32'  )
    top.set("expert.App.WaveformControl.SR0Width",				    1    			,'UINT8'   )
    top.set("expert.App.WaveformControl.AcqPolarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.AcqDelay",				    655				,'UINT32'  )
    top.set("expert.App.WaveformControl.AcqWidth",				    535				,'UINT32'  )
    top.set("expert.App.WaveformControl.R0Polarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.R0Delay",					70 			  	,"UINT32"  )
    top.set("expert.App.WaveformControl.R0Width",					1125			,"UINT32"  )
    top.set("expert.App.WaveformControl.InjPolarity",				1			  	,"boolEnum")
    top.set("expert.App.WaveformControl.InjDelay",				    660				,"UINT32"  )
    top.set("expert.App.WaveformControl.InjWidth",				    535				,"UINT32"  )
    top.set("expert.App.WaveformControl.InjEn",					    0			  	,"boolEnum")
    top.set("expert.App.WaveformControl.InjSkipFrames",			    0				,"UINT32"  )
    
    top.set("expert.App.TriggerRegisters.enable",					1			  	,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerEnable",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerDelay",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerEnable",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerDelay",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.TimingRunTriggerEnable",	0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.TimingDaqTriggerEnable",	0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerDelay",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerDelay",		    0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoRunEn",				0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoDaqEn",				0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoTrigPeriod",			42700000		,"UINT32"  )
    top.set("expert.App.TriggerRegisters.numberTrigger",			0				,"UINT8"   )
    top.set("expert.App.TriggerRegisters.PgpTrigEn",				0				,"boolEnum")
    
    top.set("expert.App.GTReadoutBoardCtrl.enable",				    1			  	,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard",	1               ,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn0",			0				,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn1",			0				,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn2",			0				,"boolEnum")
    top.set("expert.App.AsicGtClk.enable",						    1			  	,"boolEnum")
    top.set("expert.App.AsicGtClk.gtResetAll",					    0				,"boolEnum")
    for n in range(1, 5):
        top.set(f"expert.App.AsicGtData{n}.enable",					1			  	,"boolEnum")
        top.set(f"expert.App.AsicGtData{n}.gtStableRst",			0				,"boolEnum")
    
    top.set("user.App.VCALIBP_DAC.enable",                                      0				,"boolEnum")
    top.set("user.App.VCALIBP_DAC.dacEn",                                       0				,"boolEnum")
    top.set("user.App.VCALIBP_DAC.dacSingleValue",                              0				,"UINT32")
    top.set("user.App.VCALIBP_DAC.rampEn",                                      0				,"boolEnum")
    top.set("user.App.VCALIBP_DAC.dacStartValue",                               0				,"UINT32")
    top.set("user.App.VCALIBP_DAC.dacStopValue",                                0				,"UINT32")
    top.set("user.App.VCALIBP_DAC.dacStepValue",                                0				,"UINT32")
    top.set("user.App.VCALIBP_DAC.resetDacRamp",                                0				,"boolEnum")
    
    top.set("user.App.VINJ_DAC.enable",                                      0				,"boolEnum")
    top.set("user.App.VINJ_DAC.dacEn",                                       0				,"boolEnum")
    top.set("user.App.VINJ_DAC.dacSingleValue",                              0				,"UINT32")
    top.set("user.App.VINJ_DAC.rampEn",                                      0				,"boolEnum")
    top.set("user.App.VINJ_DAC.dacStartValue",                               0				,"UINT32")
    top.set("user.App.VINJ_DAC.dacStopValue",                                0				,"UINT32")
    top.set("user.App.VINJ_DAC.dacStepValue",                                0				,"UINT32")
    top.set("user.App.VINJ_DAC.resetDacRamp",                                0				,"boolEnum")
    
    
    top.set("user.App.ADS1217.enable",							    0				,"boolEnum")
    top.set("user.App.ADS1217.adcStartEnManual",				    0  				,"boolEnum")
    
    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- expert interface --"
    help_str += "\nstart_ns     : exposure start (nanoseconds)"
    top.set("help:RO", help_str, 'CHARSTR')


    top.set("user.start_ns" , 106000, 'UINT32') # taken from epixHR
    
    top.define_enum('PllRegEnum', {'temp250':1, '2_3_7':2, '0_5_7':3, '2_3_9':4, '0_5_7_v2':5})
    base = 'user.'
    
    top.set(base+'PllRegistersSel', 5, 'PllRegEnum')

    base = "user.Gain."
    top.set(base+"SetSameGain4All",			            			    0               ,"boolEnum")
    top.set(base+"UsePixelMap",             						    0               ,"boolEnum")

    top.set(base+"SetGainValue",							            48              ,'UINT8'   )
    top.set(base+'PixelBitMapSel',                                      5               , 'pixelMapEnum')

    #top.set("user.run_trigger_group",                                   6               ,'UINT32'  )
    top.set("user.asic_enable", (1<<numAsics)-1, 'UINT32')
    
    
    # timing system
    # run trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold'              ,16,'UINT32'    )
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay'                ,42,'UINT32'    )
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerSource'               ,1,'UINT32'     )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2ChannelReg[0].RateType'     ,2,'UINT32'     )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2ChannelReg[0].RateSel'      ,256,'UINT32'   )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2ChannelReg[0].EnableReg'    ,1,'UINT32'     )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2ChannelReg[0].DestType'     ,2,'UINT32'     )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].EnableTrig'   ,1,'UINT32'     )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].DelayDelta'   ,1185,'UINT32' )
    top.set('expert.App.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[0].Delay'        ,1850,'UINT32' )
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition'                   ,7,'UINT32'     )
    # daq trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].PauseThreshold'              ,16,'UINT32'    )
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay'                ,42,'UINT32'    )
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition'                   ,0,'UINT32'     )
    
    
    return top

if __name__ == "__main__":
    create = True # True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.
    
    args = cdb.createArgs().args
    
    db = 'configdb' if args.prod else 'devconfigdb'
    
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,    
                        root=dbname, user=args.user, password=args.password)

    top = epixUHR_cdict()
    top.setInfo('epixuhrhw', args.name, args.segm, args.id, 'No comment')


   # no  need for update, value are loaded at creation 
    if args.update:
        cfg = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)
        top = update_config(cfg, top.typed_json(), args.verbose)

    if not args.dryrun:
        if create:
            mycdb.add_alias(args.alias)
            mycdb.add_device_config('epixuhrhw')
        mycdb.modify_device(args.alias, top)

    