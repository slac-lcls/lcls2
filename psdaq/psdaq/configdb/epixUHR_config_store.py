'''python epixUHR_config_store.py --user tstopr --inst tst --alias BEAM --name epixuhr --segm 0 --id epixuhr_serial1234'''


from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
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
    top.setAlg('config', [0,1,0])
    top.define_enum('boolEnum', {'False':0, 'True':1})
    top.set("expert.Core.Si5345Pll.enable",						1,	'boolEnum')
    
    for n in range(1, 5):
        print(n)
        top.set(f"expert.App.Asic{n}.enable",							1   ,	"boolEnum")
        top.set(f"expert.App.Asic{n}.DacVthr",							52	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVthrGain",						2	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVfiltGain",						2	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVfilt",							28	,	'UINT8')	
        top.set(f"expert.App.Asic{n}.DacVrefCdsGain",					    2	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVrefCds",						    44	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVprechGain",						2	,	'UINT8')
        top.set(f"expert.App.Asic{n}.DacVprech",							34	,	'UINT8')
        top.set(f"expert.App.Asic{n}.CompEnGenEn",						1	,	'UINT8')
        top.set(f"expert.App.Asic{n}.CompEnGenCfg",						5	,	'UINT8')
        top.set(f"expert.App.Asic{n}.PixNumModeEn",						0	,	'boolEnum')
        # PixNumModeEn 0 takes data, 1 is buffered image
        #top.set(f"expert.App.Asic{n}.CsvFilePath",						""	,	'CHARSTR')
        #top.set(f"expert.App.Asic{n}.LoadCsvPixelBitmap",					    ,	        )
        #top.set(f"expert.App.Asic{n}.SetPixelBitmap",						    ,	        )
        # add a label SetAllMatrix to get value for comand (default 48)
        top.set(f"expert.App.Asic{n}.SetAllMatrix",							48,	  'UINT8')
        top.set(f"expert.App.BatcherEventBuilder{n}.enable",				1,	    "boolEnum")
        top.set(f"expert.App.BatcherEventBuilder{n}.Bypass",				0		,'UINT8')
        top.set(f"expert.App.BatcherEventBuilder{n}.Timeout",			    0		,'UINT8')
        top.set(f"expert.App.BatcherEventBuilder{n}.Blowoff",			    0,     'boolEnum')
        
        
    conv = functools.partial(int, base=16)
    path='/cds/home/m/melchior/git/EVERYTHING_EPIX_UHR/epix-uhr-gtreadout-dev/software/config/pll/'
    top.set('expert.Pll', np.loadtxt(path+'Si5345-B-156MHZ-out-0-5-and-7-v2-Registers.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set("expert.App.WaveformControl.enable",					1			  	,'boolEnum')
    top.set("expert.App.WaveformControl.GlblRstPolarity",			1			  	,'boolEnum')
    top.set("expert.App.WaveformControl.SR0Polarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.SR0Delay",				1195			  	,'UINT32')
    top.set("expert.App.WaveformControl.SR0Width",				1    			  	,'UINT8')
    top.set("expert.App.WaveformControl.AcqPolarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.AcqDelay",				655				  	,'UINT32')
    top.set("expert.App.WaveformControl.AcqWidth",				535				  	,'UINT32')
    top.set("expert.App.WaveformControl.R0Polarity",				0			  	,'boolEnum')
    top.set("expert.App.WaveformControl.R0Delay",					70 				  	,"UINT32")
    top.set("expert.App.WaveformControl.R0Width",					1125			  	,"UINT32")
    top.set("expert.App.WaveformControl.InjPolarity",				1			  	,"boolEnum")
    top.set("expert.App.WaveformControl.InjDelay",				660				  	,"UINT32")
    top.set("expert.App.WaveformControl.InjWidth",				535				  	,"UINT32")
    top.set("expert.App.WaveformControl.InjEn",					0			  	,"boolEnum")
    top.set("expert.App.WaveformControl.InjSkipFrames",			0				  	,"UINT32")
    
    top.set("expert.App.TriggerRegisters.enable",					1			  	,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerEnable",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerDelay",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerEnable",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerDelay",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.TimingRunTriggerEnable",	0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.TimingDaqTriggerEnable",	0				,"boolEnum")
    top.set("expert.App.TriggerRegisters.RunTriggerDelay",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.DaqTriggerDelay",		0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoRunEn",				0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoDaqEn",				0					,"boolEnum")
    top.set("expert.App.TriggerRegisters.AutoTrigPeriod",			42700000				,"UINT32")
    top.set("expert.App.TriggerRegisters.numberTrigger",			0						,"UINT8")
    top.set("expert.App.TriggerRegisters.PgpTrigEn",				0					,"boolEnum")
    
    top.set("expert.App.GTReadoutBoardCtrl.enable",				1			  	    ,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.pwrEnableAnalogBoard",	1,                   "boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn0",			0					,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn1",			0					,"boolEnum")
    top.set("expert.App.GTReadoutBoardCtrl.timingOutEn2",			0					,"boolEnum")
   # top.set("expert.App.GTReadoutBoardCtrl.DigitalBoardId",		ReadOnly				 ,)
   # top.set("expert.App.GTReadoutBoardCtrl.AnalogBoardId",		ReadOnly				,)
   # top.set("expert.App.GTReadoutBoardCtrl.CarrierBoardId",	 	ReadOnly				,)
    top.set("expert.App.AsicGtClk.enable",						1			  	,"boolEnum")
    top.set("expert.App.AsicGtClk.gtResetAll",					0					,"boolEnum")
    for n in range(1, 5):
        top.set(f"expert.App.AsicGtData{n}.enable",						1			  	,"boolEnum")
        top.set(f"expert.App.AsicGtData{n}.gtStableRst",					0					,"boolEnum")
    top.set("user.App.VINJ_DAC.enable",							0					,"boolEnum")
    top.set("user.App.VINJ_DAC.SetValue",						0						,"UINT32")
    top.set("user.App.ADS1217.enable",							0					,"boolEnum")
    top.set("user.App.ADS1217.adcStartEnManual",				0					,"boolEnum")

    pixelMap = np.zeros((elemRows*2,elemCols*2), dtype=np.uint8)
    top.set("expert.pixel_map", pixelMap)
    
    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- expert interface --"
    help_str += "\nstart_ns     : exposure start (nanoseconds)"
    top.set("help:RO", help_str, 'CHARSTR')

    # set to 88000 to get triggerDelay larger than zero when
    # L0Delay is 81 (used by TMO)
    top.set("user.start_ns" , 106000, 'UINT32') # taken from epixHR
    #top.set("expert.gate_ns" , 154000, 'UINT32') # taken from lcls1 xpptut15 run 260
    #add daqtriggerdelay and runtriggerdelay?
    
    top.set("user.run_trigger_group", 1, 'UINT32')
    top.set("user.asic_enable", (1<<numAsics)-1, 'UINT32')
    # timing system
    # run trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold',16,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay',42,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition',0,'UINT32')
    # daq trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].PauseThreshold',16,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay',42,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition',0,'UINT32')

    #top.define_enum('clkEnum', {'Si5345-B-156MHZ-out-0-5-and-7':1, 'Si5345-B-156MHZ-out-0-5-and-7-v2':2, 'Si5345-B-156MHZ-out2-3-7':3, 'Si5345-B-156MHZ-out2-3-9':4 , 'Config_Si5345_temp_250':5})
    #base = 'expert.Pll.'
    #top.set(base+'Clock', 5, 'clkEnum')
    #conv = functools.partial(int, base=16)
    #prjCfg='/cds/home/m/melchior/git/EVERYTHING_EPIX_UHR/epix-uhr-gtreadout-dev/software/config/pll'
    #top.set(base+'Si5345-B-156MHZ-out-0-5-and-7', np.loadtxt(prjCfg+'/Si5345-B-156MHZ-out-0-5-and-7-Registers.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    #top.set(base+'Si5345-B-156MHZ-out-0-5-and-7-v2', np.loadtxt(prjCfg+'/Si5345-B-156MHZ-out-0-5-and-7-v2-Registers.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    #top.set(base+'Si5345-B-156MHZ-out2-3-7', np.loadtxt(prjCfg+'/Si5345-B-156MHZ-out2-3-7-Registers.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    #top.set(base+'Si5345-B-156MHZ-out2-3-9', np.loadtxt(prjCfg+'/Si5345-B-156MHZ-out2-3-9.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    #top.set(base+'Config_Si5345_temp_250', np.loadtxt(prjCfg+'/Config_Si5345_temp_250.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    return top

if __name__ == "__main__":
    create = False# True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args
    
    db = 'configdb' if args.prod else 'devconfigdb'
    
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,    
                        root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixuhr')

    top = epixUHR_cdict()
    top.setInfo('epixuhr', args.name, args.segm, args.id, 'No comment')
    
    mycdb.modify_device(args.alias, top)

