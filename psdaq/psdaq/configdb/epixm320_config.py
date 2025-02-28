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

import epix_hr_m_320k
import ePix320kM as ePixM
import epix_hr_leap_common as leapCommon
import surf.protocols.batcher as batcher

import time
import json
import os
import numpy as np
import IPython
import datetime
import logging
import copy # deepcopy

import pprint

rogue.Version.minVersion('5.14.0')

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

#  Register ordering matters, but our configdb does not preserve the order.
#  For now, put the ordering in code here, until the configdb can be updated to preserve order.
#  Underneath, it is just storing json, so order preservation should be possible.
ordering = {}
ordering['PowerControl'] = ['DigitalSupplyEn']
ordering['RegisterControlDualClock'] = ['IDreset',
                                        'GlblRstPolarityN',
                                        'ClkSyncEn',
                                        'RoLogicRstN',
                                        'SyncPolarity',
                                        'SyncDelay',
                                        'SyncWidth',
                                        'SR0Polarity',
                                        'SR0Delay1',
                                        'SR0Width1',
                                        'ePixAdcSHPeriod',
                                        'ePixAdcSHOffset',
                                        'AcqPolarity',
                                        'AcqDelay1',
                                        'AcqWidth1',
                                        'AcqDelay2',
                                        'AcqWidth2',
                                        'R0Polarity',
                                        'R0Delay',
                                        'R0Width',
                                        'PPbePolarity',
                                        'PPbeDelay',
                                        'PPbeWidth',
                                        'PpmatPolarity',
                                        'PpmatDelay',
                                        'PpmatWidth',
                                        'SaciSyncPolarity',
                                        'SaciSyncDelay',
                                        'SaciSyncWidth',
                                        'ResetCounters',
                                        'AsicPwrEnable',
                                        'AsicPwrManual',
                                        'AsicPwrManualDig',
                                        'AsicPwrManualAna',
                                        'AsicPwrManualIo',
                                        'AsicPwrManualFpga',
                                        'DebugSel0',
                                        'DebugSel1',
                                        'getSerialNumbers',
                                        'AsicRdClk',]

for i in range(4):
    ordering[f'Mv2Asic[{i}]'] = ['shvc_DAC',
                                 'fastPP_enable',
                                 'PulserSync',
                                 'Pll_RO_Reset',
                                 'Pll_Itune',
                                 'Pll_KVCO',
                                 'Pll_filter1LSB',
                                 'Pll_filter1MSB',
                                 'Pulser',
                                 'pbit',
                                 'atest',
                                 'test',
                                 'sab_test',
                                 'hrtest',
                                 'PulserR',
                                 'DigMon1',
                                 'DigMon2',
                                 'PulserDac',
                                 'MonostPulser',
                                 'RefGenB',
                                 'Dm1En',
                                 'Dm2En',
                                 'emph_bd',
                                 'emph_bc',
                                 'VRef_DAC',
                                 'VRefLow',
                                 'trbit',
                                 'TpsMux',
                                 'RoMonost',
                                 'TpsGr',
                                 'Balcony_clk',
                                 'PpOcbS2d',
                                 'Ocb',
                                 'Monost',
                                 'mTest',
                                 'Preamp',
                                 'S2D_1_b',
                                 'Vld1_b',
                                 'CompTH_DAC',
                                 'loop_mode_sel',
                                 'TC',
                                 'S2d',
                                 'S2dDacBias',
                                 'Tsd_Tser',
                                 'Tps_DAC',
                                 'PLL_RO_filter2',
                                 'PLL_RO_divider',
                                 'TestBe',
                                 'DigRO_disable',
                                 'DelExec',
                                 'DelCCKReg',
                                 'RO_rst_en',
                                 'SlvdsBit',
                                 'FE_Autogain',
                                 'FE_Lowgain',
                                 'DCycle_DAC',
                                 'DCycle_en',
                                 'DCycle_bypass',
                                 'Debug_bit',
                                 'OSRsel',
                                 'SecondOrder',
                                 'DHg',
                                 'RefGenC',
                                 'dbus_del_sel',
                                 'SDclk_b',
                                 'SDrst_b',
                                 'Filter_DAC',
                                 'Ref_gen_d',
                                 'CompEn',
                                 'Pixel_CB',
                                 'InjEn_ePixM',
                                 'ClkInj_ePixM',
                                 'rowCK2matrix_delay',
                                 'DigRO_disable_4b',
                                 'RefinN',
                                 'RefinCompN',
                                 'RefinP',
                                 'ro_mode_i',
                                 'SDclk1_b',
                                 'SDrst1_b',
                                 'SDclk2_b',
                                 'SDrst2_b',
                                 'SDclk3_b',
                                 'SDrst3_b',
                                 'CompTH_ePixM',
                                 'Precharge_DAC_ePixM',
                                 'FE_CLK_dly',
                                 'FE_CLK_cnt_en',
                                 'FE_ACQ2GR_en',
                                 'FE_sync2GR_en',
                                 'FE_ACQ2InjEn',
                                 'pipoclk_delay_row0',
                                 'pipoclk_delay_row1',
                                 'pipoclk_delay_row2',
                                 'pipoclk_delay_row3',]

for i in range(4):
    ordering[f'SspMonGrp[{i}]'] = []
    for j in range(24):
        ordering[f'SspMonGrp[{i}]'].extend([f'UsrDlyCfg[{j}]',])
    ordering[f'SspMonGrp[{i}]'].extend(['EnUsrDlyCfg',
                                        'MinEyeWidth',
                                        'LockingCntCfg',
                                        'BypFirstBerDet',
                                        'Polarity',
                                        'GearboxSlaveBitOrder',
                                        'GearboxMasterBitOrder',
                                        'MaskOffCodeErr',
                                        'MaskOffDispErr',
                                        'MaskOffOutOfSync',
                                        'LockOnIdleOnly',
                                        'RollOverEn',])
for i in range(4):
    ordering[f'DigAsicStrmRegisters{i}'] = ['asicDataReq',
                                            'DisableLane',
                                            'EnumerateDisLane',
                                            'FillOnFailEn',
                                            'FillOnFailPersistantDisable',
                                            'SroToSofTimeout',
                                            'DataTimeout',]

ordering['TriggerRegisters'] = ['RunTriggerEnable',
                                'TimingRunTriggerEnable',
                                'RunTriggerDelay',
                                'DaqTriggerEnable',
                                'TimingDaqTriggerEnable',
                                'DaqTriggerDelay',
                                'AutoRunEn',
                                'AutoDaqEn',
                                'AutoTrigPeriod',
                                'PgpTrigEn',
                                'numberTrigger',]
for i in range(4):
    ordering[f'BatcherEventBuilder{i}'] = ['Bypass','Timeout','Blowoff',]


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

def gain_mode_name(gain_mode):
    return ('SH', 'SL', 'AHL', 'User')[gain_mode]

def gain_mode_value(gain_mode):
    return ('SH', 'SL', 'AHL', 'User').index(gain_mode)

def gain_mode_map(gain_mode):
    compTH        = ( 0,   44,   24)[gain_mode] # SoftHigh/SoftLow/Auto
    precharge_DAC = (50,   50,   50)[gain_mode]
    return (compTH, precharge_DAC)

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

#
#  Initialize the rogue accessor
#
def epixm320_init(arg,dev='/dev/datadev_0',lanemask=0xf,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv
#    logging.getLogger().setLevel(40-10*verbosity) # way too much from rogue
    logging.getLogger().setLevel(30)
    logging.warning('epixm320_init')

    base = {}
    #  Connect to the camera and the PCIe card
    cbase = ePixM.Root(top_level              = '/tmp',
                       dev                    = '/dev/datadev_0',
                       pollEn                 = False,
                       initRead               = False,
                       justCtrl               = True,
                       fullRateDataReceiverEn = False)
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
        base['bypass'] = cbase.numOfAsics * [0x3]
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True

        logging.warning('epixm320_unconfig')
        epixm320_unconfig(base)

        cbase.App.TimingRx.TimingFrameRx.ModeSelEn.set(1) # UseModeSel
        cbase.App.TimingRx.TimingFrameRx.ClkSel.set(0)    # LCLS-1 Clock
        cbase.App.TimingRx.TimingFrameRx.RxDown.set(0)
    else:
        base['bypass'] = cbase.numOfAsics * [0x3]
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        base['pcie_timing'] = False

        logging.warning('epixm320_unconfig')
        epixm320_unconfig(base)

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
#    epixm320_internal_trigger(base)
    return base

#
#  Set the PGP lane
#
def epixm320_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixm320_connectionInfo(base, alloc_json_str):

#
#  To do:  get the IDs from the detector and not the timing link
#
    txId = timTxId('epixm320')
    logging.info('TxId {:x}'.format(txId))

    cbase = base['cam']
    rxId = cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    logging.info('RxId {:x}'.format(rxId))
    cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    epixmid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixmid

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

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        rtp = origcfg['user']['run_trigger_group'] # run trigger partition
        for i,p in enumerate([rtp,group]):
            partitionDelay = getattr(cbase.App.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%p).get()
            rawStart       = cfg['user']['start_ns']
            triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
            logging.warning(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
            if triggerDelay < 0:
                logging.error(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
                logging.error('Raise start_ns >= {:}'.format(partitionDelay*base['msg_period']*base['clk_period']))
                raise ValueError('triggerDelay computes to < 0')

            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[{i}].TriggerDelay']=triggerDelay

        if full:
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition']=rtp    # Run trigger
            d[f'expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition']=group  # DAQ trigger

    calibRegsChanged = False
    if hasUser and 'gain_mode'  in cfg['user']:
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==3:  # user's choices from the configDb
            for i in range(cbase.numOfAsics):
                d[f'expert.App.Mv2Asic[{i}].CompTH_ePixM']        = origcfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']
                d[f'expert.App.Mv2Asic[{i}].Precharge_DAC_ePixM'] = origcfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM']
        else:
            compTH, precharge_DAC = gain_mode_map(gain_mode)
            for i in range(cbase.numOfAsics):
                d[f'expert.App.Mv2Asic[{i}].CompTH_ePixM']        = compTH
                d[f'expert.App.Mv2Asic[{i}].Precharge_DAC_ePixM'] = precharge_DAC

        # For charge injection
        # Run and DAQ triggers to be enabled @ 10Hz. BOTH MUST BE 10HZ
        if 'expert' in cfg and cfg['expert']['App']['FPGAChargeInjection']['step'] != 0:
            d['expert.App.FPGAChargeInjection.startCol']    = cfg['expert']['App']['FPGAChargeInjection']['startCol']
            d['expert.App.FPGAChargeInjection.endCol']      = cfg['expert']['App']['FPGAChargeInjection']['endCol']
            d['expert.App.FPGAChargeInjection.step']        = cfg['expert']['App']['FPGAChargeInjection']['step']
            d['expert.App.FPGAChargeInjection.currentAsic'] = cfg['expert']['App']['FPGAChargeInjection']['currentAsic']
        calibRegsChanged = True

    update_config_entry(cfg,origcfg,d)

    return calibRegsChanged

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writeCalibRegs=True, secondPass=False):
    global asics  # Need to maintain this across configuration updates

    #  Disable internal triggers during configuration
    epixm320_external_trigger(base)

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
                asics.append(i)
            else:
                # remove the ASIC configuration so we don't try it
                del app['Mv2Asic[{}]'.format(i)]

    # Enable batchers for all ASICs.  Data will be padded.
    base['bypass'] = cbase.numOfAsics * [0x2]  # Enable Timing (bit-0) for all ASICs
    base['batchers'] = cbase.numOfAsics * [1]  # list of active batchers
    for i in asics:
        base['bypass'][i] = 0       # Enable Data (bit-1) only for for enabled ASICs
    print(f'=== configure bypass {base["bypass"]} ===')
    for i in range(cbase.numOfAsics):
        getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').Bypass.set(base['bypass'][i])

    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    for i in asics:
        getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').Timeout.set(EventBuilderTimeout) # 400 us
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
        path = '/tmp/ePixM320_'
        ePixMTypes = cfg[':types:']['expert']['App']
        tree = ('Root','App')
        tmpfiles = []
        def toYaml(sect,keys,name):
            if sect == tree[-1]:
                tmpfiles.append(dictToYaml(app,ePixMTypes,keys,cbase,path,name,tree,ordering))
            else:
                tmpfiles.append(dictToYaml(app[sect],ePixMTypes[sect],keys,cbase,path,name,(*tree,sect),ordering))

        clk = cfg['expert']['Pll']['Clock']
        if clk != 4:            # 4 is the Default firmware setting
            freq = [None,'_250_MHz','_125_MHz','_168_MHz'][clk]
            pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
            fn = path+'PllConfig'+'.csv'
            np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
            tmpfiles.append(fn)
            setattr(cbase, 'filenamePLL', fn)

        # Generate Yaml files for all ASICs
        toYaml('App',['PowerControl'],'PowerSupply')
        toYaml('App',[f'SspMonGrp[{i}]' for i in range(cbase.numOfAsics)],'DESER')
        toYaml('AsicTop',['RegisterControlDualClock'],'WaveForms')
        toYaml('AsicTop',[f'DigAsicStrmRegisters{i}' for i in range(cbase.numOfAsics)],'PacketReg')
        toYaml('AsicTop',[f'BatcherEventBuilder{i}' for i in range(cbase.numOfAsics)],'Batcher')
        setattr(cbase, 'filenameASIC',cbase.numOfAsics*[None]) # This one is a little different
        for i in range(cbase.numOfAsics):
            toYaml('App',[f'Mv2Asic[{i}]'],f'ASIC_u{i+1}')
            cbase.filenameASIC[i] = getattr(cbase,f'filenameASIC_u{i+1}')

        arg = [clk,0,0,0,0]
        for i in asics:
            arg[1+i] = 1
        logging.warning(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.fnInitAsicScript(None,None,arg)

        # Remove the yml files
        for f in tmpfiles:
            os.remove(f)

        # Adjust for intermitent lanes of enabled ASICs
        cbase.laneDiagnostics(arg[1:5], threshold=1, loops=0, debugPrint=False)

        # Delay determination needs laneDiagnostics. Exercising the lanes seem to stablize
        # the lanes better for later evaluating the best lanes delays.
        # Adding sleeps does not seem to suffice. Images need to be sent on the lanes.

        time.sleep(1)
        logging.info("Evaluating optimal delays")

        cbase.App.FPGADelayDetermination.Start()
        time.sleep(1)
        while (cbase.App.FPGADelayDetermination.Busy.get() != 0) :
            time.sleep(1)

        for i in range(cbase.numOfAsics):
            # Prevent disabled ASICs from participating by disabling their lanes
            # It seems like disabling their Batchers should be sufficient,
            # but that prevents transitions from going through
            if i not in asics:  # Override configDb's value for disabled ASICs
                getattr(cbase.App.AsicTop, f'DigAsicStrmRegisters{i}').DisableLane.set(0xffffff)

        # Enable the batchers for all ASICs
        for i in range(cbase.numOfAsics):
            getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').enable.set(base['batchers'][i] == 1)

    if writeCalibRegs:
        gain_mode = cfg['user']['gain_mode'] if 'gain_mode' in cfg['user'] else 3
        if gain_mode == 3:
            compTH        = []
            precharge_DAC = []
            for i in asics:
                asicName = f'Mv2Asic[{i}]'
                saci = getattr(cbase.App,asicName)
                saci.enable.set(True)

                if app is not None and asicName in app:
                    di = app[asicName]
                    compTH.append       (di['CompTH_ePixM'])
                    precharge_DAC.append(di['Precharge_DAC_ePixM'])
                    setSaci(saci.CompTH_ePixM,       'CompTH_ePixM',       di)
                    setSaci(saci.Precharge_DAC_ePixM,'Precharge_DAC_ePixM',di)

                saci.enable.set(False)
            print(f'Setting gain mode {gain_mode}:  CompTH {compTH},  Precharge_DAC {precharge_DAC}')
        else:
            compTH, precharge_DAC = gain_mode_map(gain_mode)
            print(f'Setting gain mode {gain_mode}:  CompTH {compTH},  Precharge_DAC {precharge_DAC}')

            for i in asics:
                saci = getattr(cbase.App,f'Mv2Asic[{i}]')
                saci.enable.set(True)
                saci.CompTH_ePixM.set(compTH)
                saci.Precharge_DAC_ePixM.set(precharge_DAC)
                saci.enable.set(False)

        #  Enable charge injection when pulser step is non-zero
        hasChgInj = app is not None and 'FPGAChargeInjection' in app
        if hasChgInj and app['FPGAChargeInjection']['step'] != 0:
            cbase.App.FPGAChargeInjection.startCol.set   (app['FPGAChargeInjection']['startCol'])
            cbase.App.FPGAChargeInjection.endCol.set     (app['FPGAChargeInjection']['endCol'])
            cbase.App.FPGAChargeInjection.step.set       (app['FPGAChargeInjection']['step'])
            cbase.App.FPGAChargeInjection.currentAsic.set(app['FPGAChargeInjection']['currentAsic'])
            cbase.App.FPGAChargeInjection.UseTiming.set(True)
        else:
            cbase.App.FPGAChargeInjection.step.set(0) # Disable charge injection

    logging.warning('config_expert complete')

def reset_counters(base):
    cbase = base['cam']

    # Reset the timing counters
    cbase.App.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
    cbase.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

    # Reset the ASIC counters
    for i in range (cbase.numOfAsics):
        getattr(cbase.App.AsicTop, f"DigAsicStrmRegisters{i}").CountReset()

#
#  Called on Configure
#
def epixm320_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global origcfg
    global group
    global segids

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
#    epixm320_internal_trigger(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    cfg['firmwareVersion'] = cbase.Core.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cbase.Core.AxiVersion.BuildDate.get()

    #
    #  Create the segment configurations from parameters required for analysis
    #
    compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(cbase.numOfAsics) ]
    precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(cbase.numOfAsics) ]

    topname = cfg['detName:RO'].split('_')

    segcfg = {}
    segids = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    gain_mode = cfg['user']['gain_mode']
    if gain_mode != 3:
        compTH0,precharge_DAC0 = gain_mode_map(gain_mode)
        compTH        = [compTH0        for i in range(cbase.numOfAsics)]
        precharge_DAC = [precharge_DAC0 for i in range(cbase.numOfAsics)]

    print(f'gain_mode {gain_mode}  CompTH_ePixM {compTH}  Precharge_DAC_ePixM {precharge_DAC}')

    for seg in range(1):  # Loop over 'tiles', of which the ePixM has only one
        # Get serial numbers
        cbase.App.AsicTop.RegisterControlDualClock.enable.set(True)
        cbase.App.AsicTop.RegisterControlDualClock.IDreset.set(0x7)
        cbase.App.AsicTop.RegisterControlDualClock.IDreset.set(0x0)

        # Wait for hardware to get serial numbers
        time.sleep(0.1)

        #  Construct the ID
        digitalId = [0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.DigIDLow.get(),
                     0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.DigIDHigh.get()]
        pwrCommId = [0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.PowerAndCommIDLow.get(),
                     0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.PowerAndCommIDHigh.get()]
        carrierId = [0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.CarrierIDLow.get(),
                     0 if base['pcie_timing'] else cbase.App.AsicTop.RegisterControlDualClock.CarrierIDHigh.get()]
        digital = (digitalId[1] << 32) | digitalId[0]
        pwrComm = (pwrCommId[1] << 32) | pwrCommId[0]
        carrier = (carrierId[1] << 32) | carrierId[0]
        print(f'ePixM320k ids: f/w {cfg["firmwareVersion"]:x}, carrier {carrier:x}, digital {digital:x}, pwrComm {pwrComm:x}')
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(cfg['firmwareVersion'],
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          pwrCommId[0], pwrCommId[1])
        print(f'ePixM320k id: {id}')
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [1,0,0])
        top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')

        # Add some convenience variables for the offline analysis.
        # In this case, these are somewhat redundant because the information is already in
        # segcfg[0], but we're preserving the pattern in case it makes a difference downstream.
        # Note that at this point we don't know whether we are about to begin a normal data-taking
        # run or some kind of scan, so we can't conditionally add things to segcfg[1] here.
        top.set('CompTH_ePixM',        compTH,        'UINT8')
        top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
        top.set('startCol',    cfg['expert']['App']['FPGAChargeInjection']['startCol'],    'UINT32')
        top.set('endCol',      cfg['expert']['App']['FPGAChargeInjection']['endCol'],      'UINT32')
        top.set('step',        cfg['expert']['App']['FPGAChargeInjection']['step'],        'UINT32')
        top.set('currentAsic', cfg['expert']['App']['FPGAChargeInjection']['currentAsic'], 'UINT32')
        segcfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.warning('json seg {}  detname {}'.format(i, segcfg[i]['detName:RO']))
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    base['cfg']    = copy.deepcopy(cfg)
    base['result'] = copy.deepcopy(result)

    # @todo: Sanity check result here.
    # Not sure what that means.  Check that all entry's names and values are valid python, perhaps?

    return result

def epixm320_unconfig(base):
    logging.info('epixm320_unconfig')
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change in
#  response to the scan parameters.  Called on Configure, after <det>_config().
#
def epixm320_scan_keys(update):
    """Returns an updated config JSON to record in an XTC file.

    This function and the <det>_update function are used in BEBDetector config
    scans.  The update argument contains the keys of the scan parameters.
    """
    logging.warning('epixm320_scan_keys')
    global origcfg
    global base
    global segids

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
        numAsics = base['cam'].numOfAsics
        compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(numAsics) ]
        precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(numAsics) ]

        for seg in range(1):    # Loop over 'tiles', of which the ePixM has only one
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [1,0,0])
            top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')

            # Add some convenience variables for the offline
            top.set('CompTH_ePixM',        compTH,        'UINT8')
            top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')

            hasChgInj = 'FPGAChargeInjection' in cfg['expert']['App']
            if hasChgInj:
                top.set('startCol',    cfg['expert']['App']['FPGAChargeInjection']['startCol'],    'UINT32')
                top.set('endCol',      cfg['expert']['App']['FPGAChargeInjection']['endCol'],      'UINT32')
                top.set('step',        cfg['expert']['App']['FPGAChargeInjection']['step'],        'UINT32')
                top.set('currentAsic', cfg['expert']['App']['FPGAChargeInjection']['currentAsic'], 'UINT32')
            segcfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(segcfg)):
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    base['scan_keys'] = copy.deepcopy(result)

    # Sanity check result here.  All keys in result must also be in base['result'].
    if not check_json_keys(result, base['result']): # @todo: Too strict?
        logging.error('epixm320_scan_keys json is inconsistent with that of epix320_config')
        # Expect to crash in descData.hh, but let's see

    return result

#
#  Return the set of configuration updates for a scan step.
#  Called on BeginStep.
#
def epixm320_update(update):
    """Applies an updated configuration to a detector during a scan.

    This function and the <det>_scan_keys function are used in BEBDetector
    config scans.  This function is called on every scan step with step-
    dependent (key, value) pairs for the scan parameters.
    """
    logging.warning('epixm320_update')
    global origcfg
    global base

    #  Queue full configuration next Configure transition
    base['cfg'] = None

    _stop(base)

    # extract updates
    cfg = {}
    update_config_entry(cfg,origcfg,json.loads(update))
    #  Apply to expert
    writeCalibRegs = user_to_expert(base,cfg,full=False)
    print(f'Partial config writeCalibRegs {writeCalibRegs}')
    #  Apply config
    config_expert(base, cfg, writeCalibRegs, secondPass=True)

    _start(base)

    #  Enable triggers to continue monitoring
#    epixm320_internal_trigger(base)

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
        numAsics = base['cam'].numOfAsics
        compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(numAsics) ]
        precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(numAsics) ]

        for seg in range(1):    # Loop over 'tiles', of which the ePixM has only one
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [1,0,0])
            top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')

            # Add some convenience variables for the offline
            top.set('CompTH_ePixM',        compTH,        'UINT8')
            top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')

            hasChgInj = 'FPGAChargeInjection' in cfg['expert']['App']
            if hasChgInj:
                top.set('startCol',    cfg['expert']['App']['FPGAChargeInjection']['startCol'],    'UINT32')
                top.set('endCol',      cfg['expert']['App']['FPGAChargeInjection']['endCol'],      'UINT32')
                top.set('step',        cfg['expert']['App']['FPGAChargeInjection']['step'],        'UINT32')
                top.set('currentAsic', cfg['expert']['App']['FPGAChargeInjection']['currentAsic'], 'UINT32')
            segcfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(segcfg)):
        result.append( json.dumps(sanitize_config(segcfg[i])) )

    # Sanity check result.  All keys in result must also be in base['scan_keys'].
    if not check_json_keys(result, base['scan_keys'], exact=True):
        logging.error('epixm320_update json is inconsistent with that of epix320_scan_keys')
        # Expect to crash in descData.hh, but let's see

    logging.info('epixm320_update complete')

    return result

def _resetSequenceCount():
    cbase = base['cam']
    cbase.App.AsicTop.RegisterControlDualClock.ResetCounters.set(1)
    time.sleep(1.e6)
    cbase.App.AsicTop.RegisterControlDualClock.ResetCounters.set(0)

def epixm320_external_trigger(base):
    #  Switch to external triggering
    print(f"=== external triggering ===")
    cbase = base['cam']
    cbase.App.AsicTop.TriggerRegisters.SetTimingTrigger()

def epixm320_internal_trigger(base):
    ##  Disable frame readout
    #mask = 0x3
    #print('=== internal triggering with bypass {:x} ==='.format(mask))
    #cbase = base['cam']
    #for i in range(cbase.numOfAsics):
    #    # This should be base['pci'].DevPcie.Application.EventBuilder.Bypass
    #    getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').Bypass.set(mask)
    #return

    #  Switch to internal triggering
    print('=== internal triggering ===')
    cbase = base['cam']
    cbase.App.AsicTop.TriggerRegisters.SetAutoTrigger()

def epixm320_enable(base):
    logging.info('epixm320_enable')
    cbase = base['cam']

    # If charge injection was enabled, start the f/w engine
    if cbase.App.FPGAChargeInjection.step.get() != 0:
        cbase.App.FPGAChargeInjection.Start()

    epixm320_external_trigger(base)
    #_start(base)

def epixm320_disable(base):
    logging.info('epixm320_disable')
    cbase = base['cam']

    # If charge injection was enabled, stop the f/w engine
    if cbase.App.FPGAChargeInjection.step.get() != 0:
        cbase.App.FPGAChargeInjection.Stop()

    # The following prevents transitions from going through
    # epixm320_internal_trigger(base)
    # Seems like we should do the following, but it also blows off transitions
    #_stop(base)

def _stop(base):
    print('_stop')
    cbase = base['cam']
    cbase.App.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    print('_start')
    cbase = base['cam']
    cbase.App.StartRun()
    # This is unneccessary as Bypass is handled above and Blowoff in StartRun()
    #m = base['batchers']
    #for i in range(cbase.numOfAsics):
    #    getattr(cbase.App.AsicTop,f'BatcherEventBuilder{i}').Bypass.set(0x0)
    #    getattr(cbase.App.AsicTop,f'BatcherEventBuilder{i}').Blowoff.set(m[i]==0)

#
#  Test standalone
#
if __name__ == "__main__":

    _base = epixm320_init(None,dev='/dev/datadev_0')
    epixm320_init_feb()
    epixm320_connectionInfo(_base, None)

    db = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB'
    d = {'body':{'control':{'0':{'control_info':{'instrument':'tst',
                                                 'cfg_dbase' :db}}}}}

    print('***** CONFIG *****')
    _connect_str = json.dumps(d)
    epixm320_config(_base,_connect_str,'BEAM','tst_epixm',0,4)

    print('***** SCAN_KEYS *****')
    epixm320_scan_keys(json.dumps(["user.gain_mode"]))

    for i in range(100):
        print(f'***** UPDATE {i} *****')
        epixm320_update(json.dumps({'user.gain_mode':i%3}))

    print('***** DONE *****')

