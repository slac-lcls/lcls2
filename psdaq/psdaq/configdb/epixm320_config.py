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
ocfg = None
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
    compTH        = ( 0, 63, 12)[gain_mode] # SoftHigh/SoftLow/Auto
    precharge_DAC = (45, 50, 45)[gain_mode]
    return (compTH, precharge_DAC)

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
#  The cfg dictionary may be partial (scanning), so the ocfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, full=False):
    global ocfg
    global group
    global lane

    cbase = base['cam']

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        rtp = ocfg['user']['run_trigger_group'] # run trigger partition
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
    a = None
    hasUser = 'user' in cfg
    if hasUser and 'gain_mode' in cfg['user']:
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==3:  # user's choices
            # Use CompTH and Precharge_DAC from cfg['expert']
            d['user.chgInj_column_map'] = cfg['user']['chgInj_column_map']
        else:
            compTH, precharge_DAC = gain_mode_map(gain_mode)
            for i in range(cbase.numOfAsics):
                d[f'expert.App.Mv2Asic[{i}].CompTH_ePixM'] = compTH
                d[f'expert.App.Mv2Asic[{i}].Precharge_DAC_ePixM'] = precharge_DAC
        calibRegsChanged = True

    update_config_entry(cfg,ocfg,d)

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
    base['bypass'] = cbase.numOfAsics * [0x0]  # Enable Timing (bit-0) and Data (bit-1)
    base['batchers'] = cbase.numOfAsics * [1]  # list of active batchers
    print(f'=== configure bypass {base["bypass"]} ===')
    for i in asics:
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
        epixMTypes = cfg[':types:']['expert']['App']
        tree = ('Root','App')
        tmpfiles = []
        def toYaml(sect,keys,name):
            if sect == tree[-1]:
                tmpfiles.append(dictToYaml(app,epixMTypes,keys,cbase,path,name,tree,ordering))
            else:
                tmpfiles.append(dictToYaml(app[sect],epixMTypes[sect],keys,cbase,path,name,(*tree,sect),ordering))

        clk = cfg['expert']['Pll']['Clock']
        if clk != 4:            # 4 is the Default firmware setting
            freq = [None,'_250_MHz','_125_MHz','_168_MHz'][clk]
            pllCfg = np.reshape(cfg['expert']['Pll'][freq], (-1,2))
            fn = path+'PllConfig'+'.csv'
            np.savetxt(fn, pllCfg, fmt='0x%04X,0x%02X', delimiter=',', newline='\n', header='Address,Data', comments='')
            tmpfiles.append(fn)
            setattr(cbase, 'filenamePLL', fn)

        toYaml('App',['PowerControl'],'PowerSupply')
        toYaml('App',[f'SspMonGrp[{i}]' for i in asics],'DESER')
        toYaml('AsicTop',['RegisterControlDualClock'],'WaveForms')
        toYaml('AsicTop',[f'DigAsicStrmRegisters{i}' for i in asics],'PacketReg')
        toYaml('AsicTop',[f'BatcherEventBuilder{i}' for i in asics],'Batcher')
        setattr(cbase, 'filenameASIC',4*[None]) # This one is a little different
        for i in asics:
            toYaml('App',[f'Mv2Asic[{i}]'],f'ASIC_u{i+1}')
            cbase.filenameASIC[i] = getattr(cbase,f'filenameASIC_u{i+1}')

        arg = [clk,0,0,0,0]
        for i in asics:
            arg[1+i] = 1
        logging.warning(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.fnInitAsicScript(None,None,arg)

        #  Remove the yml files
        for f in tmpfiles:
            os.remove(f)

        for i in range(cbase.numOfAsics):
            # Prevent disabled ASICs from participating by disabling their lanes
            # It seems like disabling their Batchers should be sufficient,
            # but that prevents transitions from going through
            if i not in asics:  # Override configDb's value for disabled ASICs
                getattr(cbase.App.AsicTop, f'DigAsicStrmRegisters{i}').DisableLane.set(0xffffff)

            # Enable the batchers for all ASICs
            getattr(cbase.App.AsicTop, f'BatcherEventBuilder{i}').enable.set(base['batchers'][i] == 1)

        # Adjust for intermitent lanes of enabled ASICs
        cbase.laneDiagnostics(arg[1:5], threshold=20, loops=5, debugPrint=False)

    if writeCalibRegs:
        hasGainMode = 'gain_mode' in cfg['user']
        if (hasGainMode and cfg['user']['gain_mode']==3) or not hasGainMode:
            #
            #  Write the general pixel map
            #
            column_map = np.array(cfg['user']['chgInj_column_map'],dtype=np.uint8)

            for i in asics:
                #  Don't forget about the gain_mode and charge injection
                asicName = f'Mv2Asic[{i}]'
                saci = getattr(cbase.App,asicName)
                saci.enable.set(True)

                #  Don't forget about charge injection
                if app is not None and asicName in app:
                    di = app[asicName]
                    setSaci(saci.CompTH_ePixM,'CompTH_epixM',di)
                    setSaci(saci.Precharge_DAC_ePixM,'Precharge_DAC_epixM',di)

                    cbase.App.setupChargeInjection(i, column_map, di['Pulser'])
                else:
                    cbase.App.chargeInjectionCleanup(i)

                saci.enable.set(False)
        else:
            gain_mode = cfg['user']['gain_mode']
            compTH, precharge_DAC = gain_mode_map(gain_mode)
            print(f'Setting gain mode {gain_mode}:  compTH {compTH},  precharge_DAC {precharge_DAC}')

            for i in asics:
                saci = getattr(cbase.App,f'Mv2Asic[{i}]')
                saci.enable.set(True)
                cbase.App.chargeInjectionCleanup(i)
                saci.CompTH_ePixM.set(compTH)
                saci.Precharge_DAC_ePixM.set(precharge_DAC)
                saci.enable.set(False)

    logging.warning('config_expert complete')

def reset_counters(base):
    # Reset the timing counters
    base['cam'].App.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
    base['cam'].App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

#
#  Called on Configure
#
def epixm320_config(base,connect_str,cfgtype,detname,detsegm,rog):
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
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()

    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(cbase.numOfAsics) ]
    precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(cbase.numOfAsics) ]

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    gain_mode = cfg['user']['gain_mode']
    if gain_mode==3:
        column_map    = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
    else:
        compTH0,precharge_DAC0 = gain_mode_map(gain_mode)
        compTH        = [compTH0        for i in range(cbase.numOfAsics)]
        precharge_DAC = [precharge_DAC0 for i in range(cbase.numOfAsics)]

    print(f'gain_mode {gain_mode}  CompTH_ePixM {compTH}  Precharge_DAC_ePixM {precharge_DAC}  column_map shape {column_map.shape if gain_mode==3 else None}')

    for seg in range(1):
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
        print(f'ePixM320k ids: f/w {firmwareVersion:x}, carrier {carrier:x}, digital {digital:x}, pwrComm {pwrComm:x}')
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          pwrCommId[0], pwrCommId[1])
        print(f'ePixM320k id: {id}')
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [0,0,0])
        top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
        top.set('CompTH_ePixM',        compTH,        'UINT8')
        top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
        if gain_mode==3:
            top.set('chgInj_column_map', column_map)
        scfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.warning('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(sanitize_config(scfg[i])) )

    base['cfg']    = copy.deepcopy(cfg)
    base['result'] = copy.deepcopy(result)

    return result

def epixm320_unconfig(base):
    print('epixm320_unconfig')
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixm320_scan_keys(update):
    logging.warning('epixm320_scan_keys')
    global ocfg
    global base
    global segids

    cfg = {}
    copy_reconfig_keys(cfg,ocfg,json.loads(update))
    # Apply to expert
    calibRegsChanged = user_to_expert(base,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    if calibRegsChanged:
        cbase = base['cam']
        compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(cbase.numOfAsics) ]
        precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(cbase.numOfAsics) ]
        if 'chgInj_column_map' in cfg['user']:
            gain_mode = cfg['user']['gain_mode']
            if gain_mode==3:
                column_map = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
            else:
                column_map = np.zeros(nColumns, dtype=np.uint8)

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [0,0,0])
            top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            top.set('CompTH_ePixM',        compTH,        'UINT8')
            top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
            if 'chgInj_column_map' in cfg['user']:
                top.set('chgInj_column_map', column_map)
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(sanitize_config(scfg[i])) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixm320_update(update):
    logging.warning('epixm320_update')
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
    writeCalibRegs = user_to_expert(base,cfg,full=False)
    print(f'Partial config writeCalibRegs {writeCalibRegs}')
    if True:
        #  Apply config
        config_expert(base, cfg, writeCalibRegs, secondPass=True)
    else:
        ##
        ##  Try full configuration
        ##
        ncfg = ocfg.copy()
        update_config_entry(ncfg,ocfg,json.loads(update))
        _writeCalibRegs = user_to_expert(base,ncfg,full=True)
        print(f'Full config writeCalibRegs {_writeCalibRegs}')
        config_expert(base, ncfg, _writeCalibRegs, secondPass=True)

    _start(base)

    #  Enable triggers to continue monitoring
#    epixm320_internal_trigger(base)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = '_'.join(topname[:-1])+'hw_'+topname[-1]

    scfg[0] = cfg

    if writeCalibRegs:
        cbase = base['cam']
        try:
            compTH        = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['CompTH_ePixM']        for i in range(cbase.numOfAsics) ]
        except:
            compTH        = None; print('CompTH is None')
        try:
            precharge_DAC = [ cfg['expert']['App'][f'Mv2Asic[{i}]']['Precharge_DAC_ePixM'] for i in range(cbase.numOfAsics) ]
        except:
            precharge_DAC = None; print('Precharge_DAC is None')
        try:
            column_map    = np.array(cfg['user']['chgInj_column_map'], dtype=np.uint8)
        except:
            column_map    = None; print('column_map is None')

        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [0,0,0])
            top.setInfo(detType='epixm320', detName='_'.join(topname[:-1]), detSegm=seg+int(topname[-1]), detId=id, doc='No comment')
            if compTH        is not None:  top.set('CompTH_ePixM',        compTH,        'UINT8')
            if precharge_DAC is not None:  top.set('Precharge_DAC_ePixM', precharge_DAC, 'UINT8')
            if column_map    is not None:  top.set('chgInj_column_map',   column_map)
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(sanitize_config(scfg[i])) )

    logging.info('update complete')

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
    cbase.App.AsicTop.TriggerRegisters.SetTimingTrigger(1)

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
    cbase.App.AsicTop.TriggerRegisters.SetAutoTrigger(1)

def epixm320_enable(base):
    print('epixm320_enable')
    epixm320_external_trigger(base)
    _start(base)

def epixm320_disable(base):
    print('epixm320_disable')
    # Prevents transitions going through: epixm320_internal_trigger(base)

def _stop(base):
    print('_stop')
    cbase = base['cam']
    cbase.App.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    print('_start')
    cbase = base['cam']
    cbase.App.AsicTop.TriggerRegisters.SetTimingTrigger()
    cbase.App.StartRun()
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

