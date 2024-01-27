from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.det_config import *
from psdaq.cas.xpm_utils import timTxId
#from .xpmmini import *
import pyrogue as pr
import rogue
import lcls2_epix_hr_pcie
import epix_hr_single_10k
import epix_hr_core as epixHr
import ePixFpga as fpga
import surf.protocols.batcher       as batcher
import time
import json
import os
import numpy as np
import IPython
import datetime
import logging

base = None
pv = None
#lane = 0  # An element consumes all 4 lanes
chan = None
group = None
ocfg = None
segids = None
seglist = [0,1]
asics = None
readPixelMaps = False

elemRowsC = 146
elemRowsD = 144
elemCols  = 192

RTP = 6   # run trigger partition

#  Timing delay scans can be limited by this
EventBuilderTimeout = int(1.0e-3*156.25e6)

#  Register ordering matters, but our configdb does not preserve the order.
#  For now, put the ordering in code here, until the configdb can be updated to preserve order.
#  Underneath, it is just storing json, so order preservation should be possible.
ordering = {'MMCMRegisters':[]}
i=0
ordering['MMCMRegisters'].extend([f'CLKOUT{i}PhaseMux',
                                  f'CLKOUT{i}HighTime',
                                  f'CLKOUT{i}LowTime',
                                  f'CLKOUT{i}Frac',
                                  f'CLKOUT{i}FracEn',
                                  f'CLKOUT{i}Edge',
                                  f'CLKOUT{i}NoCount',
                                  f'CLKOUT{i}DelayTime',])
for i in range(1,7):
    ordering['MMCMRegisters'].extend([f'CLKOUT{i}PhaseMux',
                                      f'CLKOUT{i}HighTime',
                                      f'CLKOUT{i}LowTime',
                                      f'CLKOUT{i}Edge',
                                      f'CLKOUT{i}NoCount',
                                      f'CLKOUT{i}DelayTime',])

ordering['PowerSupply'] = ['DigitalEn','AnalogEn']
ordering['RegisterControl'] = ['GlblRstPolarity',
                               'ClkSyncEn',
                               'RoLogicRst',
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
                               'DebugSel_TG',
                               'DebugSel_MPS',
                               'StartupReq',]

for i in range(4):
    ordering[f'Hr10kTAsic{i}'] = ['shvc_DAC',
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
                                  'TC',
                                  'S2d',
                                  'S2dDacBias',
                                  'Tsd_Tser',
                                  'Tps_DAC',
                                  'PLL_RO_filter2',
                                  'PLL_RO_divider',
                                  'TestBe',
                                  'RSTreg',
                                  'DelExec',
                                  'DelCCKReg',
                                  'RO_rst_en',
                                  'SlvdsBit',
                                  'FELmode',
                                  'CompEnOn',
                                  'RowStartAddr',
                                  'RowStopAddr',
                                  'ColStartAddr',
                                  'ColStopAddr',
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
                                  'Rodis01',
                                  'CompEn',
                                  'Pixel_CB',
                                  'rodis34',
                                  'rowCK2matrix_delay',
                                  'ro_mode',
                                  'rodis5',
                                  'pipoclk_delay',]

ordering['SspLowSpeedDecoderReg'] = ['LockingCntCfg']
for i in range(2):
    ordering[f'PacketRegisters{i}'] = ['asicDataReq','DisableLane','EnumerateDisLane',]

ordering['TriggerRegisters'] = ['RunTriggerEnable',
                                'RunTriggerDelay',
                                'DaqTriggerEnable',
                                'DaqTriggerDelay',
                                'AutoRunEn',
                                'AutoDaqEn',
                                'AutoTrigPeriod',
                                'PgpTrigEn',]

def gain_mode_map(gain_mode):
    mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode] # H/M/L/AHL/AML
    trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]
    return (mapv,trbit)

class Board(pr.Root):
    def __init__(self,dev='/dev/datadev_0',lanes=4):
        super().__init__(name='ePixHr10kT',description='ePixHrGen1 board')
        self.dmaCtrlStreams = [None]
        self.dmaCtrlStreams[0] = rogue.hardware.axi.AxiStreamDma(dev,(0x100*0)+0,1)# Registers

        # Create and Connect SRP to VC1 to send commands
        self._srp = rogue.protocols.srp.SrpV3()
        pr.streamConnectBiDir(self.dmaCtrlStreams[0],self._srp)

        self.add(epixHr.SysReg  (name='Core'  , memBase=self._srp, offset=0x00000000, sim=False, expand=False, pgpVersion=4, numberOfLanes=lanes,))
        self.add(fpga.EpixHR10kT(asicVersion=4, name='EpixHR', memBase=self._srp, offset=0x80000000, hidden=False, enabled=True))
        #  Allow ePixFpga to find the yaml files for fnInitAsic()
        self.top_level = '/tmp'

        #  Remove unreliable QSFP readout
        del self.Core._nodes['QSfpI2C']

def setSaci(reg,field,di):
    if field in di:
        v = di[field]
        reg.set(v)
        print(f'Updated {field} to {v}')
    else:
        print(f'Not updating {field}')

#
#  Construct an asic pixel mask with square spacing
#
def pixel_mask_square(value0,value1,spacing,position):
    ny,nx=288,384;
    if position>=spacing**2:
        logging.error('position out of range')
        position=0;
    out=np.zeros((ny,nx),dtype=np.int)+value0
    position_x=position%spacing; position_y=position//spacing
    out[position_y::spacing,position_x::spacing]=value1
    return out

#
#  Scramble the user element pixel array into the native asic orientation
#
#
#    A1   |   A3       (A1,A3) rotated 180deg
# --------+--------
#    A0   |   A2
#
def user_to_rogue(a):
    v = a.reshape((elemRowsD*2,elemCols*2))
    s = np.zeros((4,elemRowsC,elemCols),dtype=np.uint8)
    s[0,:elemRowsD] = v[elemRowsD:,:elemCols]
    s[2,:elemRowsD] = v[elemRowsD:,elemCols:]
    vf = np.flip(v)
    s[1,:elemRowsD] = vf[elemRowsD:,elemCols:]
    s[3,:elemRowsD] = vf[elemRowsD:,:elemCols]
    return s

def rogue_to_user(s):
    vf = np.zeros((elemRowsD*2,elemCols*2),dtype=np.uint8)
    vf[elemRowsD:,:elemCols] = s[3,:elemRowsD]
    vf[elemRowsD:,elemCols:] = s[1,:elemRowsD]
    v = np.flip(vf)
    v[elemRowsD:,elemCols:] = s[2,:elemRowsD]
    v[elemRowsD:,:elemCols] = s[0,:elemRowsD]
    return v.reshape(elemRowsD*elemCols*4)

#
#  Initialize the rogue accessor
#
def epixhr2x2_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv
#    logging.getLogger().setLevel(40-10*verbosity) # way too much from rogue
    logging.getLogger().setLevel(30)
    logging.warning('epixhr2x2_init')

    base = {}
    #  Configure the PCIe card first (timing, datavctap)
    pbase = lcls2_epix_hr_pcie.DevRoot(dev           =dev,
                                       enLclsI       =False,
                                       enLclsII      =False,
                                       yamlFileLclsI =None,
                                       yamlFileLclsII=None,
                                       startupMode   =True,
                                       standAloneMode=xpmpv is not None,
                                       pgp4          =True,
                                       #dataVc        =0,
                                       pollEn        =False,
                                       initRead      =False,
                                       #numLanes      =4,
                                       pcieBoardType = 'Kcu1500')
    pbase.__enter__()
    base['pci'] = pbase

    #  Connect to the camera
    elanes = 4 if timebase=="119M" else 2
    cbase = Board(dev=dev,lanes=elanes)
    cbase.__enter__()
    base['cam'] = cbase

    #cbase.Core.enable.set(True)
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()
    buildStamp = cbase.Core.AxiVersion.BuildStamp.get()
    gitHash = cbase.Core.AxiVersion.GitHash.get()
    print(f'firmwareVersion [{firmwareVersion:x}]')
    print(f'buildStamp      [{buildStamp}]')
    print(f'gitHash         [{gitHash:x}]')
    #cbase.Core.enable.set(False)

    #  Enable the environmental monitoring
    cbase.EpixHR.SlowAdcRegisters.enable.set(1)
    cbase.EpixHR.SlowAdcRegisters.StreamPeriod.set(100000000)  # 1Hz
    cbase.EpixHR.SlowAdcRegisters.StreamEn.set(1)
    cbase.EpixHR.SlowAdcRegisters.enable.set(0)

   # configure timing
    if timebase=="119M":  # UED
        logging.warning('Using timebase 119M')
        base['bypass'] = 0x3f
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        base['pcie_timing'] = True

        logging.warning('epixhr2x2_unconfig')
        epixhr2x2_unconfig(base)

        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)
    else:
        logging.warning('Using timebase 186M')
        base['bypass'] = 0x3b  # Need the first lanes for transitions
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        base['pcie_timing'] = False

        logging.warning('epixhr2x2_unconfig')
        epixhr2x2_unconfig(base)

        cbase.EpixHR.ConfigLclsTimingV2()

    # configure internal ADC
    cbase.EpixHR.InitHSADC()

    time.sleep(1)
#    epixhr2x2_internal_trigger(base)
    return base

#
#  Set the PGP lane
#
def epixhr2x2_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixhr2x2_connectionInfo(base, alloc_json_str):

#
#  To do:  get the IDs from the detector and not the timing link
#
    txId = timTxId('epixhr2x2')
    logging.warning('TxId {:x}'.format(txId))

    if base['pcie_timing'] and 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    elif not base['pcie_timing'] and 'cam' in base:
        cbase = base['cam']
        rxId = cbase.EpixHR.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.warning('RxId {:x}'.format(rxId))
        cbase.EpixHR.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        rxId = 0xffffffff

    epixhrid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixhrid

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
        if base['pcie_timing']:
            pbase = base['pci']
            partitionDelay = getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
            rawStart       = cfg['user']['start_ns']
            triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
            logging.warning('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            if triggerDelay < 0:
                logging.error('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
                raise ValueError('triggerDelay computes to < 0')
            d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer0.TriggerDelay']=triggerDelay
            if full:
                d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer0.Partition']=group
        else:
            cbase = base['cam']
            for i,p in enumerate([RTP,group]):
                partitionDelay = getattr(cbase.EpixHR.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%p).get()
                rawStart       = cfg['user']['start_ns']
                triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
                logging.warning(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
                if triggerDelay < 0:
                    logging.error(f'partitionDelay[{p}] {partitionDelay}  rawStart {rawStart}  triggerDelay {triggerDelay}')
                    raise ValueError('triggerDelay computes to < 0')

                d[f'expert.EpixHR.TriggerEventManager.TriggerEventBuffer{i}.TriggerDelay']=triggerDelay

            if full:
                d[f'expert.EpixHR.TriggerEventManager.TriggerEventBuffer0.Partition']=RTP
                d[f'expert.EpixHR.TriggerEventManager.TriggerEventBuffer1.Partition']=group

    pixel_map_changed = False
    a = None
    hasUser = 'user' in cfg
    if hasUser and ('pixel_map' in cfg['user'] or
                    'gain_mode' in cfg['user']):
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==5:  # user map
            a  = cfg['user']['pixel_map']
            logging.warning('pixel_map len {}'.format(len(a)))
            d['user.pixel_map'] = a
            # what about associated trbit?
        else:
            mapv, trbit = gain_mode_map(gain_mode)
            for i in range(4):
                d[f'expert.EpixHR.Hr10kTAsic{i}.trbit'] = trbit
        pixel_map_changed = True

    update_config_entry(cfg,ocfg,d)

    return pixel_map_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writePixelMap=True, secondPass=False):
    global asics  # Need to maintain this across configuration updates

    #  Disable internal triggers during configuration
    epixhr2x2_external_trigger(base)

    pbase = base['pci']
    cbase = base['cam']

    # overwrite the low-level configuration parameters with calculations from the user configuration
    if 'expert' in cfg:
        if 'DevPcie' in cfg['expert']:
            apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])
        if not base['pcie_timing']:
            try:  # config update might not have this
                apply_dict('cbase.EpixHR.TriggerEventManager',cbase.EpixHR.TriggerEventManager,cfg['expert']['EpixHR']['TriggerEventManager'])
            except KeyError:
                pass

    epixHR = None
    if ('expert' in cfg and 'EpixHR' in cfg['expert']):
        epixHR = cfg['expert']['EpixHR'].copy()

    #  Make list of enabled ASICs
    if 'user' in cfg and 'asic_enable' in cfg['user']:
        asics = []
        for i in range(4):
            if cfg['user']['asic_enable']&(1<<i):
                asics.append(i)
            else:
                # remove the ASIC configuration so we don't try it
                del epixHR['Hr10kTAsic{}'.format(i)]

    #  Set the application event builder for the set of enabled asics
    if base['pcie_timing']:
        m=3
        for i in asics:
            m = m | (4<<i)
    else:
#        Enable batchers for all ASICs.  Data will be padded.
#        m=0
#        for i in asics:
#            m = m | (4<<int(i/2))
        m=3<<2
    base['bypass'] = 0x3f^m  # mask of active batcher channels
    base['batchers'] = m>>2  # mask of active batchers
    print('=== configure bypass {:x} ==='.format(base['bypass']))
    pbase.DevPcie.Application.EventBuilder.Bypass.set(base['bypass'])

    #  Use a timeout in AxiStreamBatcherEventBuilder
    #  Without a timeout, dropped contributions create an off-by-one between contributors
    pbase.DevPcie.Application.EventBuilder.Timeout.set(EventBuilderTimeout) # 400 us
    if not base['pcie_timing']:
        eventBuilder = cbase.find(typ=batcher.AxiStreamBatcherEventBuilder)
        for eb in eventBuilder:
            eb.Timeout.set(EventBuilderTimeout)
            eb.Blowoff.set(True)

    #
    #  For some unknown reason, performing this part of the configuration on BeginStep
    #  causes the readout to fail until the next Configure
    #
    if epixHR is not None and not secondPass:
        # Work hard to use the underlying rogue interface
        # Translate config data to yaml files
        path = '/tmp/epixhr'
        epixHRTypes = cfg[':types:']['expert']['EpixHR']
        tree = ('ePixHr10kT','EpixHR')
        tmpfiles = []
        def toYaml(keys,name):
            tmpfiles.append(dictToYaml(epixHR,epixHRTypes,keys,cbase.EpixHR,path,name,tree,ordering))

        toYaml(['MMCMRegisters'],'MMCM')
        toYaml(['PowerSupply'],'PowerSupply')
        toYaml(['RegisterControl'],'RegisterControl')
        for i in asics:
            toYaml([f'Hr10kTAsic{i}'],f'ASIC{i}')

        toYaml(['SspLowSpeedDecoderReg'],'SSP')

        # remove non-existent field
        for i in range(2):
            if 'gainBitRemapped' in epixHR[f'PacketRegisters{i}']:
                del epixHR[f'PacketRegisters{i}']['gainBitRemapped']

        toYaml([f'PacketRegisters{i}' for i in range(2)],'PacketReg')
        toYaml(['TriggerRegisters'],'TriggerReg')

        arg = [1,0,0,0,0]
        for i in asics:
            arg[i+1] = 1
        logging.warning(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.EpixHR.fnInitAsicScript(None,None,arg)

        #  Remove the yml files
        for f in tmpfiles:
            os.remove(f)

#       Fixup the PacketRegisters0 disable register (lane 0 is broken?)
#        lanes = cbase.EpixHR.PacketRegisters0.DisableLane.get();
#        cbase.EpixHR.PacketRegisters0.DisableLane.set(lanes | 0x33);


    if writePixelMap:
        hasGainMode = 'gain_mode' in cfg['user']
        if (hasGainMode and cfg['user']['gain_mode']==5) or not hasGainMode:
            #
            #  Write the general pixel map
            #
            pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
            pixelConfigMap = user_to_rogue(pixelConfigUsr)

            for i in asics:
                #  Write a csv file then pass to rogue
                path = '/tmp/epixhr'
                fn = path+'PixelMap{}.csv'.format(i)
                np.savetxt(fn, pixelConfigMap[i], fmt='%d', delimiter=',', newline='\n')
                print('Setting pixel bit map from {}'.format(fn))
                asicName = f'Hr10kTAsic{i}'
                saci = getattr(cbase.EpixHR,asicName)
                saci.enable.set(True)
                saci.fnSetPixelBitmap(None,None,fn)
                os.remove(fn)

                #  Don't forget about the trbit and charge injection
                #  Program these one-by-one since the second call of fnInitAsicScript breaks
                if epixHR is not None and asicName in epixHR:
                    di = epixHR[asicName]
                    setSaci(saci.atest,'atest',di)
                    setSaci(saci.test,'test',di)
                    setSaci(saci.trbit,'trbit',di)
                    setSaci(saci.Pulser,'Pulser',di)

                saci.enable.set(False)

            logging.warning('SetAsicsMatrix complete')
        else:
            #
            #  Write a uniform pixel map
            #
            gain_mode = cfg['user']['gain_mode']
            mapv, trbit = gain_mode_map(gain_mode)
            print(f'Setting uniform pixel map mode {gain_mode} mapv {mapv} trbit {trbit}')

            for i in asics:
                saci = getattr(cbase.EpixHR,f'Hr10kTAsic{i}')
                saci.enable.set(True)
                saci.fnClearMatrix(None,None,mapv)
                saci.trbit.set(trbit)
                saci.enable.set(False)

    #  read back the pixel maps
    if readPixelMaps:
        for i in asics:
            saci = getattr(cbase.EpixHR,f'Hr10kTAsic{i}')
            saci.enable.set(True)
            fname = f'/tmp/Hr10kTAsic{i}.{datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")}.csv'
            logging.warning(f'Reading pixel map to {fname}')
            saci.fnGetPixelBitmap(None,None,fname)
            cname = f'/tmp/Hr10kTAsic{i}.latest'
            try:
                os.remove(cname)
            except:
                pass
            os.symlink(fname,cname)
            saci.enable.set(False)

    logging.warning('config_expert complete')

def reset_counters(base):
    # Reset the timing counters
    base['pci'].DevPcie.Hsio.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
    base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

#
#  Called on Configure
#
def epixhr2x2_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids

    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    if not base['pcie_timing']:
        if 'TriggerEventManager' not in cfg['expert']['EpixHR']:
            #  Add missing fields
            f = {'Partition':'UINT32','PauseThreshold':'UINT32','TriggerDelay':'UINT32'}
            cfg[':types:']['expert']['EpixHR']['TriggerEventManager'] = {'TriggerEventBuffer0':f.copy(),
                                                                         'TriggerEventBuffer1':f.copy()}
            f = cfg['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer']
            cfg['expert']['EpixHR']['TriggerEventManager'] = {'TriggerEventBuffer0':f.copy(),
                                                              'TriggerEventBuffer1':f.copy()}
            #  Remove the old ones
            del cfg[':types:']['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer']
            del cfg['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer']

    ocfg = cfg

    #  Translate user settings to the expert fields
    writePixelMap=user_to_expert(base, cfg, full=True)

    #  Apply the expert settings to the device
    _stop(base)

    config_expert(base, cfg, writePixelMap)

    time.sleep(0.01)
    _start(base)

    #  Add some counter resets here
    reset_counters(base)

    #  Enable triggers to continue monitoring
#    epixhr2x2_internal_trigger(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    #cbase.Core.enable.set(True)
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()

    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]


    #  User pixel map is assumed to be 288x384 in standard element orientation
    gain_mode = cfg['user']['gain_mode']
    if not readPixelMaps:
        if gain_mode==5:
            pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8).reshape((2*elemRowsD,2*elemCols))
        else:
            mapv,trbit0 = gain_mode_map(gain_mode)
            trbit = [trbit0 for i in range(4)]
            pixelConfigUsr = np.zeros((2*elemRowsD,2*elemCols),dtype=np.uint8)+mapv
    else:
        if gain_mode==5:
            pixelConfigSet = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
        else:
            mapv,trbit0 = gain_mode_map(gain_mode)
            pixelConfigSet = np.zeros((2*elemRowsD*2*elemCols),dtype=np.uint8)+mapv

        s = user_to_rogue(pixelConfigSet)
        for i in asics:
            cname = f'/tmp/Hr10kTAsic{i}.latest'
            s[i] = np.loadtxt(cname, dtype=np.uint8, delimiter=',')

        pixelConfigUsr = rogue_to_user(s)

        # Lets do some validation

        if pixelConfigUsr.shape != pixelConfigSet.shape:
            logging.error(f'  shape error  wrote {pixelConfigSet.shape}  read {pixelConfigUsr.shape}')
        else:
            nerr = 0
            for i in range(pixelConfigUsr.shape[0]):
                if pixelConfigUsr[i] != pixelConfigSet[i]:
                    nerr += 1
                    if nerr < 20:
                        logging.error(f'  mismatch at {i}({i//(2*elemCols)},{i%(2*elemCols)})  wrote {pixelConfigSet[i]}  read {pixelConfigUsr[i]}')
            logging.warning(f'Gain map validation complete with {nerr} mismatches')

    print(f'pixelConfigUsr shape {pixelConfigUsr.shape}  trbit {trbit}')

    for seg in range(1):
        #  Construct the ID
        snCarrier = 0 if base['pcie_timing'] else cbase.Core.AxiVersion.snCarrier.get()
        snAdcCard = 0 if base['pcie_timing'] else cbase.Core.AxiVersion.snAdcCard.get()
        carrierId = [ snCarrier&0xffffffff, snCarrier>>32 ]
        digitalId = [ snAdcCard&0xffffffff, snAdcCard>>32 ]
        analogId  = [ 0, 0 ]
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          analogId [0], analogId [1])
        print(f'id {id}')
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [2,0,0])
        top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=seg+int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigUsr)
        top.set('trbit'          , trbit, 'UINT8')
        scfg[seg+1] = top.typed_json()

    #cbase.Core.enable.set(False)

    result = []
    for i in seglist:
        logging.warning('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )

    return result

def epixhr2x2_unconfig(base):
    _stop(base)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixhr2x2_scan_keys(update):
    logging.warning('epixhr2x2_scan_keys')
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
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    if pixelMapChanged:
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==5:
            pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8).reshape(2*elemRowsD,2*elemCols)
        else:
            mapv,trbit = gain_mode_map(gain_mode)
            pixelConfigUsr = np.zeros((2*elemRowsD,2*elemCols),dtype=np.uint8)+mapv

        pixelConfigMap = user_to_rogue(pixelConfigUsr)
        trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]

        cbase = base['cam']
        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigUsr)
            top.set('trbit'          , trbit                  , 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixhr2x2_update(update):
    logging.warning('epixhr2x2_update')
    global ocfg
    global base

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
#    epixhr2x2_internal_trigger(base)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    scfg[0] = cfg

    if writePixelMap:
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==5:
            pixelConfigUsr = np.array(cfg['user']['pixel_map'],dtype=np.uint8).reshape(2*elemRowsD,2*elemCols)
        else:
            mapv,trbit = gain_mode_map(gain_mode)
            pixelConfigUsr = np.zeros((2*elemRowsD,2*elemCols),dtype=np.uint8)+mapv

        pixelConfigMap = user_to_rogue(pixelConfigUsr)
        try:
            trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]
        except:
            trbit = None

        cbase = base['cam']
        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=seg+int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigUsr)
            if trbit is not None:
                top.set('trbit'      , trbit                  , 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    logging.warning('update complete')

    return result

def _resetSequenceCount():
    cbase = base['cam']
    cbase.RegisterControl.ResetCounters.set(1)
    time.sleep(1.e6)
    cbase.RegisterControl.ResetCounters.set(0)

def epixhr2x2_external_trigger(base):
    #  Switch to external triggering
    print('=== external triggering with bypass {:x} ==='.format(base['bypass']))
    cbase = base['cam'].EpixHR
    cbase.TriggerRegisters.SetTimingTrigger(1)

def epixhr2x2_internal_trigger(base):
    #  Disable frame readout
    mask = 0x3f if base['pcie_timing'] else 0x3b
    print('=== internal triggering with bypass {:x} ==='.format(mask))
    pbase = base['pci']
    pbase.DevPcie.Application.EventBuilder.Bypass.set(mask)
    return

    #  Switch to internal triggering
    print('=== internal triggering ===')
    cbase = base['cam'].EpixHR
    cbase.TriggerRegisters.SetAutoTrigger(1)

def epixhr2x2_enable(base):
    print('epixhr2x2_enable')
    epixhr2x2_external_trigger(base)
#    _start(base)

def epixhr2x2_disable(base):
    print('epixhr2x2_disable')
#    epixhr2x2_internal_trigger(base)

def _stop(base):
    pbase = base['pci']
    cbase = base['cam']
    cbase.EpixHR.StopRun()
    pbase.StopRun()
    time.sleep(0.1)  #  let last triggers pass through

def _start(base):
    pbase = base['pci']
    pbase.StartRun()
    cbase = base['cam']
    cbase.EpixHR.StartRun()
    m = base['batchers']
    cbase.EpixHR.BatcherEventBuilder0.Bypass.set(0)
    cbase.EpixHR.BatcherEventBuilder1.Bypass.set(0)
    cbase.EpixHR.BatcherEventBuilder2.Bypass.set(0)
    cbase.EpixHR.BatcherEventBuilder0.Blowoff.set((m&1)==0)
    cbase.EpixHR.BatcherEventBuilder1.Blowoff.set((m&2)==0)
    cbase.EpixHR.BatcherEventBuilder2.Blowoff.set((m&4)==0)
    print(f'Blowoff BatcherEventBuilders {m^0x7:x}')

#
#  Test standalone
#
if __name__ == "__main__":

    _base = epixhr2x2_init(None,dev='/dev/datadev_0')
    epixhr2x2_init_feb()
    epixhr2x2_connectionInfo(_base, None)

    db = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws/configDB'
    d = {'body':{'control':{'0':{'control_info':{'instrument':'tst',
                                                 'cfg_dbase' :db}}}}}

    print('***** CONFIG *****')
    _connect_str = json.dumps(d)
    epixhr2x2_config(_base,_connect_str,'BEAM','epixhr',0,4)

    print('***** SCAN_KEYS *****')
    epixhr2x2_scan_keys(json.dumps(["user.gain_mode"]))

    for i in range(100):
        print(f'***** UPDATE {i} *****')
        epixhr2x2_update(json.dumps({'user.gain_mode':i%5}))

    print('***** DONE *****')

