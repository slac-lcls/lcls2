from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
import pyrogue
import rogue
import lcls2_epix_hr_pcie
import epix100a_gen2 # necessary to pick up sys.path for ePixFpga below
import ePixFpga as fpga
import time
import json
import os
import numpy as np
import logging

base = None
pv = None
lane = 0
chan = None
group = None
segids = None
seglist = [0]
cfgfile = '/cds/group/pcds/dist/pds/tmo/misc/epix100_config.yml'
cfg = None

class EpixBoard(pyrogue.Root):
    def __init__(self, srp, **kwargs):
        super().__init__(name = 'ePixBoard', description = 'ePix 100a Board', **kwargs)

        self.add(fpga.Epix100a(name='ePix100aFPGA', offset=0, memBase=srp, hidden=False, enabled=True))

#
#  Initialize the rogue accessor
#
def epix100_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv

    assert lanemask==1,'Epix100 KCU firmware requires camera to be on lane 0'

#    logging.getLogger().setLevel(40-10*verbosity)
    logging.debug('epix100_init')

    base = {}

    #  Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_epix_hr_pcie.DevRoot(dev           =dev,
                                           enLclsI       =False,
                                           enLclsII      =True,
                                           yamlFileLclsI =None,
                                           yamlFileLclsII=None,
                                           startupMode   =True,
                                           standAloneMode=False,
                                           pgp4          =True,
                                           pollEn        =False,
                                           initRead      =False,
                                           pcieBoardType = 'Kcu1500')
        pbase.__enter__()

        base['pci'] = pbase
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(True)
        time.sleep(1)
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(False)

    # VC0 is the register interface.  See README.md here:
    # https://github.com/slaclab/lcls2-epix-hr-pcie
    VC = 0
    pgpVc0DmaDest = rogue.hardware.axi.AxiStreamDma(dev,(lane*0x100)+VC,True)

    # Create and Connect SRP to VC0 to send register read-write commands
    srp = rogue.protocols.srp.SrpV3()
    # Create and Connect SRP to VC0 DMA destination to send commands
    pyrogue.streamConnectBiDir(pgpVc0DmaDest,srp)

    cbase = EpixBoard(srp)
    cbase.__enter__()
    print('*** Loading camera configuration from',cfgfile)
    cbase.LoadConfig(cfgfile)
    print('*** Done loading camera configuration')

    print(f"KCU FpgaVersion {hex(pbase.DevPcie.AxiPcieCore.AxiVersion.FpgaVersion.get())}")
    print(f"CAM FpgaVersion {hex(cbase.ePix100aFPGA.AxiVersion.FpgaVersion.get())}")
    cbase.ePix100aFPGA.Oscilloscope.enable.set(False)

    base['cam'] = cbase
    # which event-batcher streams to ignore.  this include lowest two bits
    # (0x3, timing), third lowest-bit (0x4) is epix100. ignore upper-bit streams.
    base['bypass'] = 0x38

    #  Enable the environmental monitoring
    # cpo: currently crashes the GUI
    cbase.ePix100aFPGA.SlowAdcRegisters.enable.set(1)
    cbase.ePix100aFPGA.SlowAdcRegisters.StreamPeriod.set(100000000)  # 1Hz
    cbase.ePix100aFPGA.SlowAdcRegisters.StreamEn.set(1)
    cbase.ePix100aFPGA.SlowAdcRegisters.enable.set(0)

    logging.info('epix100_unconfig')
    epix100_unconfig(base)

    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
    if timebase=="119M":
        logging.info('Using timebase 119M')
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
    else:
        logging.info('Using timebase 186M')
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(1)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)

    time.sleep(1)
    pbase.DevPcie.Application.EventBuilder.Bypass.set(base['bypass'])
    return base

def epix100_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)
    assert lane==0,'Epix100 KCU firmware requires camera to be on lane 0'

#
#  Set the local timing ID and fetch the remote timing ID
#
def epix100_connectionInfo(base, alloc_json_str):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.debug('RxId {:x}'.format(rxId))
        txId = timTxId('epix100')
        logging.debug('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        print('*** cpohack rxid')
        #rxId = 0xffffffff
        rxId = 0xfffffffe


    epixhrid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixhrid

    return d

#
#  Called on Configure
#
def epix100_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global group
    global segids
    global cfg

    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    pbase = base['pci']
    pbase.StopRun()
    time.sleep(0.01)
    pbase.StartRun()

    partitionDelay = getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = cfg['user']['start_ns']
    triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
    logging.debug('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        logging.error('partitionDelay {:}  rawStart {:}  triggerDelay {:} clk_period {:} msg_period {:}'.format(partitionDelay,rawStart,triggerDelay,base['clk_period'],base['msg_period']))
        raise ValueError('triggerDelay computes to < 0')

    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[lane].TriggerDelay.set(triggerDelay)
    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[lane].Partition.set(group)

    cbase = base['cam']
    cbase.ePix100aFPGA.EpixFpgaRegisters.RunTriggerDelay.set(0)
    # 11905 taken from LCLS1 fixed delay offset.  see:
    # pds/epix100a/Epix100aConfigurator.cc
    # corresponds to 91.7975us give the 129.6875MHz clock
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerDelay.set(11905)
    cbase.ePix100aFPGA.EpixFpgaRegisters.RunTriggerEnable.set(True)
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerEnable.set(True)
    cbase.ePix100aFPGA.EpixFpgaRegisters.PgpTrigEn.set(True)

    # we had difficulty moving the triggers early enough with the
    # the default setting of this register.  could also put this in yml file.
    cbase.ePix100aFPGA.EpixFpgaRegisters.AcqToAsicR0Delay.set(0)

    baseClockMHz = cbase.ePix100aFPGA.EpixFpgaRegisters.BaseClockMHz.get()
    width_us = cfg['user']['gate_ns']/1000.
    width_clockticks = int(width_us*baseClockMHz)
    cbase.ePix100aFPGA.EpixFpgaRegisters.AsicAcqWidth.set(width_clockticks)

    firmwareVersion = cbase.ePix100aFPGA.AxiVersion.FpgaVersion.get()
    #  Construct the ID
    epixregs = cbase.ePix100aFPGA.EpixFpgaRegisters
    carrierId = [ epixregs.CarrierCardId0.get(),
                  epixregs.CarrierCardId1.get()]
    digitalId = [ epixregs.DigitalCardId0.get(),
                  epixregs.DigitalCardId1.get()]
    analogId  = [ epixregs.AnalogCardId0.get(),
                  epixregs.AnalogCardId1.get()]
    id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                      carrierId[0], carrierId[1],
                                                      digitalId[0], digitalId[1],
                                                      analogId [0], analogId [1])
    cfg['detId:RO']=id
    myfile = open(cfgfile)
    cfg['expert']['cfgyaml']=myfile.read()
    myfile.close()
    return json.dumps(cfg)

def epix100_unconfig(base):
    pbase = base['pci']
    pbase.StopRun()
    return base

def epix100_scan_keys(update):
    # this routine returns json with all config params that will be updated
    print('***epix100_scan_keys:',update)
    # eliminate variables that are not being scanned
    newcfg = {}
    copy_reconfig_keys(newcfg,cfg,json.loads(update))
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(newcfg,cfg,key)
        copy_config_entry(newcfg[':types:'],cfg[':types:'],key)
    return json.dumps(newcfg)

def epix100_update(update):
    # called on each step
    # receive json with the new value of the scan variable and program it
    newcfg = {}
    update_config_entry(newcfg,cfg,json.loads(update))
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(newcfg,cfg,key)
        copy_config_entry(newcfg[':types:'],cfg[':types:'],key)
    # if the input can affect multiple params, update the json if necessary

    # set the register
    partitionDelay = getattr(base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = newcfg['user']['start_ns']
    print('***epix100_update start_ns:',rawStart)
    triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
    logging.debug('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        logging.error('partitionDelay {:}  rawStart {:}  triggerDelay {:} clk_period {:} msg_period {:}'.format(partitionDelay,rawStart,triggerDelay,base['clk_period'],base['msg_period']))
        raise ValueError('triggerDelay computes to < 0')

    base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[lane].TriggerDelay.set(triggerDelay)

    return json.dumps(newcfg)
