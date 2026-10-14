import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict

from psdaq.utils import enable_epix_100a_gen2 # To find epix100_gen2
import epix100a_gen2  # To find ePixFpga
import ePixFpga as fpga
from psdaq.utils import enable_lcls2_epix_hr_pcie # To find lcls2_epix_hr_pcie
import lcls2_epix_hr_pcie
import pyrogue
import rogue

from psdaq.cas.xpm_utils import timTxId
from psdaq.configdb.det_config import intToBool
from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *

base = None
pv = None
lane = 0
chan = None
group = None
segids = None
seglist = [0]
cfg = None


class EpixBoard(pyrogue.Root):
    def __init__(self, srp, **kwargs):
        super().__init__(name="ePixBoard", description="ePix 100a Board", **kwargs)

        self.add(
            fpga.Epix100a(
                name="ePix100aFPGA", offset=0, memBase=srp, hidden=False, enabled=True
            )
        )


def epix100_init(
    arg, dev="/dev/datadev_0", lanemask=1, xpmpv=None, timebase="186M", verbosity=0
):
    """Initialize rogue root objects for KCU and FEB."""
    global base
    global pv

    assert lanemask == 1, "Epix100 KCU firmware requires camera to be on lane 0"

    logging.debug("epix100_init")

    base = {}

    # Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_epix_hr_pcie.DevRoot(
            dev=dev,
            enLclsI=False,
            enLclsII=True,
            yamlFileLclsI=None,
            yamlFileLclsII=None,
            startupMode=True,
            standAloneMode=False,
            pgp4=True,
            pollEn=False,
            initRead=False,
            pcieBoardType="XilinxKcu1500",
            serverPort=9120,
        )
        pbase.__enter__()

        base["pci"] = pbase
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(True)
        time.sleep(1)
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(False)

    # VC0 is the register interface.  See README.md here:
    # https://github.com/slaclab/lcls2-epix-hr-pcie
    VC = 0
    pgpVc0DmaDest = rogue.hardware.axi.AxiStreamDma(dev, (lane * 0x100) + VC, True)

    # Create and Connect SRP to VC0 to send register read-write commands
    srp = rogue.protocols.srp.SrpV3()
    # Create and Connect SRP to VC0 DMA destination to send commands
    pyrogue.streamConnectBiDir(pgpVc0DmaDest, srp)

    cbase = EpixBoard(srp)
    cbase.__enter__()

    print(
        f"KCU FpgaVersion {hex(pbase.DevPcie.AxiPcieCore.AxiVersion.FpgaVersion.get())}"
    )
    print(f"CAM FpgaVersion {hex(cbase.ePix100aFPGA.AxiVersion.FpgaVersion.get())}")
    cbase.ePix100aFPGA.Oscilloscope.enable.set(False)

    base["cam"] = cbase
    # Which event-batcher streams to ignore.  this include lowest two bits
    # (0x3, timing), third lowest-bit (0x4) is epix100. ignore upper-bit streams.
    base["bypass"] = 0x38

    # Enable the environmental monitoring
    # cpo: currently crashes the GUI
    cbase.ePix100aFPGA.SlowAdcRegisters.enable.set(1)
    cbase.ePix100aFPGA.SlowAdcRegisters.StreamPeriod.set(100000000)  # 1Hz
    cbase.ePix100aFPGA.SlowAdcRegisters.StreamEn.set(1)
    cbase.ePix100aFPGA.SlowAdcRegisters.enable.set(0)

    epix100_unconfig(base)

    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
    if timebase == "119M":
        logging.info("Using timebase 119M")
        base["clk_period"] = 1000 / 119.0
        base["msg_period"] = 238
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
    else:
        logging.info("Using timebase 186M")
        base["clk_period"] = 7000 / 1300.0  # default 185.7 MHz clock
        base["msg_period"] = 200
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(1)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)

    time.sleep(1)
    pbase.DevPcie.Application.EventBuilder.Bypass.set(base["bypass"])
    return base


def epix100_init_feb(slane=None, schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)
    assert lane == 0, "Epix100 KCU firmware requires camera to be on lane 0"


def epix100_connectionInfo(base, alloc_json_str):
    """Set the local timing ID and fetch the remote timing ID."""
    if "pci" in base:
        pbase = base["pci"]
        rxId = (
            pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        )
        print("RxId {:x}".format(rxId))
        txId = timTxId("epix100")
        print("TxId {:x}".format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        print("*** cpohack rxid")
        # rxId = 0xffffffff
        rxId = 0xFFFFFFFE

    epixhrid = "-"

    d = {}
    d["paddr"] = rxId
    d["serno"] = epixhrid

    # Check that the timing link is up from XPM-side
    if rxId != 0xFFFFFFFE:
        from p4p.client.thread import Context
        from p4p.nt.scalar import ntint
        xpm: int = (rxId >> 16) & 0xFF
        port: int = (rxId >> 0) & 0xFF
          
      # Full IP can be gotten with: 10.0.{:}.{:}'.format((rxId>>12)&0xf,100+((rxId>>8)&0xf))
        # We'll	map the	crate ids to various PV	prefixes
        crate_id_map: Dict[int,str] = {
            0: "DAQ:ASC:XPM",
       	    1: "DAQ:NEH:XPM", #	XTPG, RIX, TMO
       	    2: "DAQ:NEH:XPM", #	RIX
       	    3: "DAQ:NEH:XPM", #	TMO
       	    #4 - doesn't exist
      	    5: "DAQ:NEH:XPM", #	XPM10 and 11 in	the FEE
       	    6: "DAQ:FEH:XPM", #	FEH Mezz
            7: "DAQ:FEH:XPM", #	MFX
       	    8: "DAQ:FEH:XPM", #	XPP
            9: "DAQ:FEH:XPM", #XPP  
            15:"DAQ:FEH:XPM", #XPP EPX
       	}
        crate_id: int =	(rxId >> 12) & 0xF
        linkrx_pv: str = f"{crate_id_map[crate_id]}:{xpm}:LinkRxReady{port}"
        ctx: Context = Context("pva")
        ret: ntint = ctx.get(linkrx_pv)
        print(f"INFO:epix100:Checking timing link at: {linkrx_pv}")
        count: int = 0
        while ret != 1:
            print(
                "WARNING:epix100:Timing link (deadtime path) is down! "
                "Attempting to recover."
            )
            # pbase will be defined if rxId is not 0xFFFFFFFE
            #pbase.DevPcie.Hsio.TimingRx.ConfigLclsTimingV2()
            pbase.DevPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyReset()
            count += 1
            time.sleep(1)
            ret = ctx.get(linkrx_pv)
            if ret == 1:
                print(f"INFO:epix100:Timing link recovered after {count} reset(s).")

    return d


def epix100_config(base, connect_str, cfgtype, detname, detsegm, rog):
    """Called during configure transition.

    Configuration is retrieved from the configdb and written out to a temporary
    YAML file. This file is read in using the rogue LoadConfig method to configure
    the camera.
    """
    global group
    global segids
    global cfg

    group = rog

    # Retrieve the full configuration from the configDB
    cfg = get_config(connect_str, cfgtype, detname, detsegm)

    pbase = base["pci"]
    pbase.StopRun()
    time.sleep(0.01)
    pbase.StartRun()

    partitionDelay = getattr(
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
        "PartitionDelay[%d]" % group,
    ).get()
    rawStart = cfg["user"]["start_ns"]
    triggerDelay = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
    logging.debug(
        "partitionDelay {:}  rawStart {:}  triggerDelay {:}".format(
            partitionDelay, rawStart, triggerDelay
        )
    )
    if triggerDelay < 0:
        logging.error(
            "partitionDelay {:}  rawStart {:}  triggerDelay {:} clk_period {:} msg_period {:}".format(
                partitionDelay,
                rawStart,
                triggerDelay,
                base["clk_period"],
                base["msg_period"],
            )
        )
        logging.error(
            f"partitionDelay {partitionDelay}  rawStart {rawStart}  "
            f'triggerDelay {triggerDelay}  clk_period {base["clk_period"]}  '
            f'msg_period {base["msg_period"]}'
        )
        raise ValueError("triggerDelay computes to < 0")

    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[
        lane
    ].TriggerDelay.set(triggerDelay)
    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[
        lane
    ].Partition.set(group)
    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[
        lane
    ].PauseThreshold.set(8)
    cbase = base["cam"]

    # Change and remove keys to prepare dict for YAML translation
    preparedDict = {}
    preparedDict["ePixBoard"] = cfg["expert"]
    preparedDict["ePixBoard"].pop("DevPcie")
    preparedDict["ePixBoard"].pop("cfgyaml")
    # Do same for types
    types = {}
    types["ePixBoard"] = cfg[":types:"]["expert"]
    types[":enum:"] = cfg[":types:"][":enum:"]
    types["ePixBoard"].pop("DevPcie")
    types["ePixBoard"].pop("cfgyaml")
    keys = ["ePixBoard"]

    def repairKeyNames(myDict):
        newDict = {}
        for key in myDict:
            newKey = key
            if key[-2] == "_" and key[-1].isdigit():
                newKey = f"{key[:-2]}[{key[-1]}]"
            if isinstance(myDict[key], dict):
                newDict[newKey] = repairKeyNames(myDict[key])
            else:
                newDict[newKey] = myDict[key]
        return newDict

    def createYaml(inDict, types, keys):
        newDict = {}
        for key in keys:
            if key in inDict:
                newDict[key] = inDict[key]
                intToBool(newDict, types, key)
                newDict[key]["enable"] = True
        newDict = OrderedDict(newDict)
        yaml = pyrogue.dataToYaml(newDict)
        return yaml

    preparedDict = repairKeyNames(preparedDict)
    types = repairKeyNames(types)
    print(preparedDict)
    yaml = createYaml(preparedDict, types, keys)

    path = "/tmp"
    filename = "epix100_cfg.yaml"
    cfgYamlName = f"{path}/{filename}"
    with open(cfgYamlName, "w") as f:
        f.write(yaml)

    print("*** Loading camera configuration from", cfgYamlName)
    cbase.LoadConfig(cfgYamlName)
    print("*** Done loading camera configuration. Removing temporary file")
    os.remove(cfgYamlName)

    cbase.ePix100aFPGA.EpixFpgaRegisters.RunTriggerDelay.set(0)
    # 11905 taken from LCLS1 fixed delay offset.  see:
    # pds/epix100a/Epix100aConfigurator.cc
    # corresponds to 91.7975us give the 129.6875MHz clock
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerDelay.set(11905)
    cbase.ePix100aFPGA.EpixFpgaRegisters.RunTriggerEnable.set(True)
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerEnable.set(True)
    cbase.ePix100aFPGA.EpixFpgaRegisters.PgpTrigEn.set(True)

    # We had difficulty moving the triggers early enough with the
    # the default setting of this register. Could also put this in YAML file.
    cbase.ePix100aFPGA.EpixFpgaRegisters.AcqToAsicR0Delay.set(0)

    baseClockMHz = cbase.ePix100aFPGA.EpixFpgaRegisters.BaseClockMHz.get()
    width_us = cfg["user"]["gate_ns"] / 1000.0
    # width_us = 154000/1000. # Stolen from configdb cat tmo/BEAM/epix100_0
    width_clockticks = int(width_us * baseClockMHz)
    cbase.ePix100aFPGA.EpixFpgaRegisters.AsicAcqWidth.set(width_clockticks)

    firmwareVersion = cbase.ePix100aFPGA.AxiVersion.FpgaVersion.get()
    #  Construct the ID
    epixregs = cbase.ePix100aFPGA.EpixFpgaRegisters
    carrierId = [epixregs.CarrierCardId0.get(), epixregs.CarrierCardId1.get()]
    digitalId = [epixregs.DigitalCardId0.get(), epixregs.DigitalCardId1.get()]
    analogId = [epixregs.AnalogCardId0.get(), epixregs.AnalogCardId1.get()]
    id = "%010d-%010d-%010d-%010d-%010d-%010d-%010d" % (
        firmwareVersion,
        carrierId[0],
        carrierId[1],
        digitalId[0],
        digitalId[1],
        analogId[0],
        analogId[1],
    )
    cfg["detId:RO"] = id
    return json.dumps(cfg)


def epix100_unconfig(base):
    logging.debug("epix100_unconfig")
    pbase = base["pci"]
    pbase.StopRun()
    return base


def epix100_scan_keys(update):
    """Returns an updated config JSON to record in an XTC file.

    This function and the <det>_update function are used in BEBDetector
    config scans.

    This function returns the configuration parameters that will be scanned
    (plus some mandatory XTC fields).
    """
    global cfg
    assert cfg
    print("*** epix100_scan_keys:", update)

    newcfg = {}
    copy_reconfig_keys(newcfg, cfg, json.loads(update))
    # Retain mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(newcfg, cfg, key)
        copy_config_entry(newcfg[":types:"], cfg[":types:"], key)
    return json.dumps(newcfg)


def epix100_update(update):
    """Applies an updated configuration to a detector during a scan.

    This function and the <det>_scan_keys function are used in BEBDetector
    config scans.

    This function is called on each step of the scan. JSON containing the new
    value of the scan variable is passed in which is used to program the camera.
    """
    global cfg
    global base
    assert base
    assert cfg

    newcfg = {}
    update_config_entry(newcfg, cfg, json.loads(update))

    # Retain mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(newcfg, cfg, key)
        copy_config_entry(newcfg[":types:"], cfg[":types:"], key)

    partitionDelay = getattr(
        base["pci"].DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
        "PartitionDelay[%d]" % group,
    ).get()

    rawStart = newcfg["user"]["start_ns"]
    print("***epix100_update start_ns:", rawStart)
    triggerDelay = int(
        rawStart / base["clk_period"] - partitionDelay * base["msg_period"]
    )
    logging.debug(
        "partitionDelay {:}  rawStart {:}  triggerDelay {:}".format(
            partitionDelay, rawStart, triggerDelay
        )
    )
    if triggerDelay < 0:
        logging.error(
            "partitionDelay {:}  rawStart {:}  triggerDelay {:} clk_period {:} msg_period {:}".format(
                partitionDelay,
                rawStart,
                triggerDelay,
                base["clk_period"],
                base["msg_period"],
            )
        )
        raise ValueError("triggerDelay computes to < 0")

    base["pci"].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[
        lane
    ].TriggerDelay.set(triggerDelay)

    return json.dumps(newcfg)


def epix100_external_trigger(base):
    cbase = base['cam']
    # Switch to external triggering
    cbase.ePix100aFPGA.EpixFpgaRegisters.AutoRunEnable.set(0)
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerEnable.set(True)


def epix100_internal_trigger(base):
    cbase = base['cam']
    # Switch to internal triggering
    cbase.ePix100aFPGA.EpixFpgaRegisters.DaqTriggerEnable.set(False)
    cbase.ePix100aFPGA.EpixFpgaRegisters.AutoRunEnable.set(1)


def epix100_enable(base):
    epix100_external_trigger(base)


def epix100_disable(base):
    time.sleep(0.005)  # Need to make sure readout of last event is complete
    epix100_internal_trigger(base)
