import time
import json
import weakref

from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import copy_reconfig_keys, copy_config_entry, update_config_entry
from psdaq.configdb.xpmmini import *
from psdaq.cas.xpm_utils import timTxId
from psdaq.utils import enable_lcls2_udp_pcie_apps # Required to find lcls2_udp_pcie_apps

import lcls2_udp_pcie_apps # Found after importing submod_lcls2_udp_pcie_apps
import rogue

jungfrau_kcu = None
pv = None
lm: int = 1

# FEB parameters
lane: int = 0
chan: int = 0
ocfg = None
group = None


def jungfrau_init(
    arg, dev="/dev/datadev_0", lanemask=1, xpmpv=None, timebase="186M", verbosity=0
):
    global pv
    global jungfrau_kcu
    global lm
    global lane

    print("jungfrau_init")

    lm = lanemask
    lane = (lanemask & -lanemask).bit_length() - 1 # May be multiple lanes
    myargs = {
        "dev": dev,
        "enLane": lanemask,
        #'dataVcEn': False,      # Whether to open data path in devGui
        #"defaultFile": "",  # Empty string to skip config yaml
        "defaultFile": "",
        "standAloneMode": False,  # False = use fiber timing, True = local timing
        "pollEn": False,  # Enable automatic register polling (True by default)
        "initRead": False,  # Read all registers at init (True by default)
        #'promProg': False,     # Disable all devs not for PROM programming
        #'zmqSrvEn': True,        # Include ZMQ server (True by default)
    }

    jungfrau_kcu = lcls2_udp_pcie_apps.DevRoot(**myargs) # Requires the defaultFile parameter...

    weakref.finalize(jungfrau_kcu, jungfrau_kcu.stop)
    jungfrau_kcu.start()

    return jungfrau_kcu


def jungfrau_connectionInfo(jungfrau_kcu, alloc_json_str):
    print("jungfrau_connect")

    txId = timTxId("jungfrau")

    rxId = jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    jungfrau_kcu.StopRun()

    connect_info = {}
    connect_info["paddr"] = rxId

    return connect_info


def user_to_expert(jungfrau_kcu, cfg):
    global group

    d = {}
    if "user" in cfg and "delay_ns" in cfg["user"]:
        # 1300/7 MHz - Divide by 1e3 for units (ns vs MHz)
        clk_speed: float = 1300 / 7000
        target_delay_ns: int = cfg["user"]["delay_ns"]
        partition_delay: int = getattr(
            jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
            f"PartitionDelay[{group}]",
        ).get()
        target_delay_clks: int = round(
            target_delay_ns * clk_speed - partition_delay * 200
        )
        print(
            f"group: {group}\t\tdelay_ns: {target_delay_ns}\t\tdelay_clks: {target_delay_clks}"
        )
        if target_delay_clks < 0:
            raise ValueError(
                f"Requested target delay is < 0!\n"
                f"Raise delay_ns >= {partition_delay*200*7000/1300}"
            )

        d["expert.TriggerDelay"] = target_delay_clks

    update_config_entry(cfg, ocfg, d)


def config_expert(jungfrau_kcu, cfg):
    trig_event_buf = getattr(
        jungfrau_kcu.App.TimingRx.TriggerEventManager, "TriggerEventBuffer[0]"
    )

    trigger_delay: int = cfg["TriggerDelay"]
    trig_event_buf.TriggerDelay.set(trigger_delay)

    pause_thresh: int = cfg["PauseThreshold"]
    trig_event_buf.PauseThreshold.set(pause_thresh)


def jungfrau_config(jungfrau_kcu, connect_str, cfgtype, detname, detsegm, grp):
    # detsegm is either int (1 module) or "_" delimited string (multiple modules)
    global ocfg
    global group
    global lm # May be multiple lanes

    print("jungfrau_config")
    group = grp  # Assign before calling other functions.

    detsegm_list = []
    if isinstance(detsegm,int):
        detsegm_list.append(detsegm)
    else:
        for segm in detsegm.split("_"):
            detsegm_list.append(int(segm))
    ocfg = [] # per segment dicts for later use (config scans)
    cfg_list = [] # Json strings
    segm_lane = 0
    jungfrau_kcu.StopRun()
    # reset the card if needed
    jungfrau_reset(jungfrau_kcu)
    for segm in detsegm_list:
        cfg = get_config(connect_str, cfgtype, detname, segm)

        while True:
            if lm & (1 << segm_lane):
                break
            segm_lane += 1

        print(f"Configuring lane: {segm_lane}")
        udpLane = jungfrau_kcu.DevPcie.Hsio.UdpLane[segm_lane]
        udpLane.UdpEngine.SoftMac.set(cfg["user"]["kcu_mac"])
        udpLane.UdpEngine.SoftIp.set(cfg["user"]["kcu_ip"])
        if udpLane.EthPhy.phyReady.get() != 1:
            raise ValueError(f"PGP Link Down for lane: {segm_lane}")

        # Do some of the descrambling stuff here?
        #########################################
        frameReorg = jungfrau_kcu.DevPcie.UdpFrameReorg[segm_lane]

        npackets = 128
        center = (npackets // 2)
        for packet in range(npackets):
            idx = packet // 2
            if packet % 2 == 0:
                remap = center + idx
            else:
                remap = center - idx - 1
            frameReorg.RemapLut[packet].set(remap)
        #########################################

        # Need to do on all AppLanes
        appLane = jungfrau_kcu.DevPcie.Application.AppLane[segm_lane]

        appLane.EventBuilder.Bypass.set(0x0)
        appLane.EventBuilder.Timeout.set(0x0)

        trigEventBuf = jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[segm_lane]

        trigEventBuf.Partition.set(grp)
        time.sleep(0.1)

        #  Capture the firmware version to persist in the xtc
        cfg["firmwareVersion"] = jungfrau_kcu.DevPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
        cfg["firmwareBuild"] = jungfrau_kcu.DevPcie.AxiPcieCore.AxiVersion.BuildStamp.get()
        cfg[":types:"].update({"firmwareVersion": "UINT32"})
        cfg[":types:"].update({"firmwareBuild": "CHARSTR"})

        ocfg.append(cfg)
        cfg_list.append(json.dumps(cfg))
        # Increment segm_lane once at the end of this lane's configuration
        segm_lane += 1

    segm_lane = 0
    trigEventManager = jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager
    jungfrau_kcu.StartRun()
    print("Disabling all lanes")
    for i in range(7):
        trigEventManager.TriggerEventBuffer[i].MasterEnable.set(0)
    for i in range(7):
        if lm & (1 << i):
            print(f"Enabling lane {i}")
            trigEventManager.TriggerEventBuffer[i].MasterEnable.set(1)

    return cfg_list


def jungfrau_scan_keys(update):
    """Returns an updated config JSON to record in an XTC file.

    This function and the <det>_update function are used in BEBDetector
    config scans.
    """
    global ocfg
    global jungfrau_kcu
    print("jungfrau_scan_keys")

    if ocfg is None:
        raise ValueError("ocfg is None! Check jungfrau_config.py")

    updated_cfgs = []

    for segcfg in ocfg:
        cfg = {}
        copy_reconfig_keys(cfg, segcfg, json.loads(update))

        #user_to_expert(jungfrau_kcu, cfg)

        for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
            copy_config_entry(cfg, segcfg, key)
            copy_config_entry(cfg[":types:"], segcfg[":types:"], key)

        updated_cfgs.append(json.dumps(cfg))
    return updated_cfgs


def jungfrau_update(update):
    """Applies an updated configuration to a detector during a scan.

    This function and the <det>_scan_keys function are used in BEBDetector
    config scans.
    """
    global ocfg
    global jungfrau_kcu

    if ocfg is None:
        raise ValueError("ocfg is None! Check jungfrau_config.py")

    updated_cfgs = []

    for segcfg in ocfg:
        cfg = {}
        update_config_entry(cfg, segcfg, json.loads(update))

        #user_to_expert(jungfrau_kcu, cfg)
        #config_expert(jungfrau_kcu, cfg["expert"])

        for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
            copy_config_entry(cfg, segcfg, key)
            copy_config_entry(cfg[":types:"], segcfg[":types:"], key)
        updated_cfgs.append(json.dumps(cfg))
    return updated_cfgs


def jungfrau_unconfig(jungfrau_kcu):
    print("jungfrau_unconfig")

    jungfrau_kcu.StopRun()

    return jungfrau_kcu


def jungfrau_reset(jungfrau_kcu):
    """Checks if the passed jungfrau_kcu needs reset to clear batcher desync.

    This function checks if there is non-zero rxOverFlowCnt for any of the UDP
    lanes. Packets lost to overflow leaves the UDP batcher in a bad state, and
    this can be fixed by reseting the card.
    """
    global lm

    reset = False
    # check if any udp lanes have rxOverFlowCnt > 0
    for segm_lane in range(7):
        if lm & (1 << segm_lane):
            ethPhy = jungfrau_kcu.DevPcie.Hsio.UdpLane[segm_lane].EthPhy
            if ethPhy.rxOverFlowCnt.get() > 0:
                print("possible udp batcger desync on lane", segm_lane)
                reset = True

    if (reset):
        print("calling UserRst to clear udp batcher desync")
        jungfrau_kcu.DevPcie.AxiPcieCore.AxiVersion.UserRst()
        time.sleep(2.)
        jungfrau_kcu.DevPcie.Hsio.TimingRx.TimingPhyMonitor.TxUserRst()
        time.sleep(1.)
        print("finished waiting for reset to complete")

    return jungfrau_kcu
