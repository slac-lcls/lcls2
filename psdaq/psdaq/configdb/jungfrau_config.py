import time
import json
import weakref

from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.xpmmini import *
from psdaq.cas.xpm_utils import timTxId
from psdaq.utils import submod_lcls2_udp_pcie_apps # Required to find lcls2_udp_pcie_apps

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
        "enLane": lane,
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
    cfg_list = []
    segm_lane = 0
    for segm in detsegm_list:
        cfg = get_config(connect_str, cfgtype, detname, segm)
        ocfg = cfg

        while True:
            if lm & (1 << segm_lane):
                break
            segm_lane += 1

        print(f"Configuring lane: {segm_lane}")
        # Is this the correct register??
        # Not sure the correct register
        #if jungfrau_kcu.DevPcie.AxiPcieCore.AxiPciePhy.LinkStatus.get() != 1:
        #if jungfrau_kcu.Core.Pgp4AxiL.RxStatus.RemRxLinkReady.get() != 1:
        #    raise ValueError("PGP Link Down")

        # Do the descrambling stuff here?
        ###################################
        frameReorg = jungfrau_kcu.DevPcie.UdpFrameReorg[segm_lane]
        frameReorg.Blowoff.set(False)

        for i in range(128):
            frameReorg.RemapLut[i].set(i)
        ###################################

        # Need to do on all AppLanes
        appLane = jungfrau_kcu.DevPcie.Application.AppLane[segm_lane]

        appLane.EventBuilder.Blowoff.set(True)
        appLane.EventBuilder.Blowoff.set(False)
        #appLane.EventBuilder.Bypass.set(0x4)
        appLane.EventBuilder.Bypass.set(0x0)
        #appLane.EventBuilder.Timeout.set(0xffffff)

        appLane.UdpBatcher.Blowoff.set(False)

        trigEventBuf = jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[segm_lane]

        trigEventBuf.Partition.set(grp)


        udpLane = jungfrau_kcu.DevPcie.Hsio.UdpLane[segm_lane]
        # udpLane.EthPhy. ....  # Don't think we need
        # udpLane.UdpEngine ... # <- Probably not needed at all
        time.sleep(0.1)
        # Need to do on multiple lanes

        #  Capture the firmware version to persist in the xtc
        cfg["firmwareVersion"] = jungfrau_kcu.DevPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
        cfg["firmwareBuild"] = jungfrau_kcu.DevPcie.AxiPcieCore.AxiVersion.BuildStamp.get()
        cfg[":types:"].update({"firmwareVersion": "UINT32"})
        cfg[":types:"].update({"firmwareBuild": "CHARSTR"})

        cfg_list.append(json.dumps(cfg))

    segm_lane = 0
    trigEventManager = jungfrau_kcu.DevPcie.Hsio.TimingRx.TriggerEventManager
    jungfrau_kcu.StartRun()
    for i in range(4):
        print("Disabling all lanes")
        trigEventManager.TriggerEventBuffer[i].MasterEnable.set(0)
    for i in range(4):
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
    cfg = {}
    copy_reconfig_keys(cfg, ocfg, json.loads(update))

    user_to_expert(cl, cfg, full=False)

    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[":types:"], ocfg[":types"], key)
    return json.dumps(cfg)


def jungfrau_update(update):
    """Applies an updated configuration to a detector during a scan.

    This function and the <det>_scan_keys function are used in BEBDetector
    config scans.
    """
    global ocfg
    global cl
    #  extract updates
    cfg = {}
    update_config_entry(cfg, ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cl, cfg, full=False)
    #  Apply config
    config_expert(cl, cfg["expert"])
    #  Retain mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[":types:"], ocfg[":types:"], key)
    return json.dumps(cfg)


def jungfrau_unconfig(jungfrau_kcu):
    print("jungfrau_unconfig")

    jungfrau_kcu.StopRun()

    return jungfrau_kcu
