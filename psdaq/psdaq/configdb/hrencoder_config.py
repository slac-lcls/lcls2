from psdaq.configdb.get_config import get_config

from psdaq.configdb.scan_utils import *
from psdaq.configdb.xpmmini import *
from psdaq.cas.xpm_utils import timTxId
import rogue
from psdaq.utils import enable_high_rate_encoder_dev
import high_rate_encoder_dev
import time
import json
import IPython
from collections import deque
import logging
import weakref

import pyrogue as pr
import surf.protocols.clink as clink
import rogue.interfaces.stream

hr_enc = None
pv = None
lm: int = 1

# FEB parameters
lane: int = 0
chan: int = 0
ocfg = None
group = None


def cycle_timing_link(hr_enc):
    """Cycles timing link if it is stuck.

    This function should not need to be called with recent firmware
    updates. Previously empirically found by cpo that cycling LCLS1
    and XpmMini timing could be used to get the timing feedback link
    to lock. If needed, call in <det>_init().
    """
    nbad = 0
    while 1:
        # check to see if timing is stuck
        sof1 = hr_enc.App.TimingRx.TimingFrameRx.sofCount.get()
        time.sleep(0.1)
        sof2 = hr_enc.App.TimingRx.TimingFrameRx.sofCount.get()
        if sof1 != sof2:
            break
        nbad += 1
        print("*** Timing link stuck:", sof1, sof2, "resetting. Iteration:", nbad)
        hr_enc.App.TimingRx.ConfigureXpmMini()
        time.sleep(3.5)
        print("Before LCLS timing")
        hr_enc.App.TimingRx.ConfigLclsTimingV2()
        print("After LCLS timing")
        time.sleep(3.5)


def hrencoder_init(
    arg, dev="/dev/datadev_0", lanemask=1, xpmpv=None, timebase="186M", verbosity=0
):
    global pv
    global hr_enc
    global lm
    global lane

    print("hrencoder_init")

    lm = lanemask
    lane = (lm & -lm).bit_length() - 1
    assert lm == (1 << lane)  # check that lanemask only has 1 bit for hrencoder
    myargs = {
        "dev": dev,
        "lane": lane,
        #'dataVcEn': False,      # Whether to open data path in devGui
        "defaultFile": "",  # Empty string to skip config yaml
        "standAloneMode": False,  # False = use fiber timing, True = local timing
        "pollEn": False,  # Enable automatic register polling (True by default)
        "initRead": False,  # Read all registers at init (True by default)
        #'promProg': False,     # Disable all devs not for PROM programming
        #'zmqSrvEn': True,        # Include ZMQ server (True by default)
    }

    hr_enc = high_rate_encoder_dev.Root(**myargs)

    weakref.finalize(hr_enc, hr_enc.stop)
    hr_enc.start()

    return hr_enc


def hrencoder_connectionInfo(hr_enc, alloc_json_str):
    print("hrencoder_connect")

    txId = timTxId("hrencoder")

    rxId = hr_enc.App.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    hr_enc.App.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    hr_enc.StopRun()

    connect_info = {}
    connect_info["paddr"] = rxId

    return connect_info


def user_to_expert(hr_enc, cfg):
    global group

    d = {}
    if "user" in cfg and "delay_ns" in cfg["user"]:
        # 1300/7 MHz - Divide by 1e3 for units (ns vs MHz)
        clk_speed: int = 1300 / 7000
        target_delay_ns: int = cfg["user"]["delay_ns"]
        partition_delay: int = getattr(
            hr_enc.App.TimingRx.TriggerEventManager.XpmMessageAligner,
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


def config_expert(hr_enc, cfg):
    trig_event_buf = getattr(
        hr_enc.App.TimingRx.TriggerEventManager, "TriggerEventBuffer[0]"
    )

    trigger_delay: int = cfg["TriggerDelay"]
    trig_event_buf.TriggerDelay.set(trigger_delay)

    pause_thresh: int = cfg["PauseThreshold"]
    trig_event_buf.PauseThreshold.set(pause_thresh)


def hrencoder_config(hr_enc, connect_str, cfgtype, detname, detsegm, grp):
    global ocfg
    global group

    print("hrencoder_config")
    group = grp  # Assign before calling other functions.

    cfg = get_config(connect_str, cfgtype, detname, detsegm)
    ocfg = cfg

    if hr_enc.Core.Pgp4AxiL.RxStatus.RemRxLinkReady.get() != 1:
        raise ValueError("PGP Link Down")

    trig_event_buf = getattr(
        hr_enc.App.TimingRx.TriggerEventManager, "TriggerEventBuffer[0]"
    )

    trig_event_buf.TriggerSource.set(0)  # Set trigger source to XPM NOT Evr

    hr_enc.App.EventBuilder.Blowoff.set(True)
    user_to_expert(hr_enc, cfg)
    config_expert(hr_enc, cfg["expert"])

    trig_event_buf.Partition.set(grp)

    time.sleep(0.1)
    hr_enc.App.EventBuilder.Blowoff.set(False)

    # Bypass BEB 3 (full timing stream) - not needed for encoder
    # and has minimal (no) buffer.
    hr_enc.App.EventBuilder.Bypass.set(0x4)
    hr_enc.App.EventBuilder.Timeout.set(0x0)

    #  Capture the firmware version to persist in the xtc
    cfg["firmwareVersion"] = hr_enc.Core.AxiVersion.FpgaVersion.get()
    cfg[":types:"].update({"firmwareVersion": "UINT32"})

    hr_enc.StartRun()

    return json.dumps(cfg)


def hrencoder_scan_keys(update):
    """Returns an updated config JSON to record in an XTC file.

    This function and the <det>_update function are used in BEBDetector
    config scans.
    """
    global ocfg
    global hr_enc
    print("hrencoder_scan_keys")
    cfg = {}
    copy_reconfig_keys(cfg, ocfg, json.loads(update))

    user_to_expert(cl, cfg, full=False)

    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[":types:"], ocfg[":types"], key)
    return json.dumps(cfg)


def hrencoder_update(update):
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


def hrencoder_unconfig(hr_enc):
    print("hrencoder_unconfig")

    hr_enc.StopRun()

    return hr_enc
