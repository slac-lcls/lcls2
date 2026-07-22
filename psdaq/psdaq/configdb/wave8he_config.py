from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import *
from psdaq.configdb.wave8_common import (
    ctxt_get, ctxt_put, confirm_xpm_rxid, config_timing,
    retrieve_config_from_epics,
    set_system_regs, set_raw_buffers, set_batcher_event_builder,
    set_trigger_event_manager, set_adc_readout, set_adc_config,
    set_adc_pattern_tester, set_firmware_info, define_common_enums,
)
from psdaq.cas.xpm_utils import timTxId

import json
import time
import pprint
import logging

prefix = None
ocfg = None
group = None
lane = 0
timebase = "186M"
base = {"timebase": "186M", "prefix": None, "lane": 0}

#
# Wave8HE Configuration for LCLS-HE HLS Processing Firmware
#
# This configuration script supports the Wave8HE detector with HLS (High-Level Synthesis)
# firmware. Key differences from standard Wave8:
#   - Uses HlsProcessor instead of Integrators module
#   - Configures baseline/signal regions (not power-of-2 sizes)
#   - Supports dual position calculation (peak-based and integral-based)
#   - Uses HlsTrigBuffer for pause threshold (single threshold vs 2 in Wave8)
#   - Supports optional FIR filtering bypass


#  Create a dictionary of config key to PV name
def epics_get(d):
    # translate legal Python names to Rogue names
    rogue_translate = {
        "TriggerEventBuffer": "TriggerEventBuffer[0]",
        "AdcReadout0": "AdcReadout[0]",
        "AdcReadout1": "AdcReadout[1]",
        "AdcReadout2": "AdcReadout[2]",
        "AdcReadout3": "AdcReadout[3]",
        "AdcConfig0": "AdcConfig[0]",
        "AdcConfig1": "AdcConfig[1]",
        "AdcConfig2": "AdcConfig[2]",
        "AdcConfig3": "AdcConfig[3]",
    }

    rogue_not_arrays = ["BuffEn", "DelayAdcALane", "DelayAdcBLane"]

    out = {}
    for key, val in d.items():
        #  Skip these that have no PVs yet or are auto-calculated
        if (
            "AdcPatternTester" in key
            or "HlsProcessor.BaselineCnt" in key
            or "HlsProcessor.SignalCnt" in key
            or "HlsProcessor.CalcCfg_blScale" in key
            or "HlsProcessor.CalcCfg_intBlScale" in key
        ):
            continue

        pvname = rogue_translate[key] if key in rogue_translate else key
        if isinstance(val, dict):
            r = epics_get(val)
            for k, v in r.items():
                out[key + "." + k] = pvname + ":" + v
        else:
            if key in rogue_not_arrays:
                for i, v in enumerate(val):
                    out[key + f".[{i}]"] = pvname + "[%d]" % i
            else:
                out[key] = pvname
    return out


def wave8he_init(epics_prefix, dev="/dev/datadev_0", lanemask=1, xpmpv=None, timebase="186M", verbosity=0):
    global prefix
    global lane
    logging.getLogger().setLevel(40 - 10 * verbosity)
    prefix = epics_prefix
    base["prefix"] = epics_prefix
    base["timebase"] = timebase
    lm = lanemask
    lane = (lm & -lm).bit_length() - 1
    assert lm == (1 << lane)  # check that lanemask only has 1 bit for wave8he

    print(f"--- lanemask {lanemask:x}  lane {lane}  timebase {timebase} ---")

    wave8he_unconfig(base)

    return base


def wave8he_init_feb(slane=None, schan=None):
    global lane
    if slane is not None:
        lane = int(slane)


def wave8he_connectionInfo(base, alloc_json_str):
    epics_prefix = base["prefix"]

    #  Switch to LCLS2 Timing
    #    Need this to properly receive RxId
    #    Controls is no longer in-control
    config_timing(epics_prefix, timebase=base["timebase"])

    #  This fails with the current IOC, but hopefully it will be fixed.  It works directly via pgp.
    #txId = timTxId("wave8he")
    txId = timTxId("wave8")
    ctxt_put(epics_prefix + ":Top:TriggerEventManager:XpmMessageAligner:TxId", txId)
    ctxt_put(epics_prefix + ":Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable", 0)

    # Retrieve connection information from EPICS
    # May need to wait for other processes here, so poll
    for i in range(50):
        values = int(ctxt_get(epics_prefix + ":Top:TriggerEventManager:XpmMessageAligner:RxId"))
        if values != 0:
            break
        print("{:} is zero, retry".format(epics_prefix + ":Top:TriggerEventManager:XpmMessageAligner:RxId"))
        time.sleep(0.1)

    # Retrieve the XPM connection information from EPICS
    # to verify a direct connection (not through a fanout)
    #    confirm_xpm_rxid( txId, values, alloc_json_str)

    d = {}
    d["paddr"] = values
    print(f"wave8he_connect returning {d}")
    return d


def detect_version():
    """Detect if the board is a C1100 by reading /proc/datadev_0"""
    file_datadev = "/proc/datadev_0"
    isC1100 = False
    try:
        with open(file_datadev, "r", encoding="utf-8") as file:
            for line in file:
                if "Build String" in line:
                    isC1100 = "C1100" in line
                    break
        return isC1100

    except FileNotFoundError:
        logging.error(f"Error: File '{file_datadev}' not found.")
        return False
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return False


def user_to_expert(prefix, cfg, full=False):
    global group
    global ocfg
    global timebase

    d = {}

    # Calculate trigger delay (same as Wave8)
    try:
        ctrlDelay = ctxt_get(prefix + "TriggerEventManager:EvrV2CoreTriggers:EvrV2TriggerReg[0]:Delay")
        if ctrlDelay is None:
            print("Warning: Failed to retrieve control trigger delay.  Using partition delay as fallback.")
            ctrlDelay = ctxt_get(prefix + "TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay")
            delayFlag = False
        else:
            delayFlag = True
        partitionDelay = ctxt_get(prefix + "TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]" % group)

        clksPerFid = 200 if timebase == "186M" else 238
        nsPerClk = 7000 / 1300.0 if timebase == "186M" else 1000 / 119.0

        if delayFlag:
            #  LCLS2 timing. Let controls set the delay value.
            print("ctrlDelay {:}  partitionDelay {:}".format(ctrlDelay, partitionDelay))

            triggerDelay = int(ctrlDelay - partitionDelay * clksPerFid)

            print("triggerDelay {:}".format(triggerDelay))
            if triggerDelay < 0:
                print("Raise controls trigger delay >= {:} nanoseconds ({:} clock ticks)".format(-triggerDelay * nsPerClk, -triggerDelay))
                raise ValueError("triggerDelay computes to < 0")

            ctxt_put(prefix + "TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay", triggerDelay)

    except KeyError:
        pass

    # Calculate HLS auto-derived values
    # CRITICAL: These calculations MUST match C++ exactly (see HLS README)
    try:
        # Get baseline region values from config
        baseline_beg = cfg.get("expert", {}).get("HlsProcessor", {}).get("BaselineBeg")
        baseline_end = cfg.get("expert", {}).get("HlsProcessor", {}).get("BaselineEnd")

        if baseline_beg is not None and baseline_end is not None:
            # Calculate BaselineCnt
            baseline_cnt = baseline_end - baseline_beg + 1
            ctxt_put(prefix + "HlsProcessor:BaselineCnt", baseline_cnt)

            # Calculate blScale: ((1 << 26) + (cnt // 2)) / cnt
            # This matches C++: m_blScale = ((1 << BlScaleNBits) + (baselineCnt/2)) / ((float) baselineCnt)
            if baseline_cnt != 0:
                bl_scale = int(((1 << 26) + (baseline_cnt // 2)) / baseline_cnt)
                ctxt_put(prefix + "HlsProcessor:CalcCfg_blScale", bl_scale)

        # Get signal region values from config
        signal_beg = cfg.get("expert", {}).get("HlsProcessor", {}).get("SignalBeg")
        signal_end = cfg.get("expert", {}).get("HlsProcessor", {}).get("SignalEnd")

        if signal_beg is not None and signal_end is not None:
            # Calculate SignalCnt
            signal_cnt = signal_end - signal_beg + 1
            ctxt_put(prefix + "HlsProcessor:SignalCnt", signal_cnt)

            # Calculate intBlScale: (signal_cnt / baseline_cnt) * (1 << 26)
            # This matches C++: m_intBlScale = (signalCnt / (float) baselineCnt) * (1 << BlScaleNBits)
            if baseline_beg is not None and baseline_end is not None:
                baseline_cnt = baseline_end - baseline_beg + 1
                if baseline_cnt != 0:
                    int_bl_scale = int((signal_cnt / baseline_cnt) * (1 << 26))
                    ctxt_put(prefix + "HlsProcessor:CalcCfg_intBlScale", int_bl_scale)

    except KeyError:
        pass


def wave8he_config(base, connect_str, cfgtype, detname, detsegm, grp):
    global lane
    global group
    global ocfg
    global timebase

    print(f"base [{base}]")
    prefix = base["prefix"]
    timebase = base["timebase"]
    group = grp

    #  Read the configdb
    cfg = get_config(connect_str, cfgtype, detname, detsegm)
    ocfg = cfg

    #  Apply the user configs
    epics_prefix = prefix + ":Top:"
    user_to_expert(epics_prefix, cfg, full=True)

    #  Assert clears (Wave8HE uses HlsTrigBuffer:CntRst instead of Integrators:CntRst)
    names_clr = [epics_prefix + "BatcherEventBuilder:Blowoff", epics_prefix + "RawBuffers:CntRst", epics_prefix + "HlsTrigBuffer:CntRst"]
    values = [1] * len(names_clr)
    ctxt_put(names_clr, values)

    # Wave8HE: Single pause threshold in HlsTrigBuffer (vs 2 in Wave8 Integrators)
    names_cfg = [
        epics_prefix + "TriggerEventManager:TriggerEventBuffer[0]:Partition",
        epics_prefix + "TriggerEventManager:TriggerEventBuffer[0]:PauseThreshold",
        epics_prefix + "TriggerEventManager:TriggerEventBuffer[0]:MasterEnable",
        epics_prefix + "DataPathCtrl:EnableStream",  # 0x1 for Controls, 0x2 for DAQ
        epics_prefix + "RawBuffers:FifoPauseThreshold",
        epics_prefix + "HlsTrigBuffer:PauseThresh",
    ]
    values = [group, 16, 1, 0x2, 127, 127]
    ctxt_put(names_cfg, values)

    time.sleep(0.2)

    #  Deassert clears
    values = [0] * len(names_clr)
    ctxt_put(names_clr, values)

    #
    #  Now construct the configuration we will record
    #
    top = cdict()
    top.setAlg("config", [2, 0, 0])
    detname_parts = cfg["detName:RO"].rsplit("_", 1)
    top.setInfo(detType="wave8he", detName=detname_parts[0], detSegm=int(detname_parts[1]), detId=cfg["detId:RO"], doc="No comment")

    define_common_enums(top)
    set_firmware_info(top)
    set_system_regs(top)

    # Wave8HE-specific: FirFiltering configuration
    top.set("expert.FirFiltering.BypassFilter", 0, "UINT8")  # 0=use FIR, 1=bypass

    # Wave8HE-specific: HlsTrigBuffer configuration
    top.set("expert.HlsTrigBuffer.PauseThresh", 127, "UINT32")

    # Wave8HE-specific: HlsProcessor - Baseline Region
    top.set("expert.HlsProcessor.BaselineBeg", 20, "UINT8")
    top.set("expert.HlsProcessor.BaselineEnd", 52, "UINT8")
    top.set("expert.HlsProcessor.BaselineCnt", 33, "UINT8")  # auto-calculated

    # Wave8HE-specific: HlsProcessor - Signal Region
    top.set("expert.HlsProcessor.SignalBeg", 100, "UINT8")
    top.set("expert.HlsProcessor.SignalEnd", 114, "UINT8")
    top.set("expert.HlsProcessor.SignalCnt", 15, "UINT8")  # auto-calculated

    # Wave8HE-specific: HlsProcessor - Calculation Config (auto-calculated)
    top.set("expert.HlsProcessor.CalcCfg_blScale", 0, "UINT32")
    top.set("expert.HlsProcessor.CalcCfg_intBlScale", 0, "UINT32")

    # Wave8HE-specific: HlsProcessor - Peak-Based Position
    top.set("expert.HlsProcessor.PosPeak_cx", -0.0198, "FLOAT")
    top.set("expert.HlsProcessor.PosPeak_cy", 0.019, "FLOAT")
    top.set("expert.HlsProcessor.PosPeak_xm", 2, "UINT8")
    top.set("expert.HlsProcessor.PosPeak_xp", 4, "UINT8")
    top.set("expert.HlsProcessor.PosPeak_ym", 1, "UINT8")
    top.set("expert.HlsProcessor.PosPeak_yp", 3, "UINT8")

    # Wave8HE-specific: HlsProcessor - Integral-Based Position
    top.set("expert.HlsProcessor.PosIntegral_cx", -0.0198, "FLOAT")
    top.set("expert.HlsProcessor.PosIntegral_cy", 0.019, "FLOAT")
    top.set("expert.HlsProcessor.PosIntegral_xm", 2, "UINT8")
    top.set("expert.HlsProcessor.PosIntegral_xp", 4, "UINT8")
    top.set("expert.HlsProcessor.PosIntegral_ym", 1, "UINT8")
    top.set("expert.HlsProcessor.PosIntegral_yp", 3, "UINT8")

    set_raw_buffers(top)
    set_batcher_event_builder(top)
    set_trigger_event_manager(top)
    set_adc_readout(top)
    set_adc_config(top)
    set_adc_pattern_tester(top)

    scfg = top.typed_json()

    #  Retrieve full configuration for recording
    retrieve_config_from_epics(epics_prefix, scfg, epics_get)
    version = ctxt_get(prefix + ":Top:AxiVersion:FpgaVersion")
    scfg["firmwareVersion:RO"] = version if version else scfg["firmwareVersion:RO"]

    pprint.pprint(scfg)
    v = json.dumps(scfg)

    if "pci" in base:
        #  Note that other segment levels can step on EventBuilder settings (Bypass,VcDataTap)
        pbase = base["pci"]
        getattr(pbase.DevPcie.Application, f"AppLane[{lane}]").VcDataTap.Tap.set(1)
        eventBuilder = getattr(pbase.DevPcie.Application, f"AppLane[{lane}]").EventBuilder
        eventBuilder.Bypass.set(5)
        eventBuilder.Blowoff.set(False)
        eventBuilder.SoftRst()

    return v


def wave8he_scan_keys(update):
    global prefix
    global ocfg
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg, ocfg, json.loads(update))
    #  Apply group
    user_to_expert(prefix + ":Top:", cfg, full=False)
    #  Retain mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[":types:"], ocfg[":types:"], key)
    return json.dumps(cfg)


def wave8he_update(update):
    global prefix
    global ocfg
    #  extract updates
    cfg = {}
    epics_prefix = prefix + ":Top:"
    update_config_entry(cfg, ocfg, json.loads(update))
    #  Apply group
    user_to_expert(epics_prefix, cfg, full=False)
    #  Retain mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[":types:"], ocfg[":types:"], key)
    return json.dumps(cfg)


#  This is really shutdown/disconnect
def wave8he_unconfig(base):
    epics_prefix = base["prefix"]
    # cpo removed setting Partition=1 (aka readout group) here
    # because this is called in init() and writes fail before the timing
    # the timing system is initialized.  Then subsequent writes start
    # silently failing as well resulting in lost configure phase2.
    names_cfg = [epics_prefix + ":Top:TriggerEventManager:TriggerEventBuffer[0]:MasterEnable"]
    values = [0]
    ctxt_put(names_cfg, values)

    #  Leaving DAQ control.
    config_timing(epics_prefix)

    return None
