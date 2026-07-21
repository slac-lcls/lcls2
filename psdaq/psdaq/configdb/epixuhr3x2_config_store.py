"""Store the basic configuration for ePixUHR3x2 detectors in the config database.

Usage:
python epixuhr3x2_config_store.p;y --user <user> --inst tst --alias BEAM --name epixuhr3x2 --segm 0...

"""

from typing import Dict

from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import update_config
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


def ePixUHR3x2_cdict():

    top = cdict()
    top.setAlg("config", [3, 2, 0])
    top.define_enum("boolEnum", {"False": 0, "True": 1})
    top.set("expert.Core.Si5345Pll.enable", 1, "boolEnum")

    for n in range(1, 7):
        top.set(f"expert.FebFpga.App.Asic[{n}].enable", 1, "boolEnum")
        top.set(f"expert.FebFpga.App.Asic[{n}].TpsDacGain", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].TpsDac", 34, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].TpsGr", 12, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].TpsMux", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasTpsBuffer", 5, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasTps", 4, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasTpsDac", 4, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasDac", 4, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrCtrlDacTps", 3, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrCtrlDacComp", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVthrGain", 3, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVthr", 32, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].PpbitBe", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasPxlCsa", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasPxlBuf", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasAdcComp", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BiasAdcRef", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].CmlRxBias", 3, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].CmlTxBias", 3, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVfiltGain", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVfilt", 30, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVrefCdsGain", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVrefCds", 50, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVprechGain", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacVprech", 34, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrCtrlDacFilt", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrCtrlDacAdcRef", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrCtrlDacPrechCds", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrfCtrlDacAll", 2, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].BgrDisable", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefpGain", 3, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefp", 53, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefnGain", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefn", 12, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefCmGain", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].DacAdcVrefCm", 35, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].AdcCalibEn", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].CompEnGenEn", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].CompEnGenCfg", 5, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].CfgAutoflush", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].ExternalFlushN", 1, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].ClusterDvMask", 16383, "UINT32")
        top.set(f"expert.FebFpga.App.Asic[{n}].PixNumModeEn", 0, "UINT8")
        top.set(f"expert.FebFpga.App.Asic[{n}].SerializerTestEn", 0, "UINT8")

        top.set(f"expert.App.BatcherEventBuilder{n}.enable", 1, "boolEnum")
        top.set(f"expert.App.BatcherEventBuilder{n}.Timeout", 0, "UINT32")

    conv = functools.partial(int, base=16)
    pathPll = "/cds/home/p/psrel/EpixUHR/pll/"

    base = "expert.Pll."
    conv = functools.partial(int, base=16)

    top.set(
        base + "_temp250",
        np.loadtxt(
            pathPll + "PLLConfig_Si5345_temp_250.csv",
            dtype="uint16",
            delimiter=",",
            skiprows=1,
            converters=conv,
        ),
    )
    top.set(
        base + "_2_3_7",
        np.loadtxt(
            pathPll + "Si5345-B-156MHZ-out2-3-7-Registers.csv",
            dtype="uint16",
            delimiter=",",
            skiprows=1,
            converters=conv,
        ),
    )
    top.set(
        base + "_0_5_7",
        np.loadtxt(
            pathPll + "Si5345-B-156MHZ-out-0-5-and-7-Registers.csv",
            dtype="uint16",
            delimiter=",",
            skiprows=1,
            converters=conv,
        ),
    )
    top.set(
        base + "_2_3_9",
        np.loadtxt(
            pathPll + "Si5345-B-156MHZ-out2-3-9.csv",
            dtype="uint16",
            delimiter=",",
            skiprows=1,
            converters=conv,
        ),
    )
    top.set(
        base + "_0_5_7_v2",
        np.loadtxt(
            pathPll + "Si5345-B-156MHZ-out-0-5-and-7-v2-Registers.csv",
            dtype="uint16",
            delimiter=",",
            skiprows=1,
            converters=conv,
        ),
    )

    pathpix = "/cds/home/p/psrel/EpixUHR/pixelBitMaps_prod/"
    # pixelBitMapDic = {'_FL_FM_FH':0, '_FL_FM_FH_InjOff':1, '_allConfigs':2, '_allPx_52':3, '_allPx_AutoHGLG_InjOff':4, '_allPx_AutoHGLG_InjOn':5, '_allPx_AutoMGLG_InjOff':6, '_allPx_AutoMGLG_InjOn':7, '_allPx_FixedHG_InjOff':8, '_allPx_FixedHG_InjOn':9, '_allPx_FixedLG_InjOff':10, '_allPx_FixedLG_InjOn':11, '_allPx_FixedMG_InjOff':12, '_allPx_FixedMG_InjOn':13, '_crilin':14, '_crilin_epixuhr100k':15, '_defaults':16, '_injection_corners':17, '_injection_corners_px1':18, '_management':19, '_management_epixuhr100k':20, '_management_inj':21, '_maskedCSA':22, '_truck':23, '_truck_epixuhr100k':24, '_xtalk_hole':25}
    pixelBitMapDic = {
        "_0_default": 0,
        "_1_injection_truck": 1,
        "_2_injection_corners_FHG": 2,
        "_3_injection_corners_AHGLG1": 3,
        "_4_extra_config": 4,
        "_5_extra_config": 5,
        "_6_truck2": 6,
        "_7_on_the_fly": 7,
    }
    top.define_enum("pixelMapEnum", pixelBitMapDic)

    base = "expert.pixelBitMaps."
    for pixelmap in pixelBitMapDic:
        if "on_the_fly" not in pixelmap:
            top.set(
                base + pixelmap,
                np.loadtxt(
                    f"{pathpix}{pixelmap[1:]}.csv", dtype="uint16", delimiter=","
                ),
            )
    for n in range(1, 7):
        base = f"user.FebFpga.App.Asic[{n}]."
        top.set(base + "PixelBitMapSel", 5, "pixelMapEnum")
        top.set(base + "SetGainValue", 48, "UINT8")


    clkSelEnum: Dict[str, int] = {
        "Lcls1Clock": 0,
        "Lcls2Clock": 1,
    }
    top.define_enum("clkSelEnum", clkSelEnum)
    modeSelEnum: Dict[str, int] = {
        "Lcls1Protocol": 0,
        "Lcls2Protocol": 1,
    }
    top.define_enum("modeSelEnum", modeSelEnum)

    modeSelEnEnum: Dict[str, int] = {
        "UseClkSel": 0,
        "UseModeSel": 1,
    }
    top.define_enum("modeSelEnEnum", modeSelEnEnum)

    # --- FebFpga.App.TimingRx --- #
    top.set("expert.FebFpga.App.TimingRx.enable", 1, "boolEnum")
    for i in range(2):
        top.set(f"expert.FebFpga.App.TimingRx.GtRxAlignCheck[{i}].enable", 1, "boolEnum")
        top.set(f"expert.FebFpga.App.TimingRx.GtRxAlignCheck[{i}].PhaseTarget", 0x10, "UINT8") # Actually UINT7
        top.set(f"expert.FebFpga.App.TimingRx.GtRxAlignCheck[{i}].Mask", 0x7e, "UINT8") # Actually UINT7
        top.set(f"expert.FebFpga.App.TimingRx.GtRxAlignCheck[{i}].ResetLen", 0x3, "UINT8") # Actually UINT4
        top.set(f"expert.FebFpga.App.TimingRx.GtRxAlignCheck[{i}].Override", 0, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.RxPolarity", 0, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.ClkSel", 1, "clkSelEnum") # Actually UINT1
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.RxDown", 0, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.BypassRst", 0, "UINT8") # Actually UINT1
    # RxPllReset is WO
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.ModeSel", 1, "modeSelEnum") # Actually UINT1
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.ModeSelEn", 0, "modeSelEnEnum") # Actually UINT1
    top.set("expert.FebFpga.App.TimingRx.TimingFrameRx.MsgDelay", 0x0, "UINT32") # Actually UINT20
    # SKIPPING XPM MINI UNDER XpmMiniWrapper

    rateTypeEnum: Dict[str, int] = {
        "FixedRates": 0,
        "AcRates": 1,
        "ControlWord": 2,
        "INVALID": 3,
    }
    top.define_enum("rateTypeEnum", rateTypeEnum)
    # 0 == BeamRequest
    # 1 == NotBeamRequest
    # 2 == All
    destTypeEnum: Dict[str, int] = {
        "BeamRequest": 0,
        "NotBeamRequest": 1,
        "All": 2,
        "Invalid": 3,
    }
    top.define_enum("destTypeEnum", destTypeEnum)
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.enable", 1, "boolEnum")
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].enable",
        1,
        "boolEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].EnableReg",
        1,
        "boolEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].RateType",
        2,
        "rateTypeEnum", # Actually UINT2
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].RateSel",
        256, # This is the event code when using event codes
        "UINT16", # Actually UINT11
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].DestType",
        2,
        "destTypeEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[0].DestSel",
        0,
        "UINT16",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].enable",
        1,
        "boolEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].EnableReg",
        0,
        "boolEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].RateType",
        0,
        "rateTypeEnum", # Actually UINT2
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].RateSel",
        0x0, # This is the event code when using event codes
        "UINT16", # Actually UINT11
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].DestType",
        0,
        "destTypeEnum",
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.EvrV2CoreChannels.EvrV2ChannelReg[1].DestSel",
        0,
        "UINT16",
    )

    triggerSourceEnum: Dict[str, int] = {
        "XPM": 0,
        "EVR": 1,
    }
    top.define_enum("triggerSourceEnum", triggerSourceEnum)
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition", 6, "UINT8") # Actually UINT3
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerSource",
        1,
        "triggerSourceEnum"
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold",
        16,
        "UINT8" # Actually UINT5
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay",
        42,
        "UINT32"
    )

    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].MasterEnable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition", 6, "UINT8") # Actually UINT3
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerSource",
        0,
        "triggerSourceEnum"
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].PauseThreshold",
        16,
        "UINT8" # Actually UINT5
    )
    top.set(
        "expert.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay",
        42,
        "UINT32"
    )


    # --- FebFpga.App.WaveformControl --- #
    top.set("expert.FebFpga.App.WaveformControl.enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.UsrRst", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.GlblRstPolarity", 1, "boolEnum")
    # asicRefClockFreq is RO
    top.set("expert.FebFpga.App.WaveformControl.R0Polarity", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.R0Delay", 70, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.R0Width", 1125, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.AcqPolarity", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.AcqDelay", 655, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.AcqWidth", 535, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.SroPolarity", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.SroDelay", 1195, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.SroWidth", 1, "UINT8")
    top.set("expert.FebFpga.App.WaveformControl.InjPolarity", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.InjDelay", 700, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.InjWidth", 535, "UINT32")
    top.set("expert.FebFpga.App.WaveformControl.InjEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.InjSkipFrames", 0, "UINT32")
    # AcqCnt is RO
    top.set("expert.FebFpga.App.WaveformControl.ResetCounters", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.timingRxUserRst", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.timingTxUserRst", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.timingUseMiniTpg", 0, "boolEnum")
    # timingV1LinkUp is RO
    # timingV2LinkUp is RO
    top.set("expert.FebFpga.App.WaveformControl.AsicR0En", 1, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.AsicAcqEn", 1, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.AsicInjEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.AsicSroEn", 1, "boolEnum")
    top.set("expert.FebFpga.App.WaveformControl.AsicClkEn", 0, "boolEnum")

    # --- FebFpga.App.TriggerRegisters --- #
    top.set("expert.FebFpga.App.TriggerRegisters.enable", 1, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.RunTriggerEnable", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.TimingRunTriggerEnable", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.RunTriggerDelay", 0, "UINT32")

    top.set("expert.FebFpga.App.TriggerRegisters.DaqTriggerEnable", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.TimingDaqTriggerEnable", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.DaqTriggerDelay", 1210, "UINT32")

    top.set("expert.FebFpga.App.TriggerRegisters.AutoRunEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.AutoDaqEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.TriggerRegisters.AutoTrigPeriod", 42700000, "UINT32")
    top.set("expert.FebFpga.App.TriggerRegisters.PgpTrigEn", 0, "boolEnum")
    # AcqCount is RO
    # DaqCount is RO
    top.set("expert.FebFpga.App.TriggerRegisters.numberTrigger", 0, "UINT32")
    top.set("expert.FebFpga.App.TriggerRegisters.daqTriggersSkip", 0, "UINT32")
    top.set("expert.FebFpga.App.TriggerRegisters.countDaqTrigEn", 0, "boolEnum")
    # RunPauseCount is RO
    # DaqPauseCount is RO
    top.set("expert.FebFpga.App.TriggerRegisters.daqPauseEn", 0, "boolEnum")

    # --- FebFpga.App.FramerAsic[1...6] --- #
    for i in range(1, 7):
        top.set(f"expert.FebFpga.App.FramerAsic[{i}].enable", 1, "boolEnum")
        top.set(f"expert.FebFpga.App.FramerAsic[{i}].DisableLane", 0x0, "UINT8")
        top.set(f"expert.FebFpga.App.FramerAsic[{i}].EnumerateDisLane", 0x0, "UINT8")

    forcedGainBitEnum: Dict[str, int] = {
        "LowGain": 0,
        "HighGain": 1,
    }
    top.define_enum("TestForcedGainBitEnum", forcedGainBitEnum)
    # --- FebFpga.App.DescrambleAsic[1...6] --- #
    for i in range(1, 7):
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].enable", 1, "boolEnum")
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].DarkTileSub", 0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].GainEqualization", 0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].MirrorImage", 0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].TestEnumerate", 0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].TestForceGainBit", 0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.DescrambleAsic[{i}].TestForcedGainBit", 0, "TestForcedGainBitEnum") # Actually UINT1
        top.set(
            f"expert.FebFpga.App.DescrambleAsic[{i}].DarkHigh",
            "epixuhr-3x2-readout-testing/software/config/calibration/asicDarkHigh.npy",
            "CHARSTR"
        )
        top.set(
            f"expert.FebFpga.App.DescrambleAsic[{i}].GainHigh",
            "epixuhr-3x2-readout-testing/software/config/calibration/asicGainHigh.npy",
            "CHARSTR"
        )
        top.set(
            f"expert.FebFpga.App.DescrambleAsic[{i}].DarkLow",
            "epixuhr-3x2-readout-testing/software/config/calibration/asicDarkLow.npy",
            "CHARSTR"
        )
        top.set(
            f"expert.FebFpga.App.DescrambleAsic[{i}].GainLow",
            "epixuhr-3x2-readout-testing/software/config/calibration/asicGainLow.npy",
            "CHARSTR"
        )

    # --- FebFpga.App.WaveformDeskew --- #
    top.set("expert.FebFpga.App.WaveformDeskew.enable", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformDeskew.gRstLDeskew", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformDeskew.sroDeskew", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformDeskew.roDeskew", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformDeskew.acqDeskew", 0, "boolEnum")
    top.set("expert.FebFpga.App.WaveformDeskew.injDeskew", 0, "boolEnum")

    timingOutEnum: Dict[str, int] = {
        "asicR0": 0,
        "asicACQ": 1,
        "asicSRO": 2,
        "asicInj": 3,
        "asicGlbRstN": 4,
        "timingRunTrigger": 5,
        "timingDaqTrigger": 6,
        "acqStart": 7,
        "dataSend": 8,
        "_0": 9,
        "_1": 10,
    }
    top.define_enum("TimingOutMuxEnum", timingOutEnum)
    # --- FebFpga.App.BoardCtrl3x2Readout --- #
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enable", 1, "boolEnum")
    # prsntCarrierL is RO
    # hwRev is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP1V5DAsic", 1, "boolEnum")
    # pgoodP1V5DAsic is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP1V5AAsicA", 1, "boolEnum")
    # pgoodP1V5AAsicA is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP1V5AAsicB", 1, "boolEnum")
    # pgoodP1V5AAsicB is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP1V3DAsic", 0x3F, "UINT8") # Actually UINT6
    # pgoodP1V3DAsic is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP1V3AAsic", 0xF, "UINT8") # Actually UINT6
    # pgoodP1V3AAsic is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP0V8DAsic", 1, "boolEnum")
    # pgoodP0V8DAsic is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.enableP2V5AAsic", 1, "boolEnum")
    # pgoodP2V5AAsic is RO
    # readoutBoardId is RO
    # carrierBoardId is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.i2cEepromWp", 0, "boolEnum")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.eepromFirstByte", 0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.eepromLastByte", 0, "UINT8")
    for i in range(3):
        top.set(f"expert.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[{i}]", 0, "boolEnum")
        top.set(f"expert.FebFpga.App.BoardCtrl3x2Readout.timingOutSelect[{i}]", 0, "TimingOutMuxEnum")
    # --- FebFpga.App.BoardCtrl3x2Readout.LTM4664_A --- #
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.enable", 1, "boolEnum")
    # i2cAddr is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.PAGE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.OPERATION", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.ON_OFF_CONFIG", 0x1e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.CLEAR_FAULTS", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.WRITE_PROTECT", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.STORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.RESTORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    # CAPABILITY is RO
    # VOUT_MODE is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_COMMAND", 0x1800, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_MAX", 0x19cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_MARGIN_HIGH", 0x1933, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_MARGIN_LOW", 0x16cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_TRANSITION_RATE", 0x8042, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.FREQUENCY_SWITCH", 0xfabc, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VIN_ON", 0xca60, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VIN_OFF", 0xca40, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_OV_FAULT_LIMIT", 0x1a66, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_OV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_OV_WARN_LIMIT", 0x19cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_UV_WARN_LIMIT", 0x1671, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_UV_FAULT_LIMIT", 0x1652, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VOUT_UV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.IOUT_OC_FAULT_LIMIT", 0xe280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.IOUT_OC_FAULT_RESPONSE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.IOUT_OC_WARN_LIMIT", 0xdbc0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.OT_FAULT_LIMIT", 0xf200, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.OT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.OT_WARN_LIMIT", 0xebe8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.UT_FAULT_LIMIT", 0xe530, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.UT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VIN_OV_FAULT_LIMIT", 0xd3e0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VIN_OV_FAULT_RESPONSE", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.VIN_UV_WARN_LIMIT", 0xca53, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.IIN_OC_WARN_LIMIT", 0xd280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TON_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TON_RISE", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TON_MAX_FAULT_LIMIT", 0xca80, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TON_MAX_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TOFF_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TOFF_FALL", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.TOFF_MAX_WARN_LIMIT", 0x8000, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.USER_DATA_03", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.USER_DATA_04", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_CHAN_CONFIG", 0x1d, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_CONFIG_ALL", 0x21, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_FAULT_PROPAGATE", 0x6993, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_PWM_COMP", 0x8e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_PWM_MODE", 0xc7, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_FAULT_RESPONSE", 0xc0, "UINT8")
    # MFR_OT_FAULT_RESPONSE is RO
    # MFR_IOUT_PEAK is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_ADC_CONTROL", 0x0, "UINT16")
    # MFR_IOUT_CAL_GAIN is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_RETRY_DELAY", 0xf3e8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_RESTART_DELAY", 0xf258, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_ADDRESS", 0x4f, "UINT8")
    # MFR_SPECIAL_ID is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_IIN_CAL_GAIN", 0xc200, "UINT16")
    # MFR_COMMON and MFR_TEMPERATURE_2_PEAK are RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_PWM_CONFIG", 0xf3e8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_IOUT_CAL_GAIN_TC", 0xea3c, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_RVIN", 0xeae8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_TEMP_1_GAIN", 0xeaae, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_TEMP_1_OFFSET", 0xea00, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_A.MFR_RAIL_ADDRESS", 0x80, "UINT8")
    # Rest are RO or WO.

    # --- FebFpga.App.BoardCtrl3x2Readout.LTM4664_B --- #
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.enable", 1, "boolEnum")
    # i2cAddr is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.PAGE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.OPERATION", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.ON_OFF_CONFIG", 0x1e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.CLEAR_FAULTS", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.WRITE_PROTECT", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.STORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.RESTORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    # CAPABILITY is RO
    # VOUT_MODE is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_COMMAND", 0xd9b, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_MAX", 0xea0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_MARGIN_HIGH", 0xe49, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_MARGIN_LOW", 0xced, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_TRANSITION_RATE", 0x8042, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.FREQUENCY_SWITCH", 0xf3e8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VIN_ON", 0xca60, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VIN_OFF", 0xca40, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_OV_FAULT_LIMIT", 0xef7, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_OV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_OV_WARN_LIMIT", 0xea0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_UV_WARN_LIMIT", 0xcb9, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_UV_FAULT_LIMIT", 0xca7, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VOUT_UV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.IOUT_OC_FAULT_LIMIT", 0xe280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.IOUT_OC_FAULT_RESPONSE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.IOUT_OC_WARN_LIMIT", 0xdbc0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.OT_FAULT_LIMIT", 0xf200, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.OT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.OT_WARN_LIMIT", 0xebe8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.UT_FAULT_LIMIT", 0xe530, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.UT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VIN_OV_FAULT_LIMIT", 0xd3e0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VIN_OV_FAULT_RESPONSE", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.VIN_UV_WARN_LIMIT", 0xca53, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.IIN_OC_WARN_LIMIT", 0xd280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TON_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TON_RISE", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TON_MAX_FAULT_LIMIT", 0xca80, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TON_MAX_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TOFF_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TOFF_FALL", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.TOFF_MAX_WARN_LIMIT", 0x8000, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.USER_DATA_03", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.USER_DATA_04", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_CHAN_CONFIG", 0x1d, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_CONFIG_ALL", 0x21, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_FAULT_PROPAGATE", 0x6993, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_PWM_COMP", 0x8e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_PWM_MODE", 0xc7, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_FAULT_RESPONSE", 0xc0, "UINT8")
    # MFR_OT_FAULT_RESPONSE is RO
    # MFR_IOUT_PEAK is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_ADC_CONTROL", 0x0, "UINT16")
    # MFR_IOUT_CAL_GAIN is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_RETRY_DELAY", 0xf3e8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_RESTART_DELAY", 0xf258, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_ADDRESS", 0x4f, "UINT8")
    # MFR_SPECIAL_ID is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_IIN_CAL_GAIN", 0xc200, "UINT16")
    # MFR_COMMON and MFR_TEMPERATURE_2_PEAK are RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_PWM_CONFIG", 0x10, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_IOUT_CAL_GAIN_TC", 0xea3c, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_RVIN", 0xeae8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_TEMP_1_GAIN", 0xeaae, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_TEMP_1_OFFSET", 0xea00, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_B.MFR_RAIL_ADDRESS", 0x80, "UINT8")
    # Rest are RO or WO.

    # --- FebFpga.App.BoardCtrl3x2Readout.LTM4664_C --- #
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.enable", 1, "boolEnum")
    # i2cAddr is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.PAGE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.OPERATION", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.ON_OFF_CONFIG", 0x1e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.CLEAR_FAULTS", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.WRITE_PROTECT", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.STORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.RESTORE_USER_ALL", 0x1, "UINT8") # Actually UINT1
    # CAPABILITY is RO
    # VOUT_MODE is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_COMMAND", 0x1800, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_MAX", 0x19cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_MARGIN_HIGH", 0x1933, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_MARGIN_LOW", 0x16cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_TRANSITION_RATE", 0x8042, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.FREQUENCY_SWITCH", 0xfabc, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VIN_ON", 0xca60, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VIN_OFF", 0xca40, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_OV_FAULT_LIMIT", 0x1a66, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_OV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_OV_WARN_LIMIT", 0x19cd, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_UV_WARN_LIMIT", 0x1671, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_UV_FAULT_LIMIT", 0x1652, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VOUT_UV_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.IOUT_OC_FAULT_LIMIT", 0xe280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.IOUT_OC_FAULT_RESPONSE", 0x0, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.IOUT_OC_WARN_LIMIT", 0xdbc0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.OT_FAULT_LIMIT", 0xf200, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.OT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.OT_WARN_LIMIT", 0xebe8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.UT_FAULT_LIMIT", 0xe530, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.UT_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VIN_OV_FAULT_LIMIT", 0xd3e0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VIN_OV_FAULT_RESPONSE", 0x80, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.VIN_UV_WARN_LIMIT", 0xca53, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.IIN_OC_WARN_LIMIT", 0xd280, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TON_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TON_RISE", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TON_MAX_FAULT_LIMIT", 0xca80, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TON_MAX_FAULT_RESPONSE", 0xb8, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TOFF_DELAY", 0x8000, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TOFF_FALL", 0xc300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.TOFF_MAX_WARN_LIMIT", 0x8000, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.USER_DATA_03", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.USER_DATA_04", 0x0, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_CHAN_CONFIG", 0x1d, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_CONFIG_ALL", 0x21, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_FAULT_PROPAGATE", 0x6993, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_PWM_COMP", 0x8e, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_PWM_MODE", 0xc7, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_FAULT_RESPONSE", 0xc0, "UINT8")
    # MFR_OT_FAULT_RESPONSE is RO
    # MFR_IOUT_PEAK is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_ADC_CONTROL", 0x0, "UINT16")
    # MFR_IOUT_CAL_GAIN is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_RETRY_DELAY", 0xf3e8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_RESTART_DELAY", 0xf258, "UINT16")
    # ... many RO registers ...
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_ADDRESS", 0x4f, "UINT8")
    # MFR_SPECIAL_ID is RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_IIN_CAL_GAIN", 0xc200, "UINT16")
    # MFR_COMMON and MFR_TEMPERATURE_2_PEAK are RO
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_PWM_CONFIG", 0x10, "UINT8")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_IOUT_CAL_GAIN_TC", 0xe33c, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_RVIN", 0xe3e8, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_TEMP_1_GAIN", 0xe3ae, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_TEMP_1_OFFSET", 0xe300, "UINT16")
    top.set("expert.FebFpga.App.BoardCtrl3x2Readout.LTM4664_C.MFR_RAIL_ADDRESS", 0x80, "UINT8")
    # Rest are RO or WO.

    # --- FebFpga.App.AsicGtClk --- #
    top.set("expert.FebFpga.App.AsicGtClk.enable", 1, "boolEnum")
    for i in range(6):
        top.set(f"expert.FebFpga.App.AsicGtClk.gtTxData[{i}]", 0xaaaaaaaa, "UINT32")
    top.set("expert.FebFpga.App.AsicGtClk.gtRstAll", 0, "boolEnum")
    top.set("expert.FebFpga.App.AsicGtClk.gtTxPllDataPathRst", 0, "boolEnum")
    top.set("expert.FebFpga.App.AsicGtClk.gtTxDataPathRst", 0, "boolEnum")
    top.set("expert.FebFpga.App.AsicGtClk.gtTxUserClkRst", 0, "boolEnum")

    # --- FebFpga.App.AsicGtData[1...6] --- #
    for i in range(1, 7):
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].enable", 1, "boolEnum")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].gtStableRst", 0, "boolEnum")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].gtRxPllDataPathRst", 0, "boolEnum")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].gtRxDataPathRst", 0, "boolEnum")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].gtRxReset", 0, "boolEnum")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].gtRxPolarity", 0x0, "UINT8")
        top.set(f"expert.FebFpga.App.AsicGtData[{i}].reverse66bits", 0xff, "UINT8")

    # --- FebFpga.App.ClockGeneration --- #
    top.set("user.FebFpga.App.ClockGeneration.enable", 0, "boolEnum")
    top.set("user.FebFpga.App.ClockGeneration.matrixManualRst", 0, "boolEnum")
    top.set("user.FebFpga.App.ClockGeneration.sspManualRst", 0, "boolEnum")

    # --- FebFpga.App.U_matrixClk --- #
    top.set("expert.FebFpga.App.U_matrixClk.enable", 0, "boolEnum")
    for i in range(7):
        top.set(f"expert.FebFpga.App.U_matrixClk.PHASE_MUX[{i}]", 0x0, "UINT8") # Actually UINT3
        ht: int = 0xd if i ==0 else 0x1
        top.set(f"expert.FebFpga.App.U_matrixClk.HIGH_TIME[{i}]", ht, "UINT8") # Actually UINT6
        lt: int = 0xd if i ==0 else 0x1
        top.set(f"expert.FebFpga.App.U_matrixClk.LOW_TIME[{i}]", lt, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.PHASE_MUX_FB", 0x0, "UINT8") # Actually UINT3
    top.set(f"expert.FebFpga.App.U_matrixClk.HIGH_TIME_FB", 0x3, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.LOW_TIME_FB", 0x3, "UINT8") # Actually UINT6
    for i in range(7):
        top.set(f"expert.FebFpga.App.U_matrixClk.FRAC[{i}]", 0x0, "UINT8") # Actually UINT3
        top.set(f"expert.FebFpga.App.U_matrixClk.FRAC_EN[{i}]", 0x0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_matrixClk.FRAC_WF_R[{i}]", 0x0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_matrixClk.MX[{i}]", 0x0, "UINT8") # Actually UINT2
        top.set(f"expert.FebFpga.App.U_matrixClk.EDGE[{i}]", 0x0, "UINT8") # Actually UINT1
        no_count: int = 0x0 if i == 0 else 0x1
        top.set(f"expert.FebFpga.App.U_matrixClk.NO_COUNT[{i}]", no_count, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_matrixClk.DELAY_TIME[{i}]", 0x0, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.FRAC_FB", 0x0, "UINT8") # Actually UINT3
    top.set(f"expert.FebFpga.App.U_matrixClk.FRAC_EN_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_matrixClk.FRAC_WF_R_FB", 0x0, "UINT8") # Actually UINT1
    # MX_FB is WO
    top.set(f"expert.FebFpga.App.U_matrixClk.EDGE_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_matrixClk.NO_COUNT_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_matrixClk.DELAY_TIME_FB", 0x0, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.EDGE_DIV", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_matrixClk.NO_COUNT_DIV", 0x1, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_matrixClk.HIGH_TIME_DIV", 0x1, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.LOW_TIME_DIV", 0x1, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_matrixClk.LockReg[0]", 0x3e8, "UINT16")
    top.set(f"expert.FebFpga.App.U_matrixClk.LockReg[1]", 0x4401, "UINT16")
    top.set(f"expert.FebFpga.App.U_matrixClk.LockReg[2]", 0xc7e9, "UINT16")
    top.set(f"expert.FebFpga.App.U_matrixClk.FiltReg[0]", 0x9908, "UINT16")
    top.set(f"expert.FebFpga.App.U_matrixClk.FiltReg[1]", 0x9190, "UINT16")
    # POWER is WO

    # --- FebFpga.App.U_sspClk --- #
    top.set("expert.FebFpga.App.U_sspClk.enable", 0, "boolEnum")
    for i in range(7):
        top.set(f"expert.FebFpga.App.U_sspClk.PHASE_MUX[{i}]", 0x0, "UINT8") # Actually UINT3
        ht = 0x2 if i ==0 else 0x1
        top.set(f"expert.FebFpga.App.U_sspClk.HIGH_TIME[{i}]", ht, "UINT8") # Actually UINT6
        lt = 0x3 if i ==0 else 0x1
        top.set(f"expert.FebFpga.App.U_sspClk.LOW_TIME[{i}]", lt, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.PHASE_MUX_FB", 0x0, "UINT8") # Actually UINT3
    top.set(f"expert.FebFpga.App.U_sspClk.HIGH_TIME_FB", 0x3, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.LOW_TIME_FB", 0x3, "UINT8") # Actually UINT6
    for i in range(7):
        top.set(f"expert.FebFpga.App.U_sspClk.FRAC[{i}]", 0x0, "UINT8") # Actually UINT3
        top.set(f"expert.FebFpga.App.U_sspClk.FRAC_EN[{i}]", 0x0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_sspClk.FRAC_WF_R[{i}]", 0x0, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_sspClk.MX[{i}]", 0x0, "UINT8") # Actually UINT2
        edge = 0x1 if i == 0 else 0x0
        top.set(f"expert.FebFpga.App.U_sspClk.EDGE[{i}]", edge, "UINT8") # Actually UINT1
        no_count = 0x0 if i == 0 else 0x1
        top.set(f"expert.FebFpga.App.U_sspClk.NO_COUNT[{i}]", no_count, "UINT8") # Actually UINT1
        top.set(f"expert.FebFpga.App.U_sspClk.DELAY_TIME[{i}]", 0x0, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.FRAC_FB", 0x0, "UINT8") # Actually UINT3
    top.set(f"expert.FebFpga.App.U_sspClk.FRAC_EN_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_sspClk.FRAC_WF_R_FB", 0x0, "UINT8") # Actually UINT1
    # MX_FB is WO
    top.set(f"expert.FebFpga.App.U_sspClk.EDGE_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_sspClk.NO_COUNT_FB", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_sspClk.DELAY_TIME_FB", 0x0, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.EDGE_DIV", 0x0, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_sspClk.NO_COUNT_DIV", 0x1, "UINT8") # Actually UINT1
    top.set(f"expert.FebFpga.App.U_sspClk.HIGH_TIME_DIV", 0x1, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.LOW_TIME_DIV", 0x1, "UINT8") # Actually UINT6
    top.set(f"expert.FebFpga.App.U_sspClk.LockReg[0]", 0x3e8, "UINT16")
    top.set(f"expert.FebFpga.App.U_sspClk.LockReg[1]", 0x4401, "UINT16")
    top.set(f"expert.FebFpga.App.U_sspClk.LockReg[2]", 0xc7e9, "UINT16")
    top.set(f"expert.FebFpga.App.U_sspClk.FiltReg[0]", 0x9908, "UINT16")
    top.set(f"expert.FebFpga.App.U_sspClk.FiltReg[1]", 0x9190, "UINT16")
    # POWER is WO

    # --- FebFpga.App.VCALIBP_DAC --- #
    top.set("expert.FebFpga.App.VCALIBP_DAC.enable", 0, "boolEnum")
    top.set("expert.FebFpga.App.VCALIBP_DAC.dacEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.VCALIBP_DAC.dacSingleValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VCALIBP_DAC.rampEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.VCALIBP_DAC.dacStartValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VCALIBP_DAC.dacStopValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VCALIBP_DAC.dacStepValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VCALIBP_DAC.resetDacRamp", 0, "boolEnum")

    # --- FebFpga.App.VINJ_DAC --- #
    top.set("expert.FebFpga.App.VINJ_DAC.enable", 0, "boolEnum")
    top.set("expert.FebFpga.App.VINJ_DAC.dacEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.VINJ_DAC.dacSingleValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VINJ_DAC.rampEn", 0, "boolEnum")
    top.set("expert.FebFpga.App.VINJ_DAC.dacStartValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VINJ_DAC.dacStopValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VINJ_DAC.dacStepValue", 0, "UINT32")
    top.set("expert.FebFpga.App.VINJ_DAC.resetDacRamp", 0, "boolEnum")

    ## --- Create the "user" facing version of those expert registers
    top.set("user.FebFpga.App.VINJ_DAC.enable", 0, "boolEnum")
    top.set("user.FebFpga.App.VINJ_DAC.dacEn", 0, "boolEnum")
    top.set("user.FebFpga.App.VINJ_DAC.dacSingleValue", 0, "UINT32")
    top.set("user.FebFpga.App.VINJ_DAC.rampEn", 0, "UINT8")
    top.set("user.FebFpga.App.VINJ_DAC.SKIP_X", 0, "UINT8") # Pixels to skip in X
    top.set("user.FebFpga.App.VINJ_DAC.SKIP_Y", 0, "UINT8") # Pixels to skip in Y
    top.set("user.FebFpga.App.VINJ_DAC.dacStartValue", 0, "UINT32")
    top.set("user.FebFpga.App.VINJ_DAC.dacStopValue", 0, "UINT32")
    top.set("user.FebFpga.App.VINJ_DAC.dacStepValue", 0, "UINT32")
    top.set("user.FebFpga.App.VINJ_DAC.resetDacRamp", 0, "boolEnum")

    # --- FebFpga.App.AxiAds1217Core --- #
    top.set("expert.FebFpga.App.AxiAds1217Core.enable", 0, "boolEnum")
    top.set("expert.FebFpga.App.AxiAds1217Core.adcStartEnManual", 0, "boolEnum")

    top.set("firmwareBuild:RO", "-", "CHARSTR")
    top.set("firmwareVersion:RO", 0, "UINT32")

    help_str = "-- expert interface --"
    help_str += "\nstart_ns     : exposure start (nanoseconds)"
    top.set("help:RO", help_str, "CHARSTR")

    top.set("user.start_ns", 119000, "UINT32")  # taken from epixHR

    top.define_enum(
        "PllRegEnum", {"temp250": 1, "2_3_7": 2, "0_5_7": 3, "2_3_9": 4, "0_5_7_v2": 5}
    )
    base = "user."

    top.set(base + "PllRegistersSel", 5, "PllRegEnum")

    base = "user.Gain."
    top.set(base + "SetSameGain4All", 0, "boolEnum")
    top.set(base + "UsePixelMap", 0, "boolEnum")

    top.set(base + "SetGainValue", 48, "UINT8")
    top.set(base + "PixelBitMapSel", 5, "pixelMapEnum")

    # top.set("user.run_trigger_group",                                   6               ,'UINT32'  )
    top.set("user.asic_enable", (1 << numAsics) - 1, "UINT32")

    # If True (1) then you will use a detector-serial number based lookup for
    # configuration. Otherwise, the default config will be by alias (as has always
    # been the case)
    top.set("user.use_serial_db", 1, "boolEnum")

    return top


if __name__ == "__main__":
    create = True  # True
    dbname = "configDB"  # this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = "configdb" if args.prod else "devconfigdb"

    mycdb = cdb.configdb(
        f"https://pswww.slac.stanford.edu/ws-auth/{db}/ws/",
        args.inst,
        create,
        root=dbname,
        user=args.user,
        password=args.password,
    )

    top = ePixUHR3x2_cdict()
    top.setInfo("epixuhrhw", args.name, args.segm, args.id, "No comment")

    # no  need for update, value are loaded at creation
    if args.update:
        cfg = mycdb.get_configuration(args.alias, args.name + "_%d" % args.segm)
        top = update_config(cfg, top.typed_json(), args.verbose)

    if not args.dryrun:
        if create:
            mycdb.add_alias(args.alias)
            mycdb.add_device_config("epixuhrhw")
        mycdb.modify_device(args.alias, top)

