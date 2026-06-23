import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
import surf.protocols.batcher as batcher

from psdaq.utils import enable_epix_uhr3x2
import epixuhr_3x2_readout_testing as epixUhrDev


class TimingParameters(TypedDict):
    bypass: List[int]
    clk_period: float
    msg_period: int
    pcie_timing: bool


class EpixUHR3x2_Manager:
    def __init__(
        self,
        root: epixUhrDev.Root,
        nasics: int = 6,
        readout_system_num: int = 0,
        logger_name: str = "ePixUHR3x2",
        emulator: bool = True,
    ):
        self._root: epixUhrDev.Root = root
        self._nasics: int = nasics

        self._logger_name: str = logger_name

        self._timing_params: TimingParameters = {
            "bypass": [0, 0, 0, 0, 0, 0],
            "clk_period": 7000 / 1300.0,
            "msg_period": 200,
            "pcie_timing": False,
        }

        self._ros_num: int = readout_system_num
        self._emulator: bool = emulator

    @property
    def emulator(self) -> bool:
        return self._emulator

    def write_and_check(self, register, value):
        logger: logging.Logger = logging.getLogger(self._logger_name)
        if register.get() != value:
            register.set(value)

            if register.get() != value:
                # May or may not be critical...
                logger.error(f"Failed to write {value} to {register}.")
            else:
                logger.debug(f"Wrote {value} to {register} succesfully.")
        else:
            logger.debug(f"Skip writing {value} to {register}. It was already set.")

    def write_many(self, base_node, registers_and_vals: Dict[str, Any]):
        for reg_name, value in registers_and_vals.items():
            register = getattr(base_node, reg_name)
            self.write_and_check(register, value)

    def _get_nonpython_name(self, base_node, template: str, asic: int):
        return getattr(base_node, template.format(asic=asic))

    def ReadAll(self):
        self._root.ReadAll()

    def Stop(self):
        logger: logging.Logger = logging.getLogger(self._logger_name)
        self.ReadoutSystem.StopRun()
        time.sleep(0.1)
        logger.info("Stopping run.")

    def _kick_data_path(self, use_cpu: bool = True):
        """The data path may read as one or the other, but this may not be correct.

        Args:
            use_cpu (bool): If True, data will be routed to the CPU.
        """
        target_val: int = 0x0 if use_cpu else 0x1
        kick_val: int = 0x1 if use_cpu else 0x0
        if hasattr(self.ReadoutSystem, "DataDestination"):
            self.ReadoutSystem.DataDestination.set(kick_val)
            time.sleep(0.01)
            self.ReadoutSystem.DataDestination.set(target_val)

    def Start(self):
        logger: logging.Logger = logging.getLogger(self._logger_name)

        self.set_timing_trigger()
        self.setup_event_batchers()

        self.ReadoutSystem.StartTimingRun()

        # Should've been done before calling this... but double check
        trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[0]
        self.write_and_check(register=trig_event_buf.MasterEnable, value=True)

        trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[1]
        self.write_and_check(register=trig_event_buf.MasterEnable, value=True)
        logger.info("Starting run.")

    # Config C1100
    ##############

    @property
    def CfgFpga(self):
        return self._root.CfgFpga

    # Data C1100
    ###################

    @property
    def ReadoutSystem(self):
        return getattr(self._root, f"ROS[{self._ros_num}]")

    @property
    def DataFpga(self):
        return getattr(self.ReadoutSystem, "DataFpga[0]")

    @property
    def DataAxiVersion(self):
        return self.DataFpga.AxiPcieCore.AxiVersion

    @property
    def c1100_firmware_version(self):
        return self.DataAxiVersion.FpgaVersion.get()

    @property
    def c1100_build_date(self):
        return self.DataAxiVersion.BuildDate.get()

    @property
    def c1100_build_hash(self):
        return self.DataAxiVersion.GitHashShort.get()

    # EpixUHR FEB (Front end board)
    ###############################

    @property
    def FebFpga(self):
        return self.ReadoutSystem.FebFpga

    @property
    def FebTimingRx(self):
        return self.FebFpga.App.TimingRx

    @property
    def FebTriggerEventManager(self):
        return self.FebTimingRx.TriggerEventManager

    @property
    def FebXpmMessageAligner(self):
        return self.FebTriggerEventManager.XpmMessageAligner

    def _get_feb_asic(self, asic: int):
        return getattr(self.FebFpga.App, f"Asic[{asic}]")

    @property
    @lru_cache
    def FebAsics(self):
        asic_dict: Dict[int, Any] = {}
        for i in range(1, self._nasics + 1):
            # asic_dict[i] = self._get_feb_asic(asic=i)
            asic_dict[i] = self._get_nonpython_name(
                base_node=self.FebFpga.App, template="Asic[{asic}]", asic=i
            )

        return asic_dict

    @property
    @lru_cache
    def FebFramerAsics(self):
        asic_dict: Dict[int, Any] = {}
        for i in range(1, self._nasics + 1):
            # asic_dict[i] = self._get_feb_asic(asic=i)
            asic_dict[i] = self._get_nonpython_name(
                base_node=self.FebFpga.App, template="FramerAsic[{asic}]", asic=i
            )

        return asic_dict

    @classmethod
    def _setup_asic_getters(cls, nasics: int = 6) -> None:
        def make_getter(asic: int):
            def getter(self):
                return self._get_feb_asic(asic)

            return getter

        for i in range(1, nasics + 1):
            setattr(
                EpixUHR3x2_Manager,
                f"Asic{i}",
                property(
                    fget=make_getter(asic=i),
                    fset=None,
                    fdel=None,
                    doc="...placeholder...",
                ),
            )

    @property
    def FebWaveformControl(self):
        return self.FebFpga.App.WaveformControl

    @property
    def FebTriggerRegisters(self):
        return self.FebFpga.App.TriggerRegisters

    def init_waveform_control(self, raw_init: bool = False) -> None:
        logger: logging.Logger = logging.getLogger(self._logger_name)
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "GlblRstPolarity": True,
            "AsicSroEn": True,
            "AsicClkEn": True,
            "SroPolarity": False,
            "SroDelay": 1195,
            "SroWidth": 1,
            "AsicAcqEn": True,
            "AcqPolarity": False,
            "AcqDelay": 655,
            "AcqWidth": 535,
            "R0Polarity": False,
            "R0Delay": 70,
            "R0Width": 1125,
            "AsicR0En": True,
            "InjPolarity": False,
            "InjDelay": 700,  # +45 (1 us) after AcqDelay
            "InjWidth": 535,
            "InjEn": False,
            "InjSkipFrames": 0,
            "ResetCounters": False,
        }
        if raw_init:
            # Mirror panel_init defaults if they differ
            pass

        self.write_many(
            base_node=self.FebWaveformControl, registers_and_vals=registers_and_vals
        )

        time.sleep(1)
        self.write_and_check(self.FebWaveformControl.GlblRstPolarity, value=False)
        time.sleep(1)
        self.write_and_check(self.FebWaveformControl.GlblRstPolarity, value=True)
        logger.info("Done with Waveform configuration")

    def running_waveform_control(self) -> None:
        self.init_waveform_control()

    def init_trigger_registers(self) -> None:
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "RunTriggerEnable": False,
            "RunTriggerDelay": 0,
            "DaqTriggerEnable": False,
            "DaqTriggerDelay": 0,
            "TimingRunTriggerEnable": False,
            "TimingDaqTriggerEnable": False,
            "AutoRunEn": False,
            "AutoDaqEn": False,
            "AutoTrigPeriod": 42700000,
            "numberTrigger": 0,
            "PgpTrigEn": False,
        }
        self.write_many(
            base_node=self.FebTriggerRegisters, registers_and_vals=registers_and_vals
        )

    def running_trigger_registers(self, rog: int, start_ns: int, trig_cfg: Dict[str, Any]) -> Tuple[int, int]:
        logger: logging.Logger = logging.getLogger(self._logger_name)

        # Make sure triggers are not running before doing this
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "RunTriggerEnable": False,
        }
        self.write_many(
            base_node=self.FebTriggerRegisters, registers_and_vals=registers_and_vals
        )

        # Now write the currect configuration...
        # This immediately re-enables the trigger... but I am copying from rogue

        daq_delay: int = self.setup_daq_trigger(rog=rog, start_ns=start_ns, trig_cfg=trig_cfg)
        run_delay: int = self.setup_run_trigger(rog=rog, start_ns=start_ns, trig_cfg=trig_cfg)

        # TODO: At the moment overwriting the trigger delay. Will update this later
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "RunTriggerEnable": True,
            # "RunTriggerDelay": 0,
            "DaqTriggerEnable": True,
            "DaqTriggerDelay": 1210,
            "AutoRunEn": False,
            "AutoDaqEn": False,
            "AutoTrigPeriod": 12000, # 42.7 MHz / framerate. ~ 12000 frame/s
            # "AutoTrigPeriod": 42700000,  # 42.7 MHz / framerate. ~ 1 frame/s
            # "PgpTrigEn": True,
            "daqPauseEn": False,
            #"PgpTrigEn": False,
            "PgpTrigEn": True,
            #"countDaqTrigEn": False,
            "countDaqTrigEn": True,
        }
        self.write_many(
            base_node=self.FebTriggerRegisters, registers_and_vals=registers_and_vals
        )
        logger.info("Done with Trigger Registers configuration")

        return daq_delay, run_delay

    def _init_asic(self, asic: int) -> None:
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "TpsDacGain": 1,
            "TpsDac": 34,
            "TpsGr": 12,
            "TpsMux": 0,
            "BiasTpsBuffer": 5,
            "BiasTps": 4,
            "BiasDac": 4,
            "BgrCtrlDacTps": 3,
            "BgrCtrlDacComp": 0,
            "DacVthrGain": 2,
            "DacVthr": 52,
            "PpbitBe": 1,
            "BiasPxlCsa": 0,  # 1
            "BiasPxlBuf": 0,  # 1
            "BiasAdcComp": 0,
            "BiasAdcRef": 0,
            "CmlRxBias": 3,
            "CmlTxBias": 3,
            "CmlTxBias": 3,
            "DacVfiltGain": 2,
            "DacVfilt": 28,
            "DacVrefCdsGain": 2,
            "DacVrefCds": 44,
            "DacVprechGain": 2,
            "DacVprech": 34,
            "BgrCtrlDacFilt": 2,
            "BgrCtrlDacAdcRef": 2,
            "BgrCtrlDacPrechCds": 2,
            "BgrfCtrlDacAll": 2,
            "BgrDisable": 0,
            "DacAdcVrefpGain": 3,
            "DacAdcVrefp": 53,
            "DacAdcVrefnGain": 0,
            "DacAdcVrefn": 12,
            "DacAdcVrefCmGain": 1,
            "DacAdcVrefCm": 45,
            "AdcCalibEn": 0,
            "CompEnGenEn": 1,
            "CompEnGenCfg": 5,
            "CfgAutoflush": 0,
            "ExternalFlushN": 1,
            "ClusterDvMask": 16383,
            "PixNumModeEn": 0,
            "SerializerTestEn": 0,
        }

        self.write_many(
            base_node=self.FebAsics[asic], registers_and_vals=registers_and_vals
        )

    def _set_running_asic(self, asic: int, reg_cfg: Dict[str, Any] = {}) -> None:
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "TpsDacGain": 1,
            "TpsDac": 34,
            "TpsGr": 12,
            "TpsMux": 0,
            "BiasTpsBuffer": 5,
            "BiasTps": 4,
            "BiasTpsDac": 4,
            "BiasDac": 4,
            "BgrCtrlDacTps": 3,
            "BgrCtrlDacComp": 0,
            "DacVthrGain": 2,
            "DacVthr": 52,
            "PpbitBe": 1,
            "BiasPxlCsa": 1,  # 1
            "BiasPxlBuf": 1,  # 1
            "BiasAdcComp": 1,
            "BiasAdcRef": 1,
            "CmlRxBias": 3,
            "CmlTxBias": 3,
            "DacVfiltGain": 2,
            "DacVfilt": 30,  # Set tond 25 for 100 kHz
            "DacVrefCdsGain": 2,
            "DacVrefCds": 44,
            "DacVprechGain": 2,
            "DacVprech": 34,
            "BgrCtrlDacFilt": 2,
            "BgrCtrlDacAdcRef": 2,
            "BgrCtrlDacPrechCds": 2,
            "BgrfCtrlDacAll": 2,
            "BgrDisable": 0,
            "DacAdcVrefpGain": 3,
            "DacAdcVrefp": 53,
            "DacAdcVrefnGain": 0,
            "DacAdcVrefn": 12,
            "DacAdcVrefCmGain": 1,
            "DacAdcVrefCm": 45,
            "AdcCalibEn": 0,
            "CompEnGenEn": 1,
            "CompEnGenCfg": 5,
            "CfgAutoflush": 0,
            "ExternalFlushN": 1,
            "ClusterDvMask": 16383,
            # "PixNumModeEn": 1,
            # "SerializerTestEn": 0,
        }

        registers_and_vals.update(reg_cfg)

        self.write_many(
            base_node=self.FebAsics[asic], registers_and_vals=registers_and_vals
        )

    def _set_running_framer_asic(self, asic: int) -> None:
        registers_and_vals: Dict[str, Any] = {
            "enable": True,
            "DisableLane": 0,
        }

        self.write_many(
            base_node=self.FebFramerAsics[asic], registers_and_vals=registers_and_vals
        )

    def set_running_asics(self, asics: List[int], app_cfg: Dict[str, Any] = {}) -> None:
        logger: logging.Logger = logging.getLogger(self._logger_name)
        for asic in asics:
            logger.info(f"Configuring Asic {asic}")
            asic_reg_cfg: Dict[str, Any] = app_cfg[f"Asic[{asic}]"]
            self._set_running_asic(asic=asic, reg_cfg=asic_reg_cfg)
            time.sleep(0.1)
            self._set_running_framer_asic(asic=asic)

        logger.info("Done with SACI configuration")

    def _init_framer_asic(self, asic: int) -> None:
        registers_and_vals: Dict[str, Any] = {
            "enable": False,
            "DisableLane": 0,
        }

        self.write_many(
            base_node=self.FebFramerAsics[asic], registers_and_vals=registers_and_vals
        )

    def init_asics(self, asics: Tuple[int, ...]):
        for asic in asics:
            self._init_asic(asic=asic)
            self._init_framer_asic(asic=asic)

    def setup_event_batchers(self, timeout: Optional[int] = None):
        for devPtr in self.ReadoutSystem.find(typ=batcher.AxiStreamBatcherEventBuilder):
            devPtr.Blowoff.set(False)
            devPtr.SoftRst()
            devPtr.Timeout.set(timeout if timeout is not None else 0)

    def setup_bypasses_for_disabled_asics(self, asic_mask: int) -> None:
        # Mux Bypasses
        bypass0: int = ((~asic_mask) & 0b000111) << 2
        bypass1: int = ((~asic_mask) & 0b111000) >> 1
        self.FebFpga.App.EventSeqMux[0].Bypass.set(bypass0)
        self.FebFpga.App.EventSeqMux[1].Bypass.set(bypass1)

        pcie_bypass: int = ((~asic_mask) & 0x3F) << 2

        for devPtr in self.ReadoutSystem.find(typ=batcher.AxiStreamBatcherEventBuilder):
            devPtr.Bypass.set(pcie_bypass)

    def setup_debug_timing_out(self, board_ctrl: Dict[str, Any]):
        """Setup registers for debug timing outputs."""

        if board_ctrl["timingOutEn[0]"]:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[0].set(True)
            t0_sel: int = board_ctrl["timingOutSelect[0]"]
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutSelect[0].set(t0_sel)
        else:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[0].set(False)

        if board_ctrl["timingOutEn[1]"]:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[1].set(True)
            t1_sel: int = board_ctrl["timingOutSelect[1]"]
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutSelect[1].set(t1_sel)
        else:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[1].set(False)

        if board_ctrl["timingOutEn[2]"]:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[2].set(True)
            t2_sel: int = board_ctrl["timingOutSelect[2]"]
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutSelect[2].set(t2_sel)
        else:
            self.FebFpga.App.BoardCtrl3x2Readout.timingOutEn[2].set(False)

    def setup_board_control_registers(self, board_ctrl: Dict[str, Any]) -> None:
        """Setup registers for BoardCtrl3x2Readout."""

        self.setup_debug_timing_out(board_ctrl=board_ctrl)
        BoardCtrl3x2Readout = self.FebFpga.App.BoardCtrl3x2Readout

        LTM4664_A = BoardCtrl3x2Readout.LTM4664_A
        LTM4664_A_regs: Dict[str, Any] = board_ctrl["LTM4664_A"]
        self.write_many(
            base_node=LTM4664_A, registers_and_vals=LTM4664_A_regs
        )

        LTM4664_B = BoardCtrl3x2Readout.LTM4664_B
        LTM4664_B_regs: Dict[str, Any] = board_ctrl["LTM4664_B"]
        self.write_many(
            base_node=LTM4664_B, registers_and_vals=LTM4664_B_regs
        )

        LTM4664_C = BoardCtrl3x2Readout.LTM4664_C
        LTM4664_C_regs: Dict[str, Any] = board_ctrl["LTM4664_C"]
        self.write_many(
            base_node=LTM4664_C, registers_and_vals=LTM4664_C_regs
        )

    def power_on(self, asic_mask: int = 0x3F):
        self.FebFpga.App.EnableCommonAsicPower()
        self.FebFpga.App.EnableAllAsicDigitalPower()
        self.FebFpga.App.BoardCtrl3x2Readout.enableP1V3AAsic.set(asic_mask)
        self.FebFpga.App.enable.set(True)

    def reset_gt(self):
        logger: logging.Logger = logging.getLogger(self._logger_name)
        try:
            self.FebFpga.App.AsicGtClk.gtRstAll.set(True)
            time.sleep(1)
            self.FebFpga.App.AsicGtClk.gtRstAll.set(False)
        except Exception as err:
            logger.error(f"GT Reset failed: {err}")

    def reset_asic_gt(self, asics: Tuple[int, ...], emulator: bool = True):
        logger: logging.Logger = logging.getLogger(self._logger_name)
        if not emulator:
            for i in asics:
                asicGtData = self.FebFpga.App.AsicGtData[i]
                try:
                    asicGtData.enable.set(True)
                    asicGtData.gtStableRst.set(True)
                except Exception as e:
                    print(f"Failed to reset ASIC GtData {i}: {e}")

            time.sleep(1)

            for i in asics:
                asicGtData = self.FebFpga.App.AsicGtData[i]
                try:
                    asicGtData.gtStableRst.set(False)
                except Exception as e:
                    logger.error(f"Failed to unreset ASIC GtData {i}: {e}")
        else:
            logger.debug("AsicGtData registers do not exist for emulator.")

    def _enable_clock_dependencies(self):
        import uhr_integration_support as epixUhrSupport
        import pixel_camera_readout_common as ROCommon

        for device in self.ReadoutSystem.find(
            typ=epixUhrSupport.AxiStreamUhrDescrambleWrapper
        ):
            device.enable.set(True)

        for device in self.ReadoutSystem.find(typ=ROCommon.UhrAxiStreamFramer):
            device.enable.set(True)

        self.FebFpga.App.ClockGeneration.enable.set(True)
        self.FebFpga.App.U_matrixClk.enable.set(True)
        self.FebFpga.App.U_sspClk.enable.set(True)

    def _check_pll_lock(self):
        logger: logging.Logger = logging.getLogger(self._logger_name)
        # PLL initialization for setting up clocks
        # Actually only needed first time after power on, but harmless to leave
        self.write_and_check(self.FebFpga.Core.SystemDevices.Si5345Pll.enable, True)
        if self.FebFpga.Core.SystemDevices.Si5345Pll.LockedWait():
            logger.error("Failed to lock FEB (Si5345) PLL")
        else:
            logger.debug("FEB (Si5345) PLL established")
        self.write_and_check(self.FebFpga.Core.SystemDevices.Si5345Pll.enable, False)

    def init_board(self):
        # self.power_on()
        self._check_pll_lock()

        self.Stop()
        self._enable_clock_dependencies()
        self.FebFpga.App.WaveformControl.enable.set(True)

        self.FebFpga.App.TriggerRegisters.enable.set(True)
        self.init_trigger_registers()
        self.init_waveform_control()

        # self.init_asics(asics=tuple(range(1, self._nasics + 1)))

        # GT Module Setup - Emulator does not have
        if not self._emulator:
            self.write_and_check(self.FebFpga.App.AsicGtClk.enable, True)
            self.write_and_check(self.FebFpga.App.AsicGtClk.gtRstAll, True)
            time.sleep(0.1)
            self.write_and_check(self.FebFpga.App.AsicGtClk.gtRstAll, False)
            self.reset_asic_gt(
                asics=tuple(range(1, self._nasics + 1)), emulator=self._emulator
            )

        # Now can do the rest of the initialization

        self.write_and_check(self.FebFpga.App.TimingRx.enable, True)
        self.write_and_check(self.FebFpga.App.VINJ_DAC.dacEn, False)
        self.write_and_check(self.FebFpga.App.VINJ_DAC.rampEn, False)

    def reset_counters(self):
        self.FebFpga.App.TimingRx.TimingFrameRx.countReset()
        self.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].countReset()

    def set_timing_trigger(self):
        self.FebFpga.App.SetTimingTrigger()

    def start_auto_trigger(self):
        self.FebFpga.App.TriggerRegisters.StartAutoTrigger()

    @property
    def timing_params(self) -> TimingParameters:
        return self._timing_params

    @timing_params.setter
    def timing_params(self, params: TimingParameters) -> None:
        self._timing_params = params

    def initialize_timing(self, timebase: str) -> None:
        # Stop the device before doing anything elsef
        self.ReadoutSystem.StopRun()
        time.sleep(0.1)

        params: TimingParameters
        if timebase == "119M":
            # UED
            TimingFrameRx = self.FebTimingRx.TimingFrameRx
            registers_and_vals: Dict[str, Any] = {
                "ModeSelEn": 1,
                "ClkSel": 0,
                "RxDown": 0,
            }
            self.write_many(
                base_node=TimingFrameRx, registers_and_vals=registers_and_vals
            )
            params = {
                "bypass": self._nasics * [0x3],
                "clk_period": 1000 / 119.0,
                "msg_period": 238,
                "pcie_timing": True,
            }
        else:
            self.FebTimingRx.ConfigLclsTimingV2()

            params = {
                "bypass": self._nasics * [0x3],
                "clk_period": 7000 / 1300.0,
                "msg_period": 200,
                "pcie_timing": False,
            }
            self.timing_params = params

    def setup_daq_trigger(self, rog: int, start_ns: int, trig_cfg: Dict[str, Any]) -> int:
        """DAQ trigger delay is provided start_ns - L0Delay.

        If trig_cfg is empty, only set the trigger delay (like during scans).
        """
        logger: logging.Logger = logging.getLogger(self._logger_name)

        partition_delay: int = self.FebXpmMessageAligner.PartitionDelay[rog].get()
        clk_period: float = self.timing_params["clk_period"]
        msg_period: int = self.timing_params["msg_period"]

        trigger_delay: int = int(start_ns / clk_period - partition_delay * msg_period)

        msg: str = (
            f"partitionDelay[{rog}] {partition_delay} "
            f"rawStart {start_ns} "
            f"triggerDelay {trigger_delay}"
        )

        if trigger_delay < 0:
            min_start_ns: float = partition_delay * msg_period * clk_period
            logger.error(msg)
            logger.error(f"Raise start_ns >= {min_start_ns}")
            raise ValueError("triggerDelay computes to < 0")

        else:
            logger.info(msg)

        if trig_cfg:
            rate_type: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[1]"]["RateType"]
            rate_sel: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[1]"]["RateSel"]
            dest_type: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[1]"]["DestType"]
            enable: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[1]"]["enable"]
            # enable_reg: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[1]"]["EnableReg"]

            trig_reg = self.FebTriggerEventManager.EvrV2CoreChannels
            if enable:
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].enable, 1)
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].RateType, rate_type)
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].RateSel, rate_sel)
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].DestType, dest_type)
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].EnableReg, 0) # enable_reg)
            else:
                self.write_and_check(trig_reg.EvrV2ChannelReg[1].enable, 0)

            trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[1]
            trig_cfg["TriggerEventBuffer[1]"]["TriggerDelay"] = trigger_delay
            trig_cfg["TriggerEventBuffer[1]"]["TriggerSource"] = 1
            trig_cfg["TriggerEventBuffer[1]"]["Partition"] = rog
            trig_cfg["TriggerEventBuffer[1]"]["MasterEnable"] = 1

            registers_and_vals: Dict[str, Any] = {
                "TriggerDelay": trigger_delay,
                "TriggerSource": 0,
                "Partition": rog,
                "MasterEnable": 1,
            }
            self.write_many(base_node=trig_event_buf, registers_and_vals=registers_and_vals)
        else:
            trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[1]
            trig_event_buf.TriggerDelay.set(trigger_delay)
        return trigger_delay

    def setup_run_trigger(self, rog: int, start_ns: int, trig_cfg: Dict[str, Any]) -> int:
        """DAQ trigger delay is provided start_ns - fixed value of 192.

        If trig_cfg is empty, only set the trigger delay (like during scans).
        """
        logger: logging.Logger = logging.getLogger(self._logger_name)

        partition_delay: int = self.FebXpmMessageAligner.PartitionDelay[rog].get()
        clk_period: float = self.timing_params["clk_period"]
        msg_period: int = self.timing_params["msg_period"]

        fixed_delay: int = 0 # 193

        # trigger_delay: int = int(start_ns / clk_period - partition_delay * msg_period - fixed_delay)
        trigger_delay: int = int(start_ns / clk_period - fixed_delay)

        msg: str = (
            f"[Run Trigger] rawStart {start_ns} "
            f"deltaDelay {fixed_delay}"
            f"triggerDelay {trigger_delay}"
        )

        if trigger_delay < 0:
            min_start_ns: float = partition_delay * msg_period * clk_period
            logger.error(msg)
            logger.error(f"Raise start_ns >= {min_start_ns}")
            raise ValueError("triggerDelay computes to < 0")

        else:
            logger.info(msg)

        if trig_cfg:
            rate_type: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[0]"]["RateType"]
            rate_sel: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[0]"]["RateSel"]
            dest_type: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[0]"]["DestType"]
            enable: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[0]"]["enable"]
            enable_reg: int = trig_cfg["EvrV2CoreChannels"]["EvrV2ChannelReg[0]"]["EnableReg"]

            trig_reg = self.FebTriggerEventManager.EvrV2CoreChannels
            if enable:
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].enable, 1)
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].RateType, rate_type)
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].RateSel, rate_sel)
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].DestType, dest_type)
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].EnableReg, enable_reg)
            else:
                self.write_and_check(trig_reg.EvrV2ChannelReg[0].enable, 0)

            # self.write_and_check(trig_reg.EvrV2ChannelReg[1].EnableReg, 0)

            run_rog: int
            if rog == 0:
                run_rog = rog + 1
            else:
                run_rog = rog - 1

            self.FebTriggerRegisters.RunTriggerDelay.set(0)
            trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[0]
            trig_cfg["TriggerEventBuffer[0]"]["TriggerDelay"] = trigger_delay
            trig_cfg["TriggerEventBuffer[0]"]["Partition"] = run_rog
            registers_and_vals: Dict[str, Any] = {
                "TriggerDelay": trigger_delay,
                "TriggerSource": trig_cfg["TriggerEventBuffer[0]"]["TriggerSource"],
                "Partition": run_rog,
                "MasterEnable": 1, # trig_cfg["TriggerEventBuffer[0]"]["MasterEnable"],
            }
            self.write_many(base_node=trig_event_buf, registers_and_vals=registers_and_vals)
        else:
            trig_event_buf = self.FebTriggerEventManager.TriggerEventBuffer[0]
            trig_event_buf.TriggerDelay.set(trigger_delay)
        return trigger_delay

    def set_charge_injection(
        self,
        enable: bool = False,
        single_val: Optional[int] = None,
        asics: List[int] = [],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        level: Optional[int] = None,
        skip_x: int = 0,
        skip_y: int = 0,
    ) -> None:
        """Program a charge injection run."""
        if enable:
            self.write_and_check(self.FebFpga.App.VINJ_DAC.enable, True)
            self.write_and_check(self.FebFpga.App.VINJ_DAC.dacEn, True)
            self.write_and_check(self.FebWaveformControl.InjEn, True)
            self.write_and_check(self.FebWaveformControl.AsicInjEn, True)

            if single_val is None: # We are using a ramp scan
                self.write_and_check(self.FebFpga.App.VINJ_DAC.rampEn, True)

                # During scans, the reset must be called between each gain value
                self.write_and_check(self.FebFpga.App.VINJ_DAC.resetDacRamp, True)
                if start is not None:
                    self.write_and_check(self.FebFpga.App.VINJ_DAC.dacStartValue, start)
                if stop is not None:
                    self.write_and_check(self.FebFpga.App.VINJ_DAC.dacStopValue, stop)
                if step is not None:
                    self.write_and_check(self.FebFpga.App.VINJ_DAC.dacStepValue, step)
                self.write_and_check(self.FebFpga.App.VINJ_DAC.resetDacRamp, False)

                if level is not None:
                    if skip_x and skip_y:
                        # Create a new pixel map skipping some pixels for charge inj
                        # The non-injection value is level - 4
                        non_inj_level: int = level - 4

                        inj_map: npt.NDArray[np.uint8] = np.zeros((168, 192))
                        inj_map.fill(non_inj_level)
                        for y in range(0, len(inj_map), skip_y):
                            for x in range(0, len(inj_map[0]), skip_x):
                                inj_map[y, x] = level
                        self.set_pixel_gain_map(asics=asics, map_data=inj_map)
                    else:
                        self.set_pixel_gain(asics=asics, gain_value=level)
            else:
                self.write_and_check(self.FebFpga.App.VINJ_DAC.dacSingleValue, single_val)
        else:
            self.write_and_check(self.FebFpga.App.VINJ_DAC.dacEn, False)
            self.write_and_check(self.FebFpga.App.VINJ_DAC.rampEn, False)
            self.write_and_check(self.FebWaveformControl.InjEn, False)
            self.write_and_check(self.FebWaveformControl.InjEn, False)
            self.write_and_check(self.FebFpga.App.VINJ_DAC.enable, False)

    def set_pixel_gain(self, asics: List[int], gain_value: int):
        """Program a uniform gain mode using the ASIC matrix programming command."""
        logger: logging.Logger = logging.getLogger(self._logger_name)
        logger.info(f"Programming Pixel Gain {gain_value} for ASICs {asics}")
        for i in asics:
            asic = self.FebAsics[i]
            self.write_and_check(asic.PixNumModeEn, True)
            # NOTE: This register expects the bits as a string...
            asic.progPixelMatrixConstantValue(str(gain_value))
            if not self._emulator:
                self.write_and_check(asic.PixNumModeEn, False)

    def set_pixel_gain_map(
        self, asics: List[int], map_data: npt.NDArray[np.uint8], temp_dir: str = "/tmp"
    ):
        """
        Programs the pixel matrix gain bit (MSB) for the specified ASICs using a map.

        Args:
            asics (List[int]): List of ASIC indices (1-6).

            map_data (np.ndarray): 2D array (168, 192) or 3D array (6, 168, 192).

            temp_dir (str): Directory for temporary CSV files required by rogue command.
        """
        logger: logging.Logger = logging.getLogger(self._logger_name)

        self.FebFpga.App.EpixUhrMatrixConfig.enable.set(True)
        if map_data.ndim == 2:
            maps = {asic: map_data for asic in asics}
        elif map_data.ndim == 3:
            if map_data.shape[0] == len(asics):
                maps = {asic: map_data[i] for i, asic in enumerate(asics)}
            else:
                maps = {asic: map_data[asic - 1] for asic in asics}
        else:
            raise ValueError(f"map_data must be 2D or 3D, got {map_data.ndim}D")
        for asic_num in asics:
            asic_node = self.FebAsics[asic_num]

            self.write_and_check(asic_node.PixNumModeEn, False)

            fn = os.path.join(temp_dir, f"epixuhr3x2_asic{asic_num}_gain_map.csv")
            np.savetxt(
                fn, maps[asic_num], delimiter=",", newline="\n", comments="", fmt="%d"
            )

            logger.info(f"Programming ASIC {asic_num} pixel gain map from {fn}")

            try:
                matrix_config = self.FebFpga.App.EpixUhrMatrixConfig
                # prog_cmd = getattr(
                #     matrix_config, f"progPixelMatrixFromCsvAsic{asic_num}"
                # )
                # prog_cmd(fn)
                matrix_config.CsvFilePath[asic_num].set(fn)
                prog_cmd = getattr(
                    matrix_config, f"progPixelMatrixFromCsvAsic{asic_num}"
                )
                time.sleep(0.1)
                prog_cmd(fn)
                time.sleep(0.5)
                logger.info(f"ASIC {asic_num} pixel gain map programmed successfully.")
            except AttributeError as e:
                logger.error(
                    f"Failed to access EpixUhrMatrixConfig for ASIC {asic_num}: {e}"
                )
                raise
            finally:
                if os.path.exists(fn):
                    os.remove(fn)

    def set_pixel_gain_map_notmap(self, asics: List[int], map_data: np.ndarray):
        """
        Programs the pixel matrix gain bit (MSB) for the specified ASICs using direct
        memory transactions, avoiding temporary CSV files.

        Args:
            asics (List[int]): List of ASIC indices to program (1-6).

            map_data (np.ndarray): 2D array (168, 192) or 3D array (6, 168, 192).
        """
        import rogue.interfaces.memory as rim

        logger: logging.Logger = logging.getLogger(self._logger_name)

        self.FebFpga.App.EpixUhrMatrixConfig.enable.set(True)

        try:
            matrix_config = self.FebFpga.App.EpixUhrMatrixConfig
        except AttributeError:
            logger.error("EpixUhrMatrixConfig device not found in Rogue tree.")
            raise

        if map_data.ndim == 2:
            maps = {asic: map_data for asic in asics}
        elif map_data.ndim == 3:
            if map_data.shape[0] == len(asics):
                maps = {asic: map_data[i] for i, asic in enumerate(asics)}
            else:
                maps = {asic: map_data[asic - 1] for asic in asics}
        else:
            raise ValueError(f"map_data must be 2D or 3D, got {map_data.ndim}D")
        for asic_num in asics:
            asic_node = self.FebAsics[asic_num]
            matrix_cfg = maps[asic_num]

            if matrix_cfg.shape != (168, 192):
                raise ValueError(
                    f"ASIC {asic_num} map shape {matrix_cfg.shape} != (168, 192)"
                )

            logger.info(
                f"Programming ASIC {asic_num} pixel gain map via direct memory write."
            )

            self.write_and_check(asic_node.PixNumModeEn, False)
            matrix_config.selAsicMat.set(asic_num - 1)

            most_frequent = np.bincount(np.ravel(matrix_cfg.astype(int))).argmax()
            getattr(matrix_config, f"MaxFreqConfAsic[{asic_num}]").set(
                int(most_frequent)
            )

            reshaped = matrix_cfg.astype(np.uint64).reshape(-1, 8)
            packed_words = np.zeros(reshaped.shape[0], dtype=np.uint64)
            for i in range(8):
                packed_words |= reshaped[:, i] << (i * 8)

            ldata = packed_words.tobytes()
            dest_addr = matrix_config.memAddress + (asic_num - 1) * 0x80000

            matrix_config._reqTransaction(dest_addr, ldata, len(ldata), 0, rim.Write)
            matrix_config._waitTransaction(0)

            matrix_config.ConfSel.set(1 << (asic_num - 1))
            matrix_config.ConfWrReq.set(True)

            while not matrix_config.ConfDoneAll.get():
                time.sleep(0.01)

            logger.info(f"ASIC {asic_num} gain map programming complete.")

    @property
    def RxId(self) -> int:
        return self.FebTriggerEventManager.XpmMessageAligner.RxId.get()

    @property
    def TxId(self) -> int:
        return self.FebTriggerEventManager.XpmMessageAligner.TxId.get()

    @TxId.setter
    def TxId(self, value: int) -> None:
        self.write_and_check(
            register=self.FebTriggerEventManager.XpmMessageAligner.TxId, value=value
        )

    @property
    @lru_cache
    def SerNo(self) -> str:
        """Retrieve a unique identifier for the panel.

        The serial number has been decided on as the following set of numbers:
        (C1100 firmware version)-(readout board ID)-(carrier board ID)
        """
        firmwareVersion: int = self.c1100_firmware_version

        boardCtrl = getattr(self.FebFpga.App, "BoardCtrl3x2Readout")
        boardCtrl.enable.set(1)
        #  Construct the ID
        # These values are 64 bit, in hex something like 0x3000001f8b017001
        digitalId: int = boardCtrl.readoutBoardId.get()
        # pwrCommId = 0 # (Don't know what the equivalent of this is now)
        carrierId: int = boardCtrl.carrierBoardId.get()

        # detId = "%010d-%010d-%010d-%010d" % (firmwareVersion, carrierId, digitalId, pwrCommId)
        detId: str = f"{firmwareVersion:010d}-{carrierId:010d}-{digitalId:010d}"
        return detId

    @property
    def ShortSerNo(self) -> str:
        """Construct a truncated hash of the full serial number.

        Given capacity for ~10000 unique serial numbers, a 5 digit hash would have
        a very high collision rate over the full 10000 set (close to 100%).

        If we use 8 digits, it should be very low (~0.000005% if my math is good).
        I think we likely won't have 10000 serial numbers to begin with, so this should
        be okay.

        The truncated serial number will be used for the special serial number
        configdb.
        """
        import hashlib

        detId: str = self.SerNo
        hashed: str = hashlib.sha256(detId.encode()).hexdigest()
        truncatedId: str = hashed[:8]

        return truncatedId

