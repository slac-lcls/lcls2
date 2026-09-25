"""
epixquad_monitor_ioc.py

Read the ePixQuad environmental monitor stream via rogue and publish it as
EPICS PVs.

The ePixQuad sends monitor packets on a dedicated PGP virtual channel carrying
38 slow-ADC/I2C readings: sensor and electronics temperatures, humidity, LDO
currents and temperatures, power-supply rails and optical transceiver
diagnostics.  See CHANNEL_DEFS below for the full channel mapping.

Unlike the ePix100, the ePixQuad also needs a small "enable" packet written back
on the same virtual channel to start and stop the stream, in addition to the
register configuration.

Some of the readings (the thermistor sensor temperatures and the power-supply
analog/digital values) are produced by the on-board MicroBlaze.  On detectors
whose MicroBlaze is not functional, pass --no-microblaze: those channels are
then published as 0.0 and excluded from the packet and temperature checks.

Requirements (same environment as the DAQ epixquad configuration):
    rogue  pyrogue  ePixQuad  caproto

Usage examples:
    # publish decoded values as EPICS PVs under the default prefix:
    epixquad_monitor_ioc

    # non-default prefix and hardware, with a dead MicroBlaze:
    epixquad_monitor_ioc --prefix DET:EPIX:CMP004: --dev /dev/datadev_1 --no-microblaze

The monitor stream and auto trigger are enabled by writing 1 to the SET_MONITOR
PV (this needs exclusive SRP access, so no DAQ can be configuring the detector
at the same time).
"""

import time
import numpy as np
from typing import Any, Dict

# rougue imports
from psdaq.utils import enable_epix_quad
import ePixQuad
import rogue
import rogue.hardware.axi
import rogue.protocols.srp
import pyrogue
# caproto imports
import caproto as ca
from caproto.server import (
        PvpropertyEnum,
        pvproperty,
        template_arg_parser,
        run
)
from caproto.server.records import BiFields
# shared monitoring IOC code
from psdaq.IOC.epix_monitor_base import (
        EpixMonitoringIOCBase,
        MonitorPacket,
        add_common_args,
        counter_pv,
        current_pv,
        humidity_pv,
        setup_logging,
        string_pv,
        temp_pv,
        voltage_pv,
)


class EpixQuadMonitorUtils:
    """
    Namespace functions that are local functions in a constructor or lambdas in rogue so we can't use them directly.
    """
    @staticmethod
    def getPwrCurr(raw: int) -> float:
        return raw * 0.1024 / 4095 / 0.02

    @staticmethod
    def getPwrVin(raw: int) -> float:
        return raw * 102.4 / 4095

    @staticmethod
    def getPwrTemp(raw: int) -> float:
        a = 130.0 / (0.882 - 1.951)
        b = (0.882 / 0.0082) + 100
        return raw * 2.048 / 4095 * a + b

    @staticmethod
    def getShtHum(raw: int) -> float:
        return raw / 65535.0 * 100.0

    @staticmethod
    def getShtTemp(raw: int) -> float:
        return raw / 65535.0 * 175.0 - 45.0

    @staticmethod
    def getNctTempLoc(raw: int) -> float:
        return float(raw & 0xff)

    @staticmethod
    def getNctTempRem(raw: int) -> float:
        return (raw >> 8) + (raw & 0xc0) / 256

    @staticmethod
    def getLt3086DoubleCurr(raw: int) -> float:
        """
        Imon = Iin / 1000
        Rload = 330 ohm
        ADC buffer gain x 2
        Two parallel LDOs current x 2
        returns current in A
        """
        return raw / 16383.0 * 2.5 / 330.0 * 1000

    @staticmethod
    def getLt3086SingleCurr(raw: int) -> float:
        """
        Imon = Iin / 1000
        Rload = 330 ohm
        ADC buffer gain x 2
        One LDO current x 1
        returns current in mA
        """
        return raw / 16383.0 * 2.5 / 330.0 * 1000000 / 2.0

    @staticmethod
    def getAnaTemp(raw: int) -> float:
        a = 130.0 / (0.882 - 1.951)
        b = (0.882 / 0.0082) + 100
        return raw * 1.65 / 65535 * a + b

    @staticmethod
    def getLdoTemp(raw: int) -> float:
        return raw * 1.65 / 65535 * 100

    @staticmethod
    def getTrOptTemp(raw: int) -> float:
        return raw * 1.0 / 256

    @staticmethod
    def getTrOptVolt(raw: int) -> float:
        return raw * 0.0001

    @staticmethod
    def getTrOptPwr(raw: int) -> float:
        return raw * 0.1

    @staticmethod
    def getThermistorTemp(raw: int) -> float:
        # resistor divider 100k and MC65F103B (Rt25=10k)
        # Vref 2.5V
        TthermK = -273.15
        if raw != 0:
            Umeas = raw / 16383.0 * 2.5
            Itherm = Umeas / 100000
            Rtherm = (2.5 - Umeas) / Itherm
            if Rtherm > 0.0:
                LnRtR25 = np.log(Rtherm / 10000.0)
                TthermK += 1.0 / (3.3538646E-03 + 2.5654090E-04 * LnRtR25 +
                             1.9243889E-06 * (LnRtR25**2) + 1.0969244E-07 * (LnRtR25**3))

        return TthermK


CHANNEL_DEFS: Dict[int, Any] = {
    0: dict(
        name="SHT31 Humidity",
        unit="%",
        conv=EpixQuadMonitorUtils.getShtHum,
        pv_signal="humidity",
    ),
    1: dict(
        name="SHT31 Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getShtTemp,
        pv_signal="temp3",
    ),
    2: dict(
        name="NCT218 Local Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getNctTempLoc,
        pv_signal="nct_loc_temp",
    ),
    3: dict(
        name="NCT218 Remote Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getNctTempRem,
        pv_signal="nct_fpga_temp",
    ),
    4: dict(
        name="ASIC_A0_2V5 Curr.",
        unit="A",
        conv=EpixQuadMonitorUtils.getLt3086DoubleCurr,
        pv_signal="asic_a0_2v5_cur",
    ),
    5: dict(
        name="ASIC_A1_2V5 Curr.",
        unit="A",
        conv=EpixQuadMonitorUtils.getLt3086DoubleCurr,
        pv_signal="asic_a1_2v5_cur",
    ),
    6: dict(
        name="ASIC_A2_2V5 Curr.",
        unit="A",
        conv=EpixQuadMonitorUtils.getLt3086DoubleCurr,
        pv_signal="asic_a2_2v5_cur",
    ),
    7: dict(
        name="ASIC_A3_2V5 Curr.",
        unit="A",
        conv=EpixQuadMonitorUtils.getLt3086DoubleCurr,
        pv_signal="asic_a3_2v5_cur",
    ),
    8: dict(
        name="ASIC_D0_2V5 Curr.",
        unit="mA",
        conv=EpixQuadMonitorUtils.getLt3086SingleCurr,
        pv_signal="asic_d0_2v5_cur",
    ),
    9: dict(
        name="ASIC_D1_2V5 Curr.",
        unit="mA",
        conv=EpixQuadMonitorUtils.getLt3086SingleCurr,
        pv_signal="asic_d1_2v5_cur",
    ),
    10: dict(
        name="Therm0 Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getThermistorTemp,
        pv_signal="temp1",
    ),
    11: dict(
        name="Therm1 Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getThermistorTemp,
        pv_signal="temp2",
    ),
    12: dict(
        name="PwrDigCurr",
        unit="A",
        conv=EpixQuadMonitorUtils.getPwrCurr,
        pv_signal="asic_dig_cur",
    ),
    13: dict(
        name="PwrDigVin",
        unit="V",
        conv=EpixQuadMonitorUtils.getPwrVin,
        pv_signal="dig_in_v",
    ),
    14: dict(
        name="PwrDigTemp",
        unit="°C",
        conv=EpixQuadMonitorUtils.getPwrTemp,
        pv_signal="dig_temp",
    ),
    15: dict(
        name="PwrAnaCurr",
        unit="A",
        conv=EpixQuadMonitorUtils.getPwrCurr,
        pv_signal="asic_ana_cur",
    ),
    16: dict(
        name="PwrAnaVin",
        unit="V",
        conv=EpixQuadMonitorUtils.getPwrVin,
        pv_signal="ana_in_v",
    ),
    17: dict(
        name="PwrAnaTemp",
        unit="°C",
        conv=EpixQuadMonitorUtils.getPwrTemp,
        pv_signal="ana_temp",
    ),
    18: dict(
        name="A0_2_5V_H Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a0_2v5_h_temp",
    ),
    19: dict(
        name="A0_2_5V_L Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a0_2v5_l_temp",
    ),
    20: dict(
        name="A1_2_5V_H Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a1_2v5_h_temp",
    ),
    21: dict(
        name="A1_2_5V_L Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a1_2v5_l_temp",
    ),
    22: dict(
        name="A2_2_5V_H Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a2_2v5_h_temp",
    ),
    23: dict(
        name="A2_2_5V_L Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a2_2v5_l_temp",
    ),
    24: dict(
        name="A3_2_5V_H Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a3_2v5_h_temp",
    ),
    25: dict(
        name="A3_2_5V_L Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a3_2v5_l_temp",
    ),
    26: dict(
        name="D0_2_5V Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_d0_2v5_temp",
    ),
    27: dict(
        name="D1_2_5V Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_d1_2v5_temp",
    ),
    28: dict(
        name="A0_1_8V Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a0_1v8_temp",
    ),
    29: dict(
        name="A1_1_8V Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a1_1v8_temp",
    ),
    30: dict(
        name="A2_1_8V Temp.",
        unit="°C",
        conv=EpixQuadMonitorUtils.getLdoTemp,
        pv_signal="asic_a2_1v8_temp",
    ),
    31: dict(
        name="PcbAnaTemp0",
        unit="°C",
        conv=EpixQuadMonitorUtils.getAnaTemp,
        pv_signal="pcb_ana_temp0",
    ),
    32: dict(
        name="PcbAnaTemp1",
        unit="°C",
        conv=EpixQuadMonitorUtils.getAnaTemp,
        pv_signal="pcb_ana_temp1",
    ),
    33: dict(
        name="PcbAnaTemp2",
        unit="°C",
        conv=EpixQuadMonitorUtils.getAnaTemp,
        pv_signal="pcb_ana_temp2",
    ),
    34: dict(
        name="TrOptTemp",
        unit="°C",
        conv=EpixQuadMonitorUtils.getTrOptTemp,
        pv_signal="tropt_temp",
    ),
    35: dict(
        name="TrOptVcc",
        unit="V",
        conv=EpixQuadMonitorUtils.getTrOptVolt,
        pv_signal="tropt_volt",
    ),
    36: dict(
        name="TrOptTxPwr",
        unit="uW",
        conv=EpixQuadMonitorUtils.getTrOptPwr,
        pv_signal="tropt_txpwr",
    ),
    37: dict(
        name="TrOptRxPwr",
        unit="uW",
        conv=EpixQuadMonitorUtils.getTrOptPwr,
        pv_signal="tropt_rxpwr",
    ),
}


class EpixQuadMonitorPacket(MonitorPacket):
    """
    One ePixQuad monitor stream packet.

    Format (160 bytes = 80 × uint16 little-endian):
      word[0-15]      : packet header
      word[16]        : channel  0  (raw uint16)
      ...
      word[53]        : channel 37

    See CHANNEL_DEFS above for the channel mapping.  Values are uint16;
    channels 38-63 are unused/unconnected.
    """

    STRUCT_FMT = "<80H"
    HEADER_WORDS = 16
    N_CHANNELS = 38
    CHANNEL_DEFS = CHANNEL_DEFS

    @property
    def header(self) -> tuple:
        return self.raw[0:self.HEADER_WORDS]


class EpixQuadBoard(pyrogue.Root):
    def __init__(self, dev, lane, vc):
        super().__init__(name='ePixQuadBoard', pollEn=False, initRead=False)
        dma = rogue.hardware.axi.AxiStreamDma(dev, lane << 8 | vc, True)
        srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(dma, srp)
        self.addInterface(dma)
        self.addInterface(srp)
        self.add(
            ePixQuad.EpixVersion(
                name='AxiVersion',
                memBase=srp,
                offset=0x00000000,
                expand=False,
            ))
        self.add(
            ePixQuad.SystemRegs(
                 name='SystemRegs',
                 memBase=srp,
                 offset=0x00100000,
                 expand=False,
                 enabled=True,
        ))
        self.add(
            ePixQuad.EpixQuadMonitor(
                name='EpixQuadMonitor',
                memBase=srp,
                offset=0x00700000,
                expand=False,
                enabled=True,
        ))

    def check_carried_ids(self):
        for i in range(4):
            cid_lo = self.SystemRegs.CarrierIdLow[i].get()
            if (cid_lo == 0xffffffff) or (cid_lo == 0):
                return False
            cid_hi = self.SystemRegs.CarrierIdHigh[i].get()
            if (cid_hi == 0xffffffff) or (cid_hi == 0):
                return False

        return True

    @staticmethod
    def configure(dev, lane, vc, flag, mon_prescale, trig_period, queue):
        data = {'logs': {'info': [], 'warn': [], 'error': []}}
        try:
            with EpixQuadBoard(dev, lane, vc) as root:
                # read firmware info
                fw = root.AxiVersion.FpgaVersion.get()
                data["firmware_version"] = '0x%08x' % fw
                githash = root.AxiVersion.GitHash.get()
                data["firmware_githash"] = '%040x' % githash
                bldstr = root.AxiVersion.BuildStamp.get()
                data["firmware_bldstr"] = bldstr
                # read carrier id info
                if not root.check_carried_ids():
                    data['logs']['warn'].append("Board boot issue: invalid carrierId - attempting to reset")
                    root.SystemRegs.CarrierIdRst.set(True)
                    time.sleep(0.1)
                    root.SystemRegs.CarrierIdRst.set(False)
                    data['logs']['info'].append("Reset of carrierIds complete")
                for i in range(4):
                    cid_lo = root.SystemRegs.CarrierIdLow[i].get()
                    cid_hi = root.SystemRegs.CarrierIdHigh[i].get()
                    data["carrier_id_%d"%i] = '0x%08x%08x' % (cid_lo, cid_hi)
                # check if the asic mask is zero
                asic_mask = root.SystemRegs.AsicMask.get()
                if asic_mask == 0:
                    data['logs']['warn'].append("Board boot issue: asic mask is zero - attempting to reset")
                    # this needs to set to fix this just calling AdcReqStart is not enough
                    root.SystemRegs.AdcBypass.set(True)
                    root.SystemRegs.AdcReqStart.set(True)
                    time.sleep(0.1)
                    root.SystemRegs.AdcReqStart.set(False)
                    time.sleep(0.1)
                    start = time.time()
                    timeout = 1.0 # wait max one second
                    while root.SystemRegs.AdcTestDone.get() != 1:
                        time.sleep(0.1)
                        if time.time() - start > timeout:
                            data['logs']['warn'].append("Wait for AdcTestDone timed out")
                            break
                    root.SystemRegs.AdcBypass.set(False)
                    data['logs']['info'].append("Reset of asic mask complete")
                # check if the adc test is failed
                adc_fail = root.SystemRegs.AdcTestFailed.get()
                if adc_fail:
                    data['logs']['warn'].append("Board boot issue: adc test failed - attempting to rerun")
                    root.SystemRegs.TrigEn.set(False)
                    root.SystemRegs.AdcReqStart.set(True)
                    time.sleep(0.1)
                    root.SystemRegs.AdcReqStart.set(False)
                    start = time.time()
                    timeout = 1.0 # wait max one second
                    while root.SystemRegs.AdcTestDone.get() != 1:
                        time.sleep(0.1)
                        if time.time() - start > timeout:
                            data['logs']['warn'].append("Wait for AdcTestDone timed out")
                            break
                    adc_fail = root.SystemRegs.AdcTestFailed.get()
                    data['logs']['info'].append(f"AdcTest completed with result: AdcTestFailed = {adc_fail}")
                    root.SystemRegs.TrigEn.set(True)


                # configure the monitoring registers
                root.EpixQuadMonitor.MonitorEn.set(flag)
                data['logs']['info'].append(f"set EpixQuadMonitor.MonitorEn to {flag}")
                root.EpixQuadMonitor.TrigPrescaler.set(mon_prescale)
                data['logs']['info'].append(f"set EpixQuadMonitor.TrigPrescaler to {mon_prescale}")
                root.SystemRegs.TrigEn.set(1)
                data['logs']['info'].append("set SystemRegs.TrigEn to 1")
                root.SystemRegs.TrigSrcSel.set(3)
                data['logs']['info'].append("set SystemRegs.TrigSrcSel to 3")
                root.SystemRegs.AutoTrigEn.set(1)
                data['logs']['info'].append("set SystemRegs.AutoTrigEn to 1")
                root.SystemRegs.AutoTrigPerMs.set(trig_period)
                data['logs']['info'].append(f"set SystemRegs.AutoTrigPerMs to {trig_period}")
        except Exception as exc:
            data['logs']['error'].append(f"exception encountering during configuration: {exc}")
        finally:
            # send the firmware info back
            queue.put(data)


class EpixQuadMonitoringIOC(EpixMonitoringIOCBase):
    """
    EPICS IOC publishing the ePixQuad environmental monitor readings.
    """

    board_cls = EpixQuadBoard
    packet_cls = EpixQuadMonitorPacket
    uses_enable_packet = True

    # channels whose values come from the MicroBlaze, and so are meaningless on
    # detectors where it is not functioning
    microblaze_fixup_channels = {"temp1", "temp2", "ana_temp", "dig_temp"}
    # sensor temps, with the reading below which the sensor is considered invalid
    stemp_channels = {
            ("temp1", -273.15),
            ("temp2", -273.15),
    }
    # electronics temps, with the range outside which the reading is invalid
    etemp_channels = {
            ("temp3", -45.0, 130),
            ("nct_loc_temp", 0.0, 200),
            ("nct_fpga_temp", 0.0, 200),
            ("ana_temp", 0.0, 200),
            ("dig_temp", 0.0, 200),
            ("tropt_temp", 0.0, 200),
    }

    def __init__(self, *args, has_microblaze, **kwargs):
        self.has_microblaze = has_microblaze
        super().__init__(*args, **kwargs)

    @property
    def monitor_setting(self):
        """
        Monitor rate converted to a prescale value. Set to a minimum of 1.
        E.g.: a prescale of 10 means the monitoring will fire on every tenth trigger
        """
        prescale = int(self.set_auto_trig_rate.value // self.set_monitor_rate.value)
        if prescale == 0:
            prescale = 1
        return prescale

    def check_packet(self, data):
        """
        Check that the packet is valid. This check is skipped if the microblaze is set as dead.
        """
        channels = ["nct_loc_temp", "nct_fpga_temp"]
        if self.has_microblaze:
            # only check these if the detector has a working microblaze
            channels.extend(["ana_temp", "dig_temp", "ana_in_v", "dig_in_v"])
        return all([data.get(d, 0) for d in channels])

    def fixup_value(self, name, value):
        """
        Zero out the garbage values from a non-functioning microblaze.
        """
        if (not self.has_microblaze) and (name in self.microblaze_fixup_channels):
            return 0.0
        return value

    @property
    def ignored_temp_channels(self):
        """
        Temperature channels to leave out of the validity checks.  Without a
        working microblaze these are forced to 0.0 by fixup_value, which would
        otherwise read as a plausible temperature.
        """
        if self.has_microblaze:
            return frozenset()
        return self.microblaze_fixup_channels

    def check_temps(self, data):
        """
        Check that at least on the of the sensor or electronics temps are valid
        """
        stemp = False
        etemp = False
        ignored = self.ignored_temp_channels

        # loop over the sensor temps to find if at least one is valid
        for channame, lowlim in self.stemp_channels:
            if channame in ignored:
                # values in this case are bad so ignore them
                continue
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim:
                    stemp = True
                    break

        # loop over the elec temps to find if at least one is valid
        for channame, lowlim, highlim in self.etemp_channels:
            if channame in ignored:
                # values in this case are bad so ignore them
                continue
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim and chan.value < highlim:
                    etemp = True
                    break

        return stemp, etemp

    async def update_extra(self, data):
        """
        Publish the derived sensor/electronics temperature validity flags.
        """
        stemp, etemp = self.check_temps(data)
        await self.stemp_ok.write(value=stemp)
        await self.etemp_ok.write(value=etemp)

    monerrcnt = counter_pv(name="MONERRCNT",
                           doc="epix monitor error counts")
    temp1 = temp_pv(name="TEMP1",
                    alarm_group="temp1",
                    doc="Therm0 Temp")
    temp2 = temp_pv(name="TEMP2",
                    alarm_group="temp2",
                    doc="Therm1 Temp")
    temp3 = temp_pv(name="TEMP3",
                    alarm_group="temp3",
                    doc="SHT31 Temp")
    humidity = humidity_pv(name="HUMIDITY",
                           alarm_group="humidity",
                           doc="SHT31 Humidity")
    ana_temp = temp_pv(name="ANA_TEMP",
                       alarm_group="ana_temp",
                       doc="PwrAnaTemp")
    dig_temp = temp_pv(name="DIG_TEMP",
                       alarm_group="dig_temp",
                       doc="PwrDigTemp")
    nct_loc_temp = temp_pv(name="NCT_LOC_TEMP",
                           alarm_group="nct_loc_temp",
                           doc="NCT218 Local Temp.")
    nct_fpga_temp = temp_pv(name="NCT_FPGA_TEMP",
                            alarm_group="nct_fpga_temp",
                            doc="NCT218 Remote Temp.")
    asic_a0_2v5_cur = current_pv(name="ASIC_A0_2V5_CUR",
                                 alarm_group="asic_a0_2v5_cur",
                                 doc="ASIC_A0_2V5 Curr.")
    asic_a1_2v5_cur = current_pv(name="ASIC_A1_2V5_CUR",
                                 alarm_group="asic_a1_2v5_cur",
                                 doc="ASIC_A1_2V5 Curr.")
    asic_a2_2v5_cur = current_pv(name="ASIC_A2_2V5_CUR",
                                 alarm_group="asic_a2_2v5_cur",
                                 doc="ASIC_A2_2V5 Curr.")
    asic_a3_2v5_cur = current_pv(name="ASIC_A3_2V5_CUR",
                                 alarm_group="asic_a3_2v5_cur",
                                 doc="ASIC_A3_2V5 Curr.")
    asic_d0_2v5_cur = current_pv(name="ASIC_D0_2V5_CUR",
                                 units="mA",
                                 hilim=100000.0,
                                 alarm_group="asic_d0_2v5_cur",
                                 doc="ASIC_D0_2V5 Curr.")
    asic_d1_2v5_cur = current_pv(name="ASIC_D1_2V5_CUR",
                                 units="mA",
                                 hilim=100000.0,
                                 alarm_group="asic_d1_2v5_cur",
                                 doc="ASIC_D1_2V5 Curr.")
    asic_a0_2v5_h_temp = temp_pv(name="ASIC_A0_2V5_H_TEMP",
                                 alarm_group="asic_a0_2v5_h_temp",
                                 doc="ASIC_A0_2V5_H Temp.")
    asic_a0_2v5_l_temp = temp_pv(name="ASIC_A0_2V5_L_TEMP",
                                 alarm_group="asic_a0_2v5_l_temp",
                                 doc="ASIC_A0_2V5_L Temp.")
    asic_a1_2v5_h_temp = temp_pv(name="ASIC_A1_2V5_H_TEMP",
                                 alarm_group="asic_a1_2v5_h_temp",
                                 doc="ASIC_A1_2V5_H Temp.")
    asic_a1_2v5_l_temp = temp_pv(name="ASIC_A1_2V5_L_TEMP",
                                 alarm_group="asic_a1_2v5_l_temp",
                                 doc="ASIC_A1_2V5_L Temp.")
    asic_a2_2v5_h_temp = temp_pv(name="ASIC_A2_2V5_H_TEMP",
                                 alarm_group="asic_a2_2v5_h_temp",
                                 doc="ASIC_A2_2V5_H Temp.")
    asic_a2_2v5_l_temp = temp_pv(name="ASIC_A2_2V5_L_TEMP",
                                 alarm_group="asic_a2_2v5_l_temp",
                                 doc="ASIC_A2_2V5_L Temp.")
    asic_a3_2v5_h_temp = temp_pv(name="ASIC_A3_2V5_H_TEMP",
                                 alarm_group="asic_a3_2v5_h_temp",
                                 doc="ASIC_A3_2V5_H Temp.")
    asic_a3_2v5_l_temp = temp_pv(name="ASIC_A3_2V5_L_TEMP",
                                 alarm_group="asic_a3_2v5_l_temp",
                                 doc="ASIC_A3_2V5_L Temp.")
    asic_d0_2v5_temp = temp_pv(name="ASIC_D0_2V5_TEMP",
                               alarm_group="asic_d0_2v5_temp",
                               doc="ASIC_D0_2V5 Temp.")
    asic_d1_2v5_temp = temp_pv(name="ASIC_D1_2V5_TEMP",
                               alarm_group="asic_d1_2v5_temp",
                               doc="ASIC_D1_2V5 Temp.")
    asic_a0_1v8_temp = temp_pv(name="ASIC_A0_1V8_TEMP",
                               alarm_group="asic_a0_1v8_temp",
                               doc="ASIC_A0_1V8 Temp.")
    asic_a1_1v8_temp = temp_pv(name="ASIC_A1_1V8_TEMP",
                               alarm_group="asic_a1_1v8_temp",
                               doc="ASIC_A1_1V8 Temp.")
    asic_a2_1v8_temp = temp_pv(name="ASIC_A2_1V8_TEMP",
                               alarm_group="asic_a2_1v8_temp",
                               doc="ASIC_A2_1V8 Temp.")
    pcb_ana_temp0 = temp_pv(name="PCB_ANA_TEMP0",
                            alarm_group="pcb_ana_temp0",
                            doc="PcbAnaTemp0")
    pcb_ana_temp1 = temp_pv(name="PCB_ANA_TEMP1",
                            alarm_group="pcb_ana_temp1",
                            doc="PcbAnaTemp1")
    pcb_ana_temp2 = temp_pv(name="PCB_ANA_TEMP2",
                            alarm_group="pcb_ana_temp2",
                            doc="PcbAnaTemp2")
    tropt_temp = temp_pv(name="TROPT_TEMP",
                         alarm_group="tropt_temp",
                         doc="TrOptTemp")
    tropt_volt = voltage_pv(name="TROPT_VOLT",
                            alarm_group="tropt_volt",
                            doc="TrOptVcc")
    tropt_txpwr = current_pv(name="TROPT_TXPWR",
                             units="uW",
                             hilim=100000.0,
                             alarm_group="tropt_txpwr",
                             doc="TrOptTxPwr")
    tropt_rxpwr = current_pv(name="TROPT_RXPWR",
                             units="uW",
                             hilim=100000.0,
                             alarm_group="tropt_rxpwr",
                             doc="TrOptRxPwr")
    stemp_ok = counter_pv(name="STEMP_OK",
                          alarm_group="stemp_ok",
                          doc="epix sensor temp OK")
    etemp_ok = counter_pv(name="ETEMP_OK",
                          alarm_group="etemp_ok",
                          doc="epix electronics temp OK")
    microblaze = pvproperty(name="MICROBLAZE",
                            value=1,
                            dtype=PvpropertyEnum[BiFields],
                            record=BiFields,
                            alarm_group="microblaze",
                            enum_strings=["NO", "YES"],
                            doc="epix has working MicroBlaze")
    carrier_id_0 = string_pv(name="CARRIER_ID_0",
                             doc="epix carrier id 0")
    carrier_id_1 = string_pv(name="CARRIER_ID_1",
                             doc="epix carrier id 1")
    carrier_id_2 = string_pv(name="CARRIER_ID_2",
                             doc="epix carrier id 2")
    carrier_id_3 = string_pv(name="CARRIER_ID_3",
                             doc="epix carrier id 3")

    @microblaze.startup
    async def microblaze(self, instance, async_lib):
        if self.has_microblaze:
            status=ca.AlarmStatus.NO_ALARM
            severity=ca.AlarmSeverity.NO_ALARM
        else:
            status=ca.AlarmStatus.STATE
            severity=ca.AlarmSeverity.MAJOR_ALARM
        await instance.write(value=self.has_microblaze, status=status, severity=severity)


def main():
    # Parse standard EPICS IOC command-line options
    parser, split_args = template_arg_parser(
        default_prefix="DET:EPIX:CMP004:",
        desc="Read ePixQuad environmental monitor packets via rogue and publish via caproto IOC"
    )
    add_common_args(parser, regvc=1)
    parser.add_argument(
        "--no-microblaze",
        action='store_false',
        dest='microblaze',
        help="Flag to indicate the detector has a non-functional microblaze processor"
    )

    args = parser.parse_args()
    ioc_options, run_options = split_args(args)

    setup_logging(args)

    # Start the server
    ioc = EpixQuadMonitoringIOC(dev=args.dev, lane=args.lane, vc=args.vc, regvc=args.regvc, has_microblaze=args.microblaze, **ioc_options)
    run(ioc.pvdb, **run_options, startup_hook=ioc.__ainit__)


if __name__ == '__main__':
    main()
