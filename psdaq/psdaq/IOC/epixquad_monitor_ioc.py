import sys
import time
import struct
import logging
import threading
import numpy as np
import multiprocessing as mp
from typing import Any, Optional
# rougue imports
from psdaq.utils import enable_epix_quad
import ePixQuad
import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.protocols.srp
import pyrogue
# caproto imports
import caproto as ca
from caproto import config_caproto_logging
from caproto.server import (
        PVGroup, PvpropertyDouble,
        PvpropertyInteger,
        PvpropertyString,
        PvpropertyChar,
        PvpropertyEnum,
        template_arg_parser,
        pvproperty,
        run
)
from caproto.server.records import (
        AoFields,
        AiFields,
        LongoutFields,
        LonginFields,
        BiFields,
        StringinFields,
        WaveformFields,
        MbbiFields
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


CHANNEL_DEFS: dict[int, Any] = {
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


class EpixQuadMonitorPacket:
    """
    One ePixQuad monitor stream packet.

    Format (160 bytes = 80 × uint16 little-endian):
      word[0-15]      : packet header
      word[16]        : channel  0  (raw uint16)
      ...
      word[53]        : channel 37

    Channel mapping (reconstructed from old C++ IOC/AMI1):
      ch  0  (word[16])  SHT31 Humidity       %
      ch  1  (word[17])  SHT31 Temp.          °C
      ch  2  (word[18])  NCT218 Local Temp.   °C
      ch  3  (word[19])  NCT218 Remote Temp.  °C
      ch  4  (word[20])  ASIC_A0_2V5 Curr.    A
      ch  5  (word[21])  ASIC_A1_2V5 Curr.    A
      ch  6  (word[22])  ASIC_A2_2V5 Curr.    A
      ch  7  (word[23])  ASIC_A3_2V5 Curr.    A
      ch  8  (word[24])  ASIC_D0_2V5 Curr.    mA
      ch  9  (word[25])  ASIC_D1_2V5 Curr.    mA
      ch 10  (word[26])  Therm0 Temp.         °C
      ch 11  (word[27])  Therm1 Temp.         °C
      ch 12  (word[28])  PwrDigCurr           A
      ch 13  (word[29])  PwrDigVin            V
      ch 14  (word[30])  PwrDigTemp           °C
      ch 15  (word[31])  PwrAnaCurr           A
      ch 16  (word[32])  PwrAnaVin            V
      ch 17  (word[33])  PwrAnaTemp           °C
      ch 18  (word[34])  A0_2_5V_H Temp.      °C
      ch 19  (word[35])  A0_2_5V_L Temp.      °C
      ch 20  (word[36])  A1_2_5V_H Temp.      °C
      ch 21  (word[37])  A1_2_5V_L Temp.      °C
      ch 22  (word[38])  A2_2_5V_H Temp.      °C
      ch 23  (word[39])  A2_2_5V_L Temp.      °C
      ch 24  (word[40])  A3_2_5V_H Temp.      °C
      ch 25  (word[41])  A3_2_5V_L Temp.      °C
      ch 26  (word[42])  D0_2_5V Temp.        °C
      ch 27  (word[43])  D1_2_5V Temp.        °C
      ch 28  (word[44])  A0_1_8V Temp.        °C
      ch 29  (word[45])  A1_1_8V Temp.        °C
      ch 30  (word[46])  A2_1_8V Temp.        °C
      ch 31  (word[47])  PcbAnaTemp0          °C
      ch 32  (word[48])  PcbAnaTemp1          °C
      ch 33  (word[49])  PcbAnaTemp2          °C
      ch 34  (word[50])  TrOptTemp            °C
      ch 35  (word[51])  TrOptVcc             V
      ch 36  (word[52])  TrOptTxPwr           uW
      ch 37  (word[53])  TrOptRxPwr           uW

    Values are uint16.  Channels 38-63 are unused/unconnected.
    Negative readings on startup or unconnected sensors are normal.
    """

    N_WORDS = 80
    N_CHANNELS = 38
    HEADER_BYTES = 16
    PACKET_BYTES = N_WORDS * 2  # 68

    def __init__(self, data: bytes):
        if len(data) < self.PACKET_BYTES:
            raise ValueError(
                f"Packet too short: {len(data)} B  (expected {self.PACKET_BYTES} B)"
            )
        self.raw = struct.unpack_from("<80H", data)

    @property
    def counter(self) -> int:
        return self.raw[0]

    @property
    def header(self) -> tuple[int, ...]:
        return self.raw[0:self.HEADER_BYTES]

    def channel_raw(self, ch: int) -> int:
        """Raw signed int32 for channel ch (0–15)."""
        return self.raw[ch + self.HEADER_BYTES]

    def channel_value(self, ch: int) -> Optional[float]:
        """Converted physical value for a defined channel, or None if undefined."""
        if ch not in CHANNEL_DEFS:
            return None
        return CHANNEL_DEFS[ch]["conv"](self.raw[ch + self.HEADER_BYTES])

    def as_dict(self) -> dict:
        """Return {sensor_name: physical_value} for all defined channels."""
        return {
            defn["name"]: defn["conv"](self.raw[ch + self.HEADER_BYTES])
            for ch, defn in CHANNEL_DEFS.items()
        }

    def pv_data(self) -> dict:
        """Return {pv_name: physical_value} for all defined channels."""
        return {
            defn["pv_signal"]: defn["conv"](self.raw[ch + self.HEADER_BYTES])
            for ch, defn in CHANNEL_DEFS.items()
        }

    def __str__(self) -> str:
        lines = [f"  counter : {self.counter}"]
        for ch, defn in CHANNEL_DEFS.items():
            raw = self.raw[ch + self.HEADER_BYTES]
            val = defn["conv"](raw)
            lines.append(
                f"  {defn['name']:<28s}: {val:8.2f} {defn['unit']}  (raw={raw})"
            )
        return "\n".join(lines)


class MonitorStreamWriter(rogue.interfaces.stream.Master):
    def __init__(self, vc: int):
        super().__init__()
        self.vc = vc
        self.n_sent = 0
        self.n_errors = 0
        self.log = logging.getLogger(f"caproto.{__name__}")

    def enable(self, flag):
        payload = struct.pack("<4I", 0, flag, 0, 0)
        size = len(payload)
        self.log.info(f"[VC={self.vc}] sending enable packet #{self.n_sent}: {payload}")
        frame = self._reqFrame(size, True)
        with frame.lock():
            frame.write(payload, 0)
            frame.setChannel(self.vc)
        try:
            self._sendFrame(frame)
            self.n_sent += 1
        except Exception as exc:
            self.n_errors += 1
            self.log.error(f"[VC={self.vc}] enable packet write error: {exc}  ({payload})")
            self.log.exception(f"  exception traceback:")


class MonitorStreamReader(rogue.interfaces.stream.Slave):
    def __init__(self, vc: int, queue=None):
        super().__init__()
        self.vc = vc
        self.queue = queue
        self.n_received = 0
        self.n_errors = 0
        self.last_packet = None
        self.log = logging.getLogger(f"caproto.{__name__}")

    def _acceptFrame(self, frame: rogue.interfaces.stream.Frame):
        with frame.lock():
            size = frame.getPayload()
            buf = bytearray(size)
            frame.read(buf, 0)

        self.n_received += 1
        try:
            pkt = EpixQuadMonitorPacket(bytes(buf))
            self.last_packet = pkt

            self.queue.put(pkt.pv_data())

            self.log.info(
                f"\n[VC={self.vc}] packet #{self.n_received}  ({size} B  "
                f"counter=0x{pkt.counter:08x})"
            )
            self.log.debug(pkt)
        except Exception as exc:
            self.n_errors += 1
            self.log.error(f"[VC={self.vc}] decode error: {exc}  ({size} B raw)")
            if size <= 128:
                self.log.error(f"  raw bytes: {buf.hex()}")
            self.log.exception(f"  exception traceback:")


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
                data['logs']['info'].append(f"set EpixQuadMonitor.TrigPrescaler to {flag}")
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


class EpixQuadMonitoringIOC(PVGroup):
    """
    A simple EPICS IOC defining a single integer process variable.
    """
    def __init__(self, *args, dev, lane, vc, regvc, has_microblaze, **kwargs):
        self.dev = dev
        self.lane = lane
        self.vc = vc
        self.dma_dest = lane << 8 | vc
        self.regvc = regvc
        self.has_microblaze = has_microblaze
        self.mon_reader = None
        self.mon_writer = None
        self.lastmontime = None
        self.trigrateconv = 1000
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(f"caproto.{__name__}")

    @property
    def monitor_prescale(self):
        """
        Monitor rate converted to a prescale value. Set to a minimum of 1.
        E.g.: a prescale of 10 means the monitoring will fire on every tenth trigger
        """
        prescale = int(self.set_auto_trig_rate.value // self.set_monitor_rate.value)
        if prescale == 0:
            prescale = 1
        return prescale

    @property
    def auto_trigger_period(self):
        """
        Auto trigger rate converted to period
        """
        return int(self.trigrateconv/self.set_auto_trig_rate.value)

    def check_packet(self, data):
        """
        Check that the packet is valid. This check is skipped if the microblaze is set as dead.
        """
        channels = ["nct_loc_temp", "nct_fpga_temp"]
        if self.has_microblaze:
            # only check these if the detector has a working microblaze
            channels.extend(["ana_temp", "dig_temp", "ana_in_v", "dig_in_v"])
        return all([data.get(d, 0) for d in channels])

    def check_temps(self, data):
        """
        Check that at least on the of the sensor or electronics temps are valid
        """
        stemp = False
        etemp = False

        # loop over the sensor temps to find if at least one is valid
        for channame, lowlim in self.stemp_channels:
            if channame in self.microblaze_fixup_channels:
                # values in this case are bad so ignore them
                continue
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim:
                    stemp = True
                    break

        # loop over the elec temps to find if at least one is valid
        for channame, lowlim, highlim in self.etemp_channels:
            if channame in self.microblaze_fixup_channels:
                # values in this case are bad so ignore them
                continue
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim and chan.value < highlim:
                    etemp = True
                    break

        return stemp, etemp


    def configure(self, flag, mon_period, trig_period):
        self.log.debug(f"Starting register process: dev - {self.dev}, lane,vc - {self.lane},{self.regvc}")
        queue = mp.Queue()
        proc = mp.Process(target=EpixQuadBoard.configure,
                          args=(self.dev, self.lane, self.regvc, flag, mon_period, trig_period, queue))
        proc.start()

        data = queue.get()
        # wait for response from process
        proc.join()

        if data:
            self.log.debug("Register process has exitted successfully")
        else:
            self.log.error(f"Register process has failed!")

        # send the special monitor enable packet
        if self.mon_writer is not None:
            self.mon_writer.enable(flag)

        return data

    async def __ainit__(self, async_lib):
        self.async_lib = async_lib
        queue = async_lib.ThreadsafeQueue()
        self.log.info(f"Initializing monitor stream dma: dev - {self.dev}, dest,lane,vc - {self.dma_dest},{self.lane},{self.vc}")
        dma = rogue.hardware.axi.AxiStreamDma(self.dev, self.dma_dest, True)
        self.mon_reader = MonitorStreamReader(vc=self.vc, queue=queue)
        self.mon_writer = MonitorStreamWriter(vc=self.vc)
        pyrogue.streamConnect(dma, self.mon_reader)
        pyrogue.streamConnect(self.mon_writer, dma)

        try:
            count = 0
            errcount = 0
            self.lastmontime = time.time()
            while True:
                data = await queue.async_get()
                if self.check_packet(data):
                    self.lastmontime= time.time()
                    for name, value in data.items():
                        if hasattr(self, name):
                            # fixup up garbage values if microblaze is not functioning
                            if (not self.has_microblaze) and (name in self.microblaze_fixup_channels):
                                value = 0.0
                            await getattr(self, name).write(value=value)
                    stemp, etemp = self.check_temps(data)
                    await self.stemp_ok.write(value=stemp)
                    await self.etemp_ok.write(value=etemp)
                    count += 1
                    await self.moncnt.write(value=count)
                else:
                    self.log.error(f"Bad monitoring packet returned by the detector: {data}")
                    errcount += 1
                    await self.monerrcnt.write(value=errcount)
        except Exception:
            self.log.exception("Server monitoring queue reader encountered an error:")
        finally:
            self.log.info("Server monitoring queue reader exitted.")

    # This creates a PV named 'simple:number' with a default value of 42
    set_monitor = pvproperty(name="SET_MONITOR",
                             value=0,
                             dtype=PvpropertyInteger[LongoutFields],
                             record=LongoutFields,
                             doc="Start/Stop epixMon")
    moncnt = pvproperty(name="MONCNT",
                        value=0,
                        dtype=PvpropertyInteger[LonginFields],
                        record=LonginFields,
                        doc="epix monitor counts")
    monerrcnt = pvproperty(name="MONERRCNT",
                           value=0,
                           dtype=PvpropertyInteger[LonginFields],
                           record=LonginFields,
                           doc="epix monitor error counts")
    monchk = pvproperty(name="MONCHK",
                        value=0,
                        dtype=PvpropertyInteger[LonginFields],
                        record=LonginFields,
                        upper_alarm_limit=0.5,
                        lower_alarm_limit=-0.5,
                        upper_warning_limit=0.5,
                        lower_warning_limit=-0.5,
                        alarm_group="monchk",
                        doc="epixMon check")
    monchkdelay = pvproperty(name="MONCHKDELAY",
                             value=5,
                             dtype=PvpropertyInteger[LonginFields],
                             record=LonginFields,
                             doc="epix check delay")
    new_firmware = pvproperty(name="NEW_FIRMWARE",
                              value=2,
                              dtype=PvpropertyEnum[MbbiFields],
                              record=MbbiFields,
                              enum_strings=["epix100a", "epix10ka", "lcls2"],
                              doc="epix firmware type")
    set_monitor_rate = pvproperty(name="SET_MONITOR_RATE",
                                  value=1.0,
                                  dtype=PvpropertyDouble[AoFields],
                                  record=AoFields,
                                  precision=1,
                                  units="Hz",
                                  doc="Set the monitor update rate for epixMon")
    set_auto_trig_rate = pvproperty(name="SET_AUTO_TRIG_RATE",
                                    value=10.0,
                                    dtype=PvpropertyDouble[AoFields],
                                    record=AoFields,
                                    precision=1,
                                    units="Hz",
                                    doc="Set the auto trigger rate for epixMon")
    temp1 = pvproperty(name="TEMP1",
                       value=-99.0,
                       dtype=PvpropertyDouble[AiFields],
                       record=AiFields,
                       upper_alarm_limit=1000.0,
                       lower_alarm_limit=0.0,
                       upper_warning_limit=1000.0,
                       lower_warning_limit=0.0,
                       alarm_group="temp1",
                       precision=2,
                       units="C",
                       doc="Therm0 Temp")
    temp2 = pvproperty(name="TEMP2",
                       value=-99.0,
                       dtype=PvpropertyDouble[AiFields],
                       record=AiFields,
                       upper_alarm_limit=1000.0,
                       lower_alarm_limit=0.0,
                       upper_warning_limit=1000.0,
                       lower_warning_limit=0.0,
                       alarm_group="temp2",
                       precision=2,
                       units="C",
                       doc="Therm1 Temp")
    temp3 = pvproperty(name="TEMP3",
                       value=-99.0,
                       dtype=PvpropertyDouble[AiFields],
                       record=AiFields,
                       upper_alarm_limit=1000.0,
                       lower_alarm_limit=0.0,
                       upper_warning_limit=1000.0,
                       lower_warning_limit=0.0,
                       alarm_group="temp3",
                       precision=2,
                       units="C",
                       doc="SHT31 Temp")
    humidity = pvproperty(name="HUMIDITY",
                          value=0.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=101.0,
                          lower_alarm_limit=-1.0,
                          upper_warning_limit=101.0,
                          lower_warning_limit=-1.0,
                          alarm_group="humidity",
                          precision=2,
                          units="%",
                          doc="SHT31 Humidity")
    ana_in_v = pvproperty(name="ANA_IN_V",
                          value=0.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=100.0,
                          lower_alarm_limit=-1.0,
                          upper_warning_limit=100.0,
                          lower_warning_limit=-1.0,
                          alarm_group="ana_in_v",
                          precision=3,
                          units="V",
                          doc="Analog Voltage")
    dig_in_v = pvproperty(name="DIG_IN_V",
                          value=0.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=100.0,
                          lower_alarm_limit=-1.0,
                          upper_warning_limit=100.0,
                          lower_warning_limit=-1.0,
                          alarm_group="dig_in_v",
                          precision=3,
                          units="V",
                          doc="Digital Voltage")
    asic_ana_cur = pvproperty(name="ASIC_ANA_CUR",
                              value=0.0,
                              dtype=PvpropertyDouble[AiFields],
                              record=AiFields,
                              upper_alarm_limit=100.0,
                              lower_alarm_limit=-1.0,
                              upper_warning_limit=100.0,
                              lower_warning_limit=-1.0,
                              alarm_group="asic_ana_cur",
                              precision=3,
                              units="A",
                              doc="ASIC Analog Current")
    asic_dig_cur = pvproperty(name="ASIC_DIG_CUR",
                              value=0.0,
                              dtype=PvpropertyDouble[AiFields],
                              record=AiFields,
                              upper_alarm_limit=100.0,
                              lower_alarm_limit=-1.0,
                              upper_warning_limit=100.0,
                              lower_warning_limit=-1.0,
                              alarm_group="asic_dig_cur",
                              precision=3,
                              units="A",
                              doc="ASIC Digital Current")
    ana_temp = pvproperty(name="ANA_TEMP",
                          value=-99.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=1000.0,
                          lower_alarm_limit=0.0,
                          upper_warning_limit=1000.0,
                          lower_warning_limit=0.0,
                          alarm_group="ana_temp",
                          precision=2,
                          units="C",
                          doc="PwrAnaTemp")
    dig_temp = pvproperty(name="DIG_TEMP",
                          value=-99.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=1000.0,
                          lower_alarm_limit=0.0,
                          upper_warning_limit=1000.0,
                          lower_warning_limit=0.0,
                          alarm_group="dig_temp",
                          precision=2,
                          units="C",
                          doc="PwrDigTemp")
    nct_loc_temp = pvproperty(name="NCT_LOC_TEMP",
                              value=-99.0,
                              dtype=PvpropertyDouble[AiFields],
                              record=AiFields,
                              upper_alarm_limit=1000.0,
                              lower_alarm_limit=0.0,
                              upper_warning_limit=1000.0,
                              lower_warning_limit=0.0,
                              alarm_group="nct_loc_temp",
                              precision=2,
                              units="C",
                              doc="NCT218 Local Temp.")
    nct_fpga_temp = pvproperty(name="NCT_FPGA_TEMP",
                               value=-99.0,
                               dtype=PvpropertyDouble[AiFields],
                               record=AiFields,
                               upper_alarm_limit=1000.0,
                               lower_alarm_limit=0.0,
                               upper_warning_limit=1000.0,
                               lower_warning_limit=0.0,
                               alarm_group="nct_fpga_temp",
                               precision=2,
                               units="C",
                               doc="NCT218 Remote Temp.")
    asic_a0_2v5_cur = pvproperty(name="ASIC_A0_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_a0_2v5_cur",
                                 precision=3,
                                 units="A",
                                 doc="ASIC_A0_2V5 Curr.")
    asic_a1_2v5_cur = pvproperty(name="ASIC_A1_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_a1_2v5_cur",
                                 precision=3,
                                 units="A",
                                 doc="ASIC_A1_2V5 Curr.")
    asic_a2_2v5_cur = pvproperty(name="ASIC_A2_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_a2_2v5_cur",
                                 precision=3,
                                 units="A",
                                 doc="ASIC_A2_2V5 Curr.")
    asic_a3_2v5_cur = pvproperty(name="ASIC_A3_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_a3_2v5_cur",
                                 precision=3,
                                 units="A",
                                 doc="ASIC_A3_2V5 Curr.")
    asic_d0_2v5_cur = pvproperty(name="ASIC_D0_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100000.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100000.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_d0_2v5_cur",
                                 precision=3,
                                 units="mA",
                                 doc="ASIC_D0_2V5 Curr.")
    asic_d1_2v5_cur = pvproperty(name="ASIC_D1_2V5_CUR",
                                 value=0.0,
                                 dtype=PvpropertyDouble[AiFields],
                                 record=AiFields,
                                 upper_alarm_limit=100000.0,
                                 lower_alarm_limit=-1.0,
                                 upper_warning_limit=100000.0,
                                 lower_warning_limit=-1.0,
                                 alarm_group="asic_d1_2v5_cur",
                                 precision=3,
                                 units="mA",
                                 doc="ASIC_D1_2V5 Curr.")
    asic_a0_2v5_h_temp = pvproperty(name="ASIC_A0_2V5_H_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a0_2v5_h_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A0_2V5_H Temp.")
    asic_a0_2v5_l_temp = pvproperty(name="ASIC_A0_2V5_L_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a0_2v5_l_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A0_2V5_L Temp.")
    asic_a1_2v5_h_temp = pvproperty(name="ASIC_A1_2V5_H_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a1_2v5_h_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A1_2V5_H Temp.")
    asic_a1_2v5_l_temp = pvproperty(name="ASIC_A1_2V5_L_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a1_2v5_l_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A1_2V5_L Temp.")
    asic_a2_2v5_h_temp = pvproperty(name="ASIC_A2_2V5_H_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a2_2v5_h_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A2_2V5_H Temp.")
    asic_a2_2v5_l_temp = pvproperty(name="ASIC_A2_2V5_L_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a2_2v5_l_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A2_2V5_L Temp.")
    asic_a3_2v5_h_temp = pvproperty(name="ASIC_A3_2V5_H_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a3_2v5_h_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A3_2V5_H Temp.")
    asic_a3_2v5_l_temp = pvproperty(name="ASIC_A3_2V5_L_TEMP",
                                    value=-99.0,
                                    dtype=PvpropertyDouble[AiFields],
                                    record=AiFields,
                                    upper_alarm_limit=1000.0,
                                    lower_alarm_limit=0.0,
                                    upper_warning_limit=1000.0,
                                    lower_warning_limit=0.0,
                                    alarm_group="asic_a3_2v5_l_temp",
                                    precision=2,
                                    units="C",
                                    doc="ASIC_A3_2V5_L Temp.")
    asic_d0_2v5_temp = pvproperty(name="ASIC_D0_2V5_TEMP",
                                  value=-99.0,
                                  dtype=PvpropertyDouble[AiFields],
                                  record=AiFields,
                                  upper_alarm_limit=1000.0,
                                  lower_alarm_limit=0.0,
                                  upper_warning_limit=1000.0,
                                  lower_warning_limit=0.0,
                                  precision=2,
                                  alarm_group="asic_d0_2v5_temp",
                                  units="C",
                                  doc="ASIC_D0_2V5 Temp.")
    asic_d1_2v5_temp = pvproperty(name="ASIC_D1_2V5_TEMP",
                                  value=-99.0,
                                  dtype=PvpropertyDouble[AiFields],
                                  record=AiFields,
                                  upper_alarm_limit=1000.0,
                                  lower_alarm_limit=0.0,
                                  upper_warning_limit=1000.0,
                                  lower_warning_limit=0.0,
                                  alarm_group="asic_d1_2v5_temp",
                                  precision=2,
                                  units="C",
                                  doc="ASIC_D1_2V5 Temp.")
    asic_a0_1v8_temp = pvproperty(name="ASIC_A0_1V8_TEMP",
                                  value=-99.0,
                                  dtype=PvpropertyDouble[AiFields],
                                  record=AiFields,
                                  upper_alarm_limit=1000.0,
                                  lower_alarm_limit=0.0,
                                  upper_warning_limit=1000.0,
                                  lower_warning_limit=0.0,
                                  alarm_group="asic_a0_1v8_temp",
                                  precision=2,
                                  units="C",
                                  doc="ASIC_A0_1V8 Temp.")
    asic_a1_1v8_temp = pvproperty(name="ASIC_A1_1V8_TEMP",
                                  value=-99.0,
                                  dtype=PvpropertyDouble[AiFields],
                                  record=AiFields,
                                  upper_alarm_limit=1000.0,
                                  lower_alarm_limit=0.0,
                                  upper_warning_limit=1000.0,
                                  lower_warning_limit=0.0,
                                  alarm_group="asic_a1_1v8_temp",
                                  precision=2,
                                  units="C",
                                  doc="ASIC_A1_1V8 Temp.")
    asic_a2_1v8_temp = pvproperty(name="ASIC_A2_1V8_TEMP",
                                  value=-99.0,
                                  dtype=PvpropertyDouble[AiFields],
                                  record=AiFields,
                                  upper_alarm_limit=1000.0,
                                  lower_alarm_limit=0.0,
                                  upper_warning_limit=1000.0,
                                  lower_warning_limit=0.0,
                                  alarm_group="asic_a2_1v8_temp",
                                  precision=2,
                                  units="C",
                                  doc="ASIC_A2_1V8 Temp.")
    pcb_ana_temp0 = pvproperty(name="PCB_ANA_TEMP0",
                               value=-99.0,
                               dtype=PvpropertyDouble[AiFields],
                               record=AiFields,
                               upper_alarm_limit=1000.0,
                               lower_alarm_limit=0.0,
                               upper_warning_limit=1000.0,
                               lower_warning_limit=0.0,
                               alarm_group="pcb_ana_temp0",
                               precision=2,
                               units="C",
                               doc="PcbAnaTemp0")
    pcb_ana_temp1 = pvproperty(name="PCB_ANA_TEMP1",
                               value=-99.0,
                               dtype=PvpropertyDouble[AiFields],
                               record=AiFields,
                               upper_alarm_limit=1000.0,
                               lower_alarm_limit=0.0,
                               upper_warning_limit=1000.0,
                               lower_warning_limit=0.0,
                               alarm_group="pcb_ana_temp1",
                               precision=2,
                               units="C",
                               doc="PcbAnaTemp1")
    pcb_ana_temp2 = pvproperty(name="PCB_ANA_TEMP2",
                               value=-99.0,
                               dtype=PvpropertyDouble[AiFields],
                               record=AiFields,
                               upper_alarm_limit=1000.0,
                               lower_alarm_limit=0.0,
                               upper_warning_limit=1000.0,
                               lower_warning_limit=0.0,
                               alarm_group="pcb_ana_temp2",
                               precision=2,
                               units="C",
                               doc="PcbAnaTemp2")
    tropt_temp = pvproperty(name="TROPT_TEMP",
                            value=-99.0,
                            dtype=PvpropertyDouble[AiFields],
                            record=AiFields,
                            upper_alarm_limit=1000.0,
                            lower_alarm_limit=0.0,
                            upper_warning_limit=1000.0,
                            lower_warning_limit=0.0,
                            alarm_group="tropt_temp",
                            precision=2,
                            units="C",
                            doc="TrOptTemp")
    tropt_volt = pvproperty(name="TROPT_VOLT",
                            value=0.0,
                            dtype=PvpropertyDouble[AiFields],
                            record=AiFields,
                            upper_alarm_limit=100.0,
                            lower_alarm_limit=-1.0,
                            upper_warning_limit=100.0,
                            lower_warning_limit=-1.0,
                            alarm_group="tropt_volt",
                            precision=3,
                            units="V",
                            doc="TrOptVcc")
    tropt_txpwr = pvproperty(name="TROPT_TXPWR",
                             value=0.0,
                             dtype=PvpropertyDouble[AiFields],
                             record=AiFields,
                             upper_alarm_limit=100000.0,
                             lower_alarm_limit=-1.0,
                             upper_warning_limit=100000.0,
                             lower_warning_limit=-1.0,
                             alarm_group="tropt_txpwr",
                             precision=3,
                             units="uW",
                             doc="TrOptTxPwr")
    tropt_rxpwr = pvproperty(name="TROPT_RXPWR",
                             value=0.0,
                             dtype=PvpropertyDouble[AiFields],
                             record=AiFields,
                             upper_alarm_limit=100000.0,
                             lower_alarm_limit=-1.0,
                             upper_warning_limit=100000.0,
                             lower_warning_limit=-1.0,
                             alarm_group="tropt_rxpwr",
                             precision=3,
                             units="uW",
                             doc="TrOptRxPwr")
    stemp_ok = pvproperty(name="STEMP_OK",
                          value=0,
                          dtype=PvpropertyInteger[LonginFields],
                          record=LonginFields,
                          alarm_group="stemp_ok",
                          doc="epix sensor temp OK")
    etemp_ok = pvproperty(name="ETEMP_OK",
                          value=0,
                          dtype=PvpropertyInteger[LonginFields],
                          record=LonginFields,
                          alarm_group="etemp_ok",
                          doc="epix electronics temp OK")
    microblaze = pvproperty(name="MICROBLAZE",
                            value=1,
                            dtype=PvpropertyEnum[BiFields],
                            record=BiFields,
                            alarm_group="microblaze",
                            enum_strings=["NO", "YES"],
                            doc="epix has working MicroBlaze")
    firmware_version = pvproperty(name="FWVERSION",
                                  value="",
                                  dtype=PvpropertyString[StringinFields],
                                  record=StringinFields,
                                  doc="epix fw version")
    firmware_githash = pvproperty(name="FWGITHASH",
                                  value="",
                                  dtype=PvpropertyString[StringinFields],
                                  record=StringinFields,
                                  doc="epix fw githash")
    firmware_bldstr = pvproperty(name="FWBLDSTR",
                                 value="",
                                 dtype=PvpropertyChar[WaveformFields],
                                 record=WaveformFields,
                                 string_encoding='ascii',
                                 max_length=256,
                                 doc="epix fw build str")
    carrier_id_0 = pvproperty(name="CARRIER_ID_0",
                              value="",
                              dtype=PvpropertyString[StringinFields],
                              record=StringinFields,
                              doc="epix carrier id 0")
    carrier_id_1 = pvproperty(name="CARRIER_ID_1",
                              value="",
                              dtype=PvpropertyString[StringinFields],
                              record=StringinFields,
                              doc="epix carrier id 1")
    carrier_id_2 = pvproperty(name="CARRIER_ID_2",
                              value="",
                              dtype=PvpropertyString[StringinFields],
                              record=StringinFields,
                              doc="epix carrier id 2")
    carrier_id_3 = pvproperty(name="CARRIER_ID_3",
                              value="",
                              dtype=PvpropertyString[StringinFields],
                              record=StringinFields,
                              doc="epix carrier id 3")
    stemp_channels = {
            ("temp1", -273.15),
            ("temp2", -273.15),
    }
    etemp_channels = {
            ("temp3", -45.0, 130),
            ("nct_loc_temp", 0.0, 200),
            ("nct_fpga_temp", 0.0, 200),
            ("ana_temp", 0.0, 200),
            ("dig_temp", 0.0, 200),
            ("tropt_temp", 0.0, 200),
    }
    microblaze_fixup_channels = {"temp1", "temp2", "ana_temp", "dig_temp"}
 
    @monchk.scan(period=1.0, use_scan_field=True)
    async def monchk(self, instance, async_lib):
        """
        Scan this record
        """
        if self.lastmontime is not None:
            curtime = time.time()
            checkval = curtime-self.lastmontime > self.monchkdelay.value
            await instance.write(value=checkval)

    @set_monitor.putter
    async def set_monitor(self, instance, flag):
        if flag:
            state = 'on'
        else:
            state = 'off'
        self.log.info(f"Requested epix register configure - monitoring {state}")
        data = await self.async_lib.library.to_thread(self.configure, bool(flag), self.monitor_prescale, self.auto_trigger_period)
        self.log.info("Epix register configuration completed")
        for name, value in data.items():
            if name == 'logs':
                for log_name, msgs in value.items():
                    log_level = logging.getLevelName(log_name.upper())
                    for msg in msgs:
                        self.log.log(log_level, msg)
            elif hasattr(self, name):
                await getattr(self, name).write(value=value)

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
    parser.add_argument(
        "--dev",
        default="/dev/datadev_0",
        help="PCIe DMA device (default: /dev/datadev_0)",
    )
    parser.add_argument(
        "--lane",
        default=0,
        type=int,
        help="PGP lane number (default: 0)",
    )
    parser.add_argument(
        "--vc",
        default=3,
        type=int,
        help="Monitor stream virtual channel to listen on "
        "(default: 3 — first bypassed channel in EventBuilder)",
    )
    parser.add_argument(
        "--regvc",
        default=1,
        type=int,
        help="Register virtual channel (default: 1)"
    )
    parser.add_argument(
        "--no-microblaze",
        action='store_false',
        dest='microblaze',
        help="Flag to indicate the detector has a non-functional microblaze processor"
    )

    args = parser.parse_args()
    ioc_options, run_options = split_args(args)

    # Initialize the logger
    if args.verbose is not None:
        if args.verbose == 0:
            log_level = logging.WARN
        elif args.verbose == 1:
            log_level = logging.INFO
        else:
            log_level = logging.DEBUG
    else:
        log_level = logging.WARN
    config_caproto_logging(level=log_level)

    # Start the server
    ioc = EpixQuadMonitoringIOC(dev=args.dev, lane=args.lane, vc=args.vc, regvc=args.regvc, has_microblaze=args.microblaze, **ioc_options)
    run(ioc.pvdb, **run_options, startup_hook=ioc.__ainit__)


if __name__ == '__main__':
    main()
