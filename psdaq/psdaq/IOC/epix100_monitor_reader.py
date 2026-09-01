"""
epix100_monitor_reader.py

Read the ePix100 slow-ADC environmental monitor stream via rogue.

The ePix100 FPGA continuously sends small "monitor packets" on a dedicated
PGP virtual channel (separate from the main image data channel).  Each packet
carries a packet counter plus 16 slow-ADC readings that encode temperatures,
humidity, currents, and voltages.

In the normal DAQ pipeline the DRP reads only the image data and the monitor
channel is bypassed (EventBuilder.Bypass bit 3 = 0x38).  This script connects
directly to those bypassed DMA channels and decodes the packets.

Packet format (68 bytes = 17 × int32 little-endian):
  word[ 0]        : packet counter (unsigned 32-bit)
  word[ 1]        : channel  0  (raw signed int32)
  ...
  word[16]        : channel 15

Channel mapping (from epix100 viewer envConf):
  ch  7  (word[ 8])  Strong Back Temp.    raw/100   °C
  ch  8  (word[ 9])  Ambient Temp.        raw/100   °C
  ch  9  (word[10])  Humidity             raw/100   %
  ch 10  (word[11])  ASIC Analog Current  raw/1000  A
  ch 11  (word[12])  ASIC Digital Current raw/1000  A
  ch 12  (word[13])  Guard Ring Current   raw/1000  A
  ch 13  (word[14])  Analog Voltage       raw/1000  V
  ch 14  (word[15])  Digital Voltage      raw/1000  V

Channels 0-6 and 15 are unused/unconnected.

VC mapping (from epix100_config.py / firmware README):
  lane 0, VC 0  →  SRP register bus   (used for firmware register access)
  lane 0, VC 1  →  image / event data (used by DRP, virtChan=1)
  lane 0, VC 2  →  epix100 image batcher stream (EventBuilder bit 2 = 0x4)
  lane 0, VC 3+ →  monitor / slow-ADC stream    (EventBuilder bits 3-5 bypassed, 0x38)

NOTE: The exact VC for the monitor packets depends on the firmware version.
      Start with VC=3 (first bypassed channel).  Use --vcs 3,4,5 to scan all
      three bypassed channels simultaneously.

Requirements (same environment as the DAQ epix100 configuration):
    rogue  pyrogue  ePixFpga  lcls2_epix_hr_pcie

Mandatory (for --Ext EPICS publishing):
    caproto

Usage examples:
    # listen passively on VC 3 (monitor stream must already be enabled by DAQ):
    python epix100_monitor_reader.py

    # enable the stream yourself (requires exclusive SRP access, no DAQ running):
    python epix100_monitor_reader.py --enable

    # scan all three bypassed VCs at once:
    python epix100_monitor_reader.py --vcs 3,4,5

    # publish decoded values as EPICS PVAccess PVs:
    python epix100_monitor_reader.py --ext CMP004

    # non-default hardware:
    python epix100_monitor_reader.py --dev /dev/datadev_1 --lane 0 --vc 4
"""

import sys
import time
import struct
import logging
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional
# rougue imports
from psdaq.utils import enable_epix_100a_gen2
import epix100a_gen2
import ePixFpga as fpga
import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.protocols.srp
import pyrogue
# caproto imports
from caproto import config_caproto_logging
from caproto.server import PVGroup, PvpropertyDouble, PvpropertyInteger, PvpropertyString, PvpropertyChar, PvpropertyEnum, template_arg_parser, pvproperty, run
from caproto.server.records import AoFields, AiFields, LongoutFields, LonginFields, StringinFields, WaveformFields, MbbiFields


# ──────────────────────────────────────────────────────────────────────────────
# Packet decoding
# ──────────────────────────────────────────────────────────────────────────────

# Channel definitions confirmed from envConf in the epix100 viewer software.
# 'id' = channel index (0-based); maps to packet word[id+1] since word[0] is counter.
# 'conv' converts raw signed int32 to physical units.
# 'pv_signal' is the EPICS signal name used in the PV: HUTCH:EPIX100:NN:SIGNAL
CHANNEL_DEFS: Dict[int, Any] = {
    7: dict(
        name="Strong Back Temp.",
        unit="°C",
        conv=lambda d: d / 100,
        pv_signal="temp1",
    ),
    8: dict(
        name="Ambient Temp.",
        unit="°C",
        conv=lambda d: d / 100,
        pv_signal="temp2",
    ),
    9: dict(
        name="Humidity",
        unit="%",
        conv=lambda d: d / 100,
        pv_signal="humidity",
    ),
    10: dict(
        name="ASIC Analog Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="asic_ana_cur",
    ),
    11: dict(
        name="ASIC Digital Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="asic_dig_cur",
    ),
    12: dict(
        name="Guard Ring Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="asic_gr_cur",
    ),
    13: dict(
        name="Analog Voltage",
        unit="V",
        conv=lambda d: d / 1000,
        pv_signal="ana_in_v",
    ),
    14: dict(
        name="Digital Voltage",
        unit="V",
        conv=lambda d: d / 1000,
        pv_signal="dig_in_v",
    ),
}


class EpixMonitorPacket:
    """
    One ePix100 slow-ADC monitor stream packet.

    Format (68 bytes = 17 × int32 little-endian):
      word[ 0]        : packet counter (decoded as unsigned)
      word[ 1]        : channel  0  (raw signed int32)
      ...
      word[16]        : channel 15

    Channel mapping (from epix100 viewer envConf):
      ch  7  (word[ 8])  Strong Back Temp.    raw/100   °C
      ch  8  (word[ 9])  Ambient Temp.        raw/100   °C
      ch  9  (word[10])  Humidity             raw/100   %
      ch 10  (word[11])  ASIC Analog Current  raw/1000  A
      ch 11  (word[12])  ASIC Digital Current raw/1000  A
      ch 12  (word[13])  Guard Ring Current   raw/1000  A
      ch 13  (word[14])  Analog Voltage       raw/1000  V
      ch 14  (word[15])  Digital Voltage      raw/1000  V

    Values are signed int32.  Channels 0-6 and 15 are unused/unconnected.
    Negative readings on startup or unconnected sensors are normal.
    """

    N_WORDS = 17
    N_CHANNELS = 16
    PACKET_BYTES = N_WORDS * 4  # 68

    def __init__(self, data: bytes):
        if len(data) < self.PACKET_BYTES:
            raise ValueError(
                f"Packet too short: {len(data)} B  (expected {self.PACKET_BYTES} B)"
            )
        # word[0] is an unsigned counter; words[1:] are signed sensor readings.
        counter = struct.unpack_from("<I", data, 0)[0]
        signed = struct.unpack_from("<16i", data, 4)
        self.raw = (counter,) + signed

    @property
    def counter(self) -> int:
        return self.raw[0]

    def channel_raw(self, ch: int) -> int:
        """Raw signed int32 for channel ch (0–15)."""
        return self.raw[ch + 1]

    def channel_value(self, ch: int) -> Optional[float]:
        """Converted physical value for a defined channel, or None if undefined."""
        if ch not in CHANNEL_DEFS:
            return None
        return CHANNEL_DEFS[ch]["conv"](self.raw[ch + 1])

    def as_dict(self) -> dict:
        """Return {sensor_name: physical_value} for all defined channels."""
        return {
            defn["name"]: defn["conv"](self.raw[ch + 1])
            for ch, defn in CHANNEL_DEFS.items()
        }

    def pv_data(self) -> dict:
        """Return {pv_name: physical_value} for all defined channels."""
        return {
            defn["pv_signal"]: defn["conv"](self.raw[ch + 1])
            for ch, defn in CHANNEL_DEFS.items()
        }

    def __str__(self) -> str:
        lines = [f"  counter : {self.counter}"]
        for ch, defn in CHANNEL_DEFS.items():
            raw = self.raw[ch + 1]
            val = defn["conv"](raw)
            lines.append(
                f"  {defn['name']:<28s}: {val:8.2f} {defn['unit']}  (raw={raw})"
            )
        return "\n".join(lines)


class MonitorStream(rogue.interfaces.stream.Slave):
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
            pkt = EpixMonitorPacket(bytes(buf))
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


class Epix100aBoard(pyrogue.Root):
    def __init__(self, dev, lane, vc):
        super().__init__(name='ePixBoard', pollEn=False, initRead=False)
        dma = rogue.hardware.axi.AxiStreamDma(dev, lane << 8 | vc, True)
        srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(dma, srp)
        self.addInterface(dma)
        self.addInterface(srp)
        self.add(fpga.Epix100a(
            name='ePix100aFPGA', offset=0,
            memBase=srp, hidden=False, enabled=True))

    @staticmethod
    def configure(dev, lane, vc, flag, mon_period, trig_period, queue):
        data = {}
        try:
            with Epix100aBoard(dev, lane, vc) as root:
                # read firmware info
                fw = root.ePix100aFPGA.AxiVersion.FpgaVersion.get()
                data["firmware_version"] = '0x%08x' % fw
                githash = root.ePix100aFPGA.AxiVersion.GitHash.get()
                data["firmware_githash"] = '%040x' % githash
                bldstr = root.ePix100aFPGA.AxiVersion.BuildStamp.get()
                data["firmware_bldstr"] = bldstr
                # configure the monitoring registers
                root.ePix100aFPGA.EpixFpgaRegisters.RunTriggerEnable.set(1)
                root.ePix100aFPGA.EpixFpgaRegisters.PgpTrigEn.set(1)
                root.ePix100aFPGA.SlowAdcRegisters.enable.set(1)
                root.ePix100aFPGA.SlowAdcRegisters.StreamEn.set(flag)
                root.ePix100aFPGA.SlowAdcRegisters.StreamPeriod.set(mon_period)
                root.ePix100aFPGA.EpixFpgaRegisters.AutoRunEnable.set(1)
                root.ePix100aFPGA.EpixFpgaRegisters.AutoRunPeriodMs.set(trig_period)
        finally:
            # send the firmware info back
            queue.put(data)


class EpixMonitoringIOC(PVGroup):
    """
    A simple EPICS IOC defining a single integer process variable.
    """
    def __init__(self, *args, dev, lane, vc, regvc, **kwargs):
        self.dev = dev
        self.lane = lane
        self.vc = vc
        self.regvc = regvc
        self.monrateconv = 100000000
        self.trigrateconv = 1000
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(f"caproto.{__name__}")

    @property
    def monitor_period(self):
        """
        Monitor rate converted to period
        """
        return int(self.monrateconv / self.set_monitor_rate.value)

    @property
    def auto_trigger_period(self):
        """
        Auto trigger rate converted to period
        """
        return int(self.trigrateconv/self.set_auto_trig_rate.value)

    def configure(self, flag, mon_period, trig_period):
        self.log.debug(f"Starting register process: dev - {self.dev}, lane,vc - {self.lane},{self.regvc}")
        queue = mp.Queue()
        proc = mp.Process(target=Epix100aBoard.configure,
                          args=(self.dev, self.lane, self.regvc, flag, mon_period, trig_period, queue))
        proc.start()

        data = queue.get()
        # wait for response from process
        proc.join()

        if data:
            self.log.debug("Register process has exitted successfully")
        else:
            self.log.error(f"Register process has failed!")

        return data

    async def __ainit__(self, async_lib):
        self.monitoring = False
        self.async_lib = async_lib
        queue = async_lib.ThreadsafeQueue()
        dma_dest = self.lane << 8 | self.vc
        self.log.info(f"Initializing monitor stream dma: dev - {self.dev}, dest,lane,vc - {dma_dest},{self.lane},{self.vc}")
        dma = rogue.hardware.axi.AxiStreamDma(self.dev, dma_dest, True)
        mon = MonitorStream(vc=self.vc, queue=queue)
        pyrogue.streamConnect(dma, mon)

        try:
            count = 0
            self.lastmontime = time.time()
            while True:
                data = await queue.async_get()
                self.lastmontime= time.time()
                for name, value in data.items():
                    if hasattr(self, name):
                        await getattr(self, name).write(value=value)
                count += 1
                await self.moncnt.write(value=count)
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
    monchk = pvproperty(name="MONCHK",
                        value=0,
                        dtype=PvpropertyInteger[LonginFields],
                        record=LonginFields,
                        upper_alarm_limit=0.5,
                        lower_alarm_limit=-0.5,
                        upper_warning_limit=0.5,
                        lower_warning_limit=-0.5,
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
                       precision=2,
                       units="C",
                       doc="Strong Back Temp.")
    temp2 = pvproperty(name="TEMP2",
                       value=-99.0,
                       dtype=PvpropertyDouble[AiFields],
                       record=AiFields,
                       upper_alarm_limit=1000.0,
                       lower_alarm_limit=0.0,
                       upper_warning_limit=1000.0,
                       lower_warning_limit=0.0,
                       precision=2,
                       units="C",
                       doc="Ambient Temp.")
    humidity = pvproperty(name="HUMIDITY",
                          value=0.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=101.0,
                          lower_alarm_limit=-1.0,
                          upper_warning_limit=101.0,
                          lower_warning_limit=-1.0,
                          precision=2,
                          units="%",
                          doc="Humidity")
    ana_in_v = pvproperty(name="ANA_IN_V",
                          value=0.0,
                          dtype=PvpropertyDouble[AiFields],
                          record=AiFields,
                          upper_alarm_limit=100.0,
                          lower_alarm_limit=-1.0,
                          upper_warning_limit=100.0,
                          lower_warning_limit=-1.0,
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
                              precision=3,
                              units="A",
                              doc="ASIC Digital Current")
    asic_gr_cur = pvproperty(name="ASIC_GR_CUR",
                             value=0.0,
                             dtype=PvpropertyDouble[AiFields],
                             record=AiFields,
                             upper_alarm_limit=100.0,
                             lower_alarm_limit=-1.0,
                             upper_warning_limit=100.0,
                             lower_warning_limit=-1.0,
                             precision=3,
                             units="A",
                             doc="Guard Ring Current")
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
 
 
    @monchk.scan(period=1.0, use_scan_field=True)
    async def monchk(self, instance, async_lib):
        """
        Scan this record
        """
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
        data = await self.async_lib.library.to_thread(self.configure, bool(flag), self.monitor_period, self.auto_trigger_period)
        self.log.info("Epix register configuration completed")
        for name, value in data.items():
            if hasattr(self, name):
                await getattr(self, name).write(value=value)


if __name__ == '__main__':
    # Parse standard EPICS IOC command-line options
    parser, split_args = template_arg_parser(
        default_prefix="DET:EPIX:CMP004:",
        desc="Read ePix100 environmental monitor packets via rogue and publish via caproto IOC"
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
        default=0,
        type=int,
        help="Register virtual channel (default: 0)"
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
    ioc = EpixMonitoringIOC(dev=args.dev, lane=args.lane, vc=args.vc, regvc=args.regvc, **ioc_options)
    run(ioc.pvdb, **run_options, startup_hook=ioc.__ainit__)
