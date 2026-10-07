"""
epix_monitor_base.py

Shared building blocks for the ePix environmental monitoring IOCs.

The ePix detectors all send small "monitor packets" on a dedicated PGP virtual
channel carrying slow-ADC readings (temperatures, humidity, currents, voltages).
The per-detector IOCs differ only in the packet layout, the channel table, the
conversion functions and the rogue register (board) class; everything else --
the caproto PV definitions, the monitor read loop, the register-configuration
subprocess and the command-line handling -- is identical.

This module holds that common part:

  * ``pvproperty`` factories (``temp_pv``, ``current_pv``, ...) that collapse the
    repeated limit/precision/units boilerplate of the sensor PVs.
  * ``MonitorPacket``, a decoder base class parameterised by a struct format and
    a header length.
  * ``EpixMonitoringIOCBase``, a ``PVGroup`` with the PVs and behavior common to
    every ePix monitoring IOC, plus hooks for the detector-specific parts.
  * ``add_common_args`` / ``setup_logging`` helpers for ``main()``.

NOTE: This module deliberately imports neither ``rogue`` nor any detector
firmware package.  Each detector IOC pulls in its own firmware tree (via
``psdaq.utils.enable_*``, which requires ``SUBMODULEDIR``), and importing one
here would make every IOC depend on all of them.  The rogue stream classes live
in ``epix_monitor_stream``.
"""

import time
import struct
import logging
import multiprocessing as mp
from typing import Any, Dict, Optional

from caproto.server import (
        PVGroup,
        PvpropertyDouble,
        PvpropertyInteger,
        PvpropertyString,
        PvpropertyChar,
        PvpropertyEnum,
        pvproperty,
)
from caproto.server.records import (
        AoFields,
        AiFields,
        LongoutFields,
        LonginFields,
        StringinFields,
        WaveformFields,
        MbbiFields,
)


# ──────────────────────────────────────────────────────────────────────────────
# pvproperty factories
#
# Every sensor PV is an ai record whose alarm and warning limits are the same
# pair of numbers, so the factories take a single low/high and apply them to
# both.  'name' and 'alarm_group' are always passed explicitly by the caller:
# PV names do not always follow the attribute name (firmware_version ->
# FWVERSION) and a PV left without an alarm_group shares caproto's single
# default alarm group, which is meaningful for the status/config PVs.
# ──────────────────────────────────────────────────────────────────────────────

def _ai_pv(value, precision, units, lolim, hilim, doc, **kwargs):
    """An analog input record with matching alarm and warning limits."""
    return pvproperty(value=value,
                      dtype=PvpropertyDouble[AiFields],
                      record=AiFields,
                      upper_alarm_limit=hilim,
                      lower_alarm_limit=lolim,
                      upper_warning_limit=hilim,
                      lower_warning_limit=lolim,
                      precision=precision,
                      units=units,
                      doc=doc,
                      **kwargs)


def temp_pv(doc, **kwargs):
    """A temperature reading in degrees C."""
    return _ai_pv(-99.0, 2, "C", 0.0, 1000.0, doc, **kwargs)


def humidity_pv(doc, **kwargs):
    """A relative humidity reading in percent."""
    return _ai_pv(0.0, 2, "%", -1.0, 101.0, doc, **kwargs)


def voltage_pv(doc, **kwargs):
    """A voltage reading in V."""
    return _ai_pv(0.0, 3, "V", -1.0, 100.0, doc, **kwargs)


def current_pv(doc, units="A", hilim=100.0, **kwargs):
    """A current reading. Also used for the optical power readings in uW."""
    return _ai_pv(0.0, 3, units, -1.0, hilim, doc, **kwargs)


def counter_pv(doc, value=0, **kwargs):
    """An integer counter or status flag."""
    return pvproperty(value=value,
                      dtype=PvpropertyInteger[LonginFields],
                      record=LonginFields,
                      doc=doc,
                      **kwargs)


def string_pv(doc, **kwargs):
    """A short string, e.g. firmware version or carrier id."""
    return pvproperty(value="",
                      dtype=PvpropertyString[StringinFields],
                      record=StringinFields,
                      doc=doc,
                      **kwargs)


def bldstr_pv(doc, **kwargs):
    """A long string, e.g. the firmware build stamp."""
    return pvproperty(value="",
                      dtype=PvpropertyChar[WaveformFields],
                      record=WaveformFields,
                      string_encoding='ascii',
                      max_length=256,
                      doc=doc,
                      **kwargs)


def rate_pv(doc, value, **kwargs):
    """A settable rate in Hz."""
    return pvproperty(value=value,
                      dtype=PvpropertyDouble[AoFields],
                      record=AoFields,
                      precision=1,
                      units="Hz",
                      doc=doc,
                      **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Packet decoding
# ──────────────────────────────────────────────────────────────────────────────

class MonitorPacket:
    """
    Base class for one ePix monitor stream packet.

    Subclasses set:
      STRUCT_FMT    struct format for the whole packet, e.g. "<I16i" or "<80H"
      HEADER_WORDS  number of leading words before channel 0
      CHANNEL_DEFS  {channel index: {name, unit, conv, pv_signal}}

    ``PACKET_BYTES`` is derived from ``STRUCT_FMT``.

    Each channel definition's 'conv' converts the raw word to physical units and
    'pv_signal' names the attribute on the IOC that publishes it.  Channels
    absent from CHANNEL_DEFS are unused/unconnected; negative readings on
    startup or from unconnected sensors are normal.
    """

    STRUCT_FMT: str = ""
    HEADER_WORDS: int = 0
    CHANNEL_DEFS: Dict[int, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.STRUCT_FMT:
            cls.PACKET_BYTES = struct.calcsize(cls.STRUCT_FMT)

    def __init__(self, data: bytes):
        if len(data) < self.PACKET_BYTES:
            raise ValueError(
                f"Packet too short: {len(data)} B  (expected {self.PACKET_BYTES} B)"
            )
        self.raw = struct.unpack_from(self.STRUCT_FMT, data)

    @property
    def counter(self) -> int:
        return self.raw[0]

    def channel_raw(self, ch: int) -> int:
        """Raw word for channel ch."""
        return self.raw[ch + self.HEADER_WORDS]

    def channel_value(self, ch: int) -> Optional[float]:
        """Converted physical value for a defined channel, or None if undefined."""
        if ch not in self.CHANNEL_DEFS:
            return None
        return self.CHANNEL_DEFS[ch]["conv"](self.channel_raw(ch))

    def as_dict(self) -> dict:
        """Return {sensor_name: physical_value} for all defined channels."""
        return {
            defn["name"]: defn["conv"](self.channel_raw(ch))
            for ch, defn in self.CHANNEL_DEFS.items()
        }

    def pv_data(self) -> dict:
        """Return {pv_name: physical_value} for all defined channels."""
        return {
            defn["pv_signal"]: defn["conv"](self.channel_raw(ch))
            for ch, defn in self.CHANNEL_DEFS.items()
        }

    def __str__(self) -> str:
        lines = [f"  counter : {self.counter}"]
        for ch, defn in self.CHANNEL_DEFS.items():
            raw = self.channel_raw(ch)
            val = defn["conv"](raw)
            lines.append(
                f"  {defn['name']:<28s}: {val:8.2f} {defn['unit']}  (raw={raw})"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# IOC base class
# ──────────────────────────────────────────────────────────────────────────────

class EpixMonitoringIOCBase(PVGroup):
    """
    The PVs and behavior shared by the ePix monitoring IOCs.

    A subclass must provide:
      board_cls          pyrogue.Root subclass with a ``configure`` staticmethod
      packet_cls         MonitorPacket subclass for decoding the stream
      monitor_setting    property giving the detector's monitor-rate register value

    and may override these hooks:
      uses_enable_packet  set True to also send the stream-enable packet
      check_packet()      reject bad monitor packets
      fixup_value()       adjust a decoded value before it is published
      update_extra()      publish extra derived PVs after a good packet
    """

    board_cls: Any = None
    packet_cls: Any = None
    uses_enable_packet: bool = False

    def __init__(self, *args, dev, lane, vc, regvc, **kwargs):
        self.dev = dev
        self.lane = lane
        self.vc = vc
        self.dma_dest = lane << 8 | vc
        self.regvc = regvc
        self.mon_reader = None
        self.mon_writer = None
        self.lastmontime = None
        self.trigrateconv = 1000
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(f"caproto.{__name__}")

    @property
    def monitor_setting(self):
        """
        The value written to the detector's monitor rate register.  Detectors
        express this differently (a period, or a prescale of the trigger rate).
        """
        raise NotImplementedError

    @property
    def auto_trigger_period(self):
        """
        Auto trigger rate converted to period
        """
        return int(self.trigrateconv/self.set_auto_trig_rate.value)

    def check_packet(self, data):
        """
        Check that the decoded packet is valid.  Accept everything by default.
        """
        return True

    def fixup_value(self, name, value):
        """
        Adjust a decoded value before publishing it.  Pass it through by default.
        """
        return value

    async def update_extra(self, data):
        """
        Publish any extra derived PVs after a good packet.  Nothing by default.
        """

    async def publish(self, data):
        """
        Write every decoded monitor value to the PV of the same name, applying
        the detector's value fixups.  Only used for monitor stream data: the
        register data from ``configure`` needs no fixups.
        """
        for name, value in data.items():
            if hasattr(self, name):
                await getattr(self, name).write(value=self.fixup_value(name, value))

    def configure(self, flag, mon_setting, trig_period):
        """
        Configure the detector registers in a subprocess.

        The register access has to happen in a separate process: rogue's SRP
        interface cannot be opened twice in one process while the monitor stream
        DMA is attached.
        """
        self.log.debug(f"Starting register process: dev - {self.dev}, lane,vc - {self.lane},{self.regvc}")
        queue = mp.Queue()
        proc = mp.Process(target=self.board_cls.configure,
                          args=(self.dev, self.lane, self.regvc, flag, mon_setting, trig_period, queue))
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
        # imported here so that this module stays importable without rogue
        import pyrogue
        import rogue.hardware.axi
        from psdaq.IOC.epix_monitor_stream import MonitorStreamReader, MonitorStreamWriter

        self.async_lib = async_lib
        queue = async_lib.ThreadsafeQueue()
        self.log.info(f"Initializing monitor stream dma: dev - {self.dev}, dest,lane,vc - {self.dma_dest},{self.lane},{self.vc}")
        dma = rogue.hardware.axi.AxiStreamDma(self.dev, self.dma_dest, True)
        self.mon_reader = MonitorStreamReader(vc=self.vc, queue=queue, packet_cls=self.packet_cls)
        pyrogue.streamConnect(dma, self.mon_reader)
        if self.uses_enable_packet:
            self.mon_writer = MonitorStreamWriter(vc=self.vc)
            pyrogue.streamConnect(self.mon_writer, dma)

        try:
            count = 0
            errcount = 0
            self.lastmontime = time.time()
            while True:
                data = await queue.async_get()
                if self.check_packet(data):
                    self.lastmontime = time.time()
                    await self.publish(data)
                    await self.update_extra(data)
                    count += 1
                    await self.moncnt.write(value=count)
                else:
                    self.log.error(f"Bad monitoring packet returned by the detector: {data}")
                    errcount += 1
                    if hasattr(self, "monerrcnt"):
                        await self.monerrcnt.write(value=errcount)
        except Exception:
            self.log.exception("Server monitoring queue reader encountered an error:")
        finally:
            self.log.info("Server monitoring queue reader exitted.")

    set_monitor = pvproperty(name="SET_MONITOR",
                             value=0,
                             dtype=PvpropertyInteger[LongoutFields],
                             record=LongoutFields,
                             doc="Start/Stop epixMon")
    moncnt = counter_pv(name="MONCNT",
                        doc="epix monitor counts")
    monchk = counter_pv(name="MONCHK",
                        upper_alarm_limit=0.5,
                        lower_alarm_limit=-0.5,
                        upper_warning_limit=0.5,
                        lower_warning_limit=-0.5,
                        alarm_group="monchk",
                        doc="epixMon check")
    monchkdelay = counter_pv(name="MONCHKDELAY",
                             value=5,
                             doc="epix check delay")
    new_firmware = pvproperty(name="NEW_FIRMWARE",
                              value=2,
                              dtype=PvpropertyEnum[MbbiFields],
                              record=MbbiFields,
                              enum_strings=["epix100a", "epix10ka", "lcls2"],
                              doc="epix firmware type")
    set_monitor_rate = rate_pv(name="SET_MONITOR_RATE",
                               value=1.0,
                               doc="Set the monitor update rate for epixMon")
    set_auto_trig_rate = rate_pv(name="SET_AUTO_TRIG_RATE",
                                 value=10.0,
                                 doc="Set the auto trigger rate for epixMon")
    humidity = humidity_pv(name="HUMIDITY",
                           alarm_group="humidity",
                           doc="Humidity")
    ana_in_v = voltage_pv(name="ANA_IN_V",
                          alarm_group="ana_in_v",
                          doc="Analog Voltage")
    dig_in_v = voltage_pv(name="DIG_IN_V",
                          alarm_group="dig_in_v",
                          doc="Digital Voltage")
    asic_ana_cur = current_pv(name="ASIC_ANA_CUR",
                              alarm_group="asic_ana_cur",
                              doc="ASIC Analog Current")
    asic_dig_cur = current_pv(name="ASIC_DIG_CUR",
                              alarm_group="asic_dig_cur",
                              doc="ASIC Digital Current")
    firmware_version = string_pv(name="FWVERSION",
                                 doc="epix fw version")
    firmware_githash = string_pv(name="FWGITHASH",
                                 doc="epix fw githash")
    firmware_bldstr = bldstr_pv(name="FWBLDSTR",
                                doc="epix fw build str")

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
        data = await self.async_lib.library.to_thread(self.configure, bool(flag), self.monitor_setting, self.auto_trigger_period)
        self.log.info("Epix register configuration completed")
        for name, value in data.items():
            if name == 'logs':
                for log_name, msgs in value.items():
                    log_level = logging.getLevelName(log_name.upper())
                    for msg in msgs:
                        self.log.log(log_level, msg)
            elif hasattr(self, name):
                await getattr(self, name).write(value=value)


# ──────────────────────────────────────────────────────────────────────────────
# Command-line helpers
# ──────────────────────────────────────────────────────────────────────────────

def add_common_args(parser, regvc=0):
    """
    Add the hardware arguments shared by the monitoring IOCs.  ``regvc`` is the
    default register virtual channel, which is detector specific.
    """
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
        default=regvc,
        type=int,
        help=f"Register virtual channel (default: {regvc})"
    )
    return parser


def setup_logging(args):
    """
    Initialize the caproto logger from the parsed --verbose argument.
    """
    from caproto import config_caproto_logging

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
