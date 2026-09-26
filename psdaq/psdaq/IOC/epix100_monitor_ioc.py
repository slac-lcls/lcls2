"""
epix100_monitor_ioc.py

Read the ePix100 slow-ADC environmental monitor stream via rogue and publish it
as EPICS PVs.

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
      Start with VC=3 (first bypassed channel).

Requirements (same environment as the DAQ epix100 configuration):
    rogue  pyrogue  ePixFpga  lcls2_epix_hr_pcie  caproto

Usage examples:
    # publish decoded values as EPICS PVs under the default prefix:
    epix100_monitor_ioc

    # non-default prefix and hardware:
    epix100_monitor_ioc --prefix DET:EPIX:CMP004: --dev /dev/datadev_1 --lane 0 --vc 4

The monitor stream and auto trigger are enabled by writing 1 to the SET_MONITOR
PV (this needs exclusive SRP access, so no DAQ can be configuring the detector
at the same time).
"""

from typing import Any, Dict

# rougue imports
from psdaq.utils import enable_epix_100a_gen2
import epix100a_gen2
import ePixFpga as fpga
import rogue
import rogue.hardware.axi
import rogue.protocols.srp
import pyrogue
# caproto imports
from caproto.server import template_arg_parser, run
# shared monitoring IOC code
from psdaq.IOC.epix_monitor_base import (
        EpixMonitoringIOCBase,
        MonitorPacket,
        add_common_args,
        current_pv,
        setup_logging,
        temp_pv,
)


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


class EpixMonitorPacket(MonitorPacket):
    """
    One ePix100 slow-ADC monitor stream packet.

    Format (68 bytes = 17 × int32 little-endian):
      word[ 0]        : packet counter (decoded as unsigned)
      word[ 1]        : channel  0  (raw signed int32)
      ...
      word[16]        : channel 15

    See CHANNEL_DEFS above for the channel mapping.  Values are signed int32;
    channels 0-6 and 15 are unused/unconnected.
    """

    STRUCT_FMT = "<I16i"
    HEADER_WORDS = 1
    N_CHANNELS = 16
    CHANNEL_DEFS = CHANNEL_DEFS


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
        data = {'logs': {'info': [], 'warn': [], 'error': []}}
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
        except Exception as exc:
            data['logs']['error'].append(f"exception encountering during configuration: {exc}")
        finally:
            # send the firmware info back
            queue.put(data)


class EpixMonitoringIOC(EpixMonitoringIOCBase):
    """
    EPICS IOC publishing the ePix100 environmental monitor readings.
    """

    board_cls = Epix100aBoard
    packet_cls = EpixMonitorPacket
    monrateconv = 100000000

    @property
    def monitor_setting(self):
        """
        Monitor rate converted to period
        """
        return int(self.monrateconv / self.set_monitor_rate.value)

    # PVs specific to the ePix100, or needing a detector specific description
    temp1 = temp_pv(name="TEMP1",
                    alarm_group="temp1",
                    doc="Strong Back Temp.")
    temp2 = temp_pv(name="TEMP2",
                    alarm_group="temp2",
                    doc="Ambient Temp.")
    asic_gr_cur = current_pv(name="ASIC_GR_CUR",
                             alarm_group="asic_gr_cur",
                             doc="Guard Ring Current")


def main():
    # Parse standard EPICS IOC command-line options
    parser, split_args = template_arg_parser(
        default_prefix="DET:EPIX:CMP004:",
        desc="Read ePix100 environmental monitor packets via rogue and publish via caproto IOC"
    )
    add_common_args(parser, regvc=0)

    args = parser.parse_args()
    ioc_options, run_options = split_args(args)

    setup_logging(args)

    # Start the server
    ioc = EpixMonitoringIOC(dev=args.dev, lane=args.lane, vc=args.vc, regvc=args.regvc, **ioc_options)
    run(ioc.pvdb, **run_options, startup_hook=ioc.__ainit__)


if __name__ == '__main__':
    main()
