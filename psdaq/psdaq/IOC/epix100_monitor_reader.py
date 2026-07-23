#!/usr/bin/env python3
"""
epix100_monitor_reader.py

Read the ePix100 slow-ADC environmental monitor stream via rogue.

The ePix100 FPGA continuously sends small "monitor packets" on a dedicated
PGP virtual channel (separate from the main image data channel).  Each packet
carries a hardware timestamp plus 8 slow-ADC readings that encode temperatures,
humidity, currents, and voltages.

In the normal DAQ pipeline the DRP reads only the image data and the monitor
channel is bypassed (EventBuilder.Bypass bit 3 = 0x38).  This script connects
directly to those bypassed DMA channels and decodes the packets.

Packet format (mirrors EpixMonStreamAsic in psdaq/drp/pgpread_epixM320mon.cc):
  9 × uint64_t little-endian words
    word[0]  : bits [27:0]*16 = hardware tick count  (156.25 MHz clock)
    word[1]  : carrier thermistor          → temperature (°C)
    word[2]  : digital-section thermistor  → temperature (°C)
    word[3]  : humidity sensor             → %RH
    word[4]  : misc LDO/supply current     → Amps
    word[5]  : ASIC analog supply current  → Amps
    word[6]  : misc supply voltage 1       → Volts
    word[7]  : misc supply voltage 2       → Volts
    word[8]  : ASIC 2.5 V analog voltage   → Volts

VC mapping (from epix100_config.py / firmware README):
  lane 0, VC 0  →  SRP register bus   (used for firmware register access)
  lane 0, VC 1  →  image / event data (used by DRP, virtChan=1)
  lane 0, VC 2  →  epix100 image batcher stream (EventBuilder bit 2 = 0x4)
  lane 0, VC 3+ →  monitor / slow-ADC stream    (EventBuilder bits 3-5 bypassed, 0x38)

NOTE: The exact VC for the monitor packets depends on the firmware version.
      Start with VC=3 (first bypassed channel).  Use --vcs 3,4,5 to scan all
      three bypassed channels simultaneously.

Requirements (same environment as the DAQ epix100 configuration):
    rogue  pyrogue  epix100a_gen2  ePixFpga  lcls2_epix_hr_pcie

Usage examples:
    # listen passively on VC 3 (monitor stream must already be enabled by DAQ):
    python epix100_monitor_reader.py

    # enable the stream yourself (requires exclusive SRP access, no DAQ running):
    python epix100_monitor_reader.py --enable

    # scan all three bypassed VCs at once:
    python epix100_monitor_reader.py --vcs 3,4,5

    # non-default hardware:
    python epix100_monitor_reader.py --dev /dev/datadev_1 --lane 0 --vc 4
"""

import argparse
import math
import signal
import struct
import sys
import time
from typing import Dict, List, Tuple

import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.protocols.srp
import pyrogue

# ──────────────────────────────────────────────────────────────────────────────
# Packet decoding
# ──────────────────────────────────────────────────────────────────────────────

# Channel definitions confirmed from envConf in the epix100 viewer software.
# 'id' = channel index (0-based); maps to packet word[id+1] since word[0] is counter.
# 'conv' converts raw signed int32 to physical units.
CHANNEL_DEFS = {
    7: dict(name="Strong Back Temp.", unit="°C", conv=lambda d: d / 100),
    8: dict(name="Ambient Temp.", unit="°C", conv=lambda d: d / 100),
    9: dict(name="Humidity", unit="%", conv=lambda d: d / 100),
    10: dict(name="ASIC Analog Current", unit="A", conv=lambda d: d / 1000),
    11: dict(name="ASIC Digital Current", unit="A", conv=lambda d: d / 1000),
    12: dict(name="Guard Ring Current", unit="A", conv=lambda d: d / 1000),
    13: dict(name="Analog Voltage", unit="V", conv=lambda d: d / 1000),
    14: dict(name="Digital Voltage", unit="V", conv=lambda d: d / 1000),
}


class EpixMonitorPacket:
    """
    One ePix100 slow-ADC monitor stream packet.

    Format (68 bytes = 17 × int32 little-endian):
      word[ 0]        : packet counter
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
        # Signed int32 — needed for correct conversion of channels that can
        # go negative (e.g. humidity at startup, unconnected sensors).
        self.raw = struct.unpack_from(f"<{self.N_WORDS}i", data)

    @property
    def counter(self) -> int:
        return self.raw[0]

    def channel_raw(self, ch: int) -> int:
        """Raw signed int32 for channel ch (0–15)."""
        return self.raw[ch + 1]

    def channel_value(self, ch: int) -> float:
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

    def __str__(self) -> str:
        lines = [f"  counter : {self.counter}"]
        for ch, defn in CHANNEL_DEFS.items():
            raw = self.raw[ch + 1]
            val = defn["conv"](raw)
            lines.append(
                f"  {defn['name']:<28s}: {val:8.2f} {defn['unit']}  (raw={raw})"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# rogue stream receiver
# ──────────────────────────────────────────────────────────────────────────────


class MonitorStreamSlave(rogue.interfaces.stream.Slave):
    """
    rogue stream Slave that receives raw frames from an AxiStreamDma handle
    and decodes them as EpixMonitorPacket objects.

    rogue strips the AXI stream framing (header/footer) before calling
    _acceptFrame, so `frame` contains only the packet payload.
    """

    def __init__(self, vc: int):
        super().__init__()
        self.vc = vc
        self.n_received = 0
        self.n_errors = 0
        self.last_packet = None

    def _acceptFrame(self, frame: rogue.interfaces.stream.Frame) -> None:
        with frame.lock():
            size = frame.getPayload()
            buf = bytearray(size)
            frame.read(buf, 0)

        self.n_received += 1
        try:
            pkt = EpixMonitorPacket(bytes(buf))
            self.last_packet = pkt
            print(
                f"\n[VC={self.vc}] packet #{self.n_received}  ({size} B  "
                f"raw[0]=0x{pkt.raw[0]:016x})"
            )
            print(pkt)
        except Exception as exc:
            self.n_errors += 1
            print(
                f"[VC={self.vc}] decode error: {exc}  ({size} B raw)", file=sys.stderr
            )
            if size <= 128:
                print(f"  raw bytes: {buf.hex()}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Hardware setup
# ──────────────────────────────────────────────────────────────────────────────


def open_monitor_vcs(dev: str, lane: int, vcs: List[int]):
    """
    Open rogue AxiStreamDma handles for the given virtual channels and
    attach a MonitorStreamSlave to each.

    Returns (dma_handles, slaves) dicts keyed by vc number.
    """
    dma_handles: dict = {}
    slaves: dict = {}

    for vc in vcs:
        dma_dest = lane * 0x100 + vc
        print(f"  DMA dest 0x{dma_dest:03x}  (lane={lane}, VC={vc})")
        dma = rogue.hardware.axi.AxiStreamDma(dev, dma_dest, True)
        slv = MonitorStreamSlave(vc=vc)
        pyrogue.streamConnect(dma, slv)
        dma_handles[vc] = dma
        slaves[vc] = slv

    return dma_handles, slaves


def build_register_root(dev: str, lane: int):
    """
    Build a minimal pyrogue.Root that exposes the ePix100a FPGA register tree
    over SRP on VC0.

    This is only needed for --enable (to write SlowAdcRegisters).
    Follows the pattern of EpixBoard / epix100_init() in
        psdaq/psdaq/configdb/epix100_config.py

    The epix100a_gen2 and ePixFpga packages are part of the DAQ software
    stack and must be on sys.path (psdaq.utils.enable_epix_100a_gen2 does this).
    """
    from psdaq.utils import enable_epix_100a_gen2  # adds ePixFpga to sys.path
    import epix100a_gen2  # noqa: F401 (side-effect import)
    import ePixFpga as fpga

    vc0_dma = rogue.hardware.axi.AxiStreamDma(dev, lane * 0x100 + 0, True)
    srp = rogue.protocols.srp.SrpV3()
    pyrogue.streamConnectBiDir(vc0_dma, srp)

    class _EpixBoard(pyrogue.Root):
        def __init__(self, srp_bus, **kw):
            super().__init__(name="ePixBoard", description="ePix100a Board", **kw)
            self.add(
                fpga.Epix100a(
                    name="ePix100aFPGA",
                    offset=0,
                    memBase=srp_bus,
                    hidden=False,
                    enabled=True,
                )
            )

    # pollEn=False  : do not continuously poll registers in the background
    # initRead=False: do not read ALL registers on startup — that scan triggers
    #                 a Transaction Timeout if another process (e.g. the DAQ)
    #                 already owns VC0, or if the firmware isn't fully ready.
    #                 Matches the pattern used in epix100_config.py for pbase and
    #                 epixhr2x2_config.py for cbase.
    root = _EpixBoard(srp, pollEn=False, initRead=False)
    root.__enter__()
    return root, vc0_dma  # keep vc0_dma alive (rogue holds a weak ref)


def enable_monitor_stream(root, period_ticks: int = 100_000_000) -> None:
    """
    Enable the slow-ADC monitor stream via firmware register writes.

    period_ticks=100_000_000 ≈ 1 Hz (100 MHz firmware clock).

    Note: In epix100_config.py the matching block ends with enable.set(0)
    which disables *pyrogue polling* of this module but leaves the firmware
    StreamEn register set.  We leave polling enabled here so register reads
    also work for diagnostics.
    """
    regs = root.ePix100aFPGA.SlowAdcRegisters
    regs.enable.set(1)
    regs.StreamPeriod.set(period_ticks)
    regs.StreamEn.set(1)
    fw_ver = root.ePix100aFPGA.AxiVersion.FpgaVersion.get()
    print(f"  FPGA version   : 0x{fw_ver:08x}")
    print(
        f"  StreamPeriod   : {period_ticks} ticks  "
        f"≈ {period_ticks / 1e8:.2f} s per packet"
    )
    print("  StreamEn       : 1  (monitor stream active)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description="Read ePix100 environmental monitor packets via rogue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dev",
        default="/dev/datadev_0",
        help="PCIe DMA device (default: /dev/datadev_0)",
    )
    ap.add_argument("--lane", default=0, type=int, help="PGP lane number (default: 0)")
    ap.add_argument(
        "--vc",
        default=3,
        type=int,
        help="Monitor stream virtual channel to listen on "
        "(default: 3 — first bypassed channel in EventBuilder)",
    )
    ap.add_argument(
        "--vcs",
        default=None,
        help="Comma-separated list of VCs to listen on simultaneously "
        "(overrides --vc).  Example: --vcs 3,4,5",
    )
    ap.add_argument(
        "--enable",
        action="store_true",
        help="Enable the monitor stream by writing SlowAdcRegisters "
        "via SRP on VC0.  Use only when the DAQ is NOT running "
        "(requires exclusive register access).",
    )
    ap.add_argument(
        "--period",
        default=100_000_000,
        type=int,
        help="Monitor stream period in firmware ticks when --enable "
        "is used (default: 1e8 ≈ 1 Hz at 100 MHz clock).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    monitor_vcs = (
        [int(v.strip()) for v in args.vcs.split(",")] if args.vcs else [args.vc]
    )

    # ── optional: enable the stream via register writes ──────────────────
    reg_root = None
    vc0_dma = None
    if args.enable:
        print("Connecting to ePix100a FPGA registers (SRP on VC0) …")
        reg_root, vc0_dma = build_register_root(dev=args.dev, lane=args.lane)
        enable_monitor_stream(reg_root, period_ticks=args.period)
        print()

    # ── open monitor stream DMA channels ─────────────────────────────────
    print(
        f"Opening monitor virtual channel(s) {monitor_vcs} "
        f"on {args.dev} lane {args.lane} …"
    )
    dma_handles, slaves = open_monitor_vcs(
        dev=args.dev, lane=args.lane, vcs=monitor_vcs
    )
    print("Listening for monitor packets … (Ctrl-C to stop)\n")

    # ── run until Ctrl-C ─────────────────────────────────────────────────
    stop = [False]

    def _sig(sig, _frame):
        stop[0] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not stop[0]:
            time.sleep(0.1)
    finally:
        total = sum(s.n_received for s in slaves.values())
        errors = sum(s.n_errors for s in slaves.values())
        print(f"\nDone.  Received {total} packets total, {errors} decode errors.")
        if reg_root is not None:
            reg_root.__exit__(None, None, None)


if __name__ == "__main__":
    main()
