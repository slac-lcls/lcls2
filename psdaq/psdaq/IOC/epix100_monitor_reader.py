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
    p4p

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

import argparse
import contextlib
import gc
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import rogue
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.protocols.srp
import pyrogue


# ──────────────────────────────────────────────────────────────────────────────
# Embedded init script (written to a temp file by enable_monitor_stream)
# ──────────────────────────────────────────────────────────────────────────────

# This is the content of what was previously epix100_init.py.
# It is written to a NamedTemporaryFile and executed in a subprocess so that
# the VC0 file descriptor is guaranteed to be released before the parent
# process opens the monitor VCs.
#
# dma and srp are created inside _Board.__init__ and registered with
# addInterface so that root.stop() (called by __exit__) invokes _stop() on
# both objects.  _stop() calls closeAllSlave()/closeAllMaster() in C++, which
# breaks the streamConnectBiDir shared_ptr cycle and allows the C++ destructors
# to run and close the VC0 fd while the subprocess is still alive.
# OS process exit provides a second guarantee for any remaining fds.
_INIT_SCRIPT = """\
import sys
from psdaq.utils import enable_epix_100a_gen2
import epix100a_gen2, ePixFpga as fpga
import rogue, rogue.hardware.axi, rogue.protocols.srp, pyrogue

dev = sys.argv[1]
lane = int(sys.argv[2])
period_ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000_000

class _Board(pyrogue.Root):
    def __init__(self):
        super().__init__(name='ePixBoard', pollEn=False, initRead=False)
        dma = rogue.hardware.axi.AxiStreamDma(dev, lane * 0x100 + 0, True)
        srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(dma, srp)
        self.addInterface(dma)
        self.addInterface(srp)
        self.add(fpga.Epix100a(
            name='ePix100aFPGA', offset=0,
            memBase=srp, hidden=False, enabled=True))

with _Board() as root:
    fw = root.ePix100aFPGA.AxiVersion.FpgaVersion.get()
    print('  FPGA version   : 0x%08x' % fw)
    root.ePix100aFPGA.SlowAdcRegisters.enable.set(1)
    root.ePix100aFPGA.SlowAdcRegisters.StreamPeriod.set(period_ticks)
    root.ePix100aFPGA.SlowAdcRegisters.StreamEn.set(1)
    root.ePix100aFPGA.SlowAdcRegisters.enable.set(0)
    print(f'  StreamPeriod : {period_ticks} ticks')
    print('  StreamEn     : 1  (monitor stream active)')
# root.__exit__ -> root.stop() -> addInterface._stop() breaks shared_ptr cycle
# -> C++ destructors run -> VC0 fd closed before process exits
"""


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
        pv_signal="SBTEMP",
    ),
    8: dict(
        name="Ambient Temp.",
        unit="°C",
        conv=lambda d: d / 100,
        pv_signal="AMBTEMP",
    ),
    9: dict(
        name="Humidity",
        unit="%",
        conv=lambda d: d / 100,
        pv_signal="HUMD",
    ),
    10: dict(
        name="ASIC Analog Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="ASIC_ACURR",
    ),
    11: dict(
        name="ASIC Digital Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="ASIC_DCURR",
    ),
    12: dict(
        name="Guard Ring Current",
        unit="A",
        conv=lambda d: d / 1000,
        pv_signal="GRCURR",
    ),
    13: dict(
        name="Analog Voltage",
        unit="V",
        conv=lambda d: d / 1000,
        pv_signal="AVLTG",
    ),
    14: dict(
        name="Digital Voltage",
        unit="V",
        conv=lambda d: d / 1000,
        pv_signal="DVLTG",
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
# EPICS PV setup
# ──────────────────────────────────────────────────────────────────────────────


class _SetMonitorHandler:
    """p4p PV handler that accepts client PUT requests to the SET_MONITOR PV.

    Writing 1 re-enables PV posting; writing 0 suspends it.
    The new value is immediately reflected by the PV so that subsequent
    monitors/gets see the updated state.

    ``posting_enabled`` is a plain Python bool updated inside the GIL.
    MonitorStreamSlave reads it directly from the rogue callback thread,
    avoiding the p4p current()-from-foreign-thread issue that would otherwise
    cause the check to silently fall through to the default (True).
    """

    def __init__(self):
        self.posting_enabled = True  # True = PV updates active

    def put(self, pv, op):
        val = op.value()
        # op.value() may be a raw Python int (tests) or a p4p Value (production).
        # Extract the plain scalar so bool() is always reliable.
        try:
            scalar = int(val["value"])
        except Exception:
            scalar = int(val)
        self.posting_enabled = bool(scalar)
        pv.post(val)
        op.done()


def build_epics_pvs(ext: str, unit: int):
    """
    Create one p4p SharedPV per defined channel plus a writable SET_MONITOR PV,
    and register them all with a StaticProvider.

    PV names follow the LCLS convention:
        HUTCH:EPIX100:NN:SIGNAL
    e.g. TMO:EPIX100:01:SBTEMP

    Returns (provider, pvs, set_monitor_pv) where:
      pvs            – {channel_number: SharedPV}  (read-only sensor PVs)
      set_monitor_pv – writable int SharedPV; value 1 = posting enabled, 0 = suspended.

    All three must be kept alive for the duration of the server.

    Requires p4p (pip install p4p).
    """
    from p4p.nt import NTScalar
    from p4p.server import StaticProvider
    from p4p.server.thread import SharedPV

    prefix = f"{ext.upper()}:{unit:02d}:"
    # float64 with display fields (units, description, limits, alarm, timestamp)
    nt_float = NTScalar("d", display=True)
    nt_int = NTScalar("i", display=True)
    provider = StaticProvider("epix-mon")
    pvs = {}

    for ch, defn in CHANNEL_DEFS.items():
        pv_name = prefix + defn["pv_signal"]
        # initial=0.0 opens the PV immediately so clients can connect before
        # the first real packet arrives.
        pv = SharedPV(nt=nt_float, initial=0.0)
        provider.add(pv_name, pv)
        pvs[ch] = pv
        print(f"  {pv_name}  ({defn['unit']})")

    # SET_MONITOR: writable bool-as-int PV.  Initial value 1 = posting enabled.
    # A client can write 0 to suspend PV updates and 1 to resume.
    set_monitor_pv_name = prefix + "SET_MONITOR"
    _handler = _SetMonitorHandler()
    set_monitor_pv = SharedPV(nt=nt_int, initial=1, handler=_handler)
    # Attach handler as a Python attribute so MonitorStreamSlave can read
    # posting_enabled directly without calling current() from a foreign thread.
    set_monitor_pv._monitor_handler = _handler
    provider.add(set_monitor_pv_name, set_monitor_pv)
    print(f"  {set_monitor_pv_name}  (int, writable: 1=enabled 0=suspended)")

    return provider, pvs, set_monitor_pv


# ──────────────────────────────────────────────────────────────────────────────
# rogue stream receiver
# ──────────────────────────────────────────────────────────────────────────────


class MonitorStreamSlave(rogue.interfaces.stream.Slave):
    """
    rogue stream Slave that receives raw frames from an AxiStreamDma handle
    and decodes them as EpixMonitorPacket objects.

    rogue strips the AXI stream framing (header/footer) before calling
    _acceptFrame, so `frame` contains only the packet payload.

    If a pvs dict ({channel_number: SharedPV}) is supplied, decoded values
    are posted to the corresponding EPICS PVs after each packet.
    """

    def __init__(self, vc: int, pvs: Optional[dict] = None, set_monitor_pv=None):
        super().__init__()
        self.vc = vc
        self.pvs = pvs or {}
        self.set_monitor_pv = set_monitor_pv
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

            if self.pvs:
                # Check SET_MONITOR; default to enabled if PV unavailable.
                posting_enabled = True
                if self.set_monitor_pv is not None:
                    # Prefer reading the handler's Python bool directly.
                    # _acceptFrame runs on rogue's C++ callback thread; calling
                    # set_monitor_pv.current() from a foreign thread can silently
                    # fail in p4p, leaving posting_enabled stuck at True.
                    # The isinstance guard ensures the test MagicMock falls through
                    # to the current().value path, keeping tests unmodified.
                    _h = getattr(self.set_monitor_pv, "_monitor_handler", None)
                    if isinstance(_h, _SetMonitorHandler):
                        posting_enabled = _h.posting_enabled
                    else:
                        try:
                            posting_enabled = bool(self.set_monitor_pv.current().value)
                        except Exception:
                            pass  # keep posting_enabled=True on any error

                if posting_enabled:
                    ts = time.time()
                    for ch, pv in self.pvs.items():
                        pv.post(pkt.channel_value(ch), timestamp=ts)

            print(
                f"\n[VC={self.vc}] packet #{self.n_received}  ({size} B  "
                f"counter=0x{pkt.counter:08x})"
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


@contextlib.contextmanager
def _silence_stderr():
    """Redirect fd 2 to /dev/null for the duration of the block.

    Python-level sys.stderr redirection does not suppress output from C++
    libraries (such as rogue's built-in error logging).  Redirecting at the
    OS file-descriptor level silences both.  The original fd is saved and
    restored in the finally clause so that subsequent output is unaffected.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def open_monitor_vcs(
    dev: str,
    lane: int,
    vcs: List[int],
    pvs: Optional[dict] = None,
    set_monitor_pv=None,
):
    """
    Open rogue AxiStreamDma handles for the given virtual channels and
    attach a MonitorStreamSlave to each.

    Any failure to open a VC (device taken, permissions, etc.) is treated as
    non-fatal: a warning is printed and that VC is skipped.  This avoids
    fragile exception-type and message-string matching across rogue versions.

    Returns (dma_handles, slaves) dicts keyed by vc number.
    """
    dma_handles: dict = {}
    slaves: dict = {}

    for vc in vcs:
        dma_dest = lane * 0x100 + vc
        try:
            with _silence_stderr():
                dma = rogue.hardware.axi.AxiStreamDma(dev, dma_dest, True)
        except Exception as exc:
            print(
                f"  DMA dest 0x{dma_dest:03x} (lane={lane}, VC={vc}) could not be "
                f"opened — skipping.  ({exc})"
            )
            continue
        slv = MonitorStreamSlave(vc=vc, pvs=pvs, set_monitor_pv=set_monitor_pv)
        pyrogue.streamConnect(dma, slv)
        dma_handles[vc] = dma
        slaves[vc] = slv
        print(f"  DMA dest 0x{dma_dest:03x} opened  (lane={lane}, VC={vc})")

    return dma_handles, slaves


def enable_monitor_stream(dev: str, lane: int, period_ticks: int = 100_000_000) -> None:
    """
    Enable the slow-ADC monitor stream by running _INIT_SCRIPT in a subprocess.

    The script is written to a NamedTemporaryFile and executed with the current
    Python interpreter.  When the subprocess exits the OS closes every fd
    unconditionally, guaranteeing that VC0 is free when this function returns
    regardless of any Python or C++ reference-count cycles inside the child.

    period_ticks=100_000_000  ≈ 1 Hz  (100 MHz slow-ADC firmware clock).
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(_INIT_SCRIPT)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path, dev, str(lane), str(period_ticks)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f"enable_monitor_stream subprocess failed (rc={result.returncode}):\n"
                f"{result.stderr}"
            )
        print("  VC0 released (subprocess exited).")
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
    ap.add_argument(
        "--lane",
        default=0,
        type=int,
        help="PGP lane number (default: 0)",
    )
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
        help="Enable the monitor stream by writing SlowAdcRegisters via SRP on VC0. "
        "Use only when the DAQ is NOT running (requires exclusive register access).",
    )
    ap.add_argument(
        "--period",
        default=100_000_000,
        type=int,
        help="Monitor stream period in firmware ticks when --enable is used "
        "(default: 1e8 ≈ 1 Hz at 100 MHz clock).",
    )
    ap.add_argument(
        "--ext",
        required=True,
        metavar="NAME",
        help="Hutch name for EPICS PV publishing (e.g. TMO, CXI, MFX). "
        "When given, decoded values are posted as PVAccess PVs: "
        "HUTCH:EPIX100:NN:SIGNAL.  Requires p4p.",
    )
    ap.add_argument(
        "--unit",
        required=True,
        type=int,
        metavar="N",
        help="Detector unit number used in the EPICS PV prefix (default: 1)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    monitor_vcs = (
        [int(v.strip()) for v in args.vcs.split(",")] if args.vcs else [args.vc]
    )

    # ── signal handling ───────────────────────────────────────────────────
    stop = [False]

    def _sig(sig, _frame):
        stop[0] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ── optional: enable the stream via register writes ───────────────────
    # Uses a subprocess so the VC0 fd is released on process exit regardless
    # of any C++ shared_ptr cycles inside the child.
    enable_failed = False
    if args.enable:
        print(
            "Enabling monitor stream via subprocess (VC0 released on subprocess exit) …"
        )
        try:
            enable_monitor_stream(
                dev=args.dev, lane=args.lane, period_ticks=args.period
            )
        except Exception as exc:
            enable_failed = True
            print(f"  WARNING: enable_monitor_stream failed: {exc}", file=sys.stderr)
            print("  Continuing to listen anyway …", file=sys.stderr)
        print()

    # ── optional: build EPICS PVs ─────────────────────────────────────────
    epics_server = None
    pvs = {}
    set_monitor_pv = None
    if args.ext:
        print(
            f"Creating EPICS PVs  (ext={args.ext.upper()}, unit={args.unit:02d}) …"
        )
        from p4p.server import Server

        provider, pvs, set_monitor_pv = build_epics_pvs(args.ext, args.unit)
        epics_server = Server(providers=[provider])
        if enable_failed and set_monitor_pv is not None:
            set_monitor_pv.post(0)
            _h = getattr(set_monitor_pv, "_monitor_handler", None)
            if isinstance(_h, _SetMonitorHandler):
                _h.posting_enabled = False
            print("  SET_MONITOR set to 0 (enable failed).", file=sys.stderr)
        print()

    # ── open DMA channels and listen ──────────────────────────────────────
    print(
        f"Opening monitor virtual channel(s) {monitor_vcs} "
        f"on {args.dev} lane {args.lane} …"
    )
    dma_handles, slaves = open_monitor_vcs(
        dev=args.dev,
        lane=args.lane,
        vcs=monitor_vcs,
        pvs=pvs,
        set_monitor_pv=set_monitor_pv,
    )

    if not slaves:
        if epics_server is None:
            # Nothing to do: no DMA and no PV server.
            print("No DMA channels could be opened — exiting.", file=sys.stderr)
            sys.exit(1)
        # Keep the PV server alive even though DMA failed.  PVs hold their
        # initial 0.0 values and remain reachable via pvget / pvmonitor.
        # This is the common case when the DAQ holds the channels exclusively.
        print(
            "  WARNING: No DMA channels could be opened.  "
            "EPICS PV server is running but values will not update "
            "until the monitor stream becomes available.",
            file=sys.stderr,
        )

    if slaves:
        print(
            f"Opened {len(dma_handles)} of {len(monitor_vcs)} requested VC(s).  "
            f"Listening for monitor packets … (Ctrl-C to stop)\n"
        )
    else:
        print("EPICS PV server running.  (Ctrl-C to stop)\n")

    try:
        while not stop[0]:
            time.sleep(0.1)
    finally:
        total = sum(s.n_received for s in slaves.values())
        errors = sum(s.n_errors for s in slaves.values())
        print(f"\nDone.  Received {total} packets total, {errors} decode errors.")

        # Release DMA destinations so the DAQ (or any other process) can
        # reclaim them immediately after this script exits.
        #
        # Clearing the dict drops each AxiStreamDma Python wrapper's refcount
        # to zero.  In CPython this immediately triggers the C++ destructor,
        # which joins the rogue receive thread, closes the fd, and releases the
        # dmaSetMaskBytes kernel entry for that destination.
        #
        # This works because streamConnect(dma, slv) is UNIDIRECTIONAL
        # (dma→slv only): nothing holds a C++ shared_ptr back to dma, so
        # there is no cycle preventing the destructor from running.
        dma_handles.clear()
        slaves.clear()
        gc.collect()  # belt-and-suspenders: flush any deferred destructors


if __name__ == "__main__":
    main()
