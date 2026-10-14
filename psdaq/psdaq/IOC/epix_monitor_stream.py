"""
epix_monitor_stream.py

Rogue stream endpoints for the ePix monitoring IOCs.

The reader decodes incoming monitor packets and hands the converted values to
the IOC through a queue; the writer sends the stream-enable packet that some
detectors require.  Both are detector independent -- the reader is given the
packet class to decode with -- so they are shared by every ePix monitoring IOC.

This module imports ``rogue`` but no detector firmware package, so it is usable
by any of the IOCs regardless of which firmware tree is on the path.
"""

import struct
import logging

import rogue.interfaces.stream


class MonitorStreamWriter(rogue.interfaces.stream.Master):
    """
    Sends the monitor stream enable/disable packet to the detector.
    """

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
    """
    Receives monitor packets, decodes them with ``packet_cls`` and puts the
    resulting {pv_name: value} dict on ``queue`` for the IOC to publish.
    """

    def __init__(self, vc: int, queue=None, packet_cls=None):
        super().__init__()
        self.vc = vc
        self.queue = queue
        self.packet_cls = packet_cls
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
            pkt = self.packet_cls(bytes(buf))
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
