from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from psana.gpu.readers.base import BulkReader


@dataclass
class PinnedHostIngressBuffer:
    payloads: list
    storage: str = "pinned-host"


class PinnedHostBulkReader(BulkReader):
    reader_name = "pinned-host"

    def __init__(self, run=None, logger=None):
        self.run = run
        self.logger = logger or getattr(run, "logger", None)
        self._cupyx = None
        self._import_checked = False

    def allocate_ingress_buffer(self, descriptors):
        payloads = []
        for descriptor in descriptors:
            payloads.append(self._empty_payload_buffer(int(descriptor.size)))
        return PinnedHostIngressBuffer(payloads=payloads)

    def read_batch(self, descriptors, ingress_buffer):
        if self.run is None or not hasattr(self.run, "dm"):
            raise RuntimeError("PinnedHostBulkReader requires run.dm with open file descriptors")

        if len(descriptors) != len(ingress_buffer.payloads):
            raise ValueError("Descriptor batch size does not match ingress buffer payload count")

        for descriptor, payload in zip(descriptors, ingress_buffer.payloads):
            self._pread_exact(self.run.dm.fds[descriptor.file_id], payload, descriptor.offset, descriptor.size)
        return ingress_buffer

    def _empty_payload_buffer(self, size):
        cupyx = self._get_cupyx()
        if cupyx is not None:
            try:
                return cupyx.empty_pinned((size,), dtype=np.uint8)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.debug(
                        "Failed to allocate pinned host ingress buffer: %s",
                        exc,
                        exc_info=True,
                    )
        return np.empty((size,), dtype=np.uint8)

    def _get_cupyx(self):
        if self._import_checked:
            return self._cupyx

        self._import_checked = True
        try:
            import cupyx  # pylint: disable=import-outside-toplevel

            self._cupyx = cupyx
        except Exception:
            self._cupyx = None
        return self._cupyx

    def _pread_exact(self, fd, payload, offset, size):
        view = memoryview(payload).cast("B")
        total = 0
        while total < size:
            if hasattr(os, "preadv"):
                got = os.preadv(fd, [view[total:size]], offset + total)
            else:
                chunk = os.pread(fd, size - total, offset + total)
                got = len(chunk)
                if got > 0:
                    view[total : total + got] = chunk
            if got <= 0:
                raise OSError(
                    f"short read while filling pinned ingress buffer: fd={fd} offset={offset} size={size} got={total}"
                )
            total += got
