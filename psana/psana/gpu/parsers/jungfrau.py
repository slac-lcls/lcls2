from __future__ import annotations

from dataclasses import dataclass

from psana import dgram
from psana.event import Event

from psana.gpu.parsers.base import IngressParser


@dataclass(frozen=True)
class ParsedJungfrauBatch:
    descriptor_batch: object
    ingress_buffer: object
    dgrams: tuple
    event: object
    storage: str = "host-dgram"


class TransitionalJungfrauIngressParser(IngressParser):
    """Transitional parser that rebuilds host dgrams from ingress bytes."""

    parser_name = "host-dgram"

    def __init__(self, run):
        self.run = run

    def parse_batch(self, descriptor_batch, ingress_buffer):
        max_file_id = -1
        if descriptor_batch.descriptors:
            max_file_id = max(descriptor.file_id for descriptor in descriptor_batch.descriptors)
        configs = getattr(getattr(self.run, "dm", None), "configs", ()) or ()
        dgram_count = max(len(configs), max_file_id + 1)
        dgrams = [None] * dgram_count

        if ingress_buffer is not None:
            for descriptor, payload in zip(descriptor_batch.descriptors, ingress_buffer.payloads):
                config = self.run.dm.configs[descriptor.file_id]
                dgrams[descriptor.file_id] = dgram.Dgram(
                    config=config,
                    view=memoryview(payload),
                    offset=0,
                )

        event = Event(dgrams=dgrams, run=self.run._run_ctx)
        return ParsedJungfrauBatch(
            descriptor_batch=descriptor_batch,
            ingress_buffer=ingress_buffer,
            dgrams=tuple(dgrams),
            event=event,
            storage=self.parser_name,
        )
