from __future__ import annotations

from collections import deque

from psana.gpu.descriptors import JungfrauDescriptorBatch, JungfrauDescriptorBuilder
from psana.gpu.parsers import TransitionalJungfrauIngressParser
from psana.gpu.readers import PinnedHostBulkReader
from psana.gpu.runtime.base import GpuRuntime
from psana.psexp.run import Run


class BulkReaderRuntime(GpuRuntime):
    runtime_name = "bulk-reader"
    pipeline_name = "descriptor"
    uses_smdonly = True

    def __init__(self, run, profiler=None):
        super().__init__(run, profiler=profiler)
        self.batch_size = max(1, int(getattr(run.dsparms, "gpu_batch_size", 1) or 1))
        self.bulk_reader = PinnedHostBulkReader(run=run, logger=getattr(run, "logger", None))
        self.descriptor_builder = JungfrauDescriptorBuilder(run)
        self.parser = TransitionalJungfrauIngressParser(run)
        self._planned_batches = deque()
        self._ready_batches = deque()
        self._parsed_batches = deque()

    def make_detector(self, name, accept_missing=False, **kwargs):
        return Run.Detector(self.run, name, accept_missing=accept_missing, **kwargs)

    def build_smd_descriptor_batch(self, smd_dgrams, service, timestamp):
        return self.descriptor_builder.build_from_smd_dgrams(
            smd_dgrams=smd_dgrams,
            service=service,
            timestamp=timestamp,
        )

    def plan_next_smd_batch(self, smd_dgrams, service, timestamp):
        batch = self.build_smd_descriptor_batch(smd_dgrams=smd_dgrams, service=service, timestamp=timestamp)
        if len(batch) > self.batch_size:
            descriptors = batch.descriptors[: self.batch_size]
            batch = JungfrauDescriptorBatch(
                timestamp=batch.timestamp,
                service=batch.service,
                det_name=batch.det_name,
                descriptors=tuple(descriptors),
                cache_key=batch.cache_key,
            )
        self._planned_batches.append(batch)
        return batch

    def fill_next_smd_batch(self):
        if not self._planned_batches:
            return None
        batch = self._planned_batches.popleft()
        if batch.is_empty:
            self._ready_batches.append((batch, None))
            return batch, None
        ingress = self.bulk_reader.fill_from_batch(batch)
        self._ready_batches.append((batch, ingress))
        return batch, ingress

    def ingest_smd_dgrams(self, smd_dgrams, service, timestamp):
        batch = self.plan_next_smd_batch(smd_dgrams=smd_dgrams, service=service, timestamp=timestamp)
        if batch.is_empty:
            return batch, None
        return self.fill_next_smd_batch()

    def pop_filled_smd_batch(self):
        if not self._ready_batches:
            return None
        return self._ready_batches.popleft()

    def parse_next_smd_batch(self):
        if not self._ready_batches:
            return None
        batch, ingress = self._ready_batches.popleft()
        parsed = self.parser.parse_batch(batch, ingress)
        self._parsed_batches.append(parsed)
        return parsed

    def ingest_and_parse_smd_dgrams(self, smd_dgrams, service, timestamp):
        batch, _ingress = self.ingest_smd_dgrams(smd_dgrams=smd_dgrams, service=service, timestamp=timestamp)
        if batch.is_empty:
            return None
        return self.parse_next_smd_batch()

    def pop_parsed_smd_batch(self):
        if not self._parsed_batches:
            return None
        return self._parsed_batches.popleft()

    def handle_transition(self, rec):
        return []

    def submit_l1(self, rec):
        self._not_ready()

    def pop_ready(self):
        if False:
            yield None
        return

    def has_free_slot(self):
        self._not_ready()

    def wait_ready(self):
        self._not_ready()
        if False:
            yield None
        return

    def flush(self):
        if False:
            yield None
        return

    def drain(self):
        self._planned_batches.clear()
        self._ready_batches.clear()
        self._parsed_batches.clear()
        return None

    def _not_ready(self):
        raise NotImplementedError(
            "BulkReaderRuntime currently supports descriptor batching, pinned-host ingest, and transitional parse flow only; event submission/execution is not wired yet"
        )
