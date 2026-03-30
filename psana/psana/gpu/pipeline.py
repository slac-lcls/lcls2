from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from psana import utils


class GpuSlotState(Enum):
    EMPTY = auto()
    FILLING = auto()
    COPYING = auto()
    RUNNING = auto()
    DONE = auto()


@dataclass
class GpuSlot:
    slot_id: int
    state: GpuSlotState = GpuSlotState.EMPTY
    state_version: int = 0
    record: object | None = None
    metadata: dict = field(default_factory=dict)
    host_raw: object | None = None
    dev_raw: object | None = None
    dev_out: object | None = None
    stream: object | None = None
    raw_shape: tuple | None = None
    raw_dtype: object | None = None

    def reset(self):
        self.state = GpuSlotState.EMPTY
        self.record = None
        self.metadata.clear()


class GpuPipeline:
    """Generic 3-stage GPU pipeline skeleton."""

    def __init__(self, backend, queue_depth=3, profiler=None):
        self.backend = backend
        self.queue_depth = queue_depth
        self.profiler = profiler
        self.state_version = 0
        self.slots = [GpuSlot(slot_id=i) for i in range(queue_depth)]
        for slot in self.slots:
            self.backend.allocate_slot_buffers(slot)
        self._free_slots = deque(self.slots)
        self._submitted_slots = deque()
        self._ready_slots = deque()

    def has_free_slot(self):
        return bool(self._free_slots)

    def submit_l1(self, rec):
        if not self._free_slots:
            raise RuntimeError(
                f"GpuPipeline queue is full (depth={self.queue_depth}); ready events must be consumed before submitting more"
            )

        slot = self._free_slots.popleft()
        slot.state_version = self.state_version
        slot.record = rec

        stage1_start = time.perf_counter()
        slot.metadata.update(self._build_metadata(rec))
        slot.metadata["submit_t"] = time.perf_counter()

        slot.state = GpuSlotState.FILLING
        self.backend.pack_l1_to_host(rec, slot)
        stage1_dt = time.perf_counter() - stage1_start
        if self.profiler is not None:
            self.profiler.record_stage1(stage1_dt)

        slot.state = GpuSlotState.COPYING
        self.backend.ensure_device_cache(rec, slot)
        self.backend.transfer_to_device(rec, slot)
        slot.state = GpuSlotState.RUNNING
        self.backend.launch_compute(rec, slot)
        self._submitted_slots.append(slot)
        self._promote_ready_slots()
        return slot

    def pop_ready(self):
        self._promote_ready_slots()
        yield from self._consume_ready_slots()

    def wait_ready(self):
        self._promote_ready_slots()
        if not self._ready_slots and self._submitted_slots:
            slot = self._submitted_slots[0]
            self.backend.wait_slot(slot)
            self._promote_ready_slots()
            if not self._ready_slots:
                raise RuntimeError(
                    f"GPU backend failed to complete slot {slot.slot_id} after wait"
                )
        yield from self._consume_ready_slots()

    def flush(self):
        while self._submitted_slots or self._ready_slots:
            if self._ready_slots:
                yield from self._consume_ready_slots()
                continue
            yield from self.wait_ready()

    def drain(self):
        for slot in self.slots:
            if slot.state != GpuSlotState.EMPTY:
                slot.reset()
        self._submitted_slots.clear()
        self._ready_slots.clear()
        self._free_slots = deque(self.slots)

    def handle_transition(self, rec):
        drain_start = time.perf_counter()
        events = list(self.flush())
        if self.profiler is not None:
            self.profiler.record_transition_drain(time.perf_counter() - drain_start)
        self.state_version += 1
        self.backend.on_transition(rec, self.state_version)
        return events

    def _build_metadata(self, rec):
        metadata = {
            "service": rec.service,
            "state_version": self.state_version,
        }
        try:
            metadata["timestamp"] = utils.first_timestamp(rec.dgrams)
        except Exception:
            metadata["timestamp"] = None
        return metadata

    def _promote_ready_slots(self):
        while self._submitted_slots:
            slot = self._submitted_slots[0]
            if not self.backend.slot_is_ready(slot):
                break
            self._submitted_slots.popleft()
            slot.state = GpuSlotState.DONE
            self.backend.on_slot_ready(slot)
            self._ready_slots.append(slot)

    def _consume_ready_slots(self):
        while self._ready_slots:
            slot = self._ready_slots.popleft()
            record = slot.record
            submit_t = slot.metadata.get("submit_t")
            if self.profiler is not None and submit_t is not None:
                self.profiler.record_queue_wait(time.perf_counter() - submit_t)
                self.profiler.record_event_completed()
            slot.reset()
            self._free_slots.append(slot)
            if record is not None:
                yield record.event
