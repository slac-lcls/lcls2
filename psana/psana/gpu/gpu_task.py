"""Host-only task declarations and BD-owned requested calibration values.

This module does not import CuPy at declaration time. Producer dispatch lives
in gpu_producer; declarations never invoke users or create CUDA state.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _name(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('task selector components must be nonempty strings without surrounding whitespace')
    return value


def _selectors(values, *, constants=False):
    if isinstance(values, (str, bytes)):
        raise TypeError('task selectors must be a sequence; wrap a single selector in a list')
    result = []
    for value in values:
        if not constants and isinstance(value, str):
            parts = value.rsplit('.', 1)
            if len(parts) != 2 or parts[1] != 'raw':
                raise ValueError('dense input selectors must be detector.raw; use a tuple for generic fields')
            _name(parts[0])
        else:
            size = 2 if constants else 3
            if not isinstance(value, tuple) or len(value) != size:
                raise TypeError('constant selectors must be (detector, key)' if constants else
                                'field selectors must be (detector, algorithm, field) tuples')
            value = tuple(_name(part) for part in value)
            if not constants and value[1] == 'config':
                raise ValueError('Configure fields are not supported task event inputs')
        if value not in result:
            result.append(value)
    return tuple(result)


class GpuTaskWorker(ABC):
    """A stateful task: one instance per run instance and CUDA execution stream.

    Declare it with ``GpuTask(factory=...)``. The framework calls ``factory()``
    once per execution stream at run setup, on the assigned GPU worker, after
    requested constants are uploaded and before any event is admitted. The
    constructor is host-only configuration; device allocation belongs in
    ``begin_run``, through ``ctx.state``, so it is charged to the budget and
    released in ``close``. Only immutable configuration may be shared between
    instances; each owns its mutable arrays, so no cross-stream atomics are
    needed for the instance's own state.

    Every method runs with ``stream`` as the current CUDA stream (the framework
    enters ``with stream:``), so CuPy work issued in a hook is ordered after
    this instance's earlier ``process`` calls on the same stream. Hooks never
    receive scan/step payload values, SlowUpdate values, the live Run or
    EnvStore, or a communicator (PR #172 scope).

    ``process`` is the only required method; the hooks default to no-ops.
    Required science output belongs in ``end_step``/``end_run``, never in a
    destructor. Borrowed inputs, batch contexts and raw pointers from one
    ``process`` call must not be kept for a later one.

    Contracts, per instance (PR #270 as amended by its review):

    ``begin_run(ctx, stream)``   once, after placement and input metadata are
                                 available; ``ctx.step_scope`` is None.
    ``begin_step(ctx, stream)``  before this instance's first subbatch of a
                                 step, including steps this stream receives no
                                 events for; ``ctx.step_scope`` names the step.
    ``process(batch, stream)``   once per nonempty execution subbatch; the same
                                 ``batch`` a ``function`` task receives.
    ``end_step(ctx, stream)``    enqueued after this instance's last work for
                                 the step, so its contributions are complete by
                                 stream order; export is allowed; exactly once
                                 per stream per step, empty steps included.
    ``end_run(ctx, stream)``     after every admitted contribution has retired;
                                 ``ctx.run_status`` is "complete" only on
                                 EndRun; export remaining state exactly once.
    ``close()``                  idempotent reference cleanup after all GPU
                                 work and export readers finish; no science
                                 output, no MPI.
    """

    def begin_run(self, ctx, stream):
        pass

    def begin_step(self, ctx, stream):
        pass

    @abstractmethod
    def process(self, batch, stream):
        ...

    def end_step(self, ctx, stream):
        pass

    def end_run(self, ctx, stream):
        pass

    def close(self):
        pass


@dataclass(frozen=True)
class GpuTask:
    """Declare a GPU task: a ``function(batch, stream)`` or a stateful ``factory``.

    Exactly one of ``function`` and ``factory`` is given. ``function`` is the
    stateless form, invoked once per nonempty execution subbatch. ``factory``
    is called once per CUDA execution stream at run setup and must return a
    ``GpuTaskWorker``; see that class for the lifecycle. Pass configuration
    through ``functools.partial`` or a small callable class, so the declaration
    stays picklable and host-only.

    ``inputs`` contains ``"detector.raw"`` or (detector, algorithm, field)
    selectors. ``calibconst`` contains (detector, key) pairs. Both default to
    empty and duplicates are removed in declaration order. CUDA state belongs
    inside the assigned worker, not in the declaration. Published results are
    delivered through evt.gpu.get(name).on_cpu. One invocation processes all
    selected events in a memory-bounded execution subbatch. Inputs and
    publications have a leading event-row dimension.

    ``state_bytes_per_stream`` reserves that many bytes of device memory per
    worker instance at setup, before any event is admitted; ``ctx.state``
    allocations draw on it. Setup fails with a capacity diagnostic if the
    reservations do not fit alongside the pipeline's own minimum working set.
    It requires ``factory``: a stateless task has no state to reserve for.
    """
    function: object = None
    inputs: tuple = ()
    calibconst: tuple = ()
    factory: object = None
    state_bytes_per_stream: int = 0

    def __post_init__(self):
        if (self.function is None) == (self.factory is None):
            raise ValueError('GpuTask needs exactly one of function or factory')
        if self.function is not None and not callable(self.function):
            raise TypeError('GpuTask.function must be callable')
        if self.factory is not None and not callable(self.factory):
            raise TypeError('GpuTask.factory must be callable')
        n = self.state_bytes_per_stream
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError('GpuTask.state_bytes_per_stream must be an int')
        if n < 0:
            raise ValueError('GpuTask.state_bytes_per_stream must not be negative')
        if n and self.factory is None:
            raise ValueError('GpuTask.state_bytes_per_stream requires factory')
        object.__setattr__(self, 'state_bytes_per_stream', int(n))
        object.__setattr__(self, 'inputs', _selectors(self.inputs))
        object.__setattr__(self, 'calibconst', _selectors(self.calibconst, constants=True))

    @property
    def stateful(self):
        """True for a ``factory`` declaration."""
        return self.factory is not None

    def validate_detectors(self, selected):
        names = {s.rsplit('.', 1)[0] if isinstance(s, str) else s[0] for s in self.inputs}
        names.update(det for det, _ in self.calibconst)
        missing = names - set(selected)
        if missing:
            raise ValueError(f'task detectors must be selected by gpu_det or hybrid_det: {sorted(missing)}')


class RequestedConstants:
    """One admitted copy per BD; values retain their original shape and dtype.

    Caller drains all users before refresh/close. Uploads complete on the
    current stream before publication. Replacement is atomic and may require
    space for both generations; escaped aliases retain their budget charges.

    ``refresh()`` is called once per run, at setup -- the one production call
    site is ``gpu_events.py``'s setup path. Host-side changes after that are
    not picked up, so the ``_host`` copy it keeps serves only to make a second
    call a no-op rather than to watch for mutation. See
    ``ProducerContext.calibconst`` for the user-facing contract.
    """
    def __init__(self, requests, budget):
        self.requests = _selectors(requests, constants=True)
        self.budget = budget
        self._host = {}
        self._device = {}

    def get(self, detector, key):
        selector = (detector, key)
        if selector not in self.requests:
            raise KeyError(f'calibration constant {selector!r} was not declared')
        return self._device[selector]

    def refresh(self, source, *, before_upload=None):
        changed = {}
        for selector in self.requests:
            det, key = selector
            try:
                value = source[det][key]
            except KeyError:
                raise KeyError(f'requested calibration constant {selector!r} is missing') from None
            # Normal det.calibconst dictionaries hold (value, metadata) pairs.
            if isinstance(value, tuple) and len(value) == 2:
                value = value[0]
            if (not isinstance(value, np.ndarray) or not value.dtype.isnative or
                    value.dtype.char not in '?bBhHiIlLqQefdFD'):
                raise TypeError(f'{selector!r}: expected a native numeric NumPy array')
            previous = self._host.get(selector)
            if (previous is not None and previous.shape == value.shape and
                    previous.dtype == value.dtype and
                    np.array_equal(previous.reshape(-1).view(np.uint8),
                                   value.ravel(order='C').view(np.uint8))):
                continue
            changed[selector] = value.copy(order='C')
        if not changed:
            return False  # No CuPy import, allocation, or synchronization.
        if before_upload is not None:
            before_upload()
        import cupy as cp
        from .gpu_allocation import upload_owned
        arrays = upload_owned(cp, changed.values(), self.budget, category='task-constants')
        self._device.update(zip(changed, arrays))
        self._host.update(changed)
        return True

    def close(self):
        self._device.clear()
        self._host.clear()


def prepare_task_inputs(task, configs, bindings, *, n_slots, budget):
    """Resolve declared selectors once, preserving configured segment identity."""
    from .gpu_detector import DenseInputPreparer
    from .gpu_input import GpuDetectorBinding
    preparers = {}
    for selector in task.inputs:
        if isinstance(selector, str):
            name = selector.rsplit('.', 1)[0]
            binding = bindings[name]
            raw = GpuDetectorBinding(name, canonical_segment_ids=binding.canonical_segment_ids,
                                    field_handles_by_segment=binding.field('raw', 'raw').field_handles_by_segment)
            preparers[selector] = DenseInputPreparer.jungfrau_raw(
                configs, raw, n_slots=n_slots, budget=budget)
        else:
            name, algorithm, field = selector
            bindings[name].field(algorithm, field)  # Descriptor-only; no dense gather or metadata D2H.
    return preparers
