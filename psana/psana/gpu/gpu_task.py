"""Host-only task declarations and BD-owned requested calibration values.

This module does not import CuPy at declaration time. Producer dispatch lives
in gpu_producer; declarations never invoke users or create CUDA state.
"""
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


@dataclass(frozen=True)
class GpuTask:
    """Declare ``function(evt, stream)``, dense/field inputs and exact constants.

    ``inputs`` contains ``"detector.raw"`` or (detector, algorithm, field)
    selectors. ``calibconst`` contains (detector, key) pairs. Both default to
    empty and duplicates are removed in declaration order. CUDA state belongs
    inside the assigned worker, not in the declaration. Public event processing
    is rejected until Stage 4 publication delivery is available. Stage 3a
    prepares aligned BatchInputContext inputs; Stage 3b will replace this
    internal per-event callable with function(batch, stream).
    """
    function: object
    inputs: tuple = ()
    calibconst: tuple = ()

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError('GpuTask.function must be callable')
        object.__setattr__(self, 'inputs', _selectors(self.inputs))
        object.__setattr__(self, 'calibconst', _selectors(self.calibconst, constants=True))

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
    A host snapshot detects in-place source changes at transition boundaries.
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
