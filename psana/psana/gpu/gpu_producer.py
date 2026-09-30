"""One producer callback per selected execution subbatch; no host delivery.

Borrowed inputs are read-only. All work uses the supplied stream. Register
scratch/output owners before launching work; methods expire on callback return.
"""
from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np

from .gpu_task import _name


@dataclass(frozen=True)
class PublicationBatch:
    """One contiguous allocation/view and its event-row mapping.

    shape includes the result-row axis. event_indices address the selected
    batch context, not original EB event indices. Stage 4 copies this group
    under its byte cap and exposes shape[1:] for each mapped event.
    """
    name: str
    array: object
    shape: tuple
    dtype: object
    nbytes: int
    lease: object
    event_indices: tuple
    timestamps: tuple


@dataclass(frozen=True)
class _PublicationRow:
    """Host row bookkeeping; constructing it performs no CUDA operations."""
    batch: PublicationBatch
    row: int

    @property
    def array(self):
        # A slice/reshape is a borrowed view, including scalar and empty rows.
        return self.batch.array[self.row:self.row+1].reshape(self.shape)

    @property
    def shape(self):
        return self.batch.shape[1:]

    @property
    def dtype(self):
        return self.batch.dtype

    @property
    def nbytes(self):
        return prod(self.shape) * self.dtype.itemsize

    @property
    def lease(self):
        return self.batch.lease


class ProducerContext:
    """Callback-scoped access to aligned inputs and batched publication.

    publish(name, array) maps array's leading axis to all selected events.
    event_indices supplies host integer row indices for sparse/mixed-shape
    publication groups. Multiple groups may use a name only on disjoint events.
    Per-event scalars use shape (N,); aggregate output uses (1, ...) with an
    explicit single event index. Return values are ignored.
    """
    def __init__(self, inputs, owners, publications, batches, lease, reserved, device):
        self._inputs = inputs
        self._owners, self._publications, self._batches = owners, publications, batches
        self._lease, self._reserved, self._device = lease, reserved, device
        self._active = True

    def _require_active(self):
        if not self._active:
            raise RuntimeError('producer context is only valid during its callback')

    @property
    def size(self):
        self._require_active()
        return self._inputs.size

    @property
    def timestamps(self):
        self._require_active()
        return self._inputs.timestamps

    @property
    def batch_event_indices(self):
        self._require_active()
        return self._inputs.batch_event_indices

    @property
    def timestamps_gpu(self):
        self._require_active()
        return self._inputs.timestamps_gpu

    @property
    def batch_event_indices_gpu(self):
        self._require_active()
        return self._inputs.batch_event_indices_gpu

    @property
    def batch_id(self):
        self._require_active()
        return self._inputs.batch_id

    @property
    def run(self):
        self._require_active()
        return self._inputs.run

    @property
    def step_generation(self):
        self._require_active()
        return self._inputs.step_generation

    def input(self, name):
        self._require_active()
        return self._inputs.input(name)

    def present(self, name):
        self._require_active()
        return self._inputs.present(name)

    def field(self, detector, algorithm, field):
        self._require_active()
        return self._inputs.field(detector, algorithm, field)

    def calibconst(self, detector, key):
        self._require_active()
        return self._inputs.calibconst(detector, key)

    def segment_ids(self, detector):
        self._require_active()
        return self._inputs.segment_ids(detector)

    def keepalive(self, *owners):
        self._require_active()
        self._owners.extend(owners)

    def publish(self, name, array, event_indices=None):
        self._require_active()
        import cupy as cp
        _name(name)
        if name in self._reserved:
            raise ValueError(f'publication name {name!r} is reserved for inputs')
        if not isinstance(array, cp.ndarray):
            raise TypeError('publish requires a CuPy device array')
        if array.device.id != self._device:
            raise ValueError('publication is on a different device from the producer')
        dtype, shape = np.dtype(array.dtype), tuple(array.shape)
        if not dtype.isnative or dtype.char not in '?bBhHiIlLqQefdFD':
            raise TypeError('publication requires a native numeric dtype')
        if not shape:
            raise ValueError('publication requires a leading event-row axis; use (N,) for scalars')
        if not array.flags.c_contiguous:
            raise ValueError('publication must be C-contiguous')
        nbytes = int(array.nbytes)
        if any(n < 0 for n in shape) or nbytes != prod(shape) * dtype.itemsize:
            raise ValueError('publication byte extent disagrees with shape and dtype')
        if event_indices is None:
            indices = tuple(range(self.size))
        else:
            if isinstance(event_indices, cp.ndarray):
                raise TypeError('event_indices must be host integer indices')
            indices = tuple(event_indices)
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in indices):
                raise TypeError('event_indices must be host integer indices')
            indices = tuple(int(i) for i in indices)
        if shape[0] != len(indices):
            raise ValueError('publication leading axis must match event_indices or batch size')
        if any(i < 0 or i >= self.size for i in indices):
            raise ValueError('publication event index is outside the selected batch')
        if len(set(indices)) != len(indices):
            raise ValueError('duplicate publication event index')
        timestamps = tuple(self.timestamps[i] for i in indices)
        if any(name in self._publications.get(ts, {}) for ts in timestamps):
            raise ValueError(f'duplicate publication {name!r} for an event')
        batch = PublicationBatch(name, array, shape, dtype, nbytes, self._lease, indices, timestamps)
        self._owners.append(array)
        self._batches.append(batch)
        for row, ts in enumerate(timestamps):
            self._publications.setdefault(ts, {})[name] = _PublicationRow(batch, row)

    def _close(self):
        self._active = False
        self._inputs = self._owners = self._publications = self._batches = None
        self._lease = self._reserved = None


def dispatch_task(task, inputs, bindings, stream, owners, publications, batches, lease):
    """Invoke once for N>0; completion is recorded by EventPool after return."""
    if not inputs.size:
        return
    import cupy as cp
    reserved = set(bindings)
    for name, binding in bindings.items():
        reserved.add(name + '.raw')
        reserved.update(f'{name}.{alg}.{field}' for alg, field in binding.fields)
    context = ProducerContext(inputs, owners, publications, batches, lease,
                              reserved, cp.cuda.Device().id)
    try:
        with stream:
            task.function(context, stream)
    finally:
        context._close()
