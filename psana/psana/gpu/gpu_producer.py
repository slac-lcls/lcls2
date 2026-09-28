"""Internal producer callback context; host delivery is a separate stage.

Borrowed arrays and pointers are read-only by contract. Native kernels must
check locator status, type/rank and bounds on device before dereferencing.
Context methods are valid only during the callback on its supplied stream.
"""
from dataclasses import dataclass
from math import prod

import numpy as np

from .gpu_task import _name


@dataclass(frozen=True)
class DeviceFieldDescriptor:
    """One host-known physical segment, without inspecting device metadata.

    ``locator_rows[row]`` uses the LOC_* layout in gpudgram.batch. Missing
    source dgrams have no descriptor; missing/rejected fields have device status.
    Arrays retain Python allocation owners, but their execution lease is also
    required: neither a pointer nor an array reference prevents slot overwrite.
    """
    segment_id: int
    data_gpu: object
    locator_rows: object
    row: int
    xtc_type: int
    rank: int
    element_size: int

    @property
    def raw_ptr(self):
        return int(self.data_gpu.data.ptr)

    @property
    def raw_nbytes(self):
        return int(self.data_gpu.nbytes)

    @property
    def locator_ptr(self):
        return int(self.locator_rows.data.ptr)


@dataclass(frozen=True)
class _Publication:
    array: object
    shape: tuple
    dtype: object
    nbytes: int
    lease: object


class ProducerContext:
    """One selected event in an admitted execution slot.

    Register user allocations before launching work with keepalive or publish.
    Return values are ignored. Publication records metadata/ownership only here;
    the Stage 4 delivery path will queue copies after producer completion.
    """
    def __init__(self, event, *, task, dense, bindings, constants, owners,
                 publications, lease, reserved, device, run, step_generation,
                 batch_id):
        self.timestamp = event.timestamp
        self.batch_event_index = event.batch_event_index
        self.batch_id = batch_id
        self.run = run
        self.step_generation = step_generation
        self._event, self._task = event, task
        self._dense, self._bindings, self._constants = dense, bindings, constants
        self._owners, self._publications, self._lease = owners, publications, lease
        self._reserved, self._device = reserved, device
        self._fields = {}
        self._active = True

    def _require_active(self):
        if not self._active:
            raise RuntimeError('producer context is only valid during its callback')

    def _dense_input(self, name):
        self._require_active()
        if name not in self._task.inputs or not isinstance(name, str):
            raise KeyError(f'dense input {name!r} was not declared')
        return self._dense[name].get((self.batch_event_index, self.timestamp))

    def input(self, name):
        value = self._dense_input(name)
        return None if value is None else value[0].data[value[1]]

    def present(self, name):
        value = self._dense_input(name)
        return None if value is None else value[0].present[value[1]]

    def field(self, detector, algorithm, field):
        self._require_active()
        key = (detector, algorithm, field)
        if key not in self._task.inputs:
            raise KeyError(f'field {key!r} was not declared')
        if key not in self._fields:
            descriptors = []
            binding = self._bindings[detector].field(algorithm, field)
            for dgram, segment, handle in binding.iter_sources(self._event):
                batch = dgram._storage_batch()
                locations = batch.configured_locations()
                index = locations.handle_indices[handle]
                rows = locations.backing[index]
                if not 0 <= dgram.dgram_index < batch.n_dgrams <= rows.shape[0]:
                    raise ValueError('field descriptor row is outside configured storage')
                descriptors.append(DeviceFieldDescriptor(
                    segment, batch.data_gpu, rows, dgram.dgram_index,
                    handle.type, handle.rank, handle.element_size))
            self._fields[key] = tuple(descriptors)
        return self._fields[key]

    def calibconst(self, detector, key):
        self._require_active()
        selector = (detector, key)
        if selector not in self._constants:
            raise KeyError(f'calibration constant {selector!r} was not declared')
        return self._constants[selector]

    def segment_ids(self, detector):
        self._require_active()
        declared = {s.rsplit('.', 1)[0] if isinstance(s, str) else s[0]
                    for s in self._task.inputs}
        declared.update(det for det, _ in self._task.calibconst)
        if detector not in declared:
            raise KeyError(f'detector {detector!r} was not declared')
        return self._bindings[detector].canonical_segment_ids

    def keepalive(self, *owners):
        self._require_active()
        self._owners.extend(owners)

    def publish(self, name, array):
        self._require_active()
        import cupy as cp
        _name(name)
        if name in self._reserved:
            raise ValueError(f'publication name {name!r} is reserved for inputs')
        if name in self._publications:
            raise ValueError(f'duplicate publication {name!r}')
        if not isinstance(array, cp.ndarray):
            raise TypeError('publish requires a CuPy device array')
        if array.device.id != self._device:
            raise ValueError('publication is on a different device from the producer')
        dtype, shape = np.dtype(array.dtype), tuple(array.shape)
        if not dtype.isnative or dtype.char not in '?bBhHiIlLqQefdFD':
            raise TypeError('publication requires a native numeric dtype')
        if not array.flags.c_contiguous:
            raise ValueError('publication must be C-contiguous')
        nbytes = int(array.nbytes)
        if any(n < 0 for n in shape) or nbytes != prod(shape) * dtype.itemsize:
            raise ValueError('publication byte extent disagrees with shape and dtype')
        self._owners.append(array)
        self._publications[name] = _Publication(array, shape, dtype, nbytes, self._lease)

    def _close(self):
        self._active = False
        self._event = self._task = self._dense = self._bindings = None
        self._constants = self._owners = self._publications = self._lease = None
        self._fields.clear()


def dispatch_task(task, events, envelopes, prepared, bindings, constants, stream,
                  owners, publications, lease, *, batch_id, run, step_generation,
                  selected_events=None):
    """Invoke once per delivered event with GPU descriptors, after one gather.

    Selection precedes invocation, including when reads cover a max-events tail.
    Dense row maps use original identity because missing detectors compact rows.
    The caller retains owners even if this function raises after enqueueing work.
    """
    import cupy as cp
    from .gpu_task_batch import select_task_events
    if selected_events is None:
        selected_events = select_task_events(events, envelopes)
    dense = {name: {} if value is None else {
        (event.batch_event_index, event.timestamp): (value, row)
        for row, event in enumerate(value.events)
        if not getattr(value, 'source_present', ()) or value.source_present[row]}
        for name, value in prepared.items()}
    staged = {key: constants.get(*key) for key in task.calibconst}
    owners.extend(staged.values())
    reserved = set(bindings)
    for name, binding in bindings.items():
        reserved.add(name + '.raw')
        reserved.update(f'{name}.{alg}.{field}' for alg, field in binding.fields)
    device = cp.cuda.Device().id
    # Stage 3a prepares aligned batch inputs. Stage 3b replaces this historical
    # per-event invocation/publication boundary with the batch callback.
    for event in selected_events:
        outputs = {}
        publications[event.timestamp] = outputs
        context = ProducerContext(
            event, task=task, dense=dense, bindings=bindings, constants=staged,
            owners=owners, publications=outputs, lease=lease, reserved=reserved,
            device=device, run=run, step_generation=step_generation, batch_id=batch_id)
        try:
            with stream:
                task.function(context, stream)
        finally:
            context._close()
        if not outputs:
            del publications[event.timestamp]
