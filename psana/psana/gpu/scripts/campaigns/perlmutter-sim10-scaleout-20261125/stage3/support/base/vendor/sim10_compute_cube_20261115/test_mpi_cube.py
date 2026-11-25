"""CPU-only tests: deferred CUDA mocks, rendezvous MPI, and in-memory file I/O.

Requires Python >=3.8 and NumPy. Run with PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
notes/sim10_compute_cube_20261115 -p test_mpi_cube.py -v. No GPU, MPI runtime,
scratch files, bytecode, or success artifacts are required or produced.
"""

import ast
import copy
import io
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import mpi_cube


SHAPE = (2, 3, 4)
CURRENT = threading.local()


class Stream:
    def __init__(self):
        self.operations = []
        self.submitted = 0
        self.completed = 0
        self.fail_sync = False
        self.ptr = id(self)
        self.previous = []

    def __enter__(self):
        self.previous.append(getattr(CURRENT, 'stream', None))
        CURRENT.stream = self
        return self

    def __exit__(self, *args):
        CURRENT.stream = self.previous.pop()

    def enqueue(self, function):
        self.operations.append(function)
        self.submitted += 1

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError('Injected stream synchronization failure')
        while self.operations:
            self.operations.pop(0)()
            self.completed += 1


class Event:
    def __init__(self, **kwargs):
        self.stream = None

    def record(self, stream):
        self.stream = stream
        self.target = stream.submitted

    def query(self):
        return self.stream.completed >= self.target


class DeviceArray:
    def __init__(self, array):
        self.array = array
        self.device = SimpleNamespace(id=0)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def flags(self):
        return self.array.flags

    def __getitem__(self, key):
        return DeviceArray(self.array[key])

    def fill(self, value):
        CURRENT.stream.enqueue(lambda: self.array.fill(value))

    def set(self, source, stream):
        stream.enqueue(lambda: np.copyto(self.array, source))

    def get(self, out=None, stream=None, blocking=True):
        if not blocking or self.array.ndim > 3:
            raise AssertionError('Only blocking bounded-bin/count D2H is allowed')
        if stream is None:
            raise AssertionError('D2H must identify its execution stream')
        stream.synchronize()
        if out is None:
            if self.array.shape != (mpi_cube.NBINS,):
                raise AssertionError('Image D2H must reuse an output buffer')
            return self.array.copy()
        np.copyto(out, self.array)
        return out


class Kernel:
    def __init__(self, source, name, options=()):
        self.source, self.name, self.options = source, name, options
        self.compiled = False
        self.fail_launch = False

    def compile(self):
        self.compiled = True

    def __call__(self, grid, block, arguments, stream):
        if self.fail_launch:
            raise RuntimeError('Injected kernel launch failure')
        if self.name == 'accept_events':
            errors, bins, accepted, counts, event_count, panels = arguments

            def execute():
                for index in range(int(event_count)):
                    good = not np.any(errors.array[index, :int(panels)])
                    accepted.array[index] = good
                    if good:
                        counts.array[bins.array[index]] += np.uint64(1)
        else:
            images, bins, accepted, sums, pixels, event_count = arguments

            def execute():
                for index in range(int(event_count)):
                    if accepted.array[index]:
                        sums.array[bins.array[index]] += images.array[index]
        stream.enqueue(execute)


def fake_cupy():
    return SimpleNamespace(ndarray=DeviceArray, float32=np.float32, uint32=np.uint32,
                           uint64=np.uint64, int32=np.int32, uint8=np.uint8,
                           empty=lambda shape, dtype: DeviceArray(np.empty(shape, dtype)),
                           RawKernel=Kernel, cuda=SimpleNamespace(Event=Event,
                           runtime=SimpleNamespace(getDevice=lambda: 0)))


class World:
    """Blocking, one-buffer rendezvous transport with explicit abort wakeup."""

    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.aborted = False
        self.collectives = {}
        self.messages = {}
        self.receives = []
        self.send_buffers = {}
        self.receive_buffers = set()

    def wait(self, predicate):
        deadline = time.monotonic() + 8
        while not predicate() and not self.aborted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Mock MPI deadlock')
            self.condition.wait(remaining)
        if self.aborted:
            raise RuntimeError('Mock MPI aborted by runner')

    def abort(self):
        with self.condition:
            self.aborted = True
            self.condition.notify_all()

    def collective(self, sequence, rank, kind, value):
        with self.condition:
            entry = self.collectives.setdefault(sequence, {'kind': kind, 'values': {}})
            if entry['kind'] != kind or isinstance(value, np.ndarray):
                raise AssertionError('Mismatched collective or array gather')
            entry['values'][rank] = copy.deepcopy(value)
            self.condition.notify_all()
            self.wait(lambda: len(entry['values']) == self.size)
            if kind == 'allgather':
                return [copy.deepcopy(entry['values'][owner]) for owner in range(self.size)]
            return copy.deepcopy(entry['values'][0])


class Comm:
    def __init__(self, world, rank):
        self.world, self.rank = world, rank
        self.sequence = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.world.size

    def allgather(self, value):
        self.sequence += 1
        return self.world.collective(self.sequence, self.rank, 'allgather', value)

    def bcast(self, value, root):
        if root != 0:
            raise AssertionError('Collector must be global rank zero')
        self.sequence += 1
        return self.world.collective(self.sequence, self.rank, 'bcast', value)

    def Barrier(self):
        self.sequence += 1
        return self.world.collective(self.sequence, self.rank, 'barrier', None)

    def Send(self, buffer, dest, tag):
        with self.world.condition:
            if dest != 0 or tag != 0 or buffer.dtype != np.float32 or buffer.ndim != 3:
                raise AssertionError('Expected exactly one FP32 bin')
            if self.rank in self.world.messages:
                raise AssertionError('Multiple outstanding sender buffers')
            self.world.send_buffers.setdefault(self.rank, set()).add(id(buffer))
            self.world.messages[self.rank] = buffer
            self.world.condition.notify_all()
            self.world.wait(lambda: self.rank not in self.world.messages)

    def Recv(self, buffer, source, tag):
        with self.world.condition:
            if self.rank != 0 or tag != 0:
                raise AssertionError('Wrong collector/tag')
            self.world.wait(lambda: source in self.world.messages)
            np.copyto(buffer, self.world.messages.pop(source))
            self.world.receive_buffers.add(id(buffer))
            self.world.receives.append(source)
            self.world.condition.notify_all()

    def Abort(self, code):
        self.world.abort()


class MemoryFile:
    def __init__(self, filesystem, path, binary):
        self.filesystem, self.path = filesystem, str(path)
        self.buffer = io.BytesIO() if binary else io.StringIO()
        self.descriptor = len(filesystem.descriptors) + 10
        filesystem.descriptors[self.descriptor] = self.path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.filesystem.files[self.path] = self.buffer.getvalue()

    def write(self, data):
        if self.filesystem.fail_write == Path(self.path).name:
            raise OSError('Injected write failure')
        return self.buffer.write(data)

    def flush(self):
        self.filesystem.events.append(('flush', self.path))

    def fileno(self):
        return self.descriptor

    def tell(self):
        return self.buffer.tell()


class MemoryFilesystem:
    """Mock only export paths; real source reads remain available."""

    def __init__(self):
        self.directories = set()
        self.files = {}
        self.descriptors = {}
        self.events = []
        self.fail_write = self.fail_fsync = None
        self.fail_rename = False

    def mkdir(self, path, mode, exist_ok):
        if exist_ok or str(path) in self.directories:
            raise FileExistsError(str(path))
        self.directories.add(str(path))

    def open(self, path, mode):
        if mode not in ('x', 'xb') or str(path) in self.files:
            raise FileExistsError(str(path))
        self.files[str(path)] = b'' if mode == 'xb' else ''
        return MemoryFile(self, path, mode == 'xb')

    def os_open(self, path, flags):
        descriptor = len(self.descriptors) + 10
        self.descriptors[descriptor] = str(path)
        return descriptor

    def fsync(self, descriptor):
        path = self.descriptors[descriptor]
        if self.fail_fsync == Path(path).name:
            raise OSError('Injected fsync failure')
        self.events.append(('fsync', path))

    def rename(self, source, target):
        if self.fail_rename:
            raise OSError('Injected publication failure')
        if str(target) in self.files:
            raise FileExistsError(str(target))
        self.events.append(('rename', str(target)))
        self.files[str(target)] = self.files.pop(str(source))


class CubeFixture(unittest.TestCase):
    def setUp(self):
        self.patch_cupy = patch.dict(sys.modules, {'cupy': fake_cupy()})
        self.patch_cupy.start()
        self.addCleanup(self.patch_cupy.stop)

    def arrays(self, count, shape=SHAPE):
        return (DeviceArray(np.ones((count, *shape), np.float32)),
                DeviceArray(np.zeros((count, shape[0]), np.uint32)))


class CubeTests(CubeFixture):

    def test_default_admission_and_unchanged_reference(self):
        cube = mpi_cube.Cube([2, 1])
        self.assertEqual(cube.shape, (32, 512, 1024))
        self.assertEqual(cube.required_state_bytes, 100 * ((64 << 20) + 8))
        self.assertIsNone(cube.bank)
        for updates in ({'max_state_bytes': cube.required_state_bytes - 1},
                        {'shape': (33, 512, 1024)}, {'shape': (0, 1, 1)},
                        {'shape': (1., 1, 1)}, {'shape': (1, 1)},
                        {'max_batch_events': 201}, {'max_batch_events': True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                mpi_cube.Cube([], **updates)
        for stamps in ([1, 1], [1.5], [True], [-1], [2 ** 64]):
            with self.subTest(stamps=stamps), self.assertRaises(ValueError):
                mpi_cube.Cube(stamps)
        reference = mpi_cube._load_reference()
        tree = ast.parse(mpi_cube.REFERENCE_FILE.read_text())
        original = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'KERNEL'
                                for target in node.targets))
        self.assertEqual(reference.KERNEL, ast.literal_eval(original))
        self.assertEqual(reference.Bank.__init__.__code__.co_filename, str(mpi_cube.REFERENCE_FILE))
        self.assertFalse(hasattr(cube, 'sample10'))

    def test_prepare_and_stream_ownership(self):
        cube, stream = mpi_cube.Cube([40], shape=SHAPE), Stream()
        cube.prepare(stream)
        self.assertEqual(cube.initialization, 'prepared_untimed')
        self.assertTrue(cube.kernel.compiled and cube.accept_kernel.compiled)
        self.assertFalse(np.any(cube.bank.sums.array))
        self.assertIn('--fmad=false', cube.kernel.options)
        with self.assertRaises(RuntimeError):
            cube.prepare(stream)
        images, errors = self.arrays(1)
        with self.assertRaises(RuntimeError):
            cube.add(images, errors, [40], Stream(), lambda *owners: None)
        cube.add(images, errors, [40], stream, lambda *owners: None)
        cube.drain()
        self.assertEqual(cube.bank.counts.array[0], 1)

    def test_prepared_callback_never_reinitializes_or_compiles(self):
        cube, stream = mpi_cube.Cube([0, 1], shape=SHAPE), Stream()
        cube.prepare(stream)
        bank = cube.bank
        images, errors = self.arrays(1)
        with patch.object(mpi_cube, '_load_reference', side_effect=AssertionError('Reloaded source')), \
                patch.object(Kernel, 'compile', side_effect=AssertionError('Recompiled kernel')):
            cube.add(images, errors, [0], stream, lambda *owners: None)
            stream.synchronize()
            cube.add(images, errors, [1], stream, lambda *owners: None)
            cube.drain()
        self.assertIs(cube.bank, bank)
        self.assertEqual(cube.bank.counts.array[:2].tolist(), [1, 1])

    def test_registration_failure_and_failed_completion_keep_owners(self):
        cube, stream = mpi_cube.Cube([0, 1], shape=SHAPE), Stream()
        images, errors = self.arrays(1)
        with self.assertRaises(RuntimeError):
            cube.add(images, errors, [0], stream,
                     unittest.mock.Mock(side_effect=RuntimeError('Owner registration failed')))
        self.assertEqual(cube.phase, 'failed')
        self.assertTrue(cube._pending)
        self.assertFalse(stream.operations)
        cube.drain()
        cube = mpi_cube.Cube([0, 1], shape=SHAPE)
        cube.add(images, errors, [0], stream, lambda *owners: None)
        with patch.object(cube._completion, 'query', side_effect=RuntimeError('Event failure')):
            with self.assertRaises(RuntimeError):
                cube.add(images, errors, [1], stream, lambda *owners: None)
        self.assertEqual(cube.phase, 'failed')
        self.assertTrue(cube._pending)
        cube.drain()

    def test_finite_check_scratch_is_chunk_bounded(self):
        array = np.zeros((1, 1, mpi_cube.CHECK_ELEMENTS * 2 + 1), np.float64)
        original = np.isfinite
        sizes = []

        def observe(chunk):
            sizes.append(chunk.size)
            return original(chunk)

        with patch.object(mpi_cube.np, 'isfinite', observe):
            mpi_cube._check_bin(array, True)
        self.assertEqual(sizes, [mpi_cube.CHECK_ELEMENTS, mpi_cube.CHECK_ELEMENTS, 1])
        array[0, 0, -1] = np.nan
        with self.assertRaises(ValueError):
            mpi_cube._check_bin(array, False)

    def test_callback_capacity_guard(self):
        cube, stream = mpi_cube.Cube([0, 1], shape=SHAPE, max_batch_events=1), Stream()
        images, errors = self.arrays(2)
        with self.assertRaises(ValueError):
            cube.add(images, errors, [0, 1], stream, lambda *owners: None)
        self.assertIsNone(cube.bank)

    def test_identity_global_ordinal_not_local_completion(self):
        stamps = [10000 + index * 7 for index in range(202)]
        cube, stream = mpi_cube.Cube(stamps, shape=SHAPE), Stream()
        images, errors = self.arrays(3)
        images.array[:] = np.array([3, 5, 7], np.float32).reshape(3, 1, 1, 1)
        owners = []

        def keepalive(*values):
            self.assertEqual(stream.submitted, 0)
            self.assertEqual(len(values), 7)
            owners.extend(values)

        selected = [stamps[198], stamps[99], stamps[2]]
        cube.add(images, errors, selected, stream, keepalive, 8)
        self.assertEqual(cube.seen, set(selected))
        self.assertEqual(cube.bank.events, [198, 99, 2])
        self.assertEqual(cube.bank.step_labels, {8})
        self.assertEqual(len(cube._pending), 7)
        for invalid in ([stamps[198]], [stamps[3], stamps[3]], [500], [], [1.5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cube.add(images, errors, invalid, stream, lambda *values: None)
        next_images, next_errors = self.arrays(1)
        with self.assertRaisesRegex(RuntimeError, 'still in flight'):
            cube.add(next_images, next_errors, [stamps[0]], stream, lambda *values: None, 8)
        cube.drain()
        self.assertFalse(cube._pending)
        np.testing.assert_array_equal(cube.bank.sums.array[0], np.full(SHAPE, 8))
        self.assertEqual(cube.bank.counts.array[0], 2)
        with self.assertRaises(ValueError):
            cube.add(next_images, next_errors, [stamps[0]], stream, lambda *values: None, 7)
        cube.add(next_images, next_errors, [stamps[0]], stream, lambda *values: None, 9)
        cube.drain()
        self.assertEqual(cube.bank.step_labels, {8, 9})

    def test_input_guards_before_submission(self):
        images, errors = self.arrays(1)
        variants = [(DeviceArray(images.array.astype(np.float64)), errors),
                    (images, DeviceArray(errors.array.astype(np.uint64))),
                    (DeviceArray(images.array[..., ::-1]), errors),
                    (images, DeviceArray(np.zeros((1, 1), np.uint32))),
                    (images.array, errors)]
        for bad_images, bad_errors in variants:
            cube, stream = mpi_cube.Cube([0], shape=SHAPE), Stream()
            with self.subTest(images=type(bad_images)), self.assertRaises(ValueError):
                cube.add(bad_images, bad_errors, [0], stream, lambda *owners: None)
            self.assertIsNone(cube.bank)
            self.assertFalse(stream.operations)
        images.device.id = 1
        with self.assertRaises(ValueError):
            mpi_cube.Cube([0], shape=SHAPE).add(images, errors, [0], Stream(), lambda *owners: None)

    def test_launch_failure_retains_owners_until_explicit_drain(self):
        cube, stream = mpi_cube.Cube([0], shape=SHAPE), Stream()
        cube.prepare(stream)
        cube.kernel.fail_launch = True
        images, errors = self.arrays(1)
        with self.assertRaisesRegex(RuntimeError, 'launch failure'):
            cube.add(images, errors, [0], stream, lambda *owners: None)
        self.assertEqual(cube.phase, 'failed')
        self.assertTrue(cube._pending)
        self.assertFalse(cube.seen)
        stream.fail_sync = True
        with self.assertRaises(RuntimeError):
            cube.drain()
        self.assertTrue(cube._pending)
        stream.fail_sync = False
        cube.drain()
        self.assertFalse(cube._pending)
        with self.assertRaises(RuntimeError):
            cube.add(images, errors, [0], stream, lambda *owners: None)


class ExportTests(CubeFixture):
    def make_case(self, ownership, *, size=None, stamps=None):
        if stamps is None:
            stamps = [10000 + index * 7 for index in range(202)]
        size = size or len(ownership) + 2
        cubes, delivered = [None] * size, [[] for rank in range(size)]
        for offset, ordinals in enumerate(ownership):
            rank = offset + 2
            cube, stream = mpi_cube.Cube(stamps, shape=SHAPE), Stream()
            cubes[rank] = cube
            delivered[rank] = [stamps[index] for index in ordinals]
            for start in range(0, len(ordinals), 200):
                selected = ordinals[start:start + 200]
                images, errors = self.arrays(len(selected))
                for position, ordinal in enumerate(selected):
                    images.array[position] = np.arange(np.prod(SHAPE)).reshape(SHAPE) + ordinal + 1
                cube.add(images, errors, [stamps[index] for index in selected], stream,
                         lambda *owners: None, 0)
                cube.drain()
        return cubes, delivered, stamps

    def run_export(self, case, *, filesystem=None, validator=None, updates=None, wrapper=False):
        cubes, delivered, stamps = case
        filesystem = filesystem or MemoryFilesystem()
        world = World(len(cubes))
        results, failures = [None] * len(cubes), [None] * len(cubes)
        validations, array_ids = [], set()
        originals = Path.open

        def check(bin_index, array, count):
            self.assertFalse(array.flags.writeable)
            self.assertEqual(array.dtype, np.float64)
            self.assertEqual(array.shape, SHAPE)
            self.assertNotIn('/unit-cube/export/manifest.json', filesystem.files)
            array_ids.add(id(array))
            validations.append(bin_index)
            if validator is not None:
                return validator(bin_index, array, count)
            ordinals = list(range(bin_index, len(stamps), mpi_cube.ACTIVE_BINS)) if bin_index < 99 else []
            expected = np.zeros(SHAPE, np.float64)
            for ordinal in ordinals:
                expected += np.arange(np.prod(SHAPE)).reshape(SHAPE) + ordinal + 1
            self.assertEqual(count, len(ordinals))
            np.testing.assert_array_equal(array, expected)

        def run_rank(rank):
            comm = Comm(world, rank)
            arguments = dict(global_timestamps=stamps, delivered_timestamps=delivered[rank],
                             identity_gate_passed=True, expected_bds=sum(cube is not None for cube in cubes),
                             validate_bin=check if rank == 0 else None)
            if updates:
                arguments.update(updates(rank))
            try:
                if wrapper:
                    results[rank] = mpi_cube.export(
                        comm, cubes[rank], '/unit-cube/export' if rank == 0 else None,
                        arguments.pop('global_timestamps'), arguments.pop('delivered_timestamps'), **arguments)
                else:
                    results[rank] = mpi_cube.finalize(
                        comm, cubes[rank], '/unit-cube/export' if rank == 0 else None, **arguments)
            except BaseException as error:
                failures[rank] = error
                comm.Abort(1)

        def open_path(path, mode='r', *args, **kwargs):
            if str(path).startswith('/unit-cube/'):
                return filesystem.open(path, mode)
            return originals(path, mode, *args, **kwargs)

        with patch.object(Path, 'mkdir', lambda path, mode, exist_ok: filesystem.mkdir(path, mode, exist_ok)), \
                patch.object(Path, 'open', open_path), \
                patch.object(Path, 'rename', lambda source, target: filesystem.rename(source, target)), \
                patch.object(mpi_cube.os, 'open', filesystem.os_open), \
                patch.object(mpi_cube.os, 'fsync', filesystem.fsync), \
                patch.object(mpi_cube.os, 'close', lambda descriptor: None):
            threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True)
                       for rank in range(len(cubes))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertTrue(all(not thread.is_alive() for thread in threads), 'Unreleased MPI peer')
        return SimpleNamespace(results=results, failures=failures, filesystem=filesystem,
                               world=world, validations=validations, array_ids=array_ids)

    def assert_failed(self, run):
        self.assertTrue(any(run.failures), 'Expected a failed export')
        self.assertNotIn('/unit-cube/export/manifest.json', run.filesystem.files)

    def test_four_bds_uneven_zero_owner_extra_nonbd(self):
        case = self.make_case([[198, 99, 0], [2, 1], [],
                               [index for index in range(202) if index not in (198, 99, 0, 2, 1)]], size=7)
        run = self.run_export(case)
        self.assertFalse(any(run.failures), run.failures)
        result = run.results[0]
        self.assertEqual(run.world.receives, [2, 3, 4, 5] * 100)
        self.assertEqual(sum(record['kind'] == 'barrier' for record in run.world.collectives.values()), 100)
        self.assertEqual(len(run.world.receive_buffers), 1)
        self.assertTrue(all(len(buffers) == 1 for buffers in run.world.send_buffers.values()))
        self.assertEqual(len(run.array_ids), 1)
        self.assertEqual(run.validations, list(range(100)))
        self.assertEqual(result['counts'][0], 3)
        self.assertEqual(result['counts'][99], 0)
        self.assertEqual(result['counts_by_rank'][4], [0] * 100)
        self.assertEqual(result['fp32_transfer_bytes'], 4 * np.prod(SHAPE))
        self.assertEqual(result['fp64_collector_bytes'], 8 * np.prod(SHAPE))
        self.assertEqual(len(result['timings']), 7)
        for row in result['timings']:
            for key in ('d2h_seconds', 'mpi_seconds', 'merge_seconds', 'write_seconds'):
                self.assertGreaterEqual(row[key], 0.)
        for bin_index in range(100):
            stored = np.load(io.BytesIO(run.filesystem.files[
                '/unit-cube/export/bin-%05d.npy' % bin_index]))
            self.assertEqual(stored.shape, SHAPE)
            self.assertEqual(stored.dtype, np.float64)
        manifest = json.loads(run.filesystem.files['/unit-cube/export/manifest.json'])
        self.assertEqual(manifest['counts'], result['counts'])
        self.assertEqual(run.filesystem.events[-1], ('rename', '/unit-cube/export/manifest.json'))
        synced = [path for operation, path in run.filesystem.events if operation == 'fsync']
        self.assertEqual(len(synced), 102)
        self.assertEqual(synced[-2:], ['/unit-cube/export/manifest.json.partial', '/unit-cube/export'])
        self.assertTrue(all(cube.phase == 'finished' for cube in case[0] if cube is not None))

    def test_one_bd_and_root_only_wrapper_result(self):
        case = self.make_case([list(range(5))], stamps=list(range(5)))
        run = self.run_export(case, wrapper=True)
        self.assertFalse(any(run.failures), run.failures)
        self.assertEqual(run.world.receives, [2] * 100)
        self.assertIsNone(run.results[1])
        self.assertIsNone(run.results[2])
        self.assertEqual(run.results[0]['global_events'], 5)

    def test_entirely_empty_input_and_empty_banks(self):
        for ownership in ([[]], [[], [], [], []]):
            case = self.make_case(ownership, stamps=[])
            run = self.run_export(case)
            self.assertFalse(any(run.failures), run.failures)
            self.assertEqual(run.results[0]['counts'], [0] * 100)
            self.assertTrue(all(cube.bank is None for cube in case[0] if cube is not None))

    def test_global_order_mismatch_and_gate_required(self):
        for changes in (lambda rank: {'identity_gate_passed': rank != 1},
                        lambda rank: {'global_timestamps': [1, 0]} if rank == 1 else {},
                        lambda rank: {'expected_bds': 2},
                        lambda rank: {'validate_bin': None}):
            run = self.run_export(self.make_case([[0, 1]], stamps=[0, 1]), updates=changes)
            self.assert_failed(run)
            self.assertFalse(run.world.receives)
            self.assertFalse(run.filesystem.directories)

    def test_delivery_identity_mismatch_duplicate_and_nonbd_delivery(self):
        for changes in (lambda rank: {'delivered_timestamps': []} if rank == 2 else {},
                        lambda rank: {'delivered_timestamps': [0, 0]} if rank == 2 else {},
                        lambda rank: {'delivered_timestamps': [0]} if rank == 1 else {}):
            run = self.run_export(self.make_case([[0, 1]], stamps=[0, 1]), updates=changes)
            self.assert_failed(run)
            self.assertFalse(run.world.receives)

    def test_rejected_events_counts_dtype_and_uint64_corruption(self):
        for value in (np.uint64(0), np.uint64(2), np.uint64(2 ** 64 - 1)):
            case = self.make_case([[0]], stamps=[0])
            case[0][2].bank.counts.array[0] = value
            run = self.run_export(case)
            self.assert_failed(run)
            self.assertFalse(run.world.receives)
        case = self.make_case([[0]], stamps=[0])
        case[0][2].bank.counts.array = case[0][2].bank.counts.array.astype(np.float64)
        self.assert_failed(self.run_export(case))

    def test_error_kernel_rejection_fails_export(self):
        cube, stream = mpi_cube.Cube([0], shape=SHAPE), Stream()
        images, errors = self.arrays(1)
        errors.array[0, 1] = 3
        cube.add(images, errors, [0], stream, lambda *owners: None)
        case = ([None, None, cube], [[], [], [0]], [0])
        self.assert_failed(self.run_export(case))
        self.assertFalse(np.any(cube.bank.counts.array))
        self.assertFalse(np.any(cube.bank.sums.array))

    def test_missing_global_contributions_and_shape_disagreement(self):
        self.assert_failed(self.run_export(self.make_case([[0]], stamps=[0, 1])))
        case = self.make_case([[], [], [], []], stamps=[])
        case[0][3] = mpi_cube.Cube([], shape=(1, 2, 3))
        self.assert_failed(self.run_export(case))

    def test_nonfinite_and_all_empty_bin_pixels(self):
        for bin_index, value in ((0, np.nan), (0, np.inf), (98, 1), (99, 1)):
            case = self.make_case([[0]], stamps=[0])
            case[0][2].bank.sums.array[bin_index, -1, -1, -1] = value
            self.assert_failed(self.run_export(case))

    def test_independent_callback_failure_and_no_retry(self):
        for returns_false in (True, False):
            case = self.make_case([[0]], stamps=[0])

            def reject(bin_index, actual, count):
                if bin_index == 4:
                    if returns_false:
                        return False
                    raise AssertionError('Independent reference mismatch')

            run = self.run_export(case, validator=reject)
            self.assert_failed(run)
            self.assertIn('/unit-cube/export/bin-00003.npy', run.filesystem.files)
            self.assertNotIn('/unit-cube/export/bin-00004.npy', run.filesystem.files)
            self.assertEqual(case[0][2].phase, 'failed')
            retry = self.run_export(case)
            self.assert_failed(retry)
            self.assertFalse(retry.filesystem.directories)

    def test_write_fsync_publication_and_existing_path_fail_closed(self):
        for failure in ('write', 'bin_fsync', 'metadata_write', 'metadata_fsync',
                        'directory_fsync', 'rename', 'existing'):
            filesystem = MemoryFilesystem()
            if failure == 'write':
                filesystem.fail_write = 'bin-00004.npy'
            elif failure == 'bin_fsync':
                filesystem.fail_fsync = 'bin-00004.npy'
            elif failure == 'metadata_write':
                filesystem.fail_write = 'manifest.json.partial'
            elif failure == 'metadata_fsync':
                filesystem.fail_fsync = 'manifest.json.partial'
            elif failure == 'directory_fsync':
                filesystem.fail_fsync = 'export'
            elif failure == 'rename':
                filesystem.fail_rename = True
            else:
                filesystem.directories.add('/unit-cube/export')
            with self.subTest(failure=failure):
                run = self.run_export(self.make_case([[0]], stamps=[0]), filesystem=filesystem)
                self.assert_failed(run)
                if failure != 'existing':
                    self.assertIn('/unit-cube/export/bin-00000.npy', filesystem.files)

    def test_deterministic_rank_order_not_mpi_reduction(self):
        stamps = list(range(298))
        case = self.make_case([[0], [99], [198], [index for index in range(298)
                                               if index not in (0, 99, 198)]], stamps=stamps)
        for rank, value in zip((2, 3, 4, 5), (1e20, 1, -1e20, 3)):
            case[0][rank].bank.sums.array[0].fill(value)

        def check(bin_index, array, count):
            if bin_index == 0:
                self.assertEqual(count, 4)
                np.testing.assert_array_equal(array, np.full(SHAPE, 3.))

        run = self.run_export(case, validator=check)
        self.assertFalse(any(run.failures), run.failures)
        self.assertEqual(run.world.receives, [2, 3, 4, 5] * 100)


if __name__ == '__main__':
    unittest.main()
