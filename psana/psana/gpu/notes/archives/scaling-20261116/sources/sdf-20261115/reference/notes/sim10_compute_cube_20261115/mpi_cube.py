"""Bounded ECC accumulation and collective export for the four-stage Sim10 run.

BDs construct Cube(global_timestamps, max_state_bytes=...). The default cap is
exactly one full bank; the parent's physical-device admission gate remains
mandatory. prepare(stream) allocates/zeros the bank, compiles both kernels, and
joins on the SAME executor stream outside timing. If omitted, initialization is
lazy and its cost belongs to the first callback's processing interval.
The parent supplies
already calibrated arrays to add(calibrated, errors, timestamps, stream,
keepalive, step_generation). No detector adapter or calibration is called here.
The parent must retain input leases and prohibit mutation/reuse until the cube's
stream completes. Call drain() before allocating another callback's output set
or replacing step constants; add rejects overlapping unfinished submissions.
Holding Python references alone does not stop a parent's buffer recycling.

After iteration, the runner drains all consumers and performs its all-rank
disjoint-union timestamp identity gate. ALL communicator ranks then call
finalize(comm, cube_or_none, absolute_new_local_directory, ...), passing their
own delivered_timestamps and identity_gate_passed=True. Global communicator
rank zero collects; BDs are ranks [2, 2 + expected_bds), with expected_bds 1 or 4.
Other ranks pass cube=None and no delivered timestamps. Uneven and empty BD
ownership is valid. The boolean attests the runner's gate; it is not a substitute
for that gate. Counts alone cannot prove a disjoint identity union.

On root, validate_bin(index, readonly_fp64_bin, exact_python_count) is REQUIRED.
It must validate against an independent reference, return None/True on success,
and raise (or return False) on failure. It must neither retain the reused bin
nor allocate an unbounded reference cube. Pilot/allpixel versus full/sampled
reference policy belongs to the runner, not this helper.

The runner MUST wrap the entire task/export lifecycle in an exception handler
that calls comm.Abort on failure, then re-raises. In particular, root callback,
CUDA, filesystem, or MPI failure can leave peers blocked in Send/Recv or a
collective. This helper never retries, removes partial output, or calls Abort.
After a failed submission, drain() is the explicit way to join retained owners.

Each BD and the collector reuse one <=64 MiB FP32 host bin; root also reuses
one <=128 MiB FP64 bin. Checks use <=64 KiB boolean chunks. No full cube is
gathered. Blocking MPI transfers use ascending BD rank order within each bin.
An all-rank barrier after each written bin prevents eager MPI buffering from
letting senders export future bins ahead of the collector.
Output contains 100 FP64 bin-XXXXX.npy files plus manifest.json. Every bin,
staged metadata file, and directory is fsynced before the final manifest rename.
Publication is the last filesystem operation: a crash can lose the final rename
but cannot make an incomplete set of bins successful. There is no restart API.

finalize returns small metadata on every rank. Per-rank timings separate drain,
D2H, MPI, merge, validation, and bin write/fsync. The final timing rendezvous and
receipt broadcast are excluded from MPI timings. metadata_write_seconds is
returned separately (not embedded in the manifest it measures).
"""

import ast
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


FULL_SHAPE = (32, 512, 1024)
NBINS = 100
ACTIVE_BINS = 99
FP32_BIN_BYTES = 64 << 20
FP64_BIN_BYTES = 128 << 20
FULL_STATE_BYTES = NBINS * (FP32_BIN_BYTES + 8)
CHECK_ELEMENTS = 64 << 10
REFERENCE_FILE = Path(__file__).resolve().parents[1] / 'mfx_expand10_stage3a_20261105/cube.py'


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be an integer, not bool')
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(name + ' must be an integer') from error


def _timestamps(values):
    stamps = tuple(_integer(value, 'timestamp') for value in values)
    if any(value < 0 or value > np.iinfo(np.uint64).max for value in stamps):
        raise ValueError('Timestamps must fit uint64')
    if len(set(stamps)) != len(stamps):
        raise ValueError('Duplicate timestamps')
    return stamps


def _counts(stamps, ordinal):
    counts = np.zeros(NBINS, np.uint64)
    for stamp in stamps:
        if stamp not in ordinal:
            raise ValueError('Unknown timestamp')
        counts[ordinal[stamp] % ACTIVE_BINS] += np.uint64(1)
    return counts


def _load_reference():
    """Load unchanged Bank/KERNEL without legacy adapter imports or path changes.

    Importing the historical module normally executes Sample10/psana imports
    and changes sys.path. Select only these two original AST definitions instead;
    their bodies and CUDA source are compiled verbatim, not rewritten or copied.
    """
    tree = ast.parse(REFERENCE_FILE.read_text(), filename=str(REFERENCE_FILE))
    definitions = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == 'Bank') or (
        isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'KERNEL')]
    if len(definitions) != 2:
        raise RuntimeError('Historical Bank/KERNEL definitions changed')
    tree.body = definitions
    namespace = {'__name__': 'sim10_unchanged_cube_reference'}
    exec(compile(tree, str(REFERENCE_FILE), 'exec'), namespace)
    return SimpleNamespace(Bank=namespace['Bank'], KERNEL=namespace['KERNEL'])


class Cube:
    """One persistent stream-owned FP32 bank with an independent global ordinal.

    Full production layout is the default; smaller three-dimensional shapes
    exist for CPU mocks/acceptance only. State cap is explicit and covers the
    bank, not the parent's framework, inputs, constants, or safety headroom.
    """

    def __init__(self, timestamps, *, max_state_bytes=FULL_STATE_BYTES, shape=FULL_SHAPE,
                 max_batch_events=200):
        self.timestamps = _timestamps(timestamps)
        self.shape = tuple(_integer(value, 'shape') for value in shape)
        self.max_batch_events = _integer(max_batch_events, 'max_batch_events')
        self.max_state_bytes = _integer(max_state_bytes, 'max_state_bytes')
        if (len(self.shape) != 3 or any(value <= 0 for value in self.shape)
                or any(value > limit for value, limit in zip(self.shape, FULL_SHAPE))
                or not 1 <= self.max_batch_events <= 200):
            raise ValueError('Invalid bounded cube layout or callback capacity')
        self.nbins, self.active_bins = NBINS, ACTIVE_BINS
        self.required_state_bytes = NBINS * (math.prod(self.shape) * 4 + 8)
        if self.required_state_bytes > self.max_state_bytes:
            raise ValueError('Persistent bank exceeds explicit state cap')
        self.ordinal = {stamp: ordinal for ordinal, stamp in enumerate(self.timestamps)}
        self.seen = set()
        self.bank = None
        self.phase = 'open'
        self.kernel = self.accept_kernel = None
        self._pending = ()
        self._completion = None
        self._step_generation = None
        self.initialization = 'not_started'

    def _initialize(self, stream, compile_kernels):
        import cupy as cp
        reference = _load_reference()
        self.bank = reference.Bank(stream, NBINS, self.shape)
        self.kernel = cp.RawKernel(reference.KERNEL, 'accumulate',
                                   options=('--std=c++17', '--fmad=false'))
        self.accept_kernel = cp.RawKernel(reference.KERNEL, 'accept_events',
                                          options=('--std=c++17',))
        if compile_kernels:
            self.kernel.compile()
            self.accept_kernel.compile()

    def prepare(self, stream):
        """Untimed allocation/compilation/zeroing, once, on the executor stream."""
        if self.phase != 'open' or self.bank is not None:
            raise RuntimeError('Prepare requires a fresh open cube')
        try:
            with stream:
                self._initialize(stream, compile_kernels=True)
                self.bank.sums.fill(0)
                self.bank.counts.fill(0)
                stream.synchronize()
            self.initialization = 'prepared_untimed'
        except BaseException:
            self.phase = 'failed'
            raise

    def drain(self):
        """Join without closing; retain owners if synchronization fails."""
        try:
            if self.bank is not None:
                self.bank.stream.synchronize()
        except BaseException:
            self.phase = 'failed'
            raise
        self._pending = ()
        self._completion = None

    def add(self, calibrated, errors, timestamps, stream, keepalive, step_generation=0):
        """Register all owners before CUDA submission; errors reject whole events."""
        if self.phase != 'open':
            raise RuntimeError('Cube is not open')
        stamps = _timestamps(timestamps)
        if (not stamps or len(stamps) > self.max_batch_events
                or any(stamp not in self.ordinal or stamp in self.seen for stamp in stamps)):
            raise ValueError('Unknown, repeated, empty, or oversized event batch')
        generation = _integer(step_generation, 'step_generation')
        if generation < 0 or (self._step_generation is not None
                              and generation < self._step_generation):
            raise ValueError('Step generation cannot decrease')
        if self.bank is not None and self.bank.stream is not stream:
            raise RuntimeError('Only one execution stream is budgeted')
        if not callable(keepalive):
            raise ValueError('A keepalive owner registrar is required')
        if self._pending:
            try:
                complete = self._completion is not None and self._completion.query()
            except BaseException:
                self.phase = 'failed'
                raise
            if not complete:
                raise RuntimeError('Previous inputs are still in flight; drain before reuse')
            self._pending = ()
            self._completion = None
        import cupy as cp
        if (not isinstance(calibrated, cp.ndarray) or not isinstance(errors, cp.ndarray)
                or calibrated.shape != (len(stamps), *self.shape)
                or calibrated.dtype != cp.float32 or not calibrated.flags.c_contiguous
                or errors.shape != (len(stamps), self.shape[0])
                or errors.dtype != cp.uint32 or not errors.flags.c_contiguous
                or calibrated.device.id != errors.device.id
                or calibrated.device.id != cp.cuda.runtime.getDevice()):
            raise ValueError('Unexpected calibrated/error layout or CUDA device')
        if self.bank is not None and self.bank.sums.device.id != calibrated.device.id:
            raise ValueError('Cube cannot migrate between CUDA devices')
        host_bins = np.array([self.ordinal[stamp] % ACTIVE_BINS for stamp in stamps], np.int32)
        try:
            with stream:
                fresh_bank = self.bank is None
                if fresh_bank:
                    self._initialize(stream, compile_kernels=False)
                    self.initialization = 'lazy_in_first_callback'
                bins = cp.empty(host_bins.shape, cp.int32)
                accepted = cp.empty(len(stamps), cp.uint8)
                self._pending = (calibrated, errors, bins, accepted, host_bins,
                                 self.bank.sums, self.bank.counts)
                keepalive(*self._pending)
                if fresh_bank:
                    self.bank.sums.fill(0)
                    self.bank.counts.fill(0)
                bins.set(host_bins, stream=stream)
                self.accept_kernel((1,), (1,), (errors, bins, accepted, self.bank.counts,
                                   np.int32(len(stamps)), np.int32(self.shape[0])), stream=stream)
                pixels = math.prod(self.shape)
                self.kernel((min(65535, (pixels + 255) // 256),), (256,),
                            (calibrated, bins, accepted, self.bank.sums,
                             np.uint64(pixels), np.int32(len(stamps))), stream=stream)
                self._completion = cp.cuda.Event(disable_timing=True)
                self._completion.record(stream)
            self.seen.update(stamps)
            self.bank.events.extend(self.ordinal[stamp] for stamp in stamps)
            self.bank.step_labels.add(generation)
            self._step_generation = generation
        except BaseException:
            self.phase = 'failed'
            raise


def _check_bin(array, empty):
    flat = array.reshape(-1)
    for offset in range(0, flat.size, CHECK_ELEMENTS):
        chunk = flat[offset:offset + CHECK_ELEMENTS]
        if not np.isfinite(chunk).all():
            raise ValueError('Nonfinite cube bin')
        if empty and np.count_nonzero(chunk):
            raise ValueError('Nonzero empty cube bin')


def _write_bin(directory, bin_index, merged, count):
    path = directory / ('bin-%05d.npy' % bin_index)
    with path.open('xb') as output:
        np.save(output, merged, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
        size = output.tell()
    return dict(bin=bin_index, count=count, file=path.name, bytes=size)


def _publish(directory, result):
    temporary = directory / 'manifest.json.partial'
    with temporary.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.rename(directory / 'manifest.json')


def finalize(comm, cube, directory, *, global_timestamps, delivered_timestamps,
             identity_gate_passed, expected_bds, bd_start=2, validate_bin=None):
    """Collective, single-attempt export; see module contract before calling.

    directory and validate_bin are used only on root; other ranks may pass None.
    No mpi4py import is needed: comm supplies Get_rank/Get_size, allgather,
    bcast, Barrier, and blocking buffer-mode Send/Recv. Use a dedicated communicator or
    ensure no concurrent protocol uses its tag zero during this export.
    """
    try:
        return _finalize(comm, cube, directory, global_timestamps, delivered_timestamps,
                         identity_gate_passed, expected_bds, bd_start, validate_bin)
    except BaseException:
        if cube is not None:
            cube.phase = 'failed'
        raise


def _finalize(comm, cube, directory, global_timestamps, delivered_timestamps,
              identity_gate_passed, expected_bds, bd_start, validate_bin):
    rank, size = comm.Get_rank(), comm.Get_size()
    timings = dict(rank=rank, drain_seconds=0., d2h_seconds=0., mpi_seconds=0.,
                   merge_seconds=0., validation_seconds=0., write_seconds=0.)
    metadata = dict(error=None)
    try:
        expected_bds = _integer(expected_bds, 'expected_bds')
        bd_start = _integer(bd_start, 'bd_start')
        if bd_start != 2 or expected_bds not in (1, 4) or size < bd_start + expected_bds:
            raise ValueError('Expected one or four BDs starting at global rank two')
        if identity_gate_passed is not True:
            raise ValueError('Runner disjoint-union identity gate has not passed')
        stamps = _timestamps(global_timestamps)
        delivered = _timestamps(delivered_timestamps)
        ordinal = {stamp: index for index, stamp in enumerate(stamps)}
        expected = _counts(stamps, ordinal)
        is_bd = bd_start <= rank < bd_start + expected_bds
        if is_bd:
            if cube is None or cube.phase != 'open' or cube.timestamps != stamps:
                raise ValueError('BD requires an open cube with the same global timestamp order')
            if set(delivered) != cube.seen:
                raise ValueError('Local seen identities must equal local delivered identities')
            started = time.monotonic()
            cube.drain()
            timings['drain_seconds'] += time.monotonic() - started
            started = time.monotonic()
            local_counts = (cube.bank.counts.get(stream=cube.bank.stream, blocking=True)
                            if cube.bank is not None else np.zeros(NBINS, np.uint64))
            timings['d2h_seconds'] += time.monotonic() - started
            if (local_counts.shape != (NBINS,) or local_counts.dtype != np.uint64
                    or not np.array_equal(local_counts, _counts(delivered, ordinal))):
                raise ValueError('Exact local bin counts disagree with delivered ordinals')
        else:
            if cube is not None or delivered:
                raise ValueError('NonBD ranks cannot own a cube or delivered events')
            local_counts = np.zeros(NBINS, np.uint64)
        if rank == 0:
            if not callable(validate_bin):
                raise ValueError('Root requires an independent validate_bin callback')
            directory = Path(directory)
            if not directory.is_absolute():
                raise ValueError('Root output must be an absolute new local directory')
        fingerprint = hashlib.sha256(np.asarray(stamps, dtype='<u8').tobytes()).hexdigest()
        metadata.update(fingerprint=fingerprint, expected_bds=expected_bds, bd_start=bd_start,
                        shape=list(cube.shape) if is_bd else None,
                        initialization=cube.initialization if is_bd else None,
                        counts=[int(value) for value in local_counts], delivered=len(delivered))
    except Exception as error:
        metadata['error'] = '%s: %s' % (type(error).__name__, error)
    started = time.monotonic()
    gathered = comm.allgather(metadata)
    timings['mpi_seconds'] += time.monotonic() - started
    failures = [(owner, record['error']) for owner, record in enumerate(gathered) if record['error']]
    if failures:
        raise RuntimeError('Cube export preflight failed: %r' % failures)
    signature = (metadata['fingerprint'], expected_bds, bd_start)
    if any((record['fingerprint'], record['expected_bds'], record['bd_start']) != signature
           for record in gathered):
        raise ValueError('Ranks disagree on global identity order or topology')
    bd_ranks = range(bd_start, bd_start + expected_bds)
    shape = tuple(gathered[bd_start]['shape'])
    if any(tuple(gathered[owner]['shape']) != shape for owner in bd_ranks):
        raise ValueError('BD cube shapes disagree')
    counts = [sum(record['counts'][bin_index] for record in gathered) for bin_index in range(NBINS)]
    if counts != expected.tolist() or sum(record['delivered'] for record in gathered) != len(stamps):
        raise ValueError('Global bin counts disagree with global ordinals')
    if cube is not None:
        cube.phase = 'exporting'
    if rank == 0:
        directory.mkdir(mode=0o700, exist_ok=False)
    transfer = np.empty(shape, np.float32) if rank == 0 or is_bd else None
    merged = np.empty(shape, np.float64) if rank == 0 else None
    records = []
    for bin_index in range(NBINS):
        if is_bd:
            started = time.monotonic()
            if cube.bank is None:
                transfer.fill(0)
            else:
                cube.bank.sums[bin_index].get(out=transfer, stream=cube.bank.stream, blocking=True)
            timings['d2h_seconds'] += time.monotonic() - started
            started = time.monotonic()
            _check_bin(transfer, int(local_counts[bin_index]) == 0)
            timings['validation_seconds'] += time.monotonic() - started
            started = time.monotonic()
            comm.Send(transfer, dest=0, tag=0)
            timings['mpi_seconds'] += time.monotonic() - started
        if rank == 0:
            started = time.monotonic()
            merged.fill(0)
            timings['merge_seconds'] += time.monotonic() - started
            for owner in bd_ranks:
                started = time.monotonic()
                comm.Recv(transfer, source=owner, tag=0)
                timings['mpi_seconds'] += time.monotonic() - started
                started = time.monotonic()
                _check_bin(transfer, gathered[owner]['counts'][bin_index] == 0)
                timings['validation_seconds'] += time.monotonic() - started
                started = time.monotonic()
                np.add(merged, transfer, out=merged)
                timings['merge_seconds'] += time.monotonic() - started
            started = time.monotonic()
            _check_bin(merged, counts[bin_index] == 0)
            merged.flags.writeable = False
            verdict = validate_bin(bin_index, merged, counts[bin_index])
            if verdict is not None and verdict is not True:
                raise ValueError('Independent merged-bin validation failed')
            merged.flags.writeable = True
            timings['validation_seconds'] += time.monotonic() - started
            started = time.monotonic()
            records.append(_write_bin(directory, bin_index, merged, counts[bin_index]))
            timings['write_seconds'] += time.monotonic() - started
        started = time.monotonic()
        comm.Barrier()
        timings['mpi_seconds'] += time.monotonic() - started
    timing_rows = comm.allgather(timings)
    result = None
    if rank == 0:
        result = dict(status='complete', nbins=NBINS, active_bins=ACTIVE_BINS, shape=list(shape),
                      dtype='float64', count_dtype='uint64', counts=counts, bins=records,
                      global_timestamp_sha256=metadata['fingerprint'], global_events=len(stamps),
                      bd_ranks=list(bd_ranks), collector_rank=0, directory=str(directory),
                      counts_by_rank=[record['counts'] for record in gathered], timings=timing_rows,
                      initialization_by_rank=[record['initialization'] for record in gathered],
                      fp32_transfer_bytes=transfer.nbytes, fp64_collector_bytes=merged.nbytes,
                      independent_validation='every merged bin callback passed',
                      timing_scope='Bin writes include fsync; metadata publication returned separately; '
                                   'final timing rendezvous and receipt broadcast excluded')
        started = time.monotonic()
        _publish(directory, result)
        result['metadata_write_seconds'] = time.monotonic() - started
    result = comm.bcast(result, root=0)
    if cube is not None:
        cube.phase = 'finished'
    return result


def export(comm, cube, local_output, timestamps, delivered, *, validate_bin=None,
           identity_gate_passed, expected_bds=None):
    """Runner-friendly wrapper; root returns metadata, other ranks return None.

    Pass the result of the runner's prior identity gate explicitly. With extra
    nonBD ranks, specify expected_bds; otherwise size minus two defines topology.
    All ranks call even when they own no events. Failure requires runner Abort.
    """
    if expected_bds is None:
        expected_bds = comm.Get_size() - 2
    result = finalize(comm, cube, local_output, global_timestamps=timestamps,
                      delivered_timestamps=delivered, identity_gate_passed=identity_gate_passed,
                      expected_bds=expected_bds, validate_bin=validate_bin)
    return result if comm.Get_rank() == 0 else None
