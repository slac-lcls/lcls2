"""One device copy of each declared calibration constant per physical GPU.

Several BD ranks on one GPU otherwise upload identical read-only arrays: four
BDs requesting 1 GiB of constants consume ~4 GiB instead of ~1 GiB, which is
subtracted from every rank's automatic budget.

``SharedRequestedConstants`` keeps ``RequestedConstants``' public surface --
``get``, ``refresh``, ``close`` -- so ``BatchInputContext`` and the task
callback contract are unchanged. One rank per device owns the allocation and
publishes CUDA IPC handles; its peers import non-owning views.

Three properties the design turns on:

**All or nothing.** Every peer must declare the same selector set. Peers on one
device run the same ``GpuTask``, so they normally do; a group that disagrees
shares nothing and each rank uploads its own copy.

**Content disagreement is fatal.** Differing shape, dtype or digest aborts,
because sharing there would hand a follower the owner's array under the
follower's own name -- wrong results with no error.

**Un-pooled owner allocation.** ``cudaIpcGetMemHandle`` needs a pointer from
``cudaMalloc``. CuPy's default pool returns sub-blocks of a larger segment, so
exporting ``array.data.ptr`` either fails or -- worse, as measured -- succeeds
while referring to the segment base, exposing neighbouring blocks to peers and
giving the importer a pointer to the wrong data. The owner therefore allocates
outside the pool and charges the budget explicitly.

An IPC failure aborts the job. Set ``PSANA_GPU_SHARE_CONSTANTS=0`` to run
without sharing on a system where IPC is unavailable -- MIG, chiefly. See
docs/device_placement_and_shared_constants.md.
"""
import logging
from hashlib import blake2b

import numpy as np

from .gpu_placement import sharing_enabled
from .gpu_task import RequestedConstants, _selectors

# sharing_enabled lives in gpu_placement, which is where it is read and
# reduced over the device group. Re-exported here because that is where
# callers and tests look for it.
__all__ = ['SharedConstantsError', 'SharedRequestedConstants', 'digest_of',
           'sharing_enabled']


logger = logging.getLogger(__name__)

DIGEST_BYTES = 16


def digest_of(array):
    """Content digest of a C-contiguous host array."""
    source = np.ascontiguousarray(array)
    hasher = blake2b(digest_size=DIGEST_BYTES)
    hasher.update(source.view(np.uint8).reshape(-1).data)
    return hasher.hexdigest()


class SharedConstantsError(RuntimeError):
    """Peers disagree about a shared constant, or CUDA IPC is unusable."""


class _OwnedBlock:
    """An un-pooled device allocation and its explicit budget charge.

    Deliberately not routed through ``owned_empty``: that mandates the default
    CuPy pool so allocation capacity is exactly predictable, and a pooled
    pointer is not IPC-exportable. Shared constants are a handful of
    allocations made once per run, so bypassing the pool costs nothing in
    fragmentation while keeping the bytes budget-visible.
    """

    __slots__ = ('cp', 'ptr', 'nbytes', 'budget', 'array', '_released')

    def __init__(self, cp, nbytes, budget):
        self.cp = cp
        self.nbytes = int(nbytes)
        self.budget = budget
        self.ptr = 0
        self.array = None
        self._released = False
        if budget is not None:
            budget.reserve(self.nbytes)
        try:
            self.ptr = int(cp.cuda.runtime.malloc(self.nbytes))
        except BaseException:
            if budget is not None:
                budget.release(self.nbytes)
            raise

    def view(self, shape, dtype):
        memory = self.cp.cuda.UnownedMemory(self.ptr, self.nbytes, self)
        self.array = self.cp.ndarray(
            shape, dtype=dtype,
            memptr=self.cp.cuda.MemoryPointer(memory, 0))
        return self.array

    def handle(self):
        return bytes(self.cp.cuda.runtime.ipcGetMemHandle(self.ptr))

    def release(self):
        """Free the device allocation. Callers must ensure every importer has
        closed its mapping first: CUDA requires imported mappings to be closed
        before the exporter deallocates."""
        if self._released:
            return
        self._released = True
        self.array = None
        if self.ptr:
            self.cp.cuda.runtime.free(self.ptr)
            self.ptr = 0
        if self.budget is not None:
            self.budget.release(self.nbytes)


class _ImportedBlock:
    """A non-owning view of a peer's allocation.

    Never charged against this rank's budget: the memory belongs to the owner,
    and charging it would shrink the follower's real budget by the size of
    memory it does not own, which is the opposite of what sharing achieves.
    The bytes are recorded separately so a memory report is still complete.
    """

    __slots__ = ('cp', 'base', 'nbytes', 'array', '_closed')

    def __init__(self, cp, handle, nbytes):
        self.cp = cp
        self.nbytes = int(nbytes)
        self.array = None
        self._closed = False
        self.base = int(cp.cuda.runtime.ipcOpenMemHandle(
            handle, cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess))

    def view(self, shape, dtype, offset=0):
        memory = self.cp.cuda.UnownedMemory(self.base + int(offset),
                                            self.nbytes, self)
        self.array = self.cp.ndarray(
            shape, dtype=dtype,
            memptr=self.cp.cuda.MemoryPointer(memory, 0))
        return self.array

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.array = None
        if self.base:
            self.cp.cuda.runtime.ipcCloseMemHandle(self.base)
            self.base = 0


class SharedRequestedConstants(RequestedConstants):
    """Device constants shared by every BD rank on one physical GPU.

    Inherits ``RequestedConstants`` behaviour -- a private copy per rank --
    whenever sharing is unavailable: a single peer, no communicator, peers
    that declared different selector sets, or ``PSANA_GPU_SHARE_CONSTANTS=0``
    (reduced over the group into ``GpuPlacement.sharing_permitted``, so the
    whole group agrees).

    **An array returned by ``get`` is valid for the whole run.** One constant
    set per run is the supported model, so the shared allocation is written
    once at setup and released at teardown: it is never rewritten in place and
    never replaced while the run is live. Holding a reference across events or
    steps is therefore safe.

    It is NOT valid after the run. Teardown frees the owner's allocation and
    closes the importers' mappings, so a reference kept past that dangles.
    Copy with ``.get()`` to outlive the run.
    """

    def __init__(self, requests, budget, placement=None, *, cp=None,
                 sizing=None):
        # ``sizing(shared_bytes)`` is called once, before anything is
        # allocated, so the caller can set this rank's budget limit with the
        # shared bytes counted once. Optional: without it the limit is whatever
        # the caller already set, which is correct whenever an explicit
        # gpu_memory_budget_gb is in force.
        self._sizing = sizing
        self.requests = _selectors(requests, constants=True)
        self.budget = budget
        self._host = {}         # unused here; see _upload_private
        self._device = {}
        self._placement = placement
        self._cp = cp
        self._owned = {}        # selector -> _OwnedBlock
        self._imported = {}     # selector -> _ImportedBlock
        self._shared = ()       # selectors backed by one device copy
        # Whether establishment has been negotiated, tracked separately from
        # whether anything was shared. A task declaring no constants leaves
        # _shared empty forever, so deriving "not yet established" from it
        # re-ran establish on every transition.
        self._established = False
        self._closed = False
        self._ipc_error = None

    def _sync_device(self):
        """Wait for this rank's submitted work.

        Device-wide rather than null-stream: ``set`` may run on a
        non-blocking current stream, which the null stream does not order
        against.
        """
        self._cp.cuda.Device().synchronize()

    # -- introspection -------------------------------------------------
    @property
    def shared_selectors(self):
        return self._shared

    @property
    def shared_bytes(self):
        """Bytes resident on the device for shared constants, counted once."""
        if self._is_owner:
            return sum(b.nbytes for b in self._owned.values())
        return sum(b.nbytes for b in self._imported.values())

    @property
    def imported_bytes(self):
        """Bytes this rank reads but does not own. Reportable, never charged."""
        return sum(b.nbytes for b in self._imported.values())

    @property
    def _comm(self):
        return getattr(self._placement, 'device_comm', None)

    @property
    def _is_owner(self):
        return bool(getattr(self._placement, 'is_owner', True))

    @property
    def _sharing(self):
        placement = self._placement
        return bool(placement is not None and placement.can_share)

    # -- establish -----------------------------------------------------
    def refresh(self, source, *, before_upload=None):
        """Upload the declared constants once. Returns True if anything moved.

        One constant set per run is the supported model, so this establishes
        the shared copies on the first call and does nothing on any later one.
        A BeginStep does not reload the calibration DB, so there is nothing for
        a second call to pick up.

        That is also what makes a shared array safe to hold: the allocation is
        never rewritten in place and never replaced while the run is live, so
        a reference obtained from ``get`` stays valid until teardown. Mid-run
        replacement would need coordinated retirement across peers, because
        one allocation backs every peer and no rank can know when another has
        stopped reading.
        """
        if self._closed:
            raise SharedConstantsError('shared constants are closed')
        if not self._sharing:
            # No peers, so nothing to share: a private copy per rank, which is
            # the inherited behaviour. Not delegated to super().refresh()
            # because that imports cupy directly, bypassing an injected
            # stand-in and leaving the unit tests unable to exercise this
            # path at all.
            self._note_sharing('off')
            if self._established:
                return False
            self._established = True
            hosts = self._scan(source)
            if before_upload is not None:
                before_upload()
            return self._upload_private(hosts, self.requests)

        if self._established:
            # Already uploaded for this run. No collective here: every peer
            # takes this branch, because establishment happened in the same
            # call on all of them.
            return False

        # Establish runs even for an empty declaration, because _agree_on_set's
        # allgather needs every rank.
        self._established = True
        hosts = self._scan(source)
        if before_upload is not None:
            before_upload()
        moved = self._establish(hosts)
        self._check_order('establish')
        return moved

    def _scan(self, source):
        """Resolve declared selectors to validated host arrays."""
        hosts = {}
        for selector in self.requests:
            detector, key = selector
            try:
                value = source[detector][key]
            except KeyError:
                raise KeyError(
                    f'requested calibration constant {selector!r} is missing'
                ) from None
            if isinstance(value, tuple) and len(value) == 2:
                value = value[0]
            if (not isinstance(value, np.ndarray) or not value.dtype.isnative
                    or value.dtype.char not in '?bBhHiIlLqQefdFD'):
                raise TypeError(
                    f'{selector!r}: expected a native numeric NumPy array')
            hosts[selector] = np.ascontiguousarray(value)
        return hosts

    def _agree_on_set(self):
        """This rank's selectors if every peer declared the same set, else ().

        Sharing is all-or-nothing, so the group needs one answer rather than
        a per-selector split. Decided by allgather rather than by the owner
        alone, so each rank reaches the same verdict independently.
        """
        declared = set(self.requests)
        everyone = self._comm.allgather(declared)
        if any(peer != declared for peer in everyone):
            return ()
        return self.requests

    # -- agreement -----------------------------------------------------
    # Every rank in the group must execute the SAME collectives in the SAME
    # order, whatever fails. An exception that skips one leaves its peers
    # blocked in a collective that never completes -- and because MPI pairs
    # collectives by call order rather than by operation, a rank that reaches
    # a different one gives undefined results before hanging.
    #
    # So the protocol is fixed: bcast the manifest (or a failure marker),
    # then exactly one agreement step carrying both failure kinds. Nothing
    # between them may raise. This holds even though both kinds now abort:
    # the abort must happen on every rank, after the same calls.

    CONTENT, CAPABILITY = 0, 1          # agreement vector slots

    def _reduce_flags(self, content, capability):
        """One collective carrying both failure kinds, MAX over the group."""
        from mpi4py import MPI
        import numpy as _np

        from .gpu_collectives import bounded_allreduce_buffer
        local = _np.array([1 if content else 0, 1 if capability else 0],
                          dtype=_np.int32)
        total = _np.zeros_like(local)
        bounded_allreduce_buffer(self._comm, local, total, MPI.MAX,
                                 'shared-constants/settle')
        return bool(total[self.CONTENT]), bool(total[self.CAPABILITY])

    def _settle(self, content_errors, capability_error):
        """Raise on the group's behalf if any peer failed.

        Both failure kinds are fatal, and both are decided collectively so no
        rank acts on its own view -- a rank that raised alone would leave its
        peers in the next collective.

        Both also release first, through the one path whose barriers every peer
        reaches. ``any_content`` and ``any_capability`` come from the same
        reduction on every rank, so the branch taken here is identical across
        the group and the release is collectively safe.
        """
        any_content, any_capability = self._reduce_flags(
            bool(content_errors), bool(capability_error))
        if not (any_content or any_capability):
            return

        # Release whatever this rank built before anyone raises. Also clears
        # _shared, so the close() that follows an abort takes its early return
        # instead of running the release barriers a second time -- those would
        # be matched only if every peer happened to call close(), which is a
        # property of the caller rather than of this class.
        self._release_shared()
        self._shared = ()
        self._device.clear()
        if self._placement is not None:
            self._placement.imported_bytes = 0

        if any_content:
            if content_errors:
                raise SharedConstantsError('; '.join(content_errors))
            raise SharedConstantsError(
                'a peer rank rejected the shared constants; aborting this '
                'rank so the job does not hang')
        detail = self._ipc_error or 'a peer could not use CUDA IPC'
        raise SharedConstantsError(
            f'CUDA IPC is unusable on this device group: {detail}. '
            'Constant sharing cannot be established. Re-run with '
            'PSANA_GPU_SHARE_CONSTANTS=0 to use a private copy per rank '
            '(n times the device memory for constants), which is the '
            'supported configuration on MIG instances.')

    def _exchange(self, hosts):
        """Publish or import the shared constants. Never raises.

        One entry point for both roles, so a caller cannot reach agreement
        through only one of them. Always returns
        ``(content_errors, capability_error)`` and must always be followed by
        ``_settle``.
        """
        if self._is_owner:
            return self._publish(hosts)
        return self._subscribe(hosts)

    def _release_shared(self):
        """Importers close, barrier, then the owner frees.

        CUDA requires imported mappings to be closed before the exporter
        deallocates. Both callers -- ``close`` and the capability abort in
        ``_settle`` -- go through here, so neither can get the order wrong.

        Synchronises first. Callers are expected to have drained their own
        work, but this frees memory a *peer's* kernels may still be reading,
        and a closed mapping or freed allocation under a running kernel is
        undefined rather than merely late.
        """
        if self._cp is not None:
            # None when the object was constructed but never refreshed.
            self._sync_device()
        if not self._is_owner:
            self._close_imported()
        self._barrier('release-shared-closed')
        if self._is_owner:
            self._close_owned()
        self._barrier('release-shared-freed')

    def _barrier(self, step):
        """Bounded barrier: abort with a named step rather than hang."""
        from .gpu_collectives import bounded_barrier
        bounded_barrier(self._comm, f'shared-constants/{step}')

    def _check_order(self, step):
        """Confirm every peer made the same collectives, when enabled."""
        checker = getattr(self._comm, 'check_agreement', None)
        if checker is not None:
            checker(f'shared-constants/{step}', logger=logger)

    def _establish(self, hosts):
        if self._cp is None:
            import cupy as cp
            self._cp = cp

        self._shared = self._agree_on_set()
        if not self._shared:
            # Either nothing was declared, or the peers disagree about what
            # they declared. Every rank uploads its own copy. No sizing call:
            # the budget is already usable/peers, which is correct when
            # nothing is shared.
            self._note_sharing('off')
            if self.requests:
                # Say so, and say what it costs. The removed fallback logged
                # this; without it a group that quietly holds n copies of its
                # constants looks identical to one that is sharing, and the
                # only trace is shared_bytes sitting at zero.
                self._warn_not_shared(hosts)
            return self._upload_private(hosts, self.requests)

        # Resize the budget BEFORE anything is allocated. The shared copy is
        # device overhead counted once, so the owner's limit has to include it
        # and every rank's share is computed net of it. Applying this after
        # refresh() returned meant the owner allocated against
        # limit = usable/peers: a 12 GiB set on a 40 GiB four-peer device fits
        # the documented accounting (owner 7 + 12 = 19 GiB) yet was charged
        # against 10 GiB, so the copy was refused.
        #
        # Sizes come from this rank's own `hosts`. The shared SET is identical
        # on every rank by construction; a shape or dtype disagreement aborts
        # the whole group in _settle moments later.
        if self._sizing is not None:
            self._sizing(sum(int(hosts[s].nbytes) for s in self._shared
                             if s in hosts))

        self._settle(*self._exchange(hosts))
        self._note_sharing('on')
        # Mirror onto the placement, which is what the log line and the
        # harnesses read. The field was declared and documented but never
        # written, so a follower's imported bytes always reported as zero.
        if self._placement is not None:
            self._placement.imported_bytes = int(self.imported_bytes)
        return True

    def _publish(self, hosts):
        """Owner: allocate un-pooled, upload, export handles.

        Returns (content_errors, capability_error) and NEVER raises: the
        manifest bcast must happen even on failure, or followers wait in
        bcast forever. A failure publishes an empty manifest instead.
        """
        manifest = []
        capability = None
        prepared = {}
        for selector in self._shared:
            source = hosts[selector]
            block = None
            try:
                block = _OwnedBlock(self._cp, source.nbytes, self.budget)
                array = block.view(source.shape, source.dtype)
                array.set(source)
                # `set` is asynchronous on the current stream and a peer may
                # read as soon as it has the handle.
                self._sync_device()
                handle = block.handle()
            except Exception as exc:                      # noqa: BLE001
                capability = f'{type(exc).__name__}: {exc}'
                # The block never reached self._owned, so nothing else would
                # free it or return its budget charge. `block` is None when
                # the constructor itself raised.
                if block is not None:
                    block.release()
                break
            self._owned[selector] = block
            prepared[selector] = array
            manifest.append({
                'selector': selector, 'handle': handle,
                'shape': tuple(source.shape), 'dtype': source.dtype.str,
                'nbytes': int(source.nbytes), 'offset': 0,
                'digest': digest_of(source)})

        # Publish unconditionally. An empty manifest tells followers that the
        # owner failed, so they skip importing and reach agreement with us.
        self._comm.bcast([] if capability else manifest, root=0)
        if capability:
            self._ipc_error = capability
            return [], capability
        self._device.update(prepared)
        return [], None

    def _subscribe(self, hosts):
        """Follower: validate the manifest, then import non-owning views.

        Returns (content_errors, capability_error) and NEVER raises, so this
        rank always reaches the single agreement step its peers also reach.
        """
        manifest = self._comm.bcast(None, root=0)
        if not manifest:
            # The owner failed before publishing. Agree, then abort together.
            return [], 'owner published no manifest'

        errors = []
        capability = None
        for entry in manifest:
            selector = tuple(entry['selector'])
            mine = hosts[selector]
            if (tuple(mine.shape) != tuple(entry['shape'])
                    or mine.dtype.str != entry['dtype']):
                errors.append(
                    f'{selector!r}: peers disagree on layout -- mine is '
                    f'{mine.shape}/{mine.dtype.str}, the owner published '
                    f'{tuple(entry["shape"])}/{entry["dtype"]}')
                continue
            if digest_of(mine) != entry['digest']:
                errors.append(
                    f'{selector!r}: peers disagree on content; sharing would '
                    'silently substitute the owner\'s values for this rank\'s')
                continue
            block = None
            try:
                block = _ImportedBlock(self._cp, entry['handle'],
                                       entry['nbytes'])
                view = block.view(tuple(entry['shape']),
                                  np.dtype(entry['dtype']),
                                  offset=entry['offset'])
            except Exception as exc:                      # noqa: BLE001
                capability = f'{type(exc).__name__}: {exc}'
                # The block never reached self._imported, so _release_shared
                # would not close it -- and the owner frees its allocation
                # after that barrier, while this mapping is still open, which
                # CUDA forbids. Same shape as the owner's half in _publish.
                if block is not None:
                    block.close()
                break
            self._imported[selector] = block
            self._device[selector] = view
        if capability:
            self._ipc_error = capability
        return errors, capability

    def _warn_not_shared(self, hosts):
        """Report that the group declared different sets, and the cost.

        Reached only when every peer wanted constants but not the same ones,
        which is a misconfiguration rather than a design case: peers on one
        device run the same GpuTask. The job keeps running on a copy per rank,
        so without a line here the extra memory is invisible.
        """
        peers = getattr(self._placement, 'n_device_peers', 1)
        wanted = sum(int(hosts[s].nbytes) for s in self.requests if s in hosts)
        logger.warning(
            'GPU constants are NOT shared on this device: peers declared '
            'different sets, so each of %d ranks holds its own copy. '
            '%d bytes (%.1f MiB) per rank, %d bytes resident instead of %d. '
            'Peers on one device normally run the same GpuTask; check the '
            'calibconst declarations if this is unexpected.',
            peers, wanted, wanted / 1024 ** 2, wanted * peers, wanted)

    def _note_sharing(self, state):
        """Record the sharing state on the placement, for the log line."""
        if self._placement is not None:
            self._placement.sharing = state

    def _upload_private(self, hosts, selectors):
        """One copy per rank. Returns whether anything moved."""
        if not selectors:
            return False
        if self._cp is None:
            import cupy as cp
            self._cp = cp
        from .gpu_allocation import upload_owned
        arrays = upload_owned(self._cp, [hosts[s] for s in selectors],
                              self.budget, category='task-constants')
        self._device.update(zip(selectors, arrays))
        # _host is deliberately NOT populated. RequestedConstants keeps host
        # snapshots to detect a changed source between transitions; under one
        # constant set per run there is no second upload to detect anything
        # for, and retaining them would pin a full copy of every constant in
        # host memory for the life of the run. The attribute itself stays,
        # because the inherited get/close still reference it.
        #
        # This makes super().refresh() UNSAFE to call from this class: it
        # reads _host for change detection, would find it empty, and would
        # treat every constant as changed -- uploading a second full copy
        # against the same budget. refresh() below overrides it completely and
        # must keep doing so.
        return True

    # -- teardown ------------------------------------------------------
    def _close_imported(self):
        for block in self._imported.values():
            block.close()
        self._imported.clear()

    def _close_owned(self):
        for block in self._owned.values():
            block.release()
        self._owned.clear()

    def close(self):
        """Release device state. COLLECTIVE over the device group.

        Every peer must call this, because importers close before the owner
        frees -- CUDA requires it -- and that order is enforced with barriers
        inside. Calling it on one role at a time leaves those barriers
        unmatched; the bounded barrier reports that as
        ``release-shared-closed`` rather than hanging.

        A second call returns at once and performs no collective, so repeated
        cleanup is safe and need not be matched.
        """
        if self._closed:
            return
        self._closed = True
        if not self._sharing or not self._shared:
            # Nothing was shared, so there are no mappings to close in order
            # and no barriers to match.
            self._device.clear()
            self._host.clear()
            return
        self._release_shared()
        self._check_order('close')
        self._device.clear()
        self._host.clear()
        if self._placement is not None:
            self._placement.imported_bytes = 0
