"""One device copy of each declared calibration constant per physical GPU.

Several BD ranks on one GPU otherwise upload identical read-only arrays: four
BDs requesting 1 GiB of constants consume ~4 GiB instead of ~1 GiB, which is
subtracted from every rank's automatic budget.

``SharedRequestedConstants`` keeps ``RequestedConstants``' public surface --
``get``, ``refresh``, ``close`` -- so ``BatchInputContext`` and the task
callback contract are unchanged. One rank per device owns the allocation and
publishes CUDA IPC handles; its peers import non-owning views.

Three properties the design turns on:

**Intersection, not all-or-nothing.** Peers declaring different selector sets
share what they have in common and privately upload the remainder. A selector
one peer did not request must not fail the job.

**Content disagreement is fatal.** Differing shape, dtype or digest for a
*shared* selector aborts, because sharing there would hand a follower the
owner's array under the follower's own name -- wrong results with no error.

**Un-pooled owner allocation.** ``cudaIpcGetMemHandle`` needs a pointer from
``cudaMalloc``. CuPy's default pool returns sub-blocks of a larger segment, so
exporting ``array.data.ptr`` either fails or -- worse, as measured -- succeeds
while referring to the segment base, exposing neighbouring blocks to peers and
giving the importer a pointer to the wrong data. The owner therefore allocates
outside the pool and charges the budget explicitly.
"""
import logging
from hashlib import blake2b

import numpy as np

from .gpu_task import RequestedConstants, _selectors


logger = logging.getLogger(__name__)

DIGEST_BYTES = 16


def digest_of(array):
    """Content digest of a C-contiguous host array.

    Computed from the host copy ``refresh`` already retains for change
    detection, so it costs one pass over host memory per constant per
    transition -- negligible beside the ``array_equal`` already performed.
    """
    source = np.ascontiguousarray(array)
    hasher = blake2b(digest_size=DIGEST_BYTES)
    hasher.update(source.view(np.uint8).reshape(-1).data)
    return hasher.hexdigest()


class SharedConstantsError(RuntimeError):
    """Peers disagree about a shared constant, or IPC lifetime was violated."""


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

    Falls back to ``RequestedConstants`` behaviour -- a private copy per rank --
    whenever sharing is unavailable: a single peer, no communicator, a MIG
    instance, or an IPC call that fails. Capability failures degrade; only
    content disagreement aborts.
    """

    def __init__(self, requests, budget, placement=None, *, cp=None,
                 sizing=None):
        # ``sizing(shared_bytes)`` is called once, after the intersection is
        # known and before anything is allocated, so the caller can set this
        # rank's budget limit with the shared bytes counted once. Optional:
        # without it the limit is whatever the caller already set, which is
        # correct whenever an explicit gpu_memory_budget_gb is in force.
        self._sizing = sizing
        self.requests = _selectors(requests, constants=True)
        self.budget = budget
        self._host = {}
        self._device = {}
        self._placement = placement
        self._cp = cp
        self._owned = {}        # selector -> _OwnedBlock
        self._imported = {}     # selector -> _ImportedBlock
        self._shared = ()       # selectors backed by one device copy
        self._private = ()      # selectors uploaded by this rank alone
        # Whether the shared/private split has been negotiated, tracked
        # separately from whether it is non-empty. A task declaring no
        # constants leaves both tuples empty forever, so deriving "not yet
        # established" from them re-ran establish on every transition --
        # collectives and a cache trim per step for a task that asked for
        # nothing, reported as work done.
        self._established = False
        self._generation = 0
        self._closed = False
        self._ipc_error = None
        self._digests = {}      # selector -> digest of the retained snapshot
        self._new_digests = {}  # selector -> digest of the incoming value
        self._recorder = None
        if placement is not None and placement.device_comm is not None:
            from .gpu_collectives import RecordingComm, checking_enabled
            if checking_enabled():
                # Wrap the device communicator so every collective this class
                # makes is recorded, then compare the sequences across ranks
                # at the end of refresh and close. Path divergence is what
                # every hang on this branch turned out to be, and nothing
                # else detects it.
                self._recorder = RecordingComm(placement.device_comm,
                                               'shared-constants')
                # Side effect: the placement's communicator is replaced by the
                # wrapper, so every later user of placement.device_comm --
                # including self._comm, which reads through it -- is recorded
                # too. Intended, and only when checking is enabled.
                placement.device_comm = self._recorder

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
    def private_selectors(self):
        return self._private

    @property
    def shared_bytes(self):
        """Intersection bytes resident on the device, counted once."""
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
        """Upload or re-upload declared constants. Returns True if anything moved.

        Without sharing this is exactly the inherited implementation. With
        sharing, the intersection is established once and then maintained in
        place, so an unchanged transition performs no CuPy import, no
        allocation and no synchronization -- the same fast path as today.
        """
        if self._closed:
            raise SharedConstantsError('shared constants are closed')
        hosts, changed, reshaped = self._scan(source)
        if not self._sharing:
            # No peers: every selector is private. Same result as the
            # inherited path, but it honours an injected CuPy stand-in and
            # keeps one code path for the snapshot/digest bookkeeping.
            self._note_sharing('off')
            if not self._private and self.requests:
                self._private = self.requests
            if self._cp is None:
                import cupy as cp
                self._cp = cp
            moved = self._changed_private(hosts)
            if not moved:
                return False
            if before_upload is not None:
                before_upload()
            self._upload_private(hosts, moved)
            return True

        if not self._established:
            # Flag rather than `not self._shared and not self._private`: a
            # task declaring no constants leaves both tuples empty, so that
            # test re-established on every transition -- collectives and a
            # cache trim per step, reported as work done.
            #
            # Establish runs even for an empty declaration, because peers may
            # declare different sets by design and _intersect's allgather
            # needs every rank. Only the second and later transitions are
            # skipped.
            self._established = True
            if before_upload is not None:
                before_upload()
            self._establish(hosts)
            self._check_order('establish')
            # Nothing was declared, so nothing moved. Saying otherwise makes
            # the caller recompute its subbatch budget every step.
            return bool(self._shared or self._private)

        # All ranks must take the same branch or the job hangs, and the
        # branch must reflect EVERY rank's view: a follower whose shared
        # values changed while the owner's did not would otherwise be told
        # "A" and keep reading the owner's old values with no error.
        local_case = 'C' if reshaped else 'B' if changed else 'A'

        # Reduced as an ordinal rather than allgathered as a string, so the
        # collective can be bounded: MIN != MAX means the ranks disagree.
        from mpi4py import MPI

        from .gpu_collectives import bounded_allreduce
        code = 'ABC'.index(local_case)
        lowest = bounded_allreduce(self._comm, code, MPI.MIN,
                                   'shared-constants/case-min')
        highest = bounded_allreduce(self._comm, code, MPI.MAX,
                                    'shared-constants/case-max')
        if lowest != highest:
            # Disagreement means peers hold different values for a shared
            # selector. That is the fatal case, raised on every rank.
            self._settle(
                [f'peers disagree on this transition: proposed '
                 f'{"ABC"[lowest]} and {"ABC"[highest]} -- shared constants '
                 'differ between ranks'],
                None, hosts=hosts)
        case = 'ABC'[highest]

        if case == 'A':
            # Shared constants are unchanged, but a private selector may have
            # moved: those are this rank's own copies and nobody else's
            # business.
            self._check_order('refresh-A')
            moved = self._changed_private(hosts)
            if not moved:
                return False
            if before_upload is not None:
                before_upload()
            self._upload_private(hosts, moved)
            return True
        if before_upload is not None:
            before_upload()
        self._generation += 1
        if case == 'B':
            self._refresh_in_place(hosts, changed)
            self._check_order('refresh-B')
            return True
        self._reestablish(hosts)
        self._check_order('refresh-C')
        return True

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

        changed, reshaped = [], []
        # Retained for _refresh_in_place: hashing a GiB costs about a second,
        # so the value computed here is reused rather than recomputed.
        self._new_digests = {}
        for selector in self._shared:
            previous = self._host.get(selector)
            current = hosts[selector]
            current_digest = digest_of(current)
            self._new_digests[selector] = current_digest
            if previous is None:
                changed.append(selector)
                continue
            if previous.shape != current.shape or previous.dtype != current.dtype:
                reshaped.append(selector)
                changed.append(selector)
            elif self._snapshot_digest(selector, previous) != current_digest:
                changed.append(selector)
        return hosts, changed, reshaped

    def _intersect(self):
        """Selectors every peer declared, and this rank's remainder.

        Computed by allgather rather than by the owner alone, so each rank
        derives the same shared set independently and cannot be surprised by a
        manifest entry it did not expect.
        """
        declared = set(self.requests)
        everyone = self._comm.allgather(declared)
        shared = set.intersection(*everyone) if everyone else set()
        order = {selector: i for i, selector in enumerate(self.requests)}
        return (tuple(sorted(shared, key=order.__getitem__)),
                tuple(s for s in self.requests if s not in shared))

    # -- agreement -----------------------------------------------------
    # Every rank in the group must execute the SAME collectives in the SAME
    # order, whatever fails. An exception that skips one leaves its peers
    # blocked in a collective that never completes -- and because MPI pairs
    # collectives by call order rather than by operation, a rank that reaches
    # a different one gives undefined results before hanging.
    #
    # So the protocol is fixed: bcast the manifest (or a failure marker),
    # then exactly one agreement step carrying both failure kinds. Nothing
    # between them may raise.

    CONTENT, CAPABILITY = 0, 1          # agreement vector slots

    def _reduce_flags(self, content, capability):
        """One collective carrying both failure kinds, MAX over the group.

        Bounded: this is the agreement step, which is exactly where a rank
        taking a different path leaves its peers waiting.
        """
        from mpi4py import MPI
        import numpy as _np

        from .gpu_collectives import bounded_allreduce_buffer
        local = _np.array([1 if content else 0, 1 if capability else 0],
                          dtype=_np.int32)
        total = _np.zeros_like(local)
        bounded_allreduce_buffer(self._comm, local, total, MPI.MAX,
                                 'shared-constants/settle')
        return bool(total[self.CONTENT]), bool(total[self.CAPABILITY])

    def _settle(self, content_errors, capability_error, *, hosts):
        """Decide the group's outcome from one agreement step.

        Content disagreement is fatal everywhere; a capability failure
        degrades the whole group to private copies. Both are decided
        collectively so no rank acts on its own view.
        """
        any_content, any_capability = self._reduce_flags(
            bool(content_errors), bool(capability_error))
        if any_content:
            if content_errors:
                raise SharedConstantsError('; '.join(content_errors))
            raise SharedConstantsError(
                'a peer rank rejected the shared constants; aborting this '
                'rank so the job does not hang')
        if any_capability:
            self._fall_back_to_private(hosts)
            return False
        return True

    def _exchange(self, hosts):
        """Publish or import the shared intersection. Never raises.

        One entry point for both roles, so a caller cannot reach agreement
        through only one of them -- the Case C bug was exactly that: a path
        that called the role methods and then forgot ``_settle``.

        Always returns ``(content_errors, capability_error)`` and must always
        be followed by ``_settle``.
        """
        if self._is_owner:
            return self._publish(hosts)
        return self._subscribe(hosts)

    def _release_shared(self):
        """Importers close, barrier, then the owner frees.

        CUDA requires imported mappings to be closed before the exporter
        deallocates. The order appeared separately in ``close``,
        ``_reestablish`` and ``_fall_back_to_private``; keeping it in one place
        means a fourth caller cannot get it wrong.
        """
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
        if self._recorder is not None:
            self._recorder.check_agreement(f'shared-constants/{step}',
                                           logger=logger)

    def _establish(self, hosts):
        from .gpu_allocation import upload_owned
        if self._cp is None:
            import cupy as cp
            self._cp = cp

        self._shared, self._private = self._intersect()

        # Resize the budget BEFORE anything is allocated. The intersection is
        # device overhead counted once, so the owner's limit has to include it
        # and every rank's share is computed net of it -- but both were applied
        # only after refresh() returned, by which time the owner had already
        # tried to allocate against limit = usable/peers. A 12 GiB intersection
        # on a 40 GiB four-peer device fits the documented accounting (owner
        # 7 + 12 = 19 GiB) yet was charged against 10 GiB: the shared copy was
        # refused, the group degraded, and the private copy was then refused
        # by the same limit. The size is knowable here -- the selectors come
        # from _intersect and the arrays from `hosts` -- so no allocation is
        # needed to learn it.
        #
        # Sizes are taken from this rank's own `hosts`. The shared SET is
        # identical on every rank by construction; a shape or dtype
        # disagreement aborts the whole group in _settle moments later.
        if self._sizing is not None:
            self._sizing(sum(int(np.ascontiguousarray(hosts[s]).nbytes)
                             for s in self._shared if s in hosts))

        if self._private:
            arrays = upload_owned(self._cp, [hosts[s] for s in self._private],
                                  self.budget, category='task-constants')
            self._device.update(zip(self._private, arrays))
        for selector in self._private:
            self._remember(selector, hosts[selector])

        if not self._shared:
            self._note_sharing('off')
            return
        if self._settle(*self._exchange(hosts), hosts=hosts):
            self._note_sharing('on')

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
                # free it or return its budget charge -- and the fallback then
                # uploads privately, making the owner pay twice. `block` is
                # None when the constructor itself raised.
                if block is not None:
                    block.release()
                break
            self._owned[selector] = block
            prepared[selector] = array
            manifest.append({
                'selector': selector, 'handle': handle,
                'shape': tuple(source.shape), 'dtype': source.dtype.str,
                'nbytes': int(source.nbytes), 'offset': 0,
                'generation': self._generation,
                'digest': digest_of(source)})

        # Publish unconditionally. An empty manifest tells followers that the
        # owner failed, so they skip importing and reach agreement with us.
        self._comm.bcast([] if capability else manifest, root=0)
        if capability:
            self._ipc_error = capability
            return [], capability
        for selector, array in prepared.items():
            self._device[selector] = array
            # Snapshot, not alias: an in-place edit of the caller's array must
            # remain detectable at the next transition.
            self._remember(selector, hosts[selector])
        return [], None

    def _subscribe(self, hosts):
        """Follower: validate the manifest, then import non-owning views.

        Returns (content_errors, capability_error) and NEVER raises, so this
        rank always reaches the single agreement step its peers also reach.
        """
        manifest = self._comm.bcast(None, root=0)
        if not manifest:
            # The owner failed before publishing. Agree, then fall back.
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
            try:
                block = _ImportedBlock(self._cp, entry['handle'],
                                       entry['nbytes'])
                view = block.view(tuple(entry['shape']),
                                  np.dtype(entry['dtype']),
                                  offset=entry['offset'])
            except Exception as exc:                      # noqa: BLE001
                capability = f'{type(exc).__name__}: {exc}'
                break
            self._imported[selector] = block
            self._device[selector] = view
            self._remember(selector, mine)
        if capability:
            self._ipc_error = capability
        return errors, capability

    def _note_sharing(self, state):
        """Record the sharing state on the placement, for the log line."""
        if self._placement is not None:
            self._placement.sharing = state

    def _fall_back_to_private(self, hosts):
        """Abandon sharing and upload every selector privately.

        Reached when an IPC call failed on any peer. Importers close first and
        the owner frees only after a barrier: CUDA requires imported mappings
        to be closed before the exporter deallocates, and a follower may have
        opened some handles before the failure.

        Reported at WARNING: the job keeps running, but every peer now holds
        its own copy of the constants, so a device that held one copy holds n.
        Without this the only trace is shared_bytes dropping to zero.
        """
        reason = self._ipc_error or 'a peer could not use CUDA IPC'
        peers = getattr(self._placement, 'n_device_peers', 1)
        wanted = sum(int(np.ascontiguousarray(hosts[s]).nbytes)
                     for s in self._shared if s in hosts)
        logger.warning(
            'GPU constant sharing fell back to private copies: %s. '
            '%d peers on this device will each hold %d bytes (%.1f MiB) '
            'of constants: %d bytes resident instead of %d.',
            reason, peers, wanted, wanted / 1024 ** 2,
            wanted * peers, wanted)
        self._note_sharing(f'fallback({reason})')
        self._release_shared()
        self._private = self.requests
        self._shared = ()
        for selector in self.requests:
            self._device.pop(selector, None)
            self._host.pop(selector, None)
            self._digests.pop(selector, None)
        self._upload_private(hosts, list(self._private))

    def _refresh_in_place(self, hosts, changed):
        """Case B: same layout, new values. Imported views stay valid.

        The owner overwrites memory its peers are mapped into, so every peer
        must have drained its own work first. before_upload() drains only the
        calling rank, hence the barrier before the write.
        """
        # Agreeing on the case is NOT agreeing on the content: if the owner
        # and a follower both change to different values, both propose B and
        # the cases match. Without this check the follower would silently read
        # the owner's values under its own selector name.
        #
        # Only the changed selectors need comparing, and _scan already hashed
        # each one's new value, so the digests are reused rather than recomputed.
        published = self._comm.bcast(
            {sel: self._new_digests[sel] for sel in changed}
            if self._is_owner else None, root=0)
        mismatched = [
            f'{sel!r}: peers changed to different values at this transition; '
            'sharing would substitute the owner\'s for this rank\'s'
            for sel in changed
            if (published or {}).get(sel) != self._new_digests.get(sel)]
        self._settle(mismatched, None, hosts=hosts)

        self._barrier('case-B-drained')   # every peer has drained its own work
        if self._is_owner:
            for selector in changed:
                self._device[selector].set(hosts[selector])
            self._sync_device()
        # Publication point, then a second barrier so no peer reads values
        # from before the write while another has already moved on.
        # One barrier after the write: peers must not read the new values
        # until the owner's copy has completed.
        self._barrier('case-B-published')
        for selector in self._shared:
            self._remember(selector, hosts[selector])
        self._refresh_private(hosts)

    def _reestablish(self, hosts):
        """Case C: layout changed, so the allocation must be replaced.

        Importers close first and acknowledge; only then may the owner free.
        CUDA requires imported mappings to be closed before the exporter
        deallocates, so this ordering is mandatory rather than tidy.
        """
        self._release_shared()
        for selector in self._shared:
            self._device.pop(selector, None)
            self._host.pop(selector, None)
        # Same agreement as _establish, through the same entry point: without
        # it an owner failure leaves followers with no device arrays for the
        # shared selectors, so the next get() fails far from the cause.
        self._settle(*self._exchange(hosts), hosts=hosts)
        self._refresh_private(hosts)

    def _snapshot_digest(self, selector, array):
        """Digest of the retained snapshot, hashed once per replacement.

        Re-hashing both sides each transition costs about a second per GiB;
        the snapshot only changes when this class replaces it.
        """
        cached = self._digests.get(selector)
        if cached is None:
            cached = digest_of(array)
            self._digests[selector] = cached
        return cached

    def _remember(self, selector, array):
        """Retain a snapshot and invalidate its cached digest."""
        self._host[selector] = array.copy(order='C')
        self._digests.pop(selector, None)

    def _changed_private(self, hosts):
        """Private selectors whose host value differs from the snapshot."""
        return [s for s in self._private
                if self._host.get(s) is None
                or self._host[s].shape != hosts[s].shape
                or self._host[s].dtype != hosts[s].dtype
                or self._snapshot_digest(s, self._host[s]) != digest_of(hosts[s])]

    def _upload_private(self, hosts, moved):
        from .gpu_allocation import upload_owned
        arrays = upload_owned(self._cp, [hosts[s] for s in moved],
                              self.budget, category='task-constants')
        self._device.update(zip(moved, arrays))
        for selector in moved:
            self._remember(selector, hosts[selector])

    def _refresh_private(self, hosts):
        moved = self._changed_private(hosts)
        if moved:
            self._upload_private(hosts, moved)

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
        if not self._sharing:
            self._device.clear()
            self._host.clear()
            return
        self._release_shared()
        self._check_order('close')
        self._device.clear()
        self._host.clear()
