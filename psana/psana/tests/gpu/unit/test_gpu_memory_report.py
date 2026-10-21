"""The memory snapshot attributes what it can and leaves no silent residual.

`retained` is a residual -- committed minus every identified category -- so it
is the figure to watch for a leak. A category the subtraction forgets shows up
there as unattributed memory, which is indistinguishable from the thing the
field exists to reveal. Nothing exercised `_snapshot_memory` before this file,
which is how task constants came to inflate it.

No GPU: the budget's allocation ledger is pure accounting, and `owned_empty`
takes an array module, so NumPy stands in for CuPy throughout.
"""
from types import SimpleNamespace as NS

import numpy as np

from psana.gpu.gpu_allocation import owned_empty
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_events import GpuEventManager, _GpuMemStats


MIB = 1024 ** 2


def manager(budget=None, *, constants=None, reader_bytes=0):
    """A manager with only what _snapshot_memory reaches out to.

    Built with __new__ rather than through the constructor: the snapshot is a
    pure read over components, and standing up a real GpuEventManager would
    need a DataSource, configs and a device.
    """
    m = GpuEventManager.__new__(GpuEventManager)
    m.input_preparers = {}
    m.event_pool = None
    m.gpu_xtc_parser = None
    m._output_d2h = None
    m._gpu_budget = budget
    m._task_constants = constants
    m.gpu_reader = (
        NS(memory_bytes=lambda: {'raw_input_slots': reader_bytes})
        if reader_bytes else None)
    return m


def test_owned_constants_are_attributed_not_left_as_a_residual():
    """The defect: 'task-constants' was absent from the subtraction.

    An owner charged for a constant set saw it in `retained`, which reads as
    owned storage the snapshot cannot account for -- so a real leak of the
    same size would have looked identical to normal operation.
    """
    budget = _GpuBudget(64 * MIB)
    constants = owned_empty(np, 12 * MIB, np.uint8, budget, 'task-constants')

    stats = manager(budget)._snapshot_memory('owner')

    assert stats.committed == 12 * MIB
    assert stats.constants == 12 * MIB, 'the owned constants were not reported'
    assert stats.retained == 0, (
        f'{stats.retained / MIB:.0f} MiB of constants landed in `retained`, '
        'where it is indistinguishable from unattributed memory')
    del constants


def test_a_genuine_residual_still_reaches_retained():
    """The subtraction must not become a way to hide everything.

    A category the snapshot does NOT identify has to keep showing up, or the
    field stops being a leak signal. 'detector' is such a category here.
    """
    budget = _GpuBudget(64 * MIB)
    unattributed = owned_empty(np, 5 * MIB, np.uint8, budget, 'detector')

    stats = manager(budget)._snapshot_memory('residual')

    assert stats.constants == 0
    assert stats.retained == 5 * MIB, (
        'an unidentified category no longer reaches `retained`, so a leak '
        'there would be invisible')
    del unattributed


def test_constants_are_separated_from_other_identified_categories():
    """Both subtractions apply, and to the right figures."""
    budget = _GpuBudget(128 * MIB)
    held_constants = owned_empty(np, 12 * MIB, np.uint8, budget, 'task-constants')
    held_reader = owned_empty(np, 8 * MIB, np.uint8, budget, 'reader')
    held_other = owned_empty(np, 3 * MIB, np.uint8, budget, 'detector')

    stats = manager(budget, reader_bytes=8 * MIB)._snapshot_memory('mixed')

    assert stats.committed == 23 * MIB
    assert stats.constants == 12 * MIB
    assert stats.raw_input == 8 * MIB
    # Only the detector bytes are unidentified.
    assert stats.retained == 3 * MIB
    del held_constants, held_reader, held_other


def test_borrowed_reports_what_a_rank_reads_without_owning():
    """`borrowed` was declared and logged but never assigned, so it printed
    0.0 MiB always -- including for an importer reading a full constant set
    over CUDA IPC, which is the capability whose loss the metric originally
    exposed (#168, PR #147 review).

    Sourced from the constants object rather than the budget, because an
    importer is charged nothing: the bytes are real and readable but owned by
    a peer.
    """
    budget = _GpuBudget(64 * MIB)
    importer = NS(imported_bytes=384 * MIB)

    stats = manager(budget, constants=importer)._snapshot_memory('follower')

    assert stats.borrowed == 384 * MIB
    # Not charged, so it must not disturb the owned-storage figures.
    assert stats.committed == 0
    assert stats.constants == 0
    assert stats.retained == 0, \
        'imported bytes were treated as owned storage'


def test_a_sharing_owners_un_pooled_block_is_attributed_too():
    """The ledger alone misses the rank holding the physical shared copy.

    ``ipcGetMemHandle`` needs a pointer from a bare ``cudaMalloc``, so an
    IPC-sharing owner allocates outside CuPy's pool and charges the budget
    with ``reserve()`` directly. That raises ``committed`` but creates no
    ledger record -- so summing only 'task-constants' allocations reports
    constants=0 for the owner and lets ``retained`` absorb the whole set,
    which is the original defect reappearing on the sharing path.

    Simulated with reserve(), which is exactly what _OwnedBlock does.
    """
    budget = _GpuBudget(512 * MIB)
    budget.reserve(384 * MIB)              # what _OwnedBlock.__init__ does
    # No _is_owner: the role is inferred from the byte counts. An owner
    # imports nothing, so shared_bytes - imported_bytes is what it owns.
    owner = NS(shared_bytes=384 * MIB, imported_bytes=0)

    stats = manager(budget, constants=owner)._snapshot_memory('ipc-owner')

    assert stats.committed == 384 * MIB
    assert not stats.allocations, 'the un-pooled block must leave no ledger record'
    assert stats.constants == 384 * MIB, (
        'the shared copy is charged to this rank but was not attributed; '
        'a ledger-only sum reports 0 for the owner holding it')
    assert stats.retained == 0, (
        f'{stats.retained / MIB:.0f} MiB of shared constants landed in '
        '`retained`, where it is indistinguishable from a leak')
    assert stats.borrowed == 0, 'an owner borrows nothing'


def test_a_follower_does_not_claim_the_bytes_it_only_imports():
    """`shared_bytes` means different things to the two roles.

    On an owner it is the physical copy it allocated; on a follower it is the
    size of the mapping it imported, which it does not own and is not charged
    for. Treating a follower's figure as owned would subtract bytes from
    `retained` that were never in `committed` -- understating the residual by
    the whole constant set and hiding a leak of that size.
    """
    budget = _GpuBudget(512 * MIB)
    held = owned_empty(np, 8 * MIB, np.uint8, budget, 'detector')
    # A follower's shared_bytes IS its imported_bytes, so the difference is
    # zero -- no private attribute consulted.
    follower = NS(shared_bytes=384 * MIB, imported_bytes=384 * MIB)

    stats = manager(budget, constants=follower)._snapshot_memory('follower')

    assert stats.borrowed == 384 * MIB, 'the mapping is what it borrows'
    assert stats.constants == 0, \
        'a follower owns no constants; it must not claim the imported bytes'
    # The unrelated detector bytes are still the residual, undisturbed.
    assert stats.retained == 8 * MIB, (
        'the imported bytes were subtracted from `retained`, which hides '
        f'{(8 * MIB - stats.retained) / MIB:.0f} MiB of real residual')
    del held


def test_the_role_is_inferred_from_public_byte_counts_only():
    """No private attribute of the constants object is consulted.

    The owned figure is ``shared_bytes - imported_bytes``, which is correct
    for either role: an owner imports nothing, and a follower's shared_bytes
    equals its imported_bytes. Reading a private ``_is_owner`` would couple
    this to the internals of a class that does not exist on this branch at
    all -- SharedRequestedConstants lives on the IPC branch -- so a rename
    there would silently zero the figure here.

    Asserted by handing over an object that RAISES on any private access.
    """
    class PublicOnly:
        shared_bytes = 384 * MIB
        imported_bytes = 0

        def __getattr__(self, name):
            raise AssertionError(
                f'the snapshot read a private attribute {name!r}; it must '
                'work from the public byte counts alone')

    budget = _GpuBudget(512 * MIB)
    budget.reserve(384 * MIB)

    stats = manager(budget, constants=PublicOnly())._snapshot_memory('public')

    assert stats.constants == 384 * MIB
    assert stats.retained == 0


def test_a_double_counted_category_is_logged_rather_than_clamped_away():
    """`retained` is clamped at 0, so over-subtraction is invisible.

    A negative raw residual means a category was counted twice -- the live
    risk being a constants allocation appearing in both the ledger and
    shared_bytes. The clamp keeps the figure sane but destroys the evidence,
    so the raw value goes to DEBUG.
    """
    budget = _GpuBudget(512 * MIB)
    # Charged once, but reported through BOTH sources: 12 MiB in the ledger
    # and 12 MiB of shared_bytes, against 12 MiB committed.
    held = owned_empty(np, 12 * MIB, np.uint8, budget, 'task-constants')
    liar = NS(shared_bytes=12 * MIB, imported_bytes=0)

    records = []
    import psana.gpu.gpu_events as ge
    real = ge._log.debug
    ge._log.debug = lambda fmt, *a: records.append(fmt % a)
    try:
        stats = manager(budget, constants=liar)._snapshot_memory('double')
    finally:
        ge._log.debug = real

    assert stats.constants == 24 * MIB      # the double count
    assert stats.retained == 0              # clamped, as before
    assert records, (
        'the residual went negative and nothing said so; the clamp hid a '
        'category counted twice, which understates `retained`')
    assert 'counted twice' in records[0], records[0]


def test_an_ordinary_residual_logs_nothing():
    """The debug line must fire on over-subtraction only, or it is noise on
    every healthy snapshot."""
    budget = _GpuBudget(512 * MIB)
    held = owned_empty(np, 5 * MIB, np.uint8, budget, 'detector')

    records = []
    import psana.gpu.gpu_events as ge
    real = ge._log.debug
    ge._log.debug = lambda fmt, *a: records.append(fmt % a)
    try:
        stats = manager(budget)._snapshot_memory('healthy')
    finally:
        ge._log.debug = real

    assert stats.retained == 5 * MIB
    assert not records, f'logged on a healthy snapshot: {records}'
    del held


def test_borrowed_is_zero_without_an_importer():
    """The ordinary case, and the only one on a branch with no sharing: a
    constants object that owns its copy borrows nothing."""
    budget = _GpuBudget(64 * MIB)
    held = owned_empty(np, 12 * MIB, np.uint8, budget, 'task-constants')
    # A RequestedConstants has no imported_bytes attribute at all.
    owner = NS()

    stats = manager(budget, constants=owner)._snapshot_memory('solo')

    assert stats.borrowed == 0
    assert stats.constants == 12 * MIB, \
        'an owner must still report its physical copy'
    del held


def test_the_snapshot_survives_a_manager_with_no_constants():
    """Setup logs a snapshot before _task_constants is built, and a CPU-only
    or no-task run never builds one. Reading the field must not raise."""
    stats = manager(_GpuBudget(MIB))._snapshot_memory('early')
    assert stats.borrowed == 0 and stats.constants == 0


def test_both_figures_are_reported_in_the_log_line():
    """A value the snapshot computes but no line prints is how `borrowed`
    went unnoticed, so assert they reach the log rather than only the
    dataclass."""
    emitted = []
    stats = _GpuMemStats(label='x', committed=20 * MIB,
                         constants=12 * MIB, borrowed=384 * MIB)

    import psana.gpu.gpu_events as ge
    real = ge._log.info
    ge._log.info = lambda fmt, *a: emitted.append(fmt % a)
    try:
        stats.log()
    finally:
        ge._log.info = real

    ownership = emitted[0]
    assert 'constants=12.0 MiB' in ownership, ownership
    assert 'borrowed=384.0 MiB' in ownership, ownership


def test_high_water_tracks_and_prints_both():
    """Peaks are what a user reads after a run. `constants` and `borrowed`
    were tracked in the high-water dict but absent from its log line."""
    emitted = []
    m = manager(_GpuBudget(64 * MIB))
    m._high_water = {'constants': 12 * MIB, 'borrowed': 384 * MIB}

    import psana.gpu.gpu_events as ge
    real = ge._log.info
    ge._log.info = lambda fmt, *a: emitted.append(fmt % a)
    try:
        m.log_high_water()
    finally:
        ge._log.info = real

    line = emitted[0]
    assert 'constants=12.0 MiB' in line, line
    assert 'borrowed=384.0 MiB' in line, line
