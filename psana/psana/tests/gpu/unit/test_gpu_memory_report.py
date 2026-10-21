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
