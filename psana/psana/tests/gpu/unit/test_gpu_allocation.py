"""Allocation tokens preserve retained capacity and reservation provenance."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_allocation import owned_empty, allocation_requirement
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError


def test_aliases_keep_one_charge_until_last_view_dies():
    budget = _GpuBudget(100)
    root = owned_empty(np, 40, np.uint8, budget, 'input')
    view = root[4:20].reshape(4, 4)
    del root
    assert budget.committed() == 40
    assert len(budget.allocation_snapshot()) == 1
    del view
    assert budget.committed() == 0 and budget.allocation_snapshot() == ()


def test_release_does_not_credit_an_unrelated_active_hold():
    budget = _GpuBudget(100)
    root = owned_empty(np, 40, np.uint8, budget, 'input')
    hold = budget.hold(30)
    with hold:
        del root
        assert budget.committed() == 0
        assert hold.remaining == budget._held == 30
    hold.close()
    assert budget.available() == 100


def test_allocation_failure_returns_credit_to_originating_hold():
    budget = _GpuBudget(100)
    hold = budget.hold(80)
    def fail(*args, **kwargs):
        assert budget.committed() == 40 and hold.remaining == 40
        raise MemoryError('injected')
    with hold, pytest.raises(MemoryError, match='injected'):
        owned_empty(NS(empty=fail), 40, np.uint8, budget, 'input')
    assert budget.committed() == 0 and hold.remaining == budget._held == 80
    assert budget.allocation_snapshot() == ()
    hold.close()


def test_retained_generation_prevents_overcommit_before_allocator():
    budget = _GpuBudget(100)
    old = owned_empty(np, 40, np.uint8, budget, 'input')
    calls = []
    with pytest.raises(GpuMemoryPressureError):
        owned_empty(NS(empty=lambda *a, **k: calls.append(True)), 80, np.uint8, budget, 'input')
    assert not calls and budget.committed() == 40
    del old
    new = owned_empty(np, 80, np.uint8, budget, 'input')
    assert budget.committed() == 80
    del new
    assert budget.committed() == 0


def test_later_release_and_failed_rollback_are_idempotent():
    budget = _GpuBudget(100)
    charge = budget.reserve_allocation(40, requested=40, category='input')
    charge.commit()
    charge.release()
    charge.release()
    assert budget.committed() == 0
    with pytest.raises(RuntimeError, match='published'):
        charge.rollback()


def test_requirements_reserve_full_replacement_and_nothing_for_reuse():
    existing = np.empty(40, np.uint8)
    assert allocation_requirement(np, 30, existing) == (0, 0)
    assert allocation_requirement(np, 80, existing) == (80, 0)


def test_combined_locations_retire_without_retaining_storage():
    from psana.gpu.gpudgram.parser import GpuEventBatch
    budget = _GpuBudget(4096)
    batch = GpuEventBatch.__new__(GpuEventBatch)
    batch._configured_backing = owned_empty(np, (2, 3, 11), np.uint64, budget, 'parser')
    batch._configured_indices = {}
    batch._configured_ready = object()
    batch._locators = {}
    descriptor = batch.configured_locations()
    alias = descriptor.backing[:, :1]
    charged = budget.committed()
    batch.retire()
    assert budget.committed() == charged  # escaped array still owns its charge
    with pytest.raises(RuntimeError, match='released'):
        descriptor.backing
    with pytest.raises(RuntimeError, match='released'):
        batch.configured_locations()
    del alias
    assert budget.committed() == 0  # saved batch/descriptor do not retain backing
