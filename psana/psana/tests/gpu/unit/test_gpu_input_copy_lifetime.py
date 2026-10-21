"""Field copies outlive cached views, open children, and failed completion."""
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

from psana.gpu.gpu_allocation import owned_empty
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_input import GpuFieldResult, InputSlotLease, _GpuFieldViewContext
from test_gpu_input_window import Token, window
from test_gpu_retirement import cuda  # noqa: F401 -- pytest fixture


def retain(parent, budget):
    array = owned_empty(np, 40, np.uint8, budget, 'field-copy')
    parent.retain_copy(array)
    return weakref.ref(array)


@pytest.mark.parametrize('parent_first', [False, True])
@pytest.mark.parametrize('copy_first', [False, True])
def test_copy_survives_cache_retirement_and_all_open_children(parent_first, copy_first):
    owner = window()
    parent = InputSlotLease(None, (owner,))
    budget = _GpuBudget(100)
    if copy_first:
        saved = retain(parent, budget)
    first, second = parent.acquire_view(), parent.acquire_view()
    if not copy_first:
        saved = retain(parent, budget)
    result = GpuFieldResult(NS(segment_ids=(0,)), NS(), parent)
    result._views = saved()
    owner.close()
    done = Token()
    first.register_consumer_done(done)
    first.wait_until_safe_to_reuse()
    if parent_first:
        parent.wait_until_safe_to_reuse()
        assert result._views is None
    assert saved() is not None and budget.committed() == 40
    second.register_consumer_done(Token())
    second.wait_until_safe_to_reuse()
    if not parent_first:
        assert budget.committed() == 40
        parent.wait_until_safe_to_reuse()
    assert saved() is None and budget.committed() == 0 and owner.released
    parent.wait_until_safe_to_reuse()
    first.wait_until_safe_to_reuse()
    second.wait_until_safe_to_reuse()
    assert parent._view_children == 0 and budget.committed() == 0


def test_nested_child_keeps_root_copy_after_intermediate_child_retires():
    owner = window()
    parent = InputSlotLease(None, (owner,))
    child = parent.acquire_view()
    grandchild = child.acquire_view()
    budget = _GpuBudget(100)
    saved = retain(child, budget)
    owner.close()
    parent.wait_until_safe_to_reuse()
    child.wait_until_safe_to_reuse()
    assert saved() is not None and budget.committed() == 40
    grandchild.wait_until_safe_to_reuse()
    assert saved() is None and budget.committed() == 0 and owner.released


@pytest.mark.parametrize('with_child', [False, True])
def test_failed_copy_completion_retains_charge_and_can_retry(with_child):
    owner = window()
    # Releasing a use normally transfers completion without waiting here.
    owner._defer_retirement = True
    owner.batch.walk_done.done = True
    parent = InputSlotLease(None, (owner,))
    consumer = parent.acquire_view() if with_child else parent
    budget = _GpuBudget(100)
    saved = retain(parent, budget)
    done = Token()
    done.done = False
    done.fail = True
    consumer.register_consumer_done(done)
    owner.close()
    if with_child:
        parent.wait_until_safe_to_reuse()
    with pytest.raises(RuntimeError, match='completion failure'):
        consumer.wait_until_safe_to_reuse()
    assert saved() is not None and budget.committed() == 40
    with pytest.raises(RuntimeError, match='retiring'):
        parent.retain_copy(object())
    done.fail = False
    consumer.wait_until_safe_to_reuse()
    assert saved() is None and budget.committed() == 0
    assert owner.drain()
    consumer.wait_until_safe_to_reuse()
    assert parent._view_children == 0


def test_copy_producer_keeps_input_until_completion():
    budget = _GpuBudget(100)
    completed = []

    def release():
        assert completed

    owner = window(release=release)
    parent = InputSlotLease(None, (owner,))
    saved = retain(parent, budget)

    class CopyDone:
        def synchronize(self):
            if not completed:
                assert owner.batch is not None and not owner.released
                completed.append(True)
            assert saved() is not None and budget.committed() == 40

    parent.register_consumer_done(CopyDone())
    owner.close()
    parent.wait_until_safe_to_reuse()
    assert owner.released and saved() is None and budget.committed() == 0


def test_ownerless_copy_retains_storage_until_consumer_completion():
    parent = InputSlotLease(None)
    budget = _GpuBudget(100)
    saved = retain(parent, budget)
    done = Token()
    done.fail = True
    parent.register_consumer_done(done)
    with pytest.raises(RuntimeError, match='completion failure'):
        parent.wait_until_safe_to_reuse()
    assert saved() is not None and budget.committed() == 40
    done.fail = False
    parent.wait_until_safe_to_reuse()
    assert saved() is None and budget.committed() == 0


def test_no_copy_adds_no_completion_waits_or_retention_lists():
    owner = window()
    planned = owner.acquire()
    ready, done = Token(), Token()
    parent = InputSlotLease(ready, (owner,))
    child = parent.acquire_view()
    child.register_consumer_done(done)
    owner.close()
    parent.wait_until_safe_to_reuse()
    child.wait_until_safe_to_reuse()
    assert ready.waits == done.waits == 0
    assert parent._retained_copies is parent._copy_done is None
    assert child._retained_copies is child._copy_done is None
    planned.wait_until_safe_to_reuse()
    assert ready.waits == done.waits == 1


@pytest.mark.parametrize('failure', ['entry', 'body', 'record', 'drain'])
def test_field_context_failure_preserves_copy_until_safe_cleanup(cuda, failure):
    owner = window()
    parent = InputSlotLease(None, (owner,))
    budget = _GpuBudget(100)
    saved = []

    def materialize(stream):
        saved.append(retain(parent, budget))
        if failure == 'entry':
            raise ValueError('entry failed')

    context = _GpuFieldViewContext(NS(_lease=parent, _slot_views=materialize), cuda)
    if failure == 'entry':
        with pytest.raises(ValueError, match='entry failed'):
            context.__enter__()
        assert owner.references == 1
        owner.close()
        parent.wait_until_safe_to_reuse()
    else:
        context.__enter__()
        owner.close()
        parent.wait_until_safe_to_reuse()
        assert saved[0]() is not None and budget.committed() == 40
        if failure == 'body':
            context.__exit__(ValueError, ValueError('body failed'), None)
        else:
            cuda.fail = True
            cuda.drain_fail = failure == 'drain'
            with pytest.raises(RuntimeError, match='drain failed' if cuda.drain_fail else 'record failed'):
                context.__exit__(None, None, None)
            if failure == 'drain':
                assert saved[0]() is not None and budget.committed() == 40
                cuda.fail = cuda.drain_fail = False
                context.__exit__(None, None, None)
    assert saved[0]() is None and budget.committed() == 0 and owner.released
    context.__exit__(None, None, None)
