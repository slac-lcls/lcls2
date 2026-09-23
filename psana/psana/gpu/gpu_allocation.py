"""Allocation-backed charges for pipeline device storage.

The returned CuPy array uses UnownedMemory to retain an acyclic owner of the
original pooled array and its charge. All aliases of that memory keep both
alive. CUDA completion/reuse remains the caller's lease responsibility.
"""
import math
import weakref

import numpy as np


def allocation_capacity(cp, requested):
    """Capacity of the supported default CuPy pool (512-byte blocks).

    NumPy-backed injected CPU test allocators have no pool rounding. Custom CUDA
    allocators need a capacity contract and are rejected before allocation.
    """
    requested = int(requested)
    if requested < 0:
        raise ValueError('negative allocation size')
    if hasattr(cp, 'get_default_memory_pool'):
        if cp.cuda.get_allocator() != cp.get_default_memory_pool().malloc:
            raise ValueError('budgeted GPU storage requires the default CuPy memory pool')
        return ((requested + 511) // 512) * 512
    return requested


def allocation_requirement(cp, requested, existing):
    """Full replacement cost, even when growth stays in the same rounded block.

    The existing generation is already committed. A zero old-size component
    makes allocation_growth_bytes reserve the entire replacement; reuse costs
    zero.
    """
    if existing is not None and int(existing.nbytes) >= int(requested):
        return (0, 0)
    return (allocation_capacity(cp, requested), 0)


def backing_capacity(array):
    """Pool block capacity for a device array; logical bytes for CPU fixtures."""
    memory = getattr(getattr(array, 'data', None), 'mem', None)
    return int(memory.size if memory is not None else array.nbytes)


class _AllocationOwner:
    def __init__(self, backing, charge):
        self.backing = backing
        self.charge = charge

    def __del__(self):
        # No synchronization or CUDA submission here. Callers must retain an
        # array until all work using its pointer has completed.
        self.backing = None
        self.charge.release()


def owned_empty(cp, shape, dtype, budget, category):
    """Allocate with one charge that survives cache replacement and trimming."""
    if budget is None:
        return cp.empty(shape, dtype=dtype)
    dimensions = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    if any(int(n) < 0 for n in dimensions):
        raise ValueError('negative allocation dimension')
    requested = math.prod(int(n) for n in dimensions) * np.dtype(dtype).itemsize
    capacity = allocation_capacity(cp, requested)
    charge = budget.reserve_allocation(capacity, requested=requested, category=category)
    backing = owner = memory = pointer = result = None
    try:
        backing = cp.empty(shape, dtype=dtype)
        if isinstance(backing, np.ndarray):
            # CPU fault-injection backend only. NumPy views retain this owning
            # root array; no array is retained by the finalizer's charge token.
            weakref.finalize(backing, charge.release)
            charge.commit()
            return backing
        actual = int(backing.data.mem.size)
        if actual != capacity:
            raise RuntimeError(f'CuPy allocation capacity changed: expected {capacity}, got {actual}')
        owner = _AllocationOwner(backing, charge)
        memory = cp.cuda.UnownedMemory(backing.data.ptr, capacity, owner,
                                      device_id=backing.device.id)
        pointer = cp.cuda.MemoryPointer(memory, 0)
        result = cp.ndarray(shape, dtype=dtype, memptr=pointer)
        charge.commit()
        return result
    except BaseException:
        # No asynchronous work is launched by this helper. Dispose the backing
        # before rollback, and prevent the owner destructor from double release.
        if owner is not None:
            owner.backing = None
        backing = None
        result = pointer = memory = owner = None
        charge.rollback()
        raise


def upload_owned(cp, arrays, budget, category='fixed'):
    """Upload setup tables, retaining all submitted work on a failed drain.

    Setup synchronizes its own stream before releasing host upload sources.
    This is not a device-wide or event-loop synchronization.
    """
    if budget is None:
        return tuple(cp.asarray(a) for a in arrays)
    arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    stream = cp.cuda.get_current_stream()
    hold = budget.hold(sum(allocation_capacity(cp, a.nbytes) for a in arrays))
    uploaded = []
    try:
        with hold:
            for source in arrays:
                target = owned_empty(cp, source.shape, source.dtype, budget, category)
                uploaded.append(target)  # retain before the asynchronous call
                if isinstance(target, np.ndarray):
                    np.copyto(target, source)
                else:
                    target.set(source, stream=stream)
        stream.synchronize()
        return tuple(uploaded)
    except BaseException:
        try:
            stream.synchronize()
        except BaseException:
            budget.quarantine_upload(stream, uploaded, arrays)
            raise
        target = None
        uploaded.clear()
        raise
    finally:
        hold.close()
