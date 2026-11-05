"""Bit-exact packed loads at every base/offset residue and allocation boundary."""
import numpy as np
import pytest

from psana.gpu import gpu_detector as gd
from psana.gpu.gpudgram import parser as p
from test_batched_gather import _gpu_available

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason="no CUDA device available")]


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("pixels", [1, 255, 256, 257, 300])
def test_all_address_residues_exact_tail_bits_and_invalid_reuse(dtype, pixels):
    import cupy as cp

    itemsize = np.dtype(dtype).itemsize
    word_type = np.uint16 if dtype == np.uint16 else np.uint32
    # Include gain bits, signed zeros, subnormals, infinities and both quiet and
    # signaling NaN representations. Never compare float NaNs numerically.
    patterns = np.array([0, 1, 0x8000, 0xC123, 0xFFFF] if itemsize == 2 else [
        0, 0x80000000, 1, 0x80000001, 0x3F800000, 0x7F800000,
        0xFF800000, 0x7FC12345, 0x7F812345, 0xFFC54321, 0xFFFFFFFF,
    ], dtype=word_type)
    residues = [(base, offset) for base in range(itemsize)
                for offset in range(itemsize)]
    nsegments = len(residues)
    owners, locators, allocations, expected = [], [], [], []
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for i, (base, offset) in enumerate(residues):
            bits = np.resize(np.roll(patterns, i), pixels)
            payload = bits.view(np.uint8)
            # Bypass the pool: the final pixel ends at the actual CUDA
            # allocation boundary, so sanitizer can detect widened overreads.
            memory = cp.cuda.Memory(base + offset + payload.size)
            storage = cp.ndarray((memory.size,), dtype=cp.uint8,
                                 memptr=cp.cuda.MemoryPointer(memory, 0))
            data = storage[base:]
            data[offset:].set(payload, stream=stream)
            stream.synchronize()  # retain the fixture upload source until done
            assert data.data.ptr % itemsize == base
            assert (data.data.ptr + offset) % itemsize == (base + offset) % itemsize
            loc = np.zeros((1, p.LOC_NCOLS), np.uint64)
            loc[0, [p.LOC_STATUS, p.LOC_TYPE, p.LOC_RANK, p.LOC_OFFSET,
                    p.LOC_NBYTES, p.LOC_DIM0, p.LOC_DIM0 + 1]] = (
                        p.STATUS_FOUND, 1 if itemsize == 2 else 8, 2,
                        offset, payload.size, 1, pixels)
            device_loc = cp.asarray(loc)
            owners.append((data.data.ptr, data.nbytes, device_loc.data.ptr, 1, 1))
            locators.append(device_loc)
            allocations.append(storage)
            expected.append(bits)
        owner_table = cp.asarray(np.asarray(owners, np.uint64))
        plan = cp.asarray(np.asarray([(i, 0, 1 if itemsize == 2 else 8, 2)
                                     for i in range(nsegments)], np.uint64))
        rows = cp.asarray(np.asarray([(i, 0) for i in range(nsegments)], np.uint64))
        out = cp.empty((nsegments, pixels), dtype=dtype)
        present = cp.empty(nsegments, dtype=cp.uint8)
        tiles = (pixels + 255) // 256
        kernel = gd._batched_gather_kernel(np.dtype(dtype))

        def gather():
            kernel((tiles * nsegments,), (256,), (
                owner_table, np.uint64(nsegments), plan, rows,
                np.uint64(nsegments), np.uint64(nsegments), np.uint64(pixels),
                np.uint64(tiles), np.uint64(1), np.uint64(pixels), out, present),
                stream=stream)

        gather()
        stream.synchronize()
        np.testing.assert_array_equal(out.get().view(word_type), np.stack(expected))
        np.testing.assert_array_equal(present.get(), np.ones(nsegments, np.uint8))

        # Reuse the nonzero output: an absent owner, a field one byte beyond its
        # owner's extent, and a truncated field must become zero, not be loaded.
        rows[0, 0] = nsegments
        owner_table[1, 1] -= 1
        locators[2][0, p.LOC_NBYTES] -= 1
        gather()
        stream.synchronize()
        reference = np.stack(expected)
        reference[:3] = 0
        np.testing.assert_array_equal(out.get().view(word_type), reference)
        np.testing.assert_array_equal(present.get(),
                                      np.array([0, 0, 0] + [1] * (nsegments - 3), np.uint8))
