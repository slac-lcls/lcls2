import numpy as np
from psana.detector.UtilsAreaDetector import copy_plan_from_flat_index, img_from_copy_plan,\
    img_from_pixel_arrays, image_shape, flat_index_from_pixel_arrays, img_from_flat_index


def _geometry(nseg=4, R=64, C=128, T=32):
    """synthetic multi-segment geometry with ASIC gaps and flipped segments"""
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
    rows = np.empty((nseg, R, C), dtype=np.uint)
    cols = np.empty_like(rows)
    for s in range(nseg):
        r0, c0 = (s // 2)*(R + 10), (s % 2)*(C + 10)
        if s % 2:
            rows[s] = r0 + (R - 1 - rr) + (R//T - 1 - rr//T)*2
            cols[s] = c0 + (C - 1 - cc) + (C//T - 1 - cc//T)*2
        else:
            rows[s] = r0 + rr + (rr//T)*2
            cols[s] = c0 + cc + (cc//T)*2
    return rows, cols


def test_flat_index_matches_pixel_arrays():
    rows, cols = _geometry()
    rc_tot_max = [int(rows.max()), int(cols.max())]
    shape = image_shape(rows, cols, rc_tot_max)
    fidx = flat_index_from_pixel_arrays(rows, cols, shape)
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64, np.uint16):
        w = (rng.random(rows.shape)*1000).astype(dtype)
        for vbase in (0, 7.5):
            ref = img_from_pixel_arrays(rows, cols, weight=w, vbase=vbase, rc_tot_max=rc_tot_max)
            new = img_from_flat_index(fidx, shape, w, vbase=vbase)
            assert new.shape == ref.shape and new.dtype == ref.dtype
            assert np.array_equal(new, ref)


def test_flat_index_subset_of_segments():
    rows, cols = _geometry()
    rc_tot_max = [int(rows.max()), int(cols.max())]
    r, c = rows[[0, 3]], cols[[0, 3]]
    shape = image_shape(r, c, rc_tot_max)
    w = np.arange(r.size, dtype=np.float32).reshape(r.shape)
    ref = img_from_pixel_arrays(r, c, weight=w, rc_tot_max=rc_tot_max)
    new = img_from_flat_index(flat_index_from_pixel_arrays(r, c, shape), shape, w)
    assert np.array_equal(new, ref)


def test_copy_plan_matches_pixel_arrays_with_skewed_segment():
    rows, cols = _geometry()
    cols[0] += (np.arange(cols.shape[2]) // 16).astype(cols.dtype)
    rc_tot_max = [int(rows.max()), int(cols.max())]
    shape = image_shape(rows, cols, rc_tot_max)
    fidx = flat_index_from_pixel_arrays(rows, cols, shape)
    plan = copy_plan_from_flat_index(fidx, rows.shape, shape)
    for dtype in (np.float32, np.float64, np.uint16):
        weight = np.arange(rows.size, dtype=dtype).reshape(rows.shape)
        for vbase in (0, 7.5):
            ref = img_from_pixel_arrays(rows, cols, weight=weight, vbase=vbase, rc_tot_max=rc_tot_max)
            new = img_from_copy_plan(plan, shape, weight, vbase=vbase)
            assert np.array_equal(new, ref)
