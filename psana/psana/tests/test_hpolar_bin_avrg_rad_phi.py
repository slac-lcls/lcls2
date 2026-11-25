import numpy as np
from psana.pyalgos.generic.HPolar import HPolar, divide_protected


def _xy(H=200, W=220, cx=None, cy=None):
    """synthetic pixel x/y coordinate arrays, roughly centered on the array"""
    y, x = np.mgrid[0:H, 0:W]
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    return (x - cx).astype(float), (y - cy).astype(float)


def _reference_bin_avrg_rad_phi(hp, nda, do_transp=True):
    """Brute-force reference matching the pre-fix implementation: bincount over every
    pixel in the full array (not just the in-ROI subset), then drop the overflow bin.
    """
    flat = nda.ravel()
    num_full = np.bincount(hp.iseq, weights=flat, minlength=hp.ntbins + 1)
    den_full = hp.npix_per_bin
    arr_rphi = divide_protected(num_full, den_full, vsub_zero=0)[:-1]
    arr_rphi.shape = (hp.pb.nbins(), hp.rb.nbins())
    return np.transpose(arr_rphi) if do_transp else arr_rphi


def test_bin_avrg_rad_phi_matches_full_bincount_multi_bin():
    """The in-ROI-only fast path in bin_avrg_rad_phi must give bit-identical results to
    bincounting every pixel in the full array and dropping the overflow bin -- it's the same
    bins, computed over a subset of entries that excludes only out-of-ROI pixels.
    """
    xarr, yarr = _xy()
    hp = HPolar(xarr, yarr, mask=None, radedges=(5, 80), nradbins=10, phiedges=(0, 360), nphibins=16)

    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        nda = (rng.random(xarr.shape) * 1000).astype(dtype)
        out = hp.bin_avrg_rad_phi(nda, do_transp=True)
        ref = _reference_bin_avrg_rad_phi(hp, nda, do_transp=True)
        assert np.array_equal(out, ref)


def test_bin_avrg_rad_phi_matches_full_bincount_single_bin_roi():
    """Same check for a 1x1-bin annulus ROI (e.g. AMI's RoiArch node with nbins rad=1,
    nbins ang=1), the shape that made the full-array bincount a per-event bottleneck.
    """
    xarr, yarr = _xy(H=400, W=420, cx=200, cy=220)
    hp = HPolar(xarr, yarr, mask=None, radedges=(20, 160), nradbins=1, phiedges=(0, 343), nphibins=1)

    rng = np.random.default_rng(1)
    nda = rng.random(xarr.shape).astype(np.float32)
    out = hp.bin_avrg_rad_phi(nda)
    ref = _reference_bin_avrg_rad_phi(hp, nda)
    assert np.array_equal(out, ref)


def test_bin_avrg_rad_phi_matches_full_bincount_with_mask_and_no_transp():
    xarr, yarr = _xy()
    rng = np.random.default_rng(2)
    mask = (rng.random(xarr.shape) > 0.3).astype(np.uint8)
    hp = HPolar(xarr, yarr, mask=mask, radedges=(5, 80), nradbins=6, phiedges=(0, 360), nphibins=8)

    nda = rng.random(xarr.shape).astype(np.float64)
    out = hp.bin_avrg_rad_phi(nda, do_transp=False)
    ref = _reference_bin_avrg_rad_phi(hp, nda, do_transp=False)
    assert np.array_equal(out, ref)
