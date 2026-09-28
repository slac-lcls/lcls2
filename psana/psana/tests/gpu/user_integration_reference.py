"""Independent float64 CPU histogram reference and explicit count policy."""
import numpy as np


def integrate(image, raw, present, segments, constants, bin_ids, nbins,
              *, status_bits=0, stextra_bits=0):
    answer = np.zeros((len(raw), 3, nbins), np.float64)
    bins = bin_ids[list(segments)]
    good = np.ones(bins.shape, bool)
    for key, bits in (('pixel_status', status_bits), ('status_extra', stextra_bits)):
        if bits:
            good &= np.all((constants[key][:, segments].astype(np.uint64) & np.uint64(bits)) == 0, axis=0)
    for e in range(len(raw)):
        bits = raw[e] >> 14
        gain = np.zeros(bins.shape, constants['pixel_gain'].dtype)
        for mode, code in enumerate((0, 1, 3)):
            take = bits == code
            gain[take] = constants['pixel_gain'][mode, segments][take]
        valid = ((bits != 2) & present[e, :, None, None].astype(bool) & good &
                 (gain != 0) & np.isfinite(gain) & np.isfinite(image[e]) & (bins >= 0))
        counts = np.bincount(bins[valid], minlength=nbins)
        sums = np.bincount(bins[valid], weights=image[e][valid].astype(np.float64), minlength=nbins)
        answer[e, 1], answer[e, 2] = sums, counts
        np.divide(sums, counts, out=answer[e, 0], where=counts != 0)
    return answer
