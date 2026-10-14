"""Independent CPU array reference for the explicit Stage 5a user policies."""
import numpy as np


def calibrate(raw, present, segments, constants, *, use_offset=False, status_bits=0, stextra_bits=0):
    assert raw.ndim == 4 and present.shape == raw.shape[:2]
    peds = constants['pedestals'][:, segments]
    gain = constants['pixel_gain'][:, segments]
    if use_offset:
        peds = peds + constants['pixel_offset'][:, segments]
    peds = peds.astype(np.float32)
    inverse = np.divide(np.ones_like(constants['pedestals'][:, segments]),
                        np.where(gain != 0, gain, 1))
    inverse[gain == 0] = 0
    good = np.ones(raw.shape[1:], bool)
    for key, bits in (('pixel_status', status_bits), ('status_extra', stextra_bits)):
        if bits:
            flags = constants[key][:, segments].astype(np.uint64)
            good &= np.all((flags & np.uint64(bits)) == 0, axis=0)
    inverse = (inverse * good).astype(np.float32)
    answer = np.zeros(raw.shape, np.float32)
    for event in range(len(raw)):
        bits = raw[event] >> 14
        adc = (raw[event] & 0x3fff).astype(np.float32)
        for mode, code in enumerate((0, 1, 3)):
            take = (bits == code) & present[event, :, None, None].astype(bool)
            values = (adc - peds[mode]) * inverse[mode]
            answer[event][take] = values[take]
    return answer
