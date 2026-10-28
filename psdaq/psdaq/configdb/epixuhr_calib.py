"""Calibration constants of an ePixUHR3x2 panel for the GPU DRP's EpixUHR detectors.

Each function returns float32 bytes laid out [len(modes)][336*576], or None on failure.
modes indexes psana's gain modes, i.e. the first axis of the constants."""

import logging

import numpy as np

# @todo: Belongs in the detector's configDb entry
STATUS_BITS = (1 << 64) - 1     # pixel_status bits that mark a pixel bad

# psana's per-mode defaults (UtilsEpixUHR.GAIN_FACTORS), in ADU/keV, for panels without pixel_gain
DEFAULT_GAINS = (1.51, 0.54, 0.0172, 0.0086, 1.51, 1.51, 0.54, 0.54)
STATUS_MODES = 5                # psana marks a pixel bad from its status in gain modes 0-4


def _fetch(serno, exp, run, ctypes):
    from psalg.calibdb import calib_constants
    try:
        return calib_constants('epixuhr3x2_' + serno, exp or None, run, ctypes)
    except Exception as exc:
        logging.error('epixuhr: fetching %s failed: %s', ctypes, exc)
        return None


def _log(ctype, doc, modes):
    logging.info('epixuhr: %s from %s run %s, %s, gain modes %s', ctype, doc.get('experiment'),
                 doc.get('run'), doc.get('_id'), list(modes))


def epixuhr_pedestals(serno, exp, run, modes):
    cons = _fetch(serno, exp, run, ('pedestals',))
    if cons is None:
        return None
    if 'pedestals' not in cons:
        logging.error('epixuhr: no pedestals for serial %s', serno)
        return None
    nda, doc = cons['pedestals']
    if nda.ndim != 4 or max(modes) >= nda.shape[0]:
        logging.error('epixuhr: pedestals of shape %s lack gain modes %s', nda.shape, list(modes))
        return None
    _log('pedestals', doc, modes)
    return np.ascontiguousarray(nda[list(modes), 0], dtype=np.float32).tobytes()


def epixuhr_gain_factors(serno, exp, run, modes):
    """psana's calib factor 1/gain, giving keV, and 0 for pixels whose status is bad in any of
    its first STATUS_MODES gain modes, as psana's default mask does."""
    cons = _fetch(serno, exp, run, ('pixel_gain', 'pixel_status', 'status_extra'))
    if cons is None:
        return None
    shape = (len(modes), 336 * 576)
    if 'pixel_gain' in cons:
        nda, doc = cons['pixel_gain']
        if nda.ndim != 4 or max(modes) >= nda.shape[0]:
            logging.error('epixuhr: pixel_gain of shape %s lacks gain modes %s', nda.shape, list(modes))
            return None
        _log('pixel_gain', doc, modes)
        gain = np.asarray(nda[list(modes), 0], dtype=np.float32).reshape(shape)
    else:
        logging.warning('epixuhr: no pixel_gain for serial %s, using psana\'s defaults', serno)
        gain = np.repeat(np.float32([DEFAULT_GAINS[m] for m in modes])[:, None], shape[1], axis=1)
    fac = np.divide(1, gain, out=np.zeros_like(gain), where=gain != 0)

    for ctype in ('pixel_status', 'status_extra'):
        if ctype not in cons:
            continue
        nda, doc = cons[ctype]
        _log(ctype, doc, range(min(STATUS_MODES, nda.shape[0])))
        status = np.asarray(nda[:STATUS_MODES, 0], dtype=np.uint64).reshape(-1, shape[1])
        bad = np.bitwise_or.reduce(status, axis=0) & np.uint64(STATUS_BITS) != 0
        fac[:, bad] = 0
        logging.info('epixuhr: %d pixels masked by %s', int(bad.sum()), ctype)
    return np.ascontiguousarray(fac).tobytes()
