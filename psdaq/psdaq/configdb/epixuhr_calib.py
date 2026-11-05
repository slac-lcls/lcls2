"""Calibration constants of an ePixUHR3x2 panel for the GPU DRP's EpixUHR detectors.

Each function returns (bytes, warnings): float32 bytes laid out [len(modes)][336*576], or
None on failure, and messages for the operator about a failure or a default used.
modes indexes psana's gain modes, i.e. the first axis of the constants."""

import logging

import numpy as np

# @todo: Belongs in the detector's configDb entry
STATUS_BITS = (1 << 64) - 1     # pixel_status bits that mark a pixel bad

# psana's per-mode defaults (UtilsEpixUHR.GAIN_FACTORS), in ADU/keV, for panels without pixel_gain
DEFAULT_GAINS = (1.51, 0.54, 0.0172, 0.0086, 1.51, 1.51, 0.54, 0.54)
STATUS_MODES = 5                # psana marks a pixel bad from its status in gain modes 0-4


def _fail(warns, msg):
    logging.error('epixuhr: %s', msg)
    warns.append(msg)


def _warn(warns, msg):
    logging.warning('epixuhr: %s', msg)
    warns.append(msg)


def _fetch(serno, exp, run, ctypes, warns):
    from psalg.calibdb import calib_constants
    try:
        return calib_constants('epixuhr3x2_' + serno, exp or None, run, ctypes)
    except Exception as exc:
        _fail(warns, f'fetching {"/".join(ctypes)} from calibdb failed: {exc}')
        return None


def _log(ctype, doc, modes):
    logging.info('epixuhr: %s from %s run %s, %s, gain modes %s', ctype, doc.get('experiment'),
                 doc.get('run'), doc.get('_id'), list(modes))


def epixuhr_pedestals(serno, exp, run, modes):
    warns = []
    cons = _fetch(serno, exp, run, ('pedestals',), warns)
    if cons is None:
        return None, warns
    if 'pedestals' not in cons:
        _fail(warns, f'no pedestals in calibdb for {exp} run {run}')
        return None, warns
    nda, doc = cons['pedestals']
    if nda.ndim != 4 or max(modes) >= nda.shape[0]:
        _fail(warns, f'pedestals of shape {nda.shape} lack gain modes {list(modes)}')
        return None, warns
    _log('pedestals', doc, modes)
    return np.ascontiguousarray(nda[list(modes), 0], dtype=np.float32).tobytes(), warns


def epixuhr_gain_factors(serno, exp, run, modes):
    """psana's calib factor 1/gain, giving keV, and 0 for pixels whose status is bad in any of
    its first STATUS_MODES gain modes, as psana's default mask does."""
    warns = []
    cons = _fetch(serno, exp, run, ('pixel_gain', 'pixel_status', 'status_extra'), warns)
    if cons is None:
        return None, warns
    shape = (len(modes), 336 * 576)
    if 'pixel_gain' in cons:
        nda, doc = cons['pixel_gain']
        if nda.ndim != 4 or max(modes) >= nda.shape[0]:
            _fail(warns, f'pixel_gain of shape {nda.shape} lacks gain modes {list(modes)}')
            return None, warns
        _log('pixel_gain', doc, modes)
        gain = np.asarray(nda[list(modes), 0], dtype=np.float32).reshape(shape)
    else:
        _warn(warns, f'no pixel_gain in calibdb for {exp} run {run}: using default gains')
        gain = np.repeat(np.float32([DEFAULT_GAINS[m] for m in modes])[:, None], shape[1], axis=1)
    fac = np.divide(1, gain, out=np.zeros_like(gain), where=gain != 0)

    if 'pixel_status' not in cons and 'status_extra' not in cons:
        _warn(warns, f'no pixel_status in calibdb for {exp} run {run}: no pixels masked')
    for ctype in ('pixel_status', 'status_extra'):
        if ctype not in cons:
            continue
        nda, doc = cons[ctype]
        if nda.ndim != 4 or nda.shape[-1] * nda.shape[-2] != shape[1]:
            _fail(warns, f'{ctype} of shape {nda.shape} is not [modes][1][336][576]')
            return None, warns
        _log(ctype, doc, range(min(STATUS_MODES, nda.shape[0])))
        status = np.asarray(nda[:STATUS_MODES, 0], dtype=np.uint64).reshape(-1, shape[1])
        bad = np.bitwise_or.reduce(status, axis=0) & np.uint64(STATUS_BITS) != 0
        fac[:, bad] = 0
        logging.info('epixuhr: %d pixels masked by %s', int(bad.sum()), ctype)
    return np.ascontiguousarray(fac).tobytes(), warns
