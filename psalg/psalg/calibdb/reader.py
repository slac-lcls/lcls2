"""Fetch calibration constants from calib_ws, choosing documents as psana does.

Each detector has a database cdb_<short name>, and each experiment cdb_<exp>; in either,
the detector's collection holds one metadata document per stored constant, with the
array itself in GridFS.  A document is valid for runs run..run_end, run_end may be 'end'.
"""

import ast
import json
import logging
import os

import numpy as np
import requests

logger = logging.getLogger(__name__)

URL = os.environ.get('LCLS_CALIB_HTTP', 'https://pswww.slac.stanford.edu').rstrip('/') + '/calib_ws'
RUN_MAX = 9999                          # psana's stand-in for "no run" and for 'end'
TIMEOUT = 30                            # seconds per request


def _get(path, query=None):
    params = {'query_string': json.dumps(query)} if query is not None else None
    r = requests.get(f'{URL}/{path}', params=params, timeout=TIMEOUT)
    if not r.ok:
        logger.warning('calibdb: GET %s %s: %s %s', path, query, r.status_code, r.reason)
        return None
    return r


def _find_docs(dbname, colname, query):
    r = _get(f'{dbname}/{colname}', query)
    return r.json() if r is not None else None


# The long name is <type>_<serial number>; cdb_detnames maps it to the <type>_<NNNNNN>
# name the constants are filed under.
def short_name(longname):
    """The database's short detector name, e.g. 'epixuhr3x2_000002', or None."""
    docs = _find_docs('cdb_detnames', longname.split('_', 1)[0], {'long': longname})
    if not docs:
        return None
    if len(docs) > 1:
        logger.warning('calibdb: %d detnames documents for %s', len(docs), longname)
    return docs[0].get('short')


def _tsec(doc):
    return int(doc['_id'][:8], 16)      # A MongoDB ObjectId starts with its creation time


def _run_range(doc):
    """(begin, end) of a document's validity, or None if it is invalid.

    Documents are not validated on submission, so run and run_end may be missing, or
    strings rather than integers.
    """
    begin, end = str(doc.get('run', '')), str(doc.get('run_end', ''))
    if not begin.isdigit():
        return None
    end = RUN_MAX if end == 'end' else int(end) if end.isdigit() else None
    begin = int(begin)
    if end is None or begin > RUN_MAX or end > RUN_MAX:
        return None
    return begin, end


# psana's rule: the most recently started validity range wins, a narrower one breaks a
# tie, and of identical ranges the most recently stored.
def _select_for_run(docs, run):
    """The latest-beginning, then shortest, then newest document valid for run."""
    best, bestKey = None, None
    for doc in docs:
        rng = _run_range(doc)
        if rng is None or not (rng[0] <= run <= rng[1]):
            continue
        key = (rng[0], -rng[1], _tsec(doc))
        if bestKey is None or key >= bestKey:   # On a tie the last, as psana's reversed scan picks
            best, bestKey = doc, key
    return best


def _data(dbname, doc):
    """The document's ndarray, or None.  Other data types are not supported."""
    if doc.get('data_type') != 'ndarray':
        logger.warning('calibdb: %s %s has data_type %s, not ndarray',
                       doc.get('ctype'), doc.get('_id'), doc.get('data_type'))
        return None
    try:
        dtype = np.dtype(doc['data_dtype'])
        shape = tuple(ast.literal_eval(str(doc['data_shape'])))
    except (KeyError, TypeError, ValueError, SyntaxError) as exc:
        logger.warning('calibdb: %s %s has no usable dtype or shape: %s',
                       doc.get('ctype'), doc.get('_id'), exc)
        return None
    r = _get(f'{dbname}/gridfs/{doc.get("id_data")}')
    if r is None:
        return None
    if len(r.content) != dtype.itemsize * int(np.prod(shape)):
        logger.warning('calibdb: %s %s: %d bytes for shape %s of %s',
                       doc.get('ctype'), doc.get('_id'), len(r.content), shape, dtype)
        return None
    return np.frombuffer(r.content, dtype=dtype).reshape(shape)


# Groups documents by constant type, e.g. 'pedestals', keeping only the requested types
def _by_ctype(docs, ctypes):
    out = {}
    for doc in docs or []:
        ct = doc.get('ctype')
        if ct is not None and (ctypes is None or ct in ctypes):
            out.setdefault(ct, []).append(doc)
    return out


def calib_constants(longname, exp=None, run=None, ctypes=None):
    """{ctype: (ndarray, doc)} for the detector, as psana would select them.

    The experiment's database is searched first, for documents valid for run; any
    ctype it lacks comes from the detector's database, newest first.  run 0 or None
    means the latest.  ctypes limits the search, e.g. ('pedestals',).  Documents
    with an invalid run range are skipped in both.
    """
    short = short_name(longname)
    if short is None:
        logger.warning('calibdb: no short name for %s', longname)
        return {}
    run = run or RUN_MAX
    # No server-side run filter: $lte would silently drop documents whose run is a string
    query = {'detector': short}
    if ctypes is not None and len(ctypes) == 1:
        query['ctype'] = ctypes[0]

    # Experiment database first: constants taken during the experiment, valid for the run
    out = {}
    if exp:
        dbname = f'cdb_{exp}'
        for ct, docs in _by_ctype(_find_docs(dbname, short, query), ctypes).items():
            doc = _select_for_run(docs, run)
            data = _data(dbname, doc) if doc is not None else None
            if data is not None:
                out[ct] = (data, doc)

    # Then the detector database, for anything still missing: the newest, whatever its run
    dbname = f'cdb_{short}'
    for ct, docs in _by_ctype(_find_docs(dbname, short, query), ctypes).items():
        if ct in out:
            continue
        valid = [doc for doc in docs if _run_range(doc) is not None]
        if valid:
            doc = max(valid, key=_tsec)
            data = _data(dbname, doc)
            if data is not None:
                out[ct] = (data, doc)
    return out
