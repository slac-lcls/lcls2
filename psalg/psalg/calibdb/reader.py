import ast
import json
import logging

import numpy as np
import requests

logger = logging.getLogger(__name__)

URL = 'https://pswww.slac.stanford.edu/calib_ws'
RUN_MAX = 9999                          # psana's stand-in for "no run" and for 'end'
TIMEOUT = 30                            # seconds per request


def _get(path, query=None):
    params = {'query_string': json.dumps(query)} if query is not None else None
    r = requests.get(f'{URL}/{path}', params=params, timeout=TIMEOUT)
    if not r.ok:
        logger.debug('calibdb: GET %s %s: %s %s', path, query, r.status_code, r.reason)
        return None
    return r


def _find_docs(dbname, colname, query):
    r = _get(f'{dbname}/{colname}', query)
    return r.json() if r is not None else None


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
    """(begin, end) of a document's validity, or None if it is invalid."""
    begin, end = int(doc['run']), doc['run_end']
    end = RUN_MAX if end == 'end' else int(end) if str(end).isdigit() else None
    if end is None or begin > RUN_MAX or end > RUN_MAX:
        return None
    return begin, end


def _select_for_run(docs, run):
    """The latest-beginning, then shortest, then newest document valid for run."""
    best, bestKey = None, None
    for doc in docs:
        rng = _run_range(doc)
        if rng is None or not (rng[0] <= run <= rng[1]):
            continue
        key = (rng[0], -rng[1], _tsec(doc))
        if bestKey is None or key > bestKey:
            best, bestKey = doc, key
    return best


def _data(dbname, doc):
    r = _get(f'{dbname}/gridfs/{doc["id_data"]}')
    if r is None:
        return None
    if doc.get('data_type') != 'ndarray':
        return r.content.decode()
    nda = np.frombuffer(r.content, dtype=doc['data_dtype'])
    return nda.reshape(ast.literal_eval(doc['data_shape']))


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
    means the latest.  ctypes limits the search, e.g. ('pedestals',).
    """
    short = short_name(longname)
    if short is None:
        logger.warning('calibdb: no short name for %s', longname)
        return {}
    run = run or RUN_MAX
    query = {'detector': short, 'run': {'$lte': run}}
    if ctypes is not None and len(ctypes) == 1:
        query['ctype'] = ctypes[0]

    out = {}
    if exp:
        dbname = f'cdb_{exp}'
        for ct, docs in _by_ctype(_find_docs(dbname, short, query), ctypes).items():
            doc = _select_for_run(docs, run)
            if doc is not None:
                out[ct] = (_data(dbname, doc), doc)

    dbname = f'cdb_{short}'
    query['run'] = {'$lte': RUN_MAX}
    for ct, docs in _by_ctype(_find_docs(dbname, short, query), ctypes).items():
        if ct not in out:
            doc = max(docs, key=_tsec)
            out[ct] = (_data(dbname, doc), doc)
    return out
