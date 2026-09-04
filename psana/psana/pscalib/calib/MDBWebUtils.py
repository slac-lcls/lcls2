"""
Usage ::
    # see examples from Murali: /sdf/home/m/mshankar/temp/calibjwt/calibcall.py

    # Import
    import psana.pscalib.calib.MDBWebUtils as wu
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    resp = wu.check_ticket(exit_if_invalid=True)
    q = wu.query_id_pro(query) # e.i., query={"_id":doc_id}
    _ = wu.request(url, query=None)
    _ = wu.database_names(url=cc.URL)
    _ = wu.collection_names(dbname, url=cc.URL)
    _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'})
    _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'})
    _ = wu.select_latest_doc(docs, query):
    _ = wu.get_doc_for_docid(dbname, colname, docid)
    _ = wu.get_data_for_id(dbname, dataid)
    _ = wu.get_data_for_docid(dbname, colname, docid)
    _ = wu.get_data_for_doc(dbname, doc)
    data,doc = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL)

    # USED BY psana/psexp/ds_base.py TO RETRIEVE ALL CONSTANTS FROM DB
    d = wu.calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL, dbsuffix='')
    d = {ctype:(data,doc),}
    cc,doc = wu.calib_constants_for_ctype(detlongname, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL, dbsuffix='')

    id = wu.add_data_from_file(dbname, fname, sfx=None, **kwa) # DEPRECATED
    id = wu.add_data(dbname, data, **kwa)
    id = wu.add_document(dbname, colname, doc, **kwa) # url=cc.URL_KRB
    is_exp = wu.is_doc_from_exp_db(doc)
    id = replace_document(doc, **kwa)
    id_data, id_doc = wu.add_data_and_doc(data, dbname, colname, **kwargs)
    id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
      wu.add_data_and_two_docs(data, exp, det, **kwargs)

    detname_short = wu.pro_detector_name(detname, add_shortname=False) # DEPRECATED: maxsize=cc.MAX_DETNAME_SIZE)

    resp = wu.delete_database(dbname, **kwa)
    resp = wu.delete_collection(dbname, colname, **kwa)
    resp = wu.delete_document(dbname, colname, doc_id, **kwa)
    resp = wu.delete_data(dbname, data_id, **kwa)
    isok = wu.delete_document_and_data(dbname, colname, doc_id, **kwa)

    s = wu.str_formatted_list(lst, ncols=5, width=24)
    s = wu.info_docs(dbname, colname, query={}, strlen=120)
    s = wu.info_webclient(**kwargs)
    resp = wu.valid_post_privilege(dbname)

    test_*()
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
import numpy as np
import io

from psana.pscalib.calib.CalibDoc import CalibDoc
import psana.pscalib.calib.CalibConstants as cc
import requests as req

from time import time
from numpy import fromstring
import psana.pscalib.calib.MDBUtils as mu
import psana.pyalgos.generic.Utils as gu
from subprocess import call
import json as jsonmet
import psana.detector.Utils as ut
#import psana.detector.utils_psana as up # dict_filter

jwt = os.getenv('CALIB_JWT', None)
has_jwt = bool(jwt)
info_jwt = 'using jwt' if has_jwt else 'using kerberos, NO jwt available'
msg_jwt = 'importing psana.pscalib.calib.MDBWebUtils'\
    '\nusing jwt' if has_jwt else 'using kerberos, NO jwt available'\
    '\nmake env CALIB_JWT using:'\
    '\n  source psana/psana/pscalib/calib/get_JWT_from_s3df.sh'\
    '\nor'\
    '\n  source psana/psana/pscalib/calib/get_JWT_from_kerberos.sh'\

session = req.Session() if has_jwt else None
if has_jwt:
    session.headers.update({'Authorization': 'Bearer ' + jwt })
    logger.debug(f'jwt: {str(jwt)}')

print(msg_jwt)
#logger.info(msg_jwt)


def info_dict(d, cmt='', offset='  '):
    """returns (str) dict content"""
    s = '%s\n%sinfo_dict' % (cmt, offset)
    for k,v in d.items():
        if isinstance(v,dict): s = info_dict(v, cmt='', offset = offset+'  ')
        else: s = '%s\n%sk:%s t:%s v:%s' % (s, offset, str(k).ljust(10), type(v), str(v)[:60])
    return s

def info_dict_keys(d, sep=' '):
    return sep.join(d.keys())

def info_dict_for_keys(d, keys=('_id', 'experiment', 'run', 'run_orig', 'short', 'time_stamp', 'ctype'), sep='  '):
    return sep.join([f'{k}: {v}' for k,v in d.items() if k in keys])
    #d = up.dict_filter(doc, list_keys=keys, ordered=False)

info_document = info_dict_for_keys

def info_ldocs(ldocs, nmax=4, sep='\n  '):
    if ldocs is None: return 'None'
    return f'ndocs={len(ldocs)}' + sep + sep.join([info_document(d) for i,d in enumerate(ldocs) if i<nmax])

def info_docs_list(docs, strlen=150):
    if not isinstance(docs, list):
        return f'info_docs_list parameter docs is not list: {str(docs)}'
    s = ''
    for i,d in enumerate(docs):
        s += '\n%04d %s ...' % (i, str(d)[:strlen])
    return s

def info_detname(d, keys=('_id','short','time_stamp','long')):
    r = d.get('json', None)
    return info_dict_for_keys(r, keys=keys) if isinstance(r, dict) else 'None'

def info_detnames(ldocs, keys=('_id','short','time_stamp','long'), nmax=10, sep='\n  '):
    if ldocs is None: return 'None'
    return f'ndocs={len(ldocs)}' + sep + sep.join([info_detname(d, keys=keys) for i,d in enumerate(ldocs) if i<nmax])



def post(url, data=None, doc={}, **kwa):
    logger.debug(f'post url: {url}  ticket: {info_jwt}  **kwa: {str(kwa)}  doc: {str(doc)}  data: {str(data)[:200]}')
    if has_jwt:
        if data is None:
            if 'headers' in doc.keys():
                jsonmet.pop('headers')
            return session.post(url, json=doc)
        else:
            return session.post(url, data=data, headers={"Content-Type": "application/json"})
    else:
        krbh = cc.krbheaders() # krbh['Content-Type'] = 'application/octet-stream'
        logger.debug(f'post url: {url}  ticket: {info_jwt}  **kwa: {str(kwa)}  doc: {str(doc)}  data: {str(data)[:200]}')
        logger.debug(f'post krbheaders: {jsonmet.dumps(krbh, indent=2)}')
        resp = req.post(url, headers=krbh, json=dict(doc), data=data)
        logger.debug(f'post resp: {resp.text}')
        return resp


def put(url, doc, **kwa):
    logger.debug(f'put url: {url} doc: {str(doc)}')
    return session.put(url, json=doc) if has_jwt else\
           req.put(url, json=doc, headers=cc.krbheaders())


def get(url, query=None, timeout=180, **kwa):
    krbh = cc.krbheaders() # inside: krbh['Content-Type'] = 'application/octet-stream'
    logger.debug(f'get for url: {url}  query: {str(query)}  ticket: {info_jwt}')
    logger.debug(f'\nget: krbheaders {jsonmet.dumps(krbh, indent=2)}')
    return session.get(url, json=query, timeout=timeout) if has_jwt else\
           req.get(url, params=query, timeout=timeout, headers=krbh)


def delete_cmd(url):
    resp = session.delete(url) if has_jwt else\
           req.delete(url, headers=cc.krbheaders())
    logger.debug(f'delete for url: {url}  ticket: {info_jwt}  resp.ok: {resp.ok}')
    return resp


def has_kerberos_ticket():
    """Checks to see if the user has a valid Kerberos ticket."""
    return not call(["klist", "-s"])


def check_ticket(exit_if_invalid=True):
    if has_jwt:
        logger.debug('use JWT ticket')
        return True
    if has_kerberos_ticket():
        logger.debug('using kerberos, JWT ticket is missing')
        return True
    logger.error('KERBEROS AND JWT TICKETS ARE UNAVAILABLE OR EXPIRED')
    if exit_if_invalid:
        sys.exit('FIX KERBEROS OR JWT TICKET - use command "kinit" or check its status with command "klist"')
    return False


def query_id_pro_str(query):
    id = query.get('_id', None)
    if (id is None) or ('ObjectId' in id): return query
    query['_id'] = 'ObjectId(%s)'%id
    return query


def query_id_pro(query):
    id = query.get('_id', None)
    if isinstance(id, str):
        query['_id'] = mu.ObjectId(id)
    return query


def request(url, query=None, timeout=180, **kwa):
    logger.debug(f'in request for url: {url} and query: {str(query)}     {info_jwt}')
    #t0_sec = time()
    #r = req.get(url, query, timeout=180)
    #r = session.get(url, params={'query_string':str(query)}, timeout=180) if has_jwt else\
    r = get(url, query=query, timeout=timeout)
    #dt = time()-t0_sec # ~30msec
    #logger.debug('CONSUMED TIME by request %.3f sec\n  for url=%s  query=%s' % (dt, url, str(query)))
    if r.ok:
        logger.debug(f'request resp is ok: {str(r.content)[:100]}')
        return r
    s = f'get url: {url} query: {str(query)}\n  response status: {r.ok} status_code: {r.status_code} reason: {r.reason}'
    s += f'\nTry command: curl -s "{url}"'
    logger.debug(s)
    if r.status_code == 503:
        logger.warning(s)
        sys.exit(1)
    return None


# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db"
def database_names(url=cc.URL, pattern=None):
    """Returns list of database names for url."""
    r = request(url)
    #print(r.json(), type(r.json()))
    if r is None: return None
    return r.json() if pattern is None else [name for name in r.json() if str(pattern) in name]


# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll"
def collection_names(dbname, url=cc.URL):
    """Returns list of collection names for dbname and url."""
    r = request(f'{url.rstrip("/")}/{dbname}')
    if r is None: return None
    return r.json()


# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll?query_string=%7B%20%22item%22..."
def find_docs(dbname, colname, query={}, **kwa):
    """Returns list of documents for query, e.g. query={'ctype':'pedestals', "run":{ "$gte":80}}."""
    uri = f'{cc.URL.rstrip("/")}/{dbname}/{colname}'

    # WORKING OLD VERSION using query as str:
    #query_string=str(query).replace("'",'"')
    #logger.debug('find_docs uri: %s query: %s' % (uri, query_string))
    #r = request(uri, {"query_string": query_string})

    # NEW VERSION using query as dict/json:
    logger.debug(f'find_docs uri: {uri} query: {str(query)}')
    r = request(uri, query=query) # query = {'_id': bson.ObjectId(doc_id), ...}

    if ut.is_true(r is None, 'find_docs resp is None for url: {uri}', logger_method=logger.debug): return None
    ldocs = r.json()
    s = '\n\n  '.join([str(d) for d in ldocs])
    logger.debug(f"find_docs res.ok:{r.ok}  docs:\n\n  {s}")

    try:
        return r.json()
    except:
        msg = f'WARNING: find_docs responce: {str(r)}'\
            + f'\n     conversion to json failed, return None for query: {str(query)}'
        logger.debug(msg)
        return None


def find_doc(dbname, colname, query={}, **kwa): #query={'ctype':'pedestals'}
    """Returns document for query.
       1. finds all documents for query
       2. select the latest for run or time_sec
    """
    logger.debug('find_doc input pars dbname: %s colname: %s query:%s' % (dbname, colname, str(query)))

    docs = find_docs(dbname, colname, query)
    if docs is None: return None

    return select_latest_doc(docs, query)


def select_latest_doc(docs, query):
    """Returns a single document for query selected by time_sec (if available) or run."""
    if docs is None: return None

    if len(docs)==0:
        # commented out by cpo since this happens routinely the way
        # that Mona is fetching calibration constants in psana.
        #logger.warning('find_docs returns list of length 0 for query: %s' % query)
        return None

    for d in docs:
        d['tsec_id'], d['tstamp_id'] = mu.sec_and_ts_from_id(d['_id'])

    #qkeys = query.keys()
    #key_sort = 'time_sec' if 'time_sec' in qkeys else 'run'
    key_sort = 'tsec_id'

    logger.debug('select_latest_doc: %s\nkey_sort: %s' % (str(query), key_sort))
    vals = [int(d[key_sort]) for d in docs]
    vals.sort(reverse=True)
    logger.debug('find_doc values: %s' % str(vals))
    val_sel = int(vals[0])
    logger.debug('find_doc select document for %s: %s' % (key_sort,val_sel))
    for d in docs:
        if d[key_sort]==val_sel:
            return d
    return None


def select_doc_in_run_range(docs, rnum):
    """uses psana.pscalib.calib.CalibDoc in order to sort documents"""
    cdocs = [CalibDoc(d) for d in docs]
    cdocs_sorted = sorted([cd for cd in cdocs if cd.valid])
    #print('in select_doc_in_run_range - cdocs_sorted:\n  %s' % '\n  '.join([d.info_calibdoc() for d in cdocs_sorted]))
    for d in cdocs_sorted[::-1]:
        if d.valid and d.begin <= rnum and rnum <= d.end:
            logger.debug('selected calibdoc: %s' % d.info_calibdoc())
            return d.doc
    return None # if no matching found


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/5b6893e81ead141643fe4344"
def get_doc_for_docid(dbname, colname, docid, **kwa):
    """Returns document for docid."""
    r = request(f'{cc.URL.rstrip("/")}/{dbname}/{colname}/{docid}')
    if r is None: return None
    return r.json()


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a"
def get_data_for_id(dbname, dataid, **kwa):
    """Returns raw data from GridFS, at this level there is no info for parsing."""
    r = request(f'{cc.URL.rstrip("/")}/{dbname}/gridfs/{dataid}')
    if r is None: return None
    logger.debug('get_data_for_docid:'\
                +'\n  r.status_code: %s\n  r.headers: %s\n  r.encoding: %s\n  r.content: %s...\n' %
                 (str(r.status_code),  str(r.headers),  str(r.encoding),  str(r.content[:50])))
    return r


def get_data_for_docid(dbname, colname, docid, **kwa):
    """Returns data from GridFS using docid."""
    doc = get_doc_for_docid(dbname, colname, docid)
    logger.debug(f'get_data_for_docid: {str(doc)}')
    return get_data_for_doc(dbname, doc, url)


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def get_data_for_doc(dbname, doc, **kwa):
    """Returns data from GridFS using doc."""
    logger.debug(f'get_data_for_doc: {str(doc)}')
    idd = doc.get('id_data', None)
    if idd is None:
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None

    #print('curl -s "%s"' % ('%s/%s/gridfs/%s'%(cc.URL.rstrip('/'),dbname,idd)))
    url = f'{cc.URL.rstrip("/")}/{dbname}/gridfs/{idd}'
    r2 = request(url)
    if ut.is_true(r2 is None, 'resp is None for url: {url}', logger_method=logger.debug): return None
    s = r2.content

    return mu.object_from_data_string(s, doc)


def dbnames_collection_query(detname, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dtype=None, dbsuffix='', **kwa):
    """wrapper for MDBUtils.dbnames_collection_query,
       - which should receive short detector name, othervice uses direct interface to DB
    """
    logger.info('dbnames_collection_query input parameters:\n' +\
                 '    detname:%s exp:%s ctype:%s run:%s time_sec:%s vers:%s dtype:%s dbsuffix:%s kwa:%s' %\
                 (detname, exp, ctype, str(run), str(time_sec), vers, str(dtype), dbsuffix, str(kwa)))
    short = pro_detector_name(detname)
    logger.debug(f'short: {short} dbsuffix: {dbsuffix}')
    resp = list(mu.dbnames_collection_query(short, exp, ctype, run, time_sec, vers, dtype))
    if dbsuffix: resp[0] = detector_dbname(short, dbsuffix=dbsuffix)
    return resp


def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dbsuffix='', **kwa):
    """Returns calibration constants and document with metadata for specified parameters.
       To get meaningful constants, at least a few parameters must be specified, e.g.:
       - det, ctype, time_sec
       - det, ctype, version
       - det, exp, ctype, run
       - det, exp, ctype, time_sec
       - det, exp, ctype, run, version
       etc...
    """
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers, dtype=None, dbsuffix=dbsuffix)
    logger.debug('get_constants: %s %s %s %s' % (db_det, db_exp, colname, str(query)))
    dbname = db_det if dbsuffix or (exp is None) else db_exp
    doc = find_doc(dbname, colname, query)
    if doc is None:
        # commented out by cpo since this happens routinely the way
        # that Mona is fetching calibration constants in psana.
        logger.debug('document is not available for query: %s' % str(query))
        return None
    return (get_data_for_doc(dbname, doc), doc)


def calib_constants_of_missing_types(resp, det, time_sec=None, vers=None, **kwa):
    """ try to add constants of missing types in resp using detector db."""
    exp=None
    run=9999
    ctype=None
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers, dtype=None)
    dbname = db_det
    docs = find_docs(dbname, colname, query)
    #logger.debug('find_docs: number of docs found: %d' % len(docs))
    if docs is None: return None

    ctypes = set([d.get('ctype',None) for d in docs])
    ctypes.discard(None)
    logger.debug('calib_constants_missing_types - found ctypes: %s' % str(ctypes))

    ctypes_resp = resp.keys()
    _ctypes = [ct for ct in ctypes if not(ct in ctypes_resp)]

    logger.debug('calib_constants_missing_types - found additional ctypes: %s' % str(_ctypes))

    for ct in _ctypes:
        docs_for_type = [d for d in docs if d.get('ctype',None)==ct]
        doc = select_latest_doc(docs_for_type, query)
        if doc is None: continue
        resp[ct] = (get_data_for_doc(dbname, doc), doc)

    return resp


def print_docs_for_ctype(docs_for_type, ct, detname_short='epix100_000002'):
    """print for debugging"""
    print('\n\ncalib_constants_all_types docs_for_type %s' % ct)
    for i,d in enumerate(docs_for_type):
        shortname = d['shortname']
        if(shortname != detname_short): continue
        tsec_id, tstamp_id = mu.sec_and_ts_from_id(d['_id'])
        print('  doc:%02d experiment:%s ctype:%12s shortname:%s run:%3s run_end:%s tstamp_id:%s' %\
              (i, d['experiment'], ct, shortname, str(d['run']), str(d['run_end']), tstamp_id))


def calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, dbsuffix='', **kwa):
    """ USED BY psana/psexp/ds_base.py TO RETRIEVE ALL CONSTANTS FROM DB
        Returns constants for all ctype-s."""
    t0_sec = time()
    ctype=None
    longname = det
    logger.debug('detlongname: %s exp: %s run: %s time_sec: %s vers: %s' % (longname, exp, str(run), time_sec, vers))

    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers, dtype=None, dbsuffix=dbsuffix)
    dbname = db_det if dbsuffix or (exp is None) else db_exp

    logger.debug('dbname: %s db_det: %s db_exp: %s colname: %s query: %s dbsuffix: %s'%\
                (dbname, db_det, db_exp, colname, query, dbsuffix))
    logger.debug('time 1: %.6f sec - for DB %s generate query %s' % (time()-t0_sec, dbname, query))

    docs = find_docs(dbname, colname, query)
    logger.debug('find_docs: number of docs found: %s' % (str(len(docs)) if docs is not None else None))
    #print('time 2: %.6f sec - find docs for query in DB %s' % (time()-t0_sec, dbname))

    resp = {}
    if docs is not None:

        ctypes = set([d.get('ctype', None) for d in docs])
        ctypes.discard(None)
        logger.debug('calib_constants_all_types - found ctypes: %s' % str(ctypes))

        for ct in ctypes:
            docs_for_type = [d for d in docs if d.get('ctype',None)==ct]
            #print_docs_for_ctype(docs_for_type, ct, detname_short='epix100_000002')

            doc = select_doc_in_run_range(docs_for_type, run)
            #doc = select_latest_doc(docs_in_run_range, query)

            if doc is None: continue
            resp[ct] = (get_data_for_doc(dbname, doc), doc)
            #print('        %.6f sec - get data for ctype: %s' % (time()-t0_sec, ct))

        #print('time 3: %.6f sec - get data for docs total' % (time()-t0_sec))

    resp = calib_constants_of_missing_types(resp, det, time_sec, vers)

    #print('time 4: %.6f sec - check for missing types in the det DB' % (time()-t0_sec))

    return resp


def calib_constants_for_ctype(detlongname, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dbsuffix='', **kwa):
    """ USED BY psana/app/calibvalidity.py to find specific doc
        The same as calib_constants_all_types, BUT for a selected ctype"""

    logger.info('detlongname: %s exp: %s run: %s time_sec: %s vers: %s' % (detlongname, exp, str(run), time_sec, vers))
    # search in EXPERIMENT DB
    resp = calib_constants(detlongname, exp, ctype, run, time_sec, vers, dbsuffix)

    if resp is None:
        # search in DETECTOR DB
        exp=None
        run=9999
        resp = calib_constants(detlongname, exp, ctype, run, time_sec, vers, dbsuffix)
    return resp


def add_data_from_file(dbname, fname, sfx=None, **kwa):
    """DEPRECATED: Adds data from file to the database/gridfs."""
    check_ticket()

    _sfx = sfx if sfx is not None else fname.rsplit('.')[-1]
    files = [('files',  (fname, open(fname, 'rb'), 'image/'+_sfx))]
    resp = post(cc.URL_KRB+dbname+'/gridfs/', files=files)
    logger.debug('add_data_from_file: %s to %s/gridfs/ resp: %s type: %s' % (fname, dbname, resp.text, type(resp)))
    #jdic = resp.json() # type <class 'dict'>
    return resp.json().get('_id',None)


def add_data(dbname, data, **kwa):
    """Adds binary data to the database/gridfs."""
    check_ticket()

## MODERN VERSION of siving np.ndarray as entire object
#    if isinstance(data, np.ndarray):
#        b = io.BytesIO()
#        np.save(b, data)
#        b.seek(0)
#    else:
#        b = io.BytesIO(mu.encode_data(data)) # for str and any

    b = io.BytesIO(mu.encode_data(data))

    urltot = cc.URL_KRB+dbname+'/gridfs/'
    d = b.read()
    resp = post(urltot, data=d)
    logger.debug(f'add_data byte-data: {str(d[:100])}'\
                +f'\nadd_data to: {urltot}\n    resp: {resp.text[:1000]}')
    try:
        id = resp.json().get('_id',None)
    except Exception as e:
        logger.warning(f'JSONDecodeError: {str(e)}')
        return None

    if id is None: logger.warning('id_data is None')
    return id


def add_document(dbname, colname, doc, **kwa):
    """Adds document to database collection."""
    check_ticket()
    url = cc.URL_KRB+dbname+'/'+colname+'/'
    resp = post(url, doc=doc)
    logger.debug(f'add_document: {str(doc)}\n  to {url}\n  resp: {resp.text}')

    try:
        id = resp.json().get('_id',None)
    except Exception as e:
        logger.warning(f'JSONDecodeError: {str(e)}')
        return None

    if id is None: logger.warning('id_document is None')
    return id


def is_doc_from_exp_db(doc):
    """uses doc 'id_doc_exp' and 'id_data_exp' which should be 0 for exp_db"""
    id_doc_exp  = doc.get('id_doc_exp', None)
    id_data_exp = doc.get('id_data_exp', None)
    return id_doc_exp == 0 and id_data_exp == 0


def replace_document(doc, **kwa):
    """Rreplace document for database, collection using the same _id
       doc should have items for 'detector', 'experiment', '_id'
       Murali: requests.put("https://pswww_OR_psdmint/calib_ws/cdb_tstx00117/epixquad/5fc6911af587a598cbb1a601", json=doc)
    """
    check_ticket()
    shortname  = doc.get('detector', None)
    experiment = doc.get('experiment', None)
    _id        = doc.get('_id', None)
    if shortname is None:
        logger.warning('document does not have key "detector", doc: %s' % str(doc))
        return None
    dbname = mu.db_prefixed_name(experiment)
    colname = shortname
    _url = cc.URL_KRB+dbname+'/'+colname+'/'+_id
    resp = put(_url, doc)
    logger.info('replace_document: %s\n== for %s/%s\n== resp: %s' % (str(doc), dbname, colname, resp.text))
    id = resp.json().get('_id',None)
    if id is None: logger.warning('id_document is None')
    return id


def add_data_and_doc(data, _dbname, _colname, **kwargs):
    """Check permission and add data and document to the db."""
    logger.debug('add_data_and_doc kwargs: %s' % str(kwargs))

    # check permission
    t0_sec = time()
    #if not valid_post_privilege(_dbname): return None

    id_data = add_data(_dbname, data)
    if id_data is None: return None
    doc = mu.docdic(data, id_data, **kwargs) # ObjectId(id_data)???
    logger.debug(mu.doc_info(doc, fmt='  %s:%s')) #sep='\n  %16s : %s'

    id_doc = add_document(_dbname, _colname, doc)
    if id_doc is None: return None

    msg = 'Add data and doc time %.6f sec' % (time()-t0_sec)\
        + '\n  - data in %s/gridfs id: %s and doc in collection %s id: %s' % (_dbname, id_data, _colname, id_doc)
    logger.debug(msg)

    return id_data, id_doc


def insert_document_and_data(dbname, colname, dicdoc, data, **kwa):
    """DEPRECATED - wrapper for pymongo compatability - is used in graphqt/CMWDB*.py"""
    return add_data_and_doc(data, dbname, colname, **dicdoc)


def add_data_and_two_docs(data, exp, detname_long, **kwargs):
    """Add data and document to experiment and detector data bases."""
    logger.debug('add_data_and_two_docs kwargs: %s' % str(kwargs))
    shortname = pro_detector_name(detname_long, add_shortname=True)
    colname = shortname
    dbname_exp = mu.db_prefixed_name(exp)
    dbname_det = mu.db_prefixed_name(shortname)

    ctype = kwargs.get('ctype','N/A')
    logger.info(f'add_data_and_two_docs save constants: {ctype} in DBs: {dbname_exp} and {dbname_det} collection: {colname}')

    #kwargs['detector'] = detname         # ex: epix10ka
    kwargs['shortname'] = shortname       # ex: epix10ka_000001
    kwargs['longname']  = detname_long    # ex: epix10ka_<_uniqueid>

    resp = add_data_and_doc(data, dbname_exp, colname, **kwargs)
    if resp is None: return None
    id_data_exp, id_doc_exp = resp

    kwargs['id_data_exp'] = id_data_exp # override
    kwargs['id_doc_exp']  = id_doc_exp  # add
    resp = add_data_and_doc(data, dbname_det, colname, **kwargs)
    id_data_det, id_doc_det = resp if resp is not None else (None, None)
    return id_data_exp, id_data_det, id_doc_exp, id_doc_det


def detector_dbname(detname_short, **kwargs):
    """Makes detector db name depending on suffix,
       e.g. for detname_short='epixhr2x2_000001' and suffix='mytestdb'
       returns 'cdb_epixhr2x2_000001_mytestdb'
    """
    dbsuffix = kwargs.get('dbsuffix','')
    #logger.debug('detector_dbname detname: %s dbsuffix: %s' % (detname_short, dbsuffix))
    assert isinstance(dbsuffix, str)
    dbname_det = mu.db_prefixed_name(detname_short)
    logger.info('dbname_det: %s dbsuffix: %s' % (dbname_det, dbsuffix))
    if dbsuffix: dbname_det += '_%s'% dbsuffix
    assert len(dbname_det) < 50
    logger.debug('detector_dbname detname: %s dbsuffix: %s returns: %s' % (detname_short, dbsuffix, dbname_det))
    return dbname_det


def add_data_and_doc_to_detdb_extended(data, exp, detname_long, **kwargs):
    """Add data and document to the detector data base with extended name using 'dbsuffix'.
    Data and associated document added to the detector db with extended name, e.g. epix10ka_000001_mysandbox
    All document fields stay unchanged.
    """
    logger.debug('add_data_and_doc_to_detdb_extended kwargs: %s' % str(kwargs))

    short = pro_detector_name(detname_long, add_shortname=True)

    dbname_det = detector_dbname(short, **kwargs)
    colname = short

    kwargs['detector']  = short # ex: epix10ka_000001
    kwargs['shortname'] = short # ex: epix10ka_000001
    kwargs['longname']  = detname_long     # ex: epix10ka_<_uniqueid>
    #kwargs['detname']  = det_name # already in kwargs ex: epixquad
    kwargs['id_data_exp'] = 'N/A'
    kwargs['id_doc_exp']  = 'N/A'
    resp = add_data_and_doc(data, dbname_det, colname, **kwargs)
    return resp # None or (id_data_det, id_doc_det)


def deploy_constants(data, exp, detname_long, **kwa):
    """Deploys constants depending on dbsuffix."""

    detname = pro_detector_name(detname_long, add_shortname=False)
    ctype = kwa.get('ctype','')
    dbsuffix = kwa.get('dbsuffix','')

    resp = add_data_and_doc_to_detdb_extended(data, exp, detname_long, **kwa) if dbsuffix else\
           add_data_and_two_docs(data, exp, detname_long, **kwa)

    if resp is None:
        logger.warning(f'CONSTANTS ARE NOT DEPLOYED for exp:{exp} det:{detname} dbsuffix:{dbsuffix} ctype:{ctype}')
        return None

    id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
          (None, resp[0], None, resp[1]) if dbsuffix else resp

    logger.debug('deployed with id_data_exp:%s and id_data_det:%s id_doc_exp:%s id_doc_det:%s' %\
                 (id_data_exp, id_data_det, id_doc_exp, id_doc_det))
    logger.info('  constants are deployed in DB(s) for exp:%s detector:%s dbsuffix:%s ctype:%s run:%d run_beg:%s run_end:%s'%\
                (exp, detname, dbsuffix, ctype, kwa['run'], str(kwa.get('run_beg',None)), str(kwa.get('run_end',None))))

    return id_data_exp, id_data_det, id_doc_exp, id_doc_det


def _add_detector_name(dbname, colname, detname, detnum):
    """ Adds document for detector names and returns short detector name for long input name detname."""
    doc = mu._doc_detector_name(detname, colname, detnum)
    logger.debug(f'_add_detector_name doc: {str(doc)}')
    id_doc = add_document(dbname, colname, doc)
    return doc.get('short', None) if id_doc is not None else None


def _short_detector_name(detname, dbname=cc.DETNAMESDB, add_shortname=False):
    """Returns short detector name for long input name detname."""
    colname = detname.split('_',1)[0]
    # find a single doc for long detname
    query = {'long':detname}
    ldocs = find_docs(dbname, colname, query=query)

    logger.debug(f'_short_detector_name: db/collection {dbname}/{colname} query={query} list of docs: {info_ldocs(ldocs)}')

    if ldocs is None:
        logger.warning(f'db/collection {dbname}/{colname} NO DOCUMENT FOUND FOR long detname {detname}')

    elif len(ldocs)>1:
        logger.warning(f'db/collection: {dbname}/{colname} has >1 document for detname: {detname}')
        #sys.exit(f'EXIT: db/collection {dbname}/{colname} HAS TO BE FIXED')

    elif len(ldocs)==1:
        shortname = ldocs[0].get('short', None)
        if shortname is not None:
            return shortname

    # find all docs in the collection for the next detector number
    query={}
    ldocs = find_docs(dbname, colname, query=query)
    if ldocs is not None and len(ldocs)>0:
        logger.debug('doc[0] keys: ' + info_dict_keys(ldocs[0]))
    logger.debug(f'db/collection {dbname}/{colname} query={query} list of docs: {info_detnames(ldocs)}')

    # find detector for partial name
    if ldocs is not None:
        shortname = mu._short_for_partial_name(detname, ldocs)
        if shortname is not None: return shortname

    if not add_shortname: return None

    # add new short name to the db
    detnum = 0
    if not ldocs or ldocs is None: # empty list
        logger.debug(f'List of documents in db/collection: {dbname}/{colname} IS EMPTY')
        detnum = 1
    else:
        for doc in ldocs:
            num = doc.get('seqnumber', 0)
            if num > detnum: detnum = num
        detnum += 1

    logger.debug(f'new detector detnum: {str(detnum)}')
    short_name = _add_detector_name(dbname, colname, detname, detnum)
    logger.debug(f'add document to db/collection: {dbname}/{colname} doc for short name: {short_name}')

    return short_name


def pro_detector_name(detname, add_shortname=False, **kwa): # DEPRECATED: maxsize=cc.MAX_DETNAME_SIZE
    """Returns short detector name"""
    if detname is None: return None
    assert isinstance(detname, str), f'non-string detname: {str(detname)}'
    short = _short_detector_name(detname, add_shortname=add_shortname)

    logger.debug(f'pro_detector_name detname: {detname} short: {short} add_shortname: {add_shortname}')

    return short
    #return detname if len(detname)<maxsize else _short_detector_name(detname, add_shortname=add_shortname)


def delete_database(dbname, **kwa):
    """Deletes database for (str) dbname, e.g. dbname='cdb_opal_0001'."""
    check_ticket()
    r = delete_cmd(cc.URL_KRB+dbname)
    logger.debug(r.text)
    return r


def delete_databases(list_db_names, **kwa):
    """Deletes databases specified in the list_db_names."""
    msg = 'delete databases: %s' % (' '.join(list_db_names))
    for dbname in list_db_names:
        r = delete_database(dbname)
        msg += '\n  delete: %s resp: %s' % (dbname, r.text)
    logger.debug(msg)


def delete_collection(dbname, colname, **kwa):
    """Deletes collection from database."""
    check_ticket()
    r = delete_cmd(cc.URL_KRB+dbname+'/'+colname)
    logger.debug(r.text)
    return r


def delete_collections(dic_db_cols, **kwa):
    """Delete collections specified in the dic_db_cols consisting of pairs {dbname:lstcols}."""
    msg = 'Delete collections:'
    for dbname, lstcols in dic_db_cols.items():
        msg += '\nFrom database: %s delete collections: %s' % (dbname, ' '.join(lstcols))
        for colname in lstcols:
            resp = delete_collection(dbname, colname)
            msg += '\n  delete: %s resp: %s' % (colname, resp.text)
    logger.debug(msg)


def delete_document(dbname, colname, doc_id, **kwa):
    """Deletes document for specified _id from database/collection."""
    check_ticket()
    r = delete_cmd(cc.URL_KRB+dbname+'/'+colname+'/'+ doc_id)
    logger.debug(r.text)
    return r


def delete_data(dbname, data_id, **kwa):
    """Deletes data for specified data_id from database/gridfs."""
    if data_id is None:
        logger.warning('CAN NOT DELETE DATA FOR INPUT PARAMETERS DB/data_id: %s/%s' % (dbname, data_id))
        return False
    uri = cc.URL_KRB+dbname+'/gridfs/'+ data_id
    r = delete_cmd(uri)
    logger.debug(f'delete {uri} responce: {r.text}')
    return r


def delete_document_and_data(dbname, colname, doc_id, **kwa):
    """Deletes document for specified _id from database/collection and associated data from database/gridfs."""
    check_ticket()

    # find a single doc for doc_id
    ldocs = find_docs(dbname, colname, query=query_id_pro({"_id":doc_id}))
    if len(ldocs)>1:
        logger.error('UNEXPECTED ERROR: db/collection: %s/%s HAS MORE THAN ONE DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        sys.exit('EXIT: db/collection %s/%s HAS TO BE FIXED' % (dbname, colname))

    logger.debug(f'ldocs: {str(ldocs)}')

    if not ldocs:
        logger.warning('db/collection: %s/%s HAS NO DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        return False

    doc = ldocs[0]
    data_id = doc.get('id_data', None)
    uri = cc.URL_KRB+dbname+'/'+colname+'/'+ doc_id
    resp_doc = delete_cmd(uri)
    logger.debug('delete %s responce: %s' % (uri, resp_doc.text))

    if data_id is None:
        logger.warning('db/collection/doc_id: %s/%s/%s DOES NOT HAVE data_id' % (dbname, colname, doc_id))
        return False

    return delete_data(dbname, data_id).ok


def delete_documents(dbname, colname, doc_ids, **kwa):
    resp = None
    for doc_id in doc_ids:
        isok = delete_document_and_data(dbname, colname, doc_id)
        logger.debug(f'resp.ok {isok}')


def info_doc(dbname, colname, docid, strlen=150):
    ldocs = find_docs(dbname, colname, query=query_id_pro({"_id":docid}))
    if not ldocs:
        return f'db/collection: {dbname}/{colname} does not have any document'
    doc = ldocs[0]
    if not isinstance(doc, dict):
        return f'db/collection: {dbname}/{colname} document IS NOT dict: {str(doc)}'
    s = f'db/collection/Id: {dbname}/{colname}{docid} contains {str(len(doc))} items:'
    for k,v in doc.items():
        s += '\n  %s : %s' % (k.ljust(20), str(v)[:strlen])
    return s


def info_docs(dbname, colname, query={}, strlen=150):
    docs = find_docs(dbname, colname, query)
    if docs is None:
        return f'DB/collection {dbname}/{colname} DOCUMENTS NOT FOUND'
    return f'DB/collection {dbname}/{colname} contains {str(len(docs))} documents: {info_docs_list(docs, strlen=150)}'


def str_formatted_list(lst, ncols=5, width=24):
    s=''
    c=0
    for v in lst:
        s+=str(v).ljust(width)
        c+=1
        if c<ncols: continue
        s+='\n'
        c=0
    return s


def info_webclient(**kwargs):

    width = kwargs.get('width', 24)
    ptrn = mu.db_prefixed_name('') if kwargs.get('cdbonly', False) else None
    dbnames = database_names(pattern=ptrn)
    if dbnames is None: return 'NO dbnames found for url: %s pattern: %s' % (cc.URL, ptrn)

    dbname = mu.get_dbname(**kwargs)

    if dbname is None:
        s = '\n=== web client %s contains %d databases for name pattern "%s":\n%s\n\n'%\
            (cc.URL, len(dbnames), str(ptrn), str_formatted_list(dbnames))
        for name in dbnames:
             colnames = collection_names(name)
             s += '%s %2d cols: %s\n' % (str(name).ljust(width), len(colnames), str(colnames))
        return s

    if not (dbname in dbnames):
        return '\n=== database %s is not found in the list of known:\n%s'%\
               (dbname, str_formatted_list(dbnames))

    colname = mu.get_colname(**kwargs)
    colnames = collection_names(dbname)

    if colname is None:
        if colnames is None: return '\n=== colnames is None: database %s is empty ???' % (dbname)
        s = '\n=== database %s contains %d collections: %s\n' % (dbname, len(colnames), str(colnames))
        for cname in colnames:
             s += '%s\n' % info_docs(dbname, cname)
        return s

    if not(colname in colnames):
        return '\n=== database %s does not have collection %s in the list: %s' % (dbname, colname, str(colnames))

    docid = kwargs.get('docid', None)
    if docid is None: return info_docs(dbname, colname)

    return info_doc(dbname, colname, docid)


def valid_post_privilege(dbname):
    """2021-01-25 Murali suggested this interface to test privilage to write in db

    url_ws = "https://pswww.slac.stanford.edu/ws-kerb/calib_ws/cdb_xpptut15/test_post_privilege"
    krbheaders = KerberosTicket("HTTP@" + urlparse(url_ws).hostname).getAuthHeaders()
    try:
        r = req.get(url_ws, headers=cc.krbheaders())
        r.raise_for_status()
        print(r.json())
    except:
        print("Exception; possibly no privilege")
    """
    url_krb=cc.URL_KRB
    url_ws = f'{url_krb}{dbname}/test_post_privilege'
    logger.debug(f'valid_post_privilege url_ws: {url_ws}')

    if not has_jwt:
        if cc.krbheaders() is None:
           return False

    logger.info(info_jwt)

    r = request(url_ws, timeout=180)

    if r is None:
        logger.warning(f'\nrequest to {url_ws} is None    FOR NOW LET IT SLIDE IN DEVELOPMENT ??? NO PRIVILAGE TO WRITE IN DB: {dbname}')
        return True

    logger.debug(f'request url: {url_ws} response status: {r.ok} status_code: {r.status_code} reason: {r.reason}')

    if not r.ok:
        logger.warning(f'\nNO PRIVILAGE TO WRITE IN DB: {dbname}')
    return r.ok


def my_sort_parameter(e): return e['_id']


def collection_info(dbname, cname, **kwa):
    """Returns (str) info about collection documents."""
    s = 'DB %s collection %s' % (dbname, cname)

    docs = find_docs(dbname, cname)
    if not docs: return s
    s += ' contains %d docs\n' % len(docs)

    docs = sorted(docs, key=my_sort_parameter) #, reverse=True

    doc = docs[0]
    s += '\n  %s' % mu.document_keys(doc) # str(doc.keys())

    _, title = mu.document_info(doc, **kwa)
    s += '\n  doc# %s' % title

    for idoc, doc in enumerate(docs):
        vals,_ = mu.document_info(doc, **kwa)
        s += '\n  %4d %s' % (idoc, vals)

    return s


def list_of_documents(dbname, cname):
    docs = find_docs(dbname, cname)
    if not docs: return []
    docs = sorted(docs, key=my_sort_parameter) #, reverse=True
    return docs


if __name__ == "__main__":
    sys.exit('\nFor test use ./ex_%s <test-number> <mode> <...>' % sys.argv[0].rsplit('/')[-1])

# EOF
