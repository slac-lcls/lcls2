"""
Usage ::

    # Import
    import psana.pscalib.calib.MDBWebUtils as wu
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    resp = wu.check_kerberos_ticket(exit_if_invalid=True)
    q = wu.query_id_pro(query) # e.i., query={"_id":doc_id}
    _ = wu.request(url, query=None)
    _ = wu.database_names(url=cc.URL)
    _ = wu.collection_names(dbname, url=cc.URL)
    _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.select_latest_doc(docs, query):
    _ = wu.get_doc_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_id(dbname, dataid, url=cc.URL)
    _ = wu.get_data_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_doc(dbname, doc, url=cc.URL)
    data,doc = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL)
    d = wu.calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL, dbsuffix='')
    d = {ctype:(data,doc),}

    id = wu.add_data_from_file(dbname, fname, sfx=None, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    id = wu.add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    id = wu.add_document(dbname, colname, doc, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    id_data, id_doc = wu.add_data_and_doc(data, dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs)
    id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
      wu.add_data_and_two_docs(data, exp, det, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs)

    detname_short = wu.pro_detector_name(detname, add_shortname=False)

    resp = wu.delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    resp = wu.delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    resp = wu.delete_document(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    resp = wu.delete_data(dbname, data_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    resp = wu.delete_document_and_data(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)

    s = wu.str_formatted_list(lst, ncols=5, width=24)
    s = wu.info_docs(dbname, colname, query={}, url=cc.URL, strlen=120)
    s = wu.info_webclient(**kwargs)
    resp = wu.valid_post_privilege(dbname, url_krb=cc.URL_KRB)

    test_*()
"""

import logging
logger = logging.getLogger(__name__)

import sys
import numpy as np
import io

import psana.pscalib.calib.CalibConstants as cc
from requests import get, post, delete #put

from time import time
from numpy import fromstring
import psana.pscalib.calib.MDBUtils as mu
import psana.pyalgos.generic.Utils as gu
from subprocess import call

def has_kerberos_ticket():
    """Checks to see if the user has a valid Kerberos ticket."""
    return not call(["klist", "-s"])


def check_kerberos_ticket(exit_if_invalid=True):
    if has_kerberos_ticket(): return True
    logger.error('KERBEROS TICKET IS UNAVAILABLE OR EXPIRED. Requested operation requires valid kerberos ticket')
    if exit_if_invalid:
        sys.exit('FIX KERBEROS TICKET - use command "kinit" or check its status with command "klist"')
    return False


def query_id_pro(query):
    id = query.get('_id', None)
    if (id is None) or ('ObjectId' in id): return query
    query['_id'] = 'ObjectId(%s)'%id
    return query


def request(url, query=None):
    #t0_sec = time()
    r = get(url, query, timeout=180)
    #dt = time()-t0_sec # ~30msec
    #logger.debug('CONSUMED TIME by request %.3f sec\n  for url=%s  query=%s' % (dt, url, str(query)))
    if r.ok: return r
    s = 'get url: %s query: %s\n  response status: %s status_code: %s reason: %s'%\
        (url, str(query), r.ok, r.status_code, r.reason)
    s += '\nTry command: curl -s "%s"' % url
    logger.debug(s)
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
    r = request('%s/%s'%(url,dbname))
    if r is None: return None
    return r.json()


# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll?query_string=%7B%20%22item%22..."
def find_docs(dbname, colname, query={}, url=cc.URL):
    """Returns list of documents for query, e.g. query={'ctype':'pedestals', "run":{ "$gte":80}}."""
    uri = '%s/%s/%s'%(url,dbname,colname)
    query_string=str(query).replace("'",'"')
    logger.debug('find_docs uri: %s query: %s' % (uri, query_string))
    r = request(uri, {"query_string": query_string})
    if r is None: return None
    try:
        return r.json()
    except:
        msg = 'WARNING: find_docs responce: %s' % str(r)\
            + '\n     conversion to json failed, return None for query: %s' % str(query)
        logger.debug(msg)
        return None


def find_doc(dbname, colname, query={}, url=cc.URL): #query={'ctype':'pedestals'}
    """Returns document for query.
       1. finds all documents for query
       2. select the latest for run or time_sec
    """

    logger.debug('find_doc input pars dbname: %s colname: %s query:%s' % (dbname, colname, str(query)))

    docs = find_docs(dbname, colname, query, url)
    if docs is None: return None

    return select_latest_doc(docs, query)


def select_latest_doc(docs, query):
    """Returns a single document for query selected by time_sec (if available) or run."""
    if len(docs)==0:
        # commented out by cpo since this happens routinely the way
        # that Mona is fetching calibration constants in psana.
        #logger.warning('find_docs returns list of length 0 for query: %s' % query)
        return None

    qkeys = query.keys()
    key_sort = 'time_sec' if 'time_sec' in qkeys else 'run'

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


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/5b6893e81ead141643fe4344"
def get_doc_for_docid(dbname, colname, docid, url=cc.URL):
    """Returns document for docid."""
    r = request('%s/%s/%s/%s'%(url,dbname,colname,docid))
    if r is None: return None
    return r.json()


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a"
def get_data_for_id(dbname, dataid, url=cc.URL):
    """Returns raw data from GridFS, at this level there is no info for parsing."""
    r = request('%s/%s/gridfs/%s'%(url,dbname,dataid))
    if r is None: return None
    logger.debug('get_data_for_docid:'\
                +'\n  r.status_code: %s\n  r.headers: %s\n  r.encoding: %s\n  r.content: %s...\n' %
                 (str(r.status_code),  str(r.headers),  str(r.encoding),  str(r.content[:50])))
    return r.content


def get_data_for_docid(dbname, colname, docid, url=cc.URL):
    """Returns data from GridFS using docid."""
    doc = get_doc_for_docid(dbname, colname, docid, url)
    logger.debug('get_data_for_docid: %s' % str(doc))
    return get_data_for_doc(dbname, doc, url)


# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def get_data_for_doc(dbname, doc, url=cc.URL):
    """Returns data from GridFS using doc."""
    logger.debug('get_data_for_doc: %s', str(doc))
    idd = doc.get('id_data', None)
    if idd is None:
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None

    #print('curl -s "%s"' % ('%s/%s/gridfs/%s'%(url,dbname,idd)))
    r2 = request('%s/%s/gridfs/%s'%(url,dbname,idd))
    if r2 is None: return None
    s = r2.content

    return mu.object_from_data_string(s, doc)


def dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dtype=None, dbsuffix=''):
    """wrapper for MDBUtils.dbnames_collection_query,
       - which should receive short detector name, othervice uses direct interface to DB
    """
    short = pro_detector_name(det)
    logger.debug('short: %s dbsuffix: %s' % (short, dbsuffix))
    resp = list(mu.dbnames_collection_query(short, exp, ctype, run, time_sec, vers, dtype))
    if dbsuffix: resp[0] = detector_dbname(short, dbsuffix=dbsuffix)

    return resp


def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL, dbsuffix=''):
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
    doc = find_doc(dbname, colname, query, url)
    if doc is None:
        # commented out by cpo since this happens routinely the way
        # that Mona is fetching calibration constants in psana.
        logger.debug('document is not available for query: %s' % str(query))
        return (None, None)
    return (get_data_for_doc(dbname, doc, url), doc)


def calib_constants_of_missing_types(resp, det, time_sec=None, vers=None, url=cc.URL):
    """ try to add constants of missing types in resp using detector db."""
    exp=None
    run=9999
    ctype=None
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers, dtype=None)
    dbname = db_det
    docs = find_docs(dbname, colname, query, url)
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
        resp[ct] = (get_data_for_doc(dbname, doc, url), doc)

    return resp


def calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL, dbsuffix=''):
    """Returns constants for all ctype-s."""
    t0_sec = time()
    ctype=None
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers, dtype=None, dbsuffix=dbsuffix)
    dbname = db_det if dbsuffix or (exp is None) else db_exp

    #print('time 1: %.6f sec - for DB %s generate query %s' % (time()-t0_sec, dbname, query))

    docs = find_docs(dbname, colname, query, url)
    #logger.debug('find_docs: number of docs found: %d' % len(docs))
    #print('time 2: %.6f sec - find docs for query in DB %s' % (time()-t0_sec, dbname))
    if docs is None: return None

    ctypes = set([d.get('ctype',None) for d in docs])
    ctypes.discard(None)
    logger.debug('calib_constants_all_types - found ctypes: %s' % str(ctypes))

    resp = {}
    for ct in ctypes:
        docs_for_type = [d for d in docs if d.get('ctype',None)==ct]
        doc = select_latest_doc(docs_for_type, query)
        if doc is None: continue
        resp[ct] = (get_data_for_doc(dbname, doc, url), doc)
        #print('        %.6f sec - get data for ctype: %s' % (time()-t0_sec, ct))

    #print('time 3: %.6f sec - get data for docs total' % (time()-t0_sec))

    resp = calib_constants_of_missing_types(resp, det, time_sec, vers, url)

    #print('time 4: %.6f sec - check for missing types in the det DB' % (time()-t0_sec))

    return resp


def add_data_from_file(dbname, fname, sfx=None, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds data from file to the database/gridfs."""
    check_kerberos_ticket()

    _sfx = sfx if sfx is not None else fname.rsplit('.')[-1]
    files = [('files',  (fname, open(fname, 'rb'), 'image/'+_sfx))]
    resp = post(url+dbname+'/gridfs/', headers=krbheaders, files=files)
    logger.debug('add_data_from_file: %s to %s/gridfs/ resp: %s type: %s' % (fname, dbname, resp.text, type(resp)))
    #jdic = resp.json() # type <class 'dict'>
    return resp.json().get('_id',None)


def add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds binary data to the database/gridfs."""
    check_kerberos_ticket()

    headers = dict(krbheaders) # krbheaders <class 'dict'>
    headers['Content-Type'] = 'application/octet-stream'
    f = io.BytesIO(mu.encode_data(data))   # io.StringIO(data)
    d = f.read()
    #logger.debug('add_data byte-data:',d)
    resp = post(url+dbname+'/gridfs/', headers=headers, data=d)
    logger.debug('add_data: to %s/gridfs/ resp: %s' % (dbname, resp.text))
    id = resp.json().get('_id',None)
    if id is None: logger.warning('id_data is None')
    return id


def add_document(dbname, colname, doc, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds document to database collection."""
    check_kerberos_ticket()

    resp = post(url+dbname+'/'+colname+'/', headers=krbheaders, json=doc)
    logger.debug('add_document: %s\n  to %s/%s resp: %s' % (str(doc), dbname, colname, resp.text))
    id = resp.json().get('_id',None)
    if id is None: logger.warning('id_document is None')
    return id


def add_data_and_doc(data, _dbname, _colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs):
    """Check permission and add data and document to the db."""
    logger.debug('add_data_and_doc kwargs: %s' % str(kwargs))

    # check permission
    t0_sec = time()
    if not valid_post_privilege(_dbname, url_krb=url): return None

    id_data = add_data(_dbname, data, url, krbheaders)
    if id_data is None: return None
    doc = mu.docdic(data, id_data, **kwargs) # ObjectId(id_data)???
    logger.debug(mu.doc_info(doc, fmt='  %s:%s')) #sep='\n  %16s : %s'

    id_doc = add_document(_dbname, _colname, doc, url, krbheaders)
    if id_doc is None: return None

    msg = 'Add data and doc time %.6f sec' % (time()-t0_sec)\
        + '\n  - data in %s/gridfs id: %s and doc in collection %s id: %s' % (_dbname, id_data, _colname, id_doc)
    logger.debug(msg)

    return id_data, id_doc


def insert_document_and_data(dbname, colname, dicdoc, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """DEPRECATED - wrapper for pymongo compatability - is used in graphqt/CMWDB*.py"""
    return add_data_and_doc(data, dbname, colname, url, krbheaders, **dicdoc)


def add_data_and_two_docs(data, exp, detname_long, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs):
    """Add data and document to experiment and detector data bases."""
    logger.debug('add_data_and_two_docs kwargs: %s' % str(kwargs))

    shortname = pro_detector_name(detname_long, add_shortname=True)

    colname = shortname
    dbname_exp = mu.db_prefixed_name(exp)
    dbname_det = mu.db_prefixed_name(shortname)
    
    ctype = kwargs.get('ctype','N/A')
    logger.info('add_data_and_two_docs save constants: %s in DBs: %s %s collection: %s' % (ctype, dbname_exp, dbname_det, colname))

    #kwargs['detector'] = detname         # ex: epix10ka
    kwargs['shortname'] = shortname       # ex: epix10ka_000001
    kwargs['longname']  = detname_long    # ex: epix10ka_<_uniqueid>

    resp = add_data_and_doc(data, dbname_exp, colname, url=url, krbheaders=krbheaders, **kwargs)
    if resp is None: return None
    id_data_exp, id_doc_exp = resp

    kwargs['id_data_exp'] = id_data_exp # override
    kwargs['id_doc_exp']  = id_doc_exp  # add
    resp = add_data_and_doc(data, dbname_det, colname, url=url, krbheaders=krbheaders, **kwargs)
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
    if dbsuffix: dbname_det += '_%s'% dbsuffix
    assert len(dbname_det) < 50
    logger.debug('detector_dbname detname: %s dbsuffix: %s returns: %s' % (detname_short, dbsuffix, dbname_det))
    return dbname_det


def add_data_and_doc_to_detdb_extended(data, exp, detname_long, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs):
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
    resp = add_data_and_doc(data, dbname_det, colname, url=url, krbheaders=krbheaders, **kwargs)
    return resp # None or (id_data_det, id_doc_det)


def deploy_constants(data, exp, detname_long, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwa):
    """Deploys constants depending on dbsuffix."""

    detname = pro_detector_name(detname_long, add_shortname=False)
    ctype = kwa.get('ctype','')
    dbsuffix = kwa.get('dbsuffix','')

    resp = add_data_and_doc_to_detdb_extended(data, exp, detname_long, url=url, krbheaders=krbheaders, **kwa) if dbsuffix else\
           add_data_and_two_docs(data, exp, detname_long, url=url, krbheaders=krbheaders, **kwa)

    if resp is None:
        logger.warning('CONSTANTS ARE NOT DEPLOYED for exp:%s det:%s dbsuffix:%s ctype:%s' %\
                       (exp, detname, dbsuffix, ctype))
        return None

    id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
          (None, resp[0], None, resp[1]) if dbsuffix else resp

    logger.debug('deployed with id_data_exp:%s and id_data_det:%s id_doc_exp:%s id_doc_det:%s' %\
                 (id_data_exp, id_data_det, id_doc_exp, id_doc_det))
    logger.info('  constants are deployed in DB(s) for exp:%s det:%s dbsuffix:%s ctype:%s' % (exp, detname, dbsuffix, ctype))

    return id_data_exp, id_data_det, id_doc_exp, id_doc_det


def _add_detector_name(dbname, colname, detname, detnum):
    """ Adds document for detector names and returns short detector name for long input name detname."""
    check_kerberos_ticket()
    doc = mu._doc_detector_name(detname, colname, detnum)
    id_doc = add_document(dbname, colname, doc) #, url, krbheaders)
    return doc.get('short', None) if id_doc is not None else None


def _short_detector_name(detname, dbname=cc.DETNAMESDB, add_shortname=False):
    """Returns short detector name for long input name detname."""
    colname = detname.split('_',1)[0]
    # find a single doc for long detname
    query = {'long':detname}
    ldocs = find_docs(dbname, colname, query=query)

    logger.debug('db/collection %s/%s query=%s list of docs: %s' % (dbname, colname, query, str(ldocs)))

    if ldocs is None:
        logger.warning('db/collection %s/%s NO DOCUMENT FOUND FOR long detname %s' % (dbname, colname, detname))

    if len(ldocs)>1:
        logger.warning('db/collection: %s/%s has >1 document for detname: %s' % (dbname, colname, detname))
        #sys.exit('EXIT: db/collection %s/%s HAS TO BE FIXED' % (dbname, colname))

    if len(ldocs)==1:
        shortname = ldocs[0].get('short', None)
        if shortname is not None:
            return shortname

    # find all docs in the collection
    query={}
    ldocs = find_docs(dbname, colname, query=query)
    logger.debug('db/collection %s/%s query=%s list of docs: %s' % (dbname, colname, query, str(ldocs)))

    # find detector for partial name
    shortname = mu._short_for_partial_name(detname, ldocs)
    if shortname is not None: return shortname

    if not add_shortname: return None

    # add new short name to the db
    detnum = 0
    if not ldocs or ldocs is None: # empty list
        logger.debug('List of documents in db/collection: %s/%s IS EMPTY' % (dbname, colname))
        detnum = 1
    else:
        for doc in ldocs:
            num = doc.get('seqnumber', 0)
            if num > detnum: detnum = num
        detnum += 1

    short_name = _add_detector_name(dbname, colname, detname, detnum)
    logger.debug('add document to db/collection: %s/%s doc for short name:%s' % (dbname, colname, short_name))

    return short_name


def pro_detector_name(detname, maxsize=cc.MAX_DETNAME_SIZE, add_shortname=False):
    """Returns short detector name if its length exceeds cc.MAX_DETNAME_SIZE chars."""
    assert isinstance(detname,str), 'unexpected detname: %s' % str(detname)
    return detname if len(detname)<maxsize else _short_detector_name(detname, add_shortname=add_shortname)


def delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes database for (str) dbname, e.g. dbname='cdb_opal_0001'."""
    check_kerberos_ticket()
    resp = delete(url+dbname, headers=krbheaders)
    logger.debug(resp.text)
    return resp


def delete_databases(list_db_names, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes databases specified in the list_db_names."""
    msg = 'delete databases: %s' % (' '.join(list_db_names))
    for dbname in list_db_names:
        resp = delete_database(dbname, url, krbheaders)
        msg += '\n  delete: %s resp: %s' % (dbname, resp.text)
    logger.debug(msg)


def delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes collection from database."""
    check_kerberos_ticket()
    resp = delete(url+dbname+'/'+colname, headers=krbheaders)
    logger.debug(resp.text)
    return resp


def delete_collections(dic_db_cols, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Delete collections specified in the dic_db_cols consisting of pairs {dbname:lstcols}."""
    msg = 'Delete collections:'
    for dbname, lstcols in dic_db_cols.items():
        msg += '\nFrom database: %s delete collections: %s' % (dbname, ' '.join(lstcols))
        for colname in lstcols:
            resp = delete_collection(dbname, colname, url, krbheaders)
            msg += '\n  delete: %s resp: %s' % (colname, resp.text)
    logger.debug(msg)


def delete_document(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes document for specified _id from database/collection."""
    check_kerberos_ticket()
    resp = delete(url+dbname+'/'+colname+'/'+ doc_id, headers=krbheaders)
    logger.debug(resp.text)
    return resp


def delete_data(dbname, data_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes data for specified data_id from database/gridfs."""
    if data_id is None:
        logger.warning('CAN NOT DELETE DATA FOR INPUT PARAMETERS DB/data_id: %s/%s' % (dbname, data_id))
        return False
    uri = url+dbname+'/gridfs/'+ data_id
    resp_data = delete(uri, headers=krbheaders)
    logger.debug('delete %s responce: %s' % (uri, resp_data.text))
    return resp_data


def delete_document_and_data(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes document for specified _id from database/collection and associated data from database/gridfs."""
    check_kerberos_ticket()

    # find a single doc for doc_id
    ldocs = find_docs(dbname, colname, query=query_id_pro({"_id":doc_id}))
    if len(ldocs)>1:
        logger.error('UNEXPECTED ERROR: db/collection: %s/%s HAS MORE THAN ONE DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        sys.exit('EXIT: db/collection %s/%s HAS TO BE FIXED' % (dbname, colname))

    logger.debug('ldocs: %s' % str(ldocs))

    if not ldocs:
        logger.warning('db/collection: %s/%s HAS NO DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        return False

    doc = ldocs[0]
    data_id = doc.get('id_data', None)
    uri = url+dbname+'/'+colname+'/'+ doc_id
    resp_doc = delete(uri, headers=krbheaders)
    logger.debug('delete %s responce: %s' % (uri, resp_doc.text))

    if data_id is None:
        logger.warning('db/collection/doc_id: %s/%s/%s DOES NOT HAVE data_id' % (dbname, colname, doc_id))
        return False

    return delete_data(dbname, data_id, url, krbheaders)


def delete_documents(dbname, colname, doc_ids, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    resp = None
    for doc_id in doc_ids:
        resp = delete_document_and_data(dbname, colname, doc_id, url, krbheaders)
        logger.debug(resp.text)


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


def info_doc(dbname, colname, docid, strlen=150):
    ldocs = find_docs(dbname, colname, query=query_id_pro({"_id":docid}), url=cc.URL)
    if not ldocs:
        return 'db/collection: %s/%s does not have any document' % (dbname, colname)
    doc = ldocs[0]
    if not isinstance(doc, dict):
        return 'db/collection: %s/%s document IS NOT dict: %s' % (dbname, colname, str(doc))
    s = 'db/collection/Id: %s/%s/%s contains %d items:' % (dbname, colname, docid, len(doc))
    for k,v in doc.items():
        s += '\n  %s : %s' % (k.ljust(20), str(v)[:strlen])
    return s


def info_docs_list(docs, strlen=150):
    if not isinstance(docs, list):
        return 'info_docs_list parameter docs is not list: %s' % str(docs)
    s = ''
    for i,d in enumerate(docs):
        s += '\n%04d %s ...' % (i, str(d)[:strlen])
    return s


def info_docs(dbname, colname, query={}, url=cc.URL, strlen=150):
    docs = find_docs(dbname, colname, query, url=cc.URL)
    if docs is None:
        return 'DB/collection %s/%s DOCUMENTS NOT FOUND' % (dbname, colname)
    return 'DB/collection %s/%s contains %d documents:%s' %\
           (dbname, colname, len(docs), info_docs_list(docs, strlen=150))


def info_webclient(**kwargs):

    width = kwargs.get('width', 24)
    ptrn = mu.db_prefixed_name('') if kwargs.get('cdbonly', False) else None
    dbnames = database_names(url=cc.URL, pattern=ptrn)
    if dbnames is None: return 'NO dbnames found for url: %s pattern: %s' % (cc.URL, ptrn)

    dbname = mu.get_dbname(**kwargs)
    if dbname is None:
        s = '\n=== web client %s contains %d databases for name pattern "%s":\n%s\n\n'%\
            (cc.URL, len(dbnames), str(ptrn), str_formatted_list(dbnames))
        for name in dbnames:
             colnames = collection_names(name, url=cc.URL)
             s += '%s %2d cols: %s\n' % (str(name).ljust(width), len(colnames), str(colnames))
        return s

    if not (dbname in dbnames):
        return '\n=== database %s is not found in the list of known:\n%s'%\
               (dbname, str_formatted_list(dbnames))

    colname = mu.get_colname(**kwargs)
    colnames = collection_names(dbname, url=cc.URL)

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


def valid_post_privilege(dbname, url_krb=cc.URL_KRB):
    """2021-01-25 Murali suggested this interface to test privilage to write in db

    ws_url = "https://pswww.slac.stanford.edu/ws-kerb/calib_ws/cdb_xpptut15/test_post_privilege"
    krbheaders = KerberosTicket("HTTP@" + urlparse(ws_url).hostname).getAuthHeaders()
    try:
        r = get(ws_url, headers=krbh_test)
        r.raise_for_status()
        print(r.json())
    except:
        print("Exception; possibly no privilege")
    """
    ws_url = "%s%s/test_post_privilege" % (url_krb, dbname)
    logger.debug('valid_post_privilege ws_url: %s'% ws_url)

    try:
        krbh_test = cc.KerberosTicket("HTTP@" + cc.urlparse(ws_url).hostname).getAuthHeaders()
    except Exception as err: #except kerberos.GSSError as err:
        logger.warning('KerberosTicket error: %s' % str(err))
        logger.warning('BEFORE RUNNING THIS SCRIPT TRY COMMAND: kinit')
        return False

    r = get(ws_url, headers=krbh_test, timeout=180)

    logger.debug('get url: %s response status: %s status_code: %s reason: %s'%\
            (ws_url, r.ok, r.status_code, r.reason))
    if not r.ok:
        logger.warning('\nNO PRIVILAGE TO WRITE IN DB: %s' % dbname)
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
