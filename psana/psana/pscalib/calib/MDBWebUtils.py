"""
Usage ::

    # Import
    import psana.pscalib.calib.MDBWebUtils as wu
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    _ = wu.request(url, query=None)
    _ = wu.database_names(url=cc.URL)
    _ = wu.collection_names(dbname, url=cc.URL)
    _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.select_latest_doc(docs, query):
    _ = wu.get_doc_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_id(dbname, dataid, url=cc.URL)
    _ = wu.get_data_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_doc(dbname, colname, doc, url=cc.URL)
    data,doc = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL)
    d = wu.calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL)
    d = {ctype:(data,doc),}

    test_*()
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import numpy as np

import psana.pscalib.calib.CalibConstants as cc
from requests import get, post, delete #put
import json
from time import time
from numpy import fromstring
#from psana.pscalib.calib.MDBUtils import dbnames_collection_query, object_from_data_string
import psana.pscalib.calib.MDBUtils as mu
#from bson.objectid import ObjectId

import psana.pyalgos.generic.Utils as gu

#------------------------------

from subprocess import call
#from psana.pyalgos.generic.Utils import has_kerberos_ticket
def has_kerberos_ticket():
    """Checks to see if the user has a valid Kerberos ticket"""
    return not call(["klist", "-s"])

#------------------------------

def check_kerberos_ticket(exit_if_invalid=True):
    if has_kerberos_ticket(): return True
    logger.warning('KERBEROS TICKET IS UNAVAILABLE OR EXPIRED. Requested operation requires valid kerberos ticket')
    if exit_if_invalid : 
        import sys
        sys.exit('FIX KERBEROS TICKET - use command "kinit" or check its status with command "klist"')
    return False

#------------------------------

def request(url, query=None):
    #logger.debug('==== query: %s' % str(query))
    t0_sec = time()
    r = get(url, query)
    dt = time()-t0_sec
    logger.debug('CONSUMED TIME by request %.6f sec\n  for url=%s  query=%s' % (dt, url, str(query)))
    return r

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db"
def database_names(url=cc.URL):
    """Returns list of database names for url.
    """
    r = request(url)
    return r.json()

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll"
def collection_names(dbname, url=cc.URL):
    """Returns list of collection names for dbname and url.
    """
    r = request('%s/%s'%(url,dbname))
    return r.json()

#------------------------------
# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll?query_string=%7B%20%22item%22..."
def find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL):
    """Returns list of documents for query, e.g. query={'ctype':'pedestals', "run":{ "$gte":80}}
    """
    query_string=str(query).replace("'",'"')
    logger.debug('find_docs query: %s' % query_string)
    r = request('%s/%s/%s'%(url,dbname,colname),{"query_string": query_string})
    try:
        return r.json()
    except:
        msg = '**** find_docs responce: %s' % str(r)\
            + '\n     conversion to json failed, return None for query: %s' % str(query)
        logger.warning(msg)
        return None

#------------------------------

def find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL):
    """Returns document for query.
       1. finds all documents for query
       2. select the latest for run or time_sec
    """
    docs = find_docs(dbname, colname, query, url)
    if docs is None : return None

    return select_latest_doc(docs, query)

#------------------------------

def select_latest_doc(docs, query):
    """Returns a single document for query selected by time_sec (if available) or run
    """
    if len(docs)==0 :
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
    for d in docs : 
        if d[key_sort]==val_sel : 
            return d
    return None

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/5b6893e81ead141643fe4344"
def get_doc_for_docid(dbname, colname, docid, url=cc.URL):
    """Returns document for docid.
    """
    r = request('%s/%s/%s/%s'%(url,dbname,colname,docid))
    return r.json()

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a" 
def get_data_for_id(dbname, dataid, url=cc.URL):
    """Returns raw data from GridFS, at this level there is no info for parsing.
    """
    r = request('%s/%s/gridfs/%s'%(url,dbname,dataid))
    logger.debug('get_data_for_docid:'\
                +'\n  r.status_code: %s\n  r.headers: %s\n  r.encoding: %s\n  r.content: %s...\n' % 
                 (str(r.status_code),  str(r.headers),  str(r.encoding),  str(r.content[:50])))
    return r.content

#------------------------------

def get_data_for_docid(dbname, colname, docid, url=cc.URL):
    """Returns data from GridFS using docid.
    """
    doc = get_doc_for_docid(dbname, colname, docid, url)
    logger.debug('get_data_for_docid: %s' % str(doc))
    return get_data_for_doc(dbname, colname, doc, url)

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def get_data_for_doc(dbname, colname, doc, url=cc.URL):
    """Returns data from GridFS using doc.
    """
    logger.debug('get_data_for_doc: %s', str(doc))
    idd = doc.get('id_data', None)
    if idd is None :
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None

    r2 = request('%s/%s/gridfs/%s'%(url,dbname,idd))
    s = r2.content

    return mu.object_from_data_string(s, doc)

#------------------------------

def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL):
    """Returns calibration constants and document with metadata for specified parameters. 
       To get meaningful constants, at least a few parameters must be specified, e.g.:
       - det, ctype, time_sec
       - det, ctype, version
       - det, exp, ctype, run
       - det, exp, ctype, time_sec
       - det, exp, ctype, run, version
       etc...
    """
    db_det, db_exp, colname, query = mu.dbnames_collection_query(det, exp, ctype, run, time_sec, vers)
    logger.debug('get_constants: %s %s %s %s' % (db_det, db_exp, colname, str(query)))
    dbname = db_det if exp is None else db_exp
    doc = find_doc(dbname, colname, query, url)
    if doc is None :
        # commented out by cpo since this happens routinely the way
        # that Mona is fetching calibration constants in psana.
        #logger.warning('document is not available for query: %s' % str(query))
        return (None, None)
    return (get_data_for_doc(dbname, colname, doc, url), doc)

#------------------------------

def calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL):
    """ returns constants for all ctype-s
    """
    ctype=None
    db_det, db_exp, colname, query = mu.dbnames_collection_query(det, exp, ctype, run, time_sec, vers)
    dbname = db_det if exp is None else db_exp
    docs = find_docs(dbname, colname, query, url)
    #logger.debug('find_docs: number of docs found: %d' % len(docs))
    if docs is None : return None

    ctypes = set([d.get('ctype',None) for d in docs])
    ctypes.discard(None)
    logger.debug('calib_constants_all_types - found ctypes: %s' % str(ctypes))

    resp = {}
    for ct in ctypes :
        docs_for_type = [d for d in docs if d.get('ctype',None)==ct]
        doc = select_latest_doc(docs_for_type, query)
        if doc is None : continue
        resp[ct] = (get_data_for_doc(dbname, colname, doc, url), doc)

    return resp

#------------------------------
#-------- 2020-04-30 ----------
#------------------------------

def add_data_from_file(dbname, fname, sfx=None, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds data from file to the database/gridfs.
    """
    check_kerberos_ticket()

    _sfx = sfx if sfx is not None else fname.rsplit('.')[-1]
    files = [('files',  (fname, open(fname, 'rb'), 'image/'+_sfx))]
    resp = post(url+dbname+'/gridfs/', headers=krbheaders, files=files)
    logger.debug('add_data_from_file: %s to %s/gridfs/ resp: %s type: %s' % (fname, dbname, resp.text, type(resp)))
    #jdic = resp.json() # type <class 'dict'>
    return resp.json().get('_id',None)

#------------------------------

def add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds binary data to the database/gridfs.
    """
    check_kerberos_ticket()

    import io
    headers = dict(krbheaders) # krbheaders <class 'dict'>
    headers['Content-Type'] = 'application/octet-stream'
    f = io.BytesIO(mu.encode_data(data))   # io.StringIO(data)
    d = f.read()
    #logger.debug('add_data byte-data:',d)
    resp = post(url+dbname+'/gridfs/', headers=headers, data=d)
    logger.debug('add_data: to %s/gridfs/ resp: %s' % (dbname, resp.text))
    id = resp.json().get('_id',None)
    if id is None : logger.warning('id_data is None')
    return id

#------------------------------

def add_document(dbname, colname, doc, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Adds document to database collection.
    """
    check_kerberos_ticket()

    resp = post(url+dbname+'/'+colname+'/', headers=krbheaders, json=doc)
    logger.debug('add_document: %s\n  to %s/%s resp: %s' % (str(doc), dbname, colname, resp.text))
    id = resp.json().get('_id',None)
    if id is None : logger.warning('id_document is None')
    return id

#------------------------------

def add_data_and_doc(data, dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs):
    """Adds data and document to the db
    """
    id_data = add_data(dbname, data, url, krbheaders)
    if id_data is None : return None
    doc = mu.docdic(data, id_data, **kwargs) # ObjectId(id_data)???
    id_doc = add_document(dbname, colname, doc, url, krbheaders)
    if id_doc is None : return None
    return id_data, id_doc

#------------------------------

def _add_detector_name(dbname, colname, detname, detnum):
    check_kerberos_ticket()
    short_name = '%s_%06d'%(colname,detnum)
    t0_sec = time()
    doc = {'long'       : detname,\
           'short'      : short_name,\
           'seqnumber'  : detnum,\
           'uid'        : gu.get_login(),
           'host'       : gu.get_hostname(),
           'cwd'        : gu.get_cwd(),
           'time_sec'   : t0_sec,
           'time_stamp' : mu._timestamp(int(t0_sec))
           }
    id_doc = add_document(dbname, colname, doc) #, url, krbheaders)
    return short_name if id_doc is not None else None   

#------------------------------

def _short_detector_name(detname, dbname=cc.DETNAMESDB):
    colname = detname.split('_',1)[0]
    # find a single doc for long detname
    ldocs = find_docs(dbname, colname, query={'long':detname})
    if len(ldocs)>1:
        logger.error('UNEXPECTED ERROR: db/collection: %s/%s has >1 document for detname: %s' % (dbname, colname, detname))
        sys.exit('db/collection: %s/%s HAS TO BE FIXED' % (dbname, colname))

    if ldocs:
        return ldocs[0].get('short', None)

    # find all docs in the collection
    ldocs = find_docs(dbname, colname, query={})

    detnum = 0
    if not ldocs: # empty list
        logger.debug('List of documents in db/collection: %s/%s IS EMPTY' % (dbname, colname))
        detnum = 1
    else:
        for doc in ldocs :
            num = doc.get('seqnumber', 0)
            if num > detnum: detnum = num
        detnum += 1

    short_name = _add_detector_name(dbname, colname, detname, detnum)
    logger.debug('add document to db/collection: %s/%s doc for short name:%s' % (dbname, colname, short_name))

    return short_name

#------------------------------

def pro_detector_name(detname, maxsize=55):
    """ Returns short detector name if its length exceeds 55 chars.
    """
    nchars = len(detname)
    #assert nchars < maxsize, 'name length should be <%d characters' % maxsize
    return detname if nchars<maxsize else _short_detector_name(detname)

#------------------------------

def add_data_and_two_docs(data, exp, det, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwargs):
    """ Adds data and document to experiment and detector data bases.
    """
    t0_sec = time()

    detname = pro_detector_name(det)
    colname = detname
    dbname_exp = mu.db_prefixed_name(exp)
    dbname_det = mu.db_prefixed_name(detname)

    id_data_exp = add_data(dbname_exp, data, url, krbheaders)
    id_data_det = add_data(dbname_det, data, url, krbheaders)
    if None in (id_data_exp, id_data_det): return None

    doc = mu.docdic(data, id_data_exp, **kwargs)
    logger.debug(mu.doc_info(doc, fmt='  %s:%s')) #sep='\n  %16s : %s'

    id_doc_exp = add_document(dbname_exp, colname, doc, url, krbheaders)
    doc['id_data'] = id_data_det # override
    doc['id_exp']  = id_doc_exp  # add
    id_doc_det = add_document(dbname_det, colname, doc, url, krbheaders)
    if None in (id_doc_exp, id_doc_det): return None

    msg = 'Add 2 data and docs time %.6f sec' % (time()-t0_sec)\
        + '\n  - data in %s/gridfs id: %s and doc in collection %s id: %s' % (dbname_exp, id_data_exp, colname, id_doc_exp)\
        + '\n  - data in %s/gridfs id: %s and doc in collection %s id: %s' % (dbname_det, id_data_det, colname, id_doc_det)
    logger.debug(msg)

    return id_data_exp, id_data_det, id_doc_exp, id_doc_det

#------------------------------

def delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes database for (str) dbname, e.g. dbname='cdb_opal_0001'.
    """
    check_kerberos_ticket()
    resp = delete(url+dbname, headers=krbheaders)
    logger.debug(resp.text)
    return resp

#------------------------------

def delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """ Deletes collection from database.
    """
    check_kerberos_ticket()
    resp = delete(url+dbname+'/'+colname, headers=krbheaders)
    logger.debug(resp.text)
    return resp

#------------------------------

def delete_document(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """Deletes document for specified _id from database/collection.
    """
    check_kerberos_ticket()
    resp = delete(url+dbname+'/'+colname+'/'+ doc_id, headers=krbheaders)
    logger.debug(resp.text)
    return resp


#------------------------------

def delete_document_and_data(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS):
    """TBD: Deletes document for specified _id from database/collection and associated data from database/gridfs.
    """
    check_kerberos_ticket()

    # find a single doc for doc_id
    ldocs = find_docs(dbname, colname, query={'_id':doc_id})
    if len(ldocs)>1:
        logger.error('UNEXPECTED ERROR: db/collection: %s/%s HAS MORE THAN ONE DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        sys.exit('db/collection: %s/%s HAS TO BE FIXED' % (dbname, colname))

    print('XXXX ldocs:', ldocs)

    if not ldocs:
        logger.warning('db/collection: %s/%s HAS NO DOCUMENT FOR _id: %s' % (dbname, colname, doc_id))
        return False

    doc = ldocs[0]
    data_id = doc.get('id_data', None)

    resp_doc = delete(url+dbname+'/'+colname+'/'+ doc_id, headers=krbheaders)
    logger.debug('delete db/collection/doc_id: %s/%s/%s responce: %s' (dbname, colname, doc_id, resp_doc.text))

    if data_id is None:
        logger.warning('db/collection/doc_id: %s/%s/%s DOES NOT HAVE data_id' % (dbname, colname, doc_id))
        return False

    resp_data = delete(url+dbname+'/gridfs/'+ data_id, headers=krbheaders)
    logger.debug('delete db/gridfs/data_id: %s/%s/%s responce: %s' (dbname, data_id, resp_data.text))
    return resp_data

#------------------------------
#---------  TESTS  ------------
#------------------------------

if __name__ == "__main__" :

  TEST_FNAME_PNG = '/reg/g/psdm/detector/data2_test/misc/small_img.png'

  def test_database_names():
    print('test_database_names:', database_names())

#------------------------------

  def test_collection_names():
    dbname = sys.argv[2] if len(sys.argv) > 2 else 'cdb_cspad_0001'
    print('test_collection_names:', collection_names(dbname))

#------------------------------

  def test_find_docs():
    docs = find_docs('cdb_cspad_0001', 'cspad_0001')
    print('find_docs: number of docs found: %d' % len(docs))
    print('test_find_docs returns:', type(docs))
    for i,d in enumerate(docs):
        print('%04d %12s %10s run:%04d time_sec:%10s %s' % (i, d['ctype'], d['experiment'], d['run'], str(d['time_sec']), d['detector']))

    if len(docs)==0 : return
    doc0 = docs[0]
    print('doc0 type:', type(doc0))
    print('doc0:', doc0)
    print('doc0.keys():', doc0.keys())

#------------------------------

  def test_get_random_doc_and_data_ids(det='cspad_0001'):
    dbname = mu.db_prefixed_name(det)
    colname = det
    doc = find_doc(dbname, colname, query={'ctype':'pedestals'})
    print('Pick up any doc for dbname:%s colname:%s pedestals: ' % (dbname,colname))
    print('Document: %s' % str(doc))
    id_doc  = doc.get('_id', None)
    id_data = doc.get('id_data', None)
    print('_id : %s   id_data : %s' % (id_doc, id_data))
    return id_doc, id_data, dbname, colname

#------------------------------

  def test_find_doc():
    #doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals', 'run':{'$lte':40}})
    #print('====> test_find_doc for run: %s' % str(doc))

    #doc = find_doc('cdb_cxid9114', 'cspad_0001', query={'ctype':'pedestals', 'time_sec':{'$lte':1402851400}})
    #print('====> test_find_doc for time_sec: %s' % str(doc))

    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0001') 
    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0002') 

#------------------------------

  def test_get_data_for_id():
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_id(dbname, id_data)
    print('test_get_data_for_id: r.content raw data: %s ...' % str(o[:500]))

#------------------------------

  def test_get_data_for_docid():
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_docid(dbname, colname, id_doc)
    #o = get_data_for_docid('cdb_cxid9114', 'cspad_0001', '5b6cdde71ead144f115319be')
    print_ndarr(o, 'test_get_data_for_docid o:', first=0, last=10)

#------------------------------

  def test_dbnames_collection_query():
    det='cspad_0001'
    db_det, db_exp, colname, query = mu.dbnames_collection_query(det, exp=None, ctype='pedestals', run=50, time_sec=None, vers=None)
    print('test_dbnames_collection_query:', db_det, db_exp, colname, query)

#------------------------------

  def test_calib_constants():
    det = 'cspad_0001'
    data, doc = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print_ndarr(data, '==== test_calib_constants', first=0, last=5)
    print('==== doc: %s' % str(doc))

#------------------------------

  def test_calib_constants_text():
    det = 'cspad_0001'
    data, doc = calib_constants(det, exp='cxic0415', ctype='geometry', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

    det = 'tmo_quadanode'
    data, doc = calib_constants(det, exp='amox27716', ctype='calibcfg', run=100, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

#------------------------------

  def test_calib_constants_dict():
    det = 'opal1000_0059'
    #data, doc = calib_constants(det, exp='amox23616', ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    print('==== test_calib_constants_dict data:', data)
    print('XXXX ==== type(data)', type(data))
    print('XXXX ==== type(doc) ', type(doc))
    print('==== doc: %s' % doc)

#------------------------------

  def test_calib_constants_all_types():
    #resp = calib_constants_all_types('tmo_quadanode', exp='amox27716', run=100, time_sec=None, vers=None) #, url=cc.URL)

    resp = calib_constants_all_types('pnccd_0001', exp='amo86615', run=200, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:') #, resp)

    for k,v in resp.items():
        print('ctype:%16s    data and meta:' % k, type(v[0]), type(v[1]))

    import pickle
    s = pickle.dumps(resp)
    print('IF YOU SEE THIS, dict FOR ctypes SHOULD BE pickle-d')

#------------------------------

  def test_insert_constants(expname='testexper', detname='testdet_1234', ctype='test_ctype', runnum=10, data='test text sampele'):
    """ Inserts constants using direct MongoDB interface from MDBUtils.
    """
    import psana.pyalgos.generic.Utils as gu

    print('test_delete_database 1:', database_names())
    #txt = '%s\nThis is a string\n to test\ncalibration storage' % gu.str_tstamp()
    #data, ctype = txt, 'testtext'; logger.debug('txt: %s' % str(data))
    #data, ctype = get_test_nda(), 'testnda';  logger.debug(info_ndarr(data, 'nda'))
    #data, ctype = get_test_dic(), 'testdict'; logger.debug('dict: %s' % str(data))

    kwa = {'user' : gu.get_login()}
    t0_sec = time()
    ts = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=t0_sec)
    mu.insert_constants('%s - saved at %s'%(data,ts), expname, detname, ctype, runnum+int(tname), int(t0_sec),\
                        time_stamp=ts, **kwa)
    print('test_delete_database 2:', database_names())

#------------------------------

  def test_delete_database(dbname='cdb_testexper'):
    print('test_delete_database %s' % dbname)
    print('test_delete_database BEFORE:', database_names())
    resp = delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_database AFTER :', database_names())

#------------------------------

  def test_delete_collection(dbname='cdb_testexper', colname='testdet_1234'):
    print('test_delete_collection %s collection: %s' % (dbname, colname))
    print('test_delete_collection BEFORE:', collection_names(dbname, url=cc.URL))
    resp = delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_collection AFTER :', collection_names(dbname, url=cc.URL))

#------------------------------

  def test_delete_document(dbname='cdb_testexper', colname='testdet_1234', query={'ctype':'test_ctype'}):
    doc = find_doc(dbname, colname, query=query, url=cc.URL)
    print('find_doc:', doc)
    if doc is None : 
        logger.warning('test_delete_document: Non-found document in db:%s col:%s query:%s' % (dbname,colname,str(query)))
        return
    id = doc.get('_id', None)
    print('test_delete_document for doc _id:', id)
    resp = delete_document(dbname, colname, id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_document resp:', resp)

#------------------------------

  def test_delete_document_and_data(dbname='cdb_testexper', colname='testdet_1234'):
    ldocs = find_docs(dbname, colname, query={}, url=cc.URL)
    if not ldocs :
        print('test_delete_document_and_data db/collection: %s/%s does not have any document' % (dbname, colname))
        return
    doc = ldocs[0]
    print('test_delete_document_and_data db/collection: %s/%s contains %d documents\n  try to delete doc: %s' % (dbname, colname, len(ldocs), str(doc)))
    doc_id = doc.get('_id', None)
    resp = delete_document_and_data(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_document_and_data resp:', resp)

#------------------------------

  def test_add_data_from_file(dbname='cdb_testexper', fname=TEST_FNAME_PNG):
    resp = add_data_from_file(dbname, fname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data_from_file resp: %s of type: %s' % (resp, type(resp)))

#------------------------------

  def test_add_data(dbname='cdb_testexper'):
    #data = 'some text is here'
    data = mu.get_test_nda() # np.array(range(12))
    resp = add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data: %s\n  to: %s/gridfs/\n  resp: %s' % (str(data), dbname, resp))

#------------------------------

  def test_add_document(dbname='cdb_testexper', colname='testdet_1234', doc={'ctype':'test_ctype'}):
    from psana.pyalgos.generic.Utils import str_tstamp
    doc['time_stamp'] = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z')
    resp = add_document(dbname, colname, doc, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('\ntest_add_document: %s\n  to: %s/%s\n  resp: %s' % (str(doc), dbname, colname, resp))

#------------------------------

  def test_add_data_and_two_docs(exp='testexper', det='testdet_1234'):
    from psana.pyalgos.generic.Utils import get_login
    t0_sec = time()
    kwa = {'user'      : get_login(),
           'experiment': exp,
           'detector'  : det,
           'ctype'     : 'testnda',
           'run'       : 123,
           'time_sec'  : t0_sec,
           'time_stamp': mu._timestamp(int(t0_sec)),
          }
    data = mu.get_test_nda()
    id_data_exp, id_data_det, id_doc_exp, id_doc_det =\
      add_data_and_two_docs(data, exp, det, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwa)

    print('time to insert data and two docs: %.6f sec' % (time()-t0_sec))

#------------------------------

  def test_pro_detector_name(shortname='testdet_1234'):
    longname = shortname + '_this_is_insane_long_detector_name_exceeding_55_characters_in_length_or_longer'
    tmode = sys.argv[2] if len(sys.argv) > 2 else '0'
    dname = shortname if tmode=='0' else\
            longname  if tmode=='1' else\
            longname + '_' + tmode # mu._timestamp(int(time()))
    name = pro_detector_name(dname)
    print('XXX protected detector name:', name)

#------------------------------

if __name__ == "__main__" :
  def usage(): 
      return 'Use command: python %s <test-number>, where <test-number> = 0,1,2,...,9' % sys.argv[0]\
           + '\n  0: test_database_names'\
           + '\n  1: test_collection_names [dbname]'\
           + '\n  2: test_find_docs'\
           + '\n  3: test_find_doc'\
           + '\n  4: test_get_data_for_id'\
           + '\n  5: test_get_data_for_docid'\
           + '\n  6: test_dbnames_collection_query'\
           + '\n  7: test_calib_constants'\
           + '\n  8: test_calib_constants_text'\
           + '\n  9: test_calib_constants_dict'\
           + '\n 10: test_calib_constants_all_types'\
           + '\n 11: test_insert_constants [using direct access methods of MDBUtils]'\
           + '\n 12: test_delete_database'\
           + '\n 13: test_delete_collection'\
           + '\n 14: test_delete_document'\
           + '\n 15: test_delete_document_and_data'\
           + '\n 16: test_add_data_from_file'\
           + '\n 17: test_add_data'\
           + '\n 18: test_add_document'\
           + '\n 19: test_add_data_and_two_docs'\
           + '\n 20: test_pro_detector_name [test-number=0-short name, 1-fixed long name, n-long name +"_n"]'\
           + ''

#------------------------------

if __name__ == "__main__":
    import os
    import sys
    from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr, print_ndarr
    global print_ndarr
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG) # logging.INFO

    logger.info('\n%s\n' % usage())
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_database_names()
    elif tname == '1' : test_collection_names()
    elif tname == '2' : test_find_docs()
    elif tname == '3' : test_find_doc()
    elif tname == '4' : test_get_data_for_id()
    elif tname == '5' : test_get_data_for_docid()
    elif tname == '6' : test_dbnames_collection_query()
    elif tname == '7' : test_calib_constants()
    elif tname == '8' : test_calib_constants_text()
    elif tname == '9' : test_calib_constants_dict()
    elif tname =='10' : test_calib_constants_all_types()
    elif tname =='11' : test_insert_constants()
    elif tname =='12' : test_delete_database()
    elif tname =='13' : test_delete_collection()
    elif tname =='14' : test_delete_document()
    elif tname =='15' : test_delete_document_and_data()
    elif tname =='16' : test_add_data_from_file()
    elif tname =='17' : test_add_data()
    elif tname =='18' : test_add_document()
    elif tname =='19' : test_add_data_and_two_docs()
    elif tname =='20' : test_pro_detector_name()
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
