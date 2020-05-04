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
    _ = wu.select_latest_doc(docs, query) :
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
from psana.pscalib.calib.MDBUtils import dbnames_collection_query, object_from_data_string

#------------------------------
#------------------------------

def request(url, query=None) :
    #logger.debug('==== query: %s' % str(query))
    t0_sec = time()
    r = get(url, query)
    dt = time()-t0_sec
    logger.debug('CONSUMED TIME by request %.6f sec\n  for url=%s  query=%s' % (dt, url, str(query)))
    return r

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db"
def database_names(url=cc.URL) :
    """Returns list of database names for url.
    """
    r = request(url)
    return r.json()

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll"
def collection_names(dbname, url=cc.URL) :
    """Returns list of collection names for dbname and url.
    """
    r = request('%s/%s'%(url,dbname))
    return r.json()

#------------------------------
# curl -s "https://pswww.slac.stanford.edu/calib_ws/test_db/test_coll?query_string=%7B%20%22item%22..."
def find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL) :
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

def find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL) :
    """Returns document for query.
       1. finds all documents for query
       2. select the latest for run or time_sec
    """
    docs = find_docs(dbname, colname, query, url)
    if docs is None : return None

    return select_latest_doc(docs, query)

#------------------------------

def select_latest_doc(docs, query) :
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
def get_doc_for_docid(dbname, colname, docid, url=cc.URL) :
    """Returns document for docid.
    """
    r = request('%s/%s/%s/%s'%(url,dbname,colname,docid))
    return r.json()

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a" 
def get_data_for_id(dbname, dataid, url=cc.URL) :
    """Returns raw data from GridFS, at this level there is no info for parsing.
    """
    r = request('%s/%s/gridfs/%s'%(url,dbname,dataid))
    logger.debug('get_data_for_docid:'\
                +'\n  r.status_code: %s\n  r.headers: %s\n  r.encoding: %s\n  r.content: %s...\n' % 
                 (str(r.status_code),  str(r.headers),  str(r.encoding),  str(r.content[:50])))
    return r.content

#------------------------------

def get_data_for_docid(dbname, colname, docid, url=cc.URL) :
    """Returns data from GridFS using docid.
    """
    doc = get_doc_for_docid(dbname, colname, docid, url)
    logger.debug('get_data_for_docid: %s' % str(doc))
    return get_data_for_doc(dbname, colname, doc, url)

#------------------------------

# curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def get_data_for_doc(dbname, colname, doc, url=cc.URL) :
    """Returns data from GridFS using doc.
    """
    logger.debug('get_data_for_doc: %s', str(doc))
    idd = doc.get('id_data', None)
    if idd is None :
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None

    r2 = request('%s/%s/gridfs/%s'%(url,dbname,idd))
    s = r2.content

    return object_from_data_string(s, doc)

#------------------------------

def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL) :
    """Returns calibration constants and document with metadata for specified parameters. 
       To get meaningful constants, at least a few parameters must be specified, e.g.:
       - det, ctype, time_sec
       - det, ctype, version
       - det, exp, ctype, run
       - det, exp, ctype, time_sec
       - det, exp, ctype, run, version
       etc...
    """
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers)
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

def calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL) :
    """ returns constants for all ctype-s
    """
    ctype=None
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers)
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

def encode_data(data) :
    """Converts any data type into octal string to save in gridfs.
    """
    s = None
    if   isinstance(data, np.ndarray) : s = data.tobytes()
    elif isinstance(data, str) :        s = str.encode(data)
    else :
        logger.warning('DATA TYPE "%s" IS NOT "str" OR "numpy.ndarray" CONVERTED BY pickle.dumps ...'%\
                       type(data).__name__)
        s = pickle.dumps(data)
    return s      

#------------------------------

def add_data_from_file(dbname, fname, sfx=None, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """Adds data from file to the database/gridfs.
    """
    _sfx = sfx if sfx is not None else fname.rsplit('.')[-1]
    files = [('files',  (fname, open(fname, 'rb'), 'image/'+_sfx))]
    resp = post(url+dbname+'/gridfs/', headers=krbheaders, files=files)
    logger.debug('add_data_from_file: %s to %s/gridfs/ resp: %s type: %s' % (fname, dbname, resp.text, type(resp)))
    #jdic = resp.json() # type <class 'dict'>
    return resp.json().get('_id',None)

#------------------------------

def add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """Adds binary data to the database/gridfs.
    """
    import io
    headers = dict(krbheaders) # krbheaders <class 'dict'>
    headers['Content-Type'] = 'application/octet-stream'
    #f = io.StringIO(data)
    f = io.BytesIO(encode_data(data))
    d = f.read()
    print('XXXX',d)
    resp = post(url+dbname+'/gridfs/', headers=headers, data=d)
    logger.debug('add_data: to %s/gridfs/ resp: %s' % (dbname, resp.text))
    return resp.json().get('_id',None)

#------------------------------

def add_document(dbname, colname, jdic, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """Adds document to database collection.
    """
    resp = post(url+dbname+'/'+colname+'/', headers=krbheaders, json=jdic)
    logger.debug('add_document: %s\n  to %s/%s resp: %s' % (str(jdic), dbname, colname, resp.text))
    return resp.json().get('_id',None)

#------------------------------

def delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """Deletes database for (str) dbname, e.g. dbname='cdb_opal_0001'.
    """
    resp = delete(url+dbname, headers=krbheaders)
    logger.debug(resp.text)
    return resp

#------------------------------

def delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """ Deletes collection from database.
    """
    resp = delete(url+dbname+'/'+colname, headers=krbheaders)
    logger.debug(resp.text)
    return resp

#------------------------------

def delete_document(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS) :
    """Deletes document for specified _id from database/collection.
    """
    resp = delete(url+dbname+'/'+colname+'/'+ doc_id, headers=krbheaders)
    logger.debug(resp.text)
    return resp

#------------------------------
#---------  TESTS  ------------
#------------------------------

if __name__ == "__main__" :

  TEST_FNAME_PNG = '/reg/g/psdm/detector/data2_test/misc/small_img.png'

  def test_database_names() :
    print('test_database_names:', database_names())

#------------------------------

  def test_collection_names() :
    print('test_collection_names:', collection_names('cdb_cspad_0001'))

#------------------------------

  def test_find_docs() :
    docs = find_docs('cdb_cspad_0001', 'cspad_0001')
    print('find_docs: number of docs found: %d' % len(docs))
    print('test_find_docs returns:', type(docs))
    for i,d in enumerate(docs) :
        print('%04d %12s %10s run:%04d time_sec:%10s %s' % (i, d['ctype'], d['experiment'], d['run'], str(d['time_sec']), d['detector']))

    if len(docs)==0 : return
    doc0 = docs[0]
    print('doc0 type:', type(doc0))
    print('doc0:', doc0)
    print('doc0.keys():', doc0.keys())

#------------------------------

  def test_get_random_doc_and_data_ids(det='cspad_0001') :
    from psana.pscalib.calib.MDBUtils import db_prefixed_name
    dbname = db_prefixed_name(det)
    colname = det
    doc = find_doc(dbname, colname, query={'ctype':'pedestals'})
    print('Pick up any doc for dbname:%s colname:%s pedestals: ' % (dbname,colname))
    print('Document: %s' % str(doc))
    id_doc  = doc.get('_id', None)
    id_data = doc.get('id_data', None)
    print('_id : %s   id_data : %s' % (id_doc, id_data))
    return id_doc, id_data, dbname, colname

#------------------------------

  def test_find_doc() :
    #doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals', 'run':{'$lte':40}})
    #print('====> test_find_doc for run: %s' % str(doc))

    #doc = find_doc('cdb_cxid9114', 'cspad_0001', query={'ctype':'pedestals', 'time_sec':{'$lte':1402851400}})
    #print('====> test_find_doc for time_sec: %s' % str(doc))

    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0001') 
    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0002') 

#------------------------------

  def test_get_data_for_id() :
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_id(dbname, id_data)
    print('test_get_data_for_id: r.content raw data: %s ...' % str(o[:500]))

#------------------------------

  def test_get_data_for_docid() :
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_docid(dbname, colname, id_doc)
    #o = get_data_for_docid('cdb_cxid9114', 'cspad_0001', '5b6cdde71ead144f115319be')
    print_ndarr(o, 'test_get_data_for_docid o:', first=0, last=10)

#------------------------------

  def test_dbnames_collection_query() :
    det='cspad_0001'
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp=None, ctype='pedestals', run=50, time_sec=None, vers=None)
    print('test_dbnames_collection_query:', db_det, db_exp, colname, query)

#------------------------------

  def test_calib_constants() :
    det = 'cspad_0001'
    data, doc = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print_ndarr(data, '==== test_calib_constants', first=0, last=5)
    print('==== doc: %s' % str(doc))

#------------------------------

  def test_calib_constants_text() :
    det = 'cspad_0001'
    data, doc = calib_constants(det, exp='cxic0415', ctype='geometry', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

    det = 'tmo_quadanode'
    data, doc = calib_constants(det, exp='amox27716', ctype='calibcfg', run=100, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

#------------------------------

  def test_calib_constants_dict() :
    det = 'opal1000_0059'
    #data, doc = calib_constants(det, exp='amox23616', ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    print('==== test_calib_constants_dict data:', data)
    print('XXXX ==== type(data)', type(data))
    print('XXXX ==== type(doc) ', type(doc))
    print('==== doc: %s' % doc)

#------------------------------

  def test_calib_constants_all_types() :
    #resp = calib_constants_all_types('tmo_quadanode', exp='amox27716', run=100, time_sec=None, vers=None) #, url=cc.URL)

    resp = calib_constants_all_types('pnccd_0001', exp='amo86615', run=200, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:') #, resp)

    for k,v in resp.items() :
        print('ctype:%16s    data and meta:' % k, type(v[0]), type(v[1]))

    import pickle
    s = pickle.dumps(resp)
    print('IF YOU SEE THIS, dict FOR ctypes SHOULD BE pickle-d')

#------------------------------

  def test_insert_constants(expname='test_exp', detname='test_det', ctype='test_ctype', runnum=10, data='test text sampele') :
    """ Inserts constants using direct MongoDB interface from MDBUtils.
    """
    import psana.pscalib.calib.MDBUtils as mu
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

  def test_delete_database(dbname='cdb_test_det'):
    print('test_delete_database %s' % dbname)
    print('test_delete_database BEFORE:', database_names())
    resp = delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_database AFTER :', database_names())

#------------------------------

  def test_delete_collection(dbname='cdb_test_exp', colname='test_det'):
    print('test_delete_collection %s collection: %s' % (dbname, colname))
    print('test_delete_collection BEFORE:', collection_names(dbname, url=cc.URL))
    resp = delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_collection AFTER :', collection_names(dbname, url=cc.URL))

#------------------------------

  def test_delete_document(dbname='cdb_test_exp', colname='test_det', query={'ctype':'test_ctype'}):
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

  def test_add_data_from_file(dbname='cdb_test_det', fname=TEST_FNAME_PNG) :
    resp = add_data_from_file(dbname, fname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data_from_file resp: %s of type: %s' % (resp, type(resp)))

#------------------------------

  def test_add_data(dbname='cdb_test_det') :
    #data = 'some text is here'
    data = np.array(range(12))
    resp = add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data: %s\n  to: %s/gridfs/\n  resp: %s' % (str(data), dbname, resp))

#------------------------------

  def test_add_document(dbname='cdb_test_det', colname='test_det', jdic={'ctype':'test_ctype'}):
    from psana.pyalgos.generic.Utils import str_tstamp
    jdic['time_stamp'] = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z')
    resp = add_document(dbname, colname, jdic, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('\ntest_add_document: %s\n  to: %s/%s\n  resp: %s' % (str(jdic), dbname, colname, resp))

#------------------------------

if __name__ == "__main__" :
  def usage() : 
      return 'Use command: python %s <test-number>, where <test-number> = 0,1,2,...,9' % sys.argv[0]\
           + '\n  0: test_database_names'\
           + '\n  1: test_collection_names'\
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
           + '\n 15: test_add_data_from_file'\
           + '\n 16: test_add_data'\
           + '\n 17: test_add_document'\
           + ''

#------------------------------

if __name__ == "__main__" :
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
    elif tname =='15' : test_add_data_from_file()
    elif tname =='16' : test_add_data()
    elif tname =='17' : test_add_document()

    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
