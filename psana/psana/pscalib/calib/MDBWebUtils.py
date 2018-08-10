"""
Usage ::

    # Import
    import psana.pscalib.calib.MDBWebUtils as wu
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    _ = wu.requests_get(url, query=None)
    _ = wu.database_names(url=cc.URL)
    _ = wu.collection_names(dbname, url=cc.URL)
    _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
    _ = wu.get_doc_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_id(dbname, dataid, url=cc.URL)
    _ = wu.get_data_for_docid(dbname, colname, docid, url=cc.URL)
    _ = wu.get_data_for_doc(dbname, colname, doc, url=cc.URL)
    o = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL)

    test_*()
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import psana.pscalib.calib.CalibConstants as cc
import requests
#import json
import pickle
import numpy as np
from psana.pscalib.calib.MDBUtils import dbnames_collection_query

#------------------------------

def requests_get(url, query=None) :
    #logger.debug('==== query: %s' % str(query))
    r = requests.get(url, query)
    logger.debug('URL: %s' % r.url)
    #logger.debug('Response: %s' % r.json())
    return r

#------------------------------

def database_names(url=cc.URL) :
    """Returns list of database names for url.
    """
    r = requests_get(url)
    return r.json()

#------------------------------

def collection_names(dbname, url=cc.URL) :
    """Returns list of collection names for dbname and url.
    """
    r = requests_get('%s/%s'%(url,dbname))
    return r.json()

#------------------------------
# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/test_db/test_coll?query_string=%7B%20%22item%22..."
def find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL) :
    """Returns list of documents for query, e.g. query={'ctype':'pedestals', "run":{ "$gte":80}}
       #r = requests_get("https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxid9114/cspad_0001",\
                         {"query_string": '{"ctype": "pedestals", "run" : { "$gte": 80}}'})
    """
    str_query=str(query).replace("'",'"')
    logger.debug('find_docs str_query: %s' % str_query)
    r = requests_get('%s/%s/%s'%(url,dbname,colname),{"query_string": str_query})
    return r.json()

#------------------------------

def find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL) :
    """Returns document for query.
    """
    docs = find_docs(dbname, colname, query, url)
    if docs is None : return None
    if len(docs)==0 : return None
    qkeys = query.keys()
    key_sort = 'time_sec' if 'time_sec' in qkeys else 'run'

    logger.debug('find_doc query: %s\n  key_sort: %s' % (str(query), key_sort))
    vals = [int(d[key_sort]) for d in docs]
    vals.sort(reverse=True)
    logger.debug('find_doc values: %s' % str(vals))
    val_sel = int(vals[0])
    logger.debug('find_doc select document for %s:%s' % (key_sort,val_sel))
    for d in docs : 
        if d[key_sort]==val_sel : 
            return d
    return None

#------------------------------

# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/5b6893e81ead141643fe4344"
def get_doc_for_docid(dbname, colname, docid, url=cc.URL) :
    """Returns document for docid.
    """
    r = requests_get('%s/%s/%s/%s'%(url,dbname,colname,docid))
    return r.json()

#------------------------------

# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a" 
def get_data_for_id(dbname, dataid, url=cc.URL) :
    """Returns raw data from GridFS, at this level there is no info for parsing.
    """
    r = requests_get('%s/%s/gridfs/%s'%(url,dbname,dataid))
    return r.content

#------------------------------

def get_data_for_docid(dbname, colname, docid, url=cc.URL) :
    """Returns data from GridFS using docid.
    """
    doc = get_doc_for_docid(dbname, colname, docid, url)
    logger.debug('get_data_for_docid: %s', str(doc))
    return get_data_for_doc(dbname, colname, doc, url)

#------------------------------

# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def get_data_for_doc(dbname, colname, doc, url=cc.URL) :
    """Returns data from GridFS using doc.
    """
    logger.debug('get_data_for_doc: %s', str(doc))

    idd = doc.get('id_data', None)
    if idd is None :
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None

    r2 = requests_get('%s/%s/gridfs/%s'%(url,dbname,idd))

    s = r2.content
    data_type = doc['data_type']
    if data_type == 'str'     : return s.decode()
    if data_type == 'ndarray' : 
        str_dtype = doc['data_dtype']
        nda = np.fromstring(s, dtype=str_dtype)
        nda.shape = eval(doc['data_shape']) # eval converts string shape to tuple
        return nda
    return pickle.loads(s)

#------------------------------

def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL) :

    db_det, db_exp, colname, query = dbnames_collection_query(det, exp, ctype, run, time_sec, vers)
    logger.debug('get_constants: %s %s %s %s' % (db_det, db_exp, colname, str(query)))

    dbname = db_det if exp is None else db_exp

    doc = find_doc(dbname, colname, query, url)

    if doc is None :
        logger.warning('document is not available for query: %s' % str(query))
        return None

    return get_data_for_doc(dbname, colname, doc, url=cc.URL)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def test_database_names() :
    print('test_database_names:', database_names())

#------------------------------

def test_collection_names() :
    print('test_collection_names:', collection_names('cdb_cspad_0001'))

#------------------------------

def test_find_docs() :
    jo = find_docs('cdb_cspad_0001', 'cspad_0001')
    logger.info('find_docs: number of docs found: %d' % len(jo))
    print('test_find_docs:', type(jo))
    dic0 = jo[0]
    print('dic0:', type(dic0))
    print('dic0:', dic0)
    print('dic0.keys():', dic0.keys())

#------------------------------

def test_find_doc() :
    #doc = find_doc('cdb_cspad_0001', 'cspad_0001', query={'ctype':'pedestals'}) #, 'run':{'$lte':40}})
    #doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals'}) #, 'run':{'$lte':40}})
    #doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals'}) #, 'run':{'$lte':40}})
    doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals', 'run':{'$lte':40}})
    logger.info('test_find_doc: %s' % str(doc))

#------------------------------

# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a" 
def test_get_data_for_id() :
    o = get_data_for_id('cdb_cxid9114', '5b6cdde71ead144f11531974')
    print('test_get_data_for_id: r.content:', o)

#------------------------------

# curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/gridfs/5b6893e81ead141643fe4344"
def test_get_data_for_docid() :
    #o = get_data_for_docid('cdb_cspad_0001', 'cspad_0001', '5b6896fc1ead142459f10138')
    #o = get_data_for_docid('cdb_cxic0415', 'cspad_0001', '5b6893e81ead141643fe4344')
    o = get_data_for_docid('cdb_cxid9114', 'cspad_0001', '5b6cdde71ead144f115319be')
    print('test_get_data_for_docid: o:', o)

#------------------------------

def test_dbnames_collection_query() :
    det='cspad_0001'
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp=None, ctype='pedestals', run=50, time_sec=None, vers=None)
    print('test_dbnames_collection_query:', db_det, db_exp, colname, query)

#------------------------------

def test_calib_constants() :
    det = 'cspad_0001'
    #o = calib_constants(det, exp=None, ctype='pedestals', run=50, time_sec=None, vers=None, url=cc.URL)
    o = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None) #, url=cc.URL)
    #print('test_calib_constants: o:', o)
    print_ndarr(o, 'test_calib_constants', first=0, last=5)

#------------------------------

if __name__ == "__main__" :
    import sys
    from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr, print_ndarr
    global print_ndarr
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    #logging.basicConfig(format='%(message)s', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_database_names();
    elif tname == '1' : test_collection_names();
    elif tname == '2' : test_find_docs();
    elif tname == '3' : test_find_doc();
    elif tname == '4' : test_get_data_for_id();
    elif tname == '5' : test_get_data_for_docid();
    elif tname == '6' : test_dbnames_collection_query();
    elif tname == '7' : test_calib_constants();
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
