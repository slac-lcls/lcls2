
"""
1. Start server
2. Use this API test: python lcls2/psana/pscalib/calib/MDBUtils.py 1 ... 12

Usage ::

    # Import
    import psana.pscalib.calib.MDBUtils as mu

    # Connect to server etc.
    client = mu.connect_to_server(host=cc.HOST, port=cc.PORT, )
    db = mu.db(client, dbname='calib-cspad-0-cxids1-0')
    db, fs = mu.db_and_fs(client, dbname='calib-cxi12345')
    col = mu.collection(db, cname='camera-0-cxids1-0')

    # All connect methods in one call
    detname, expname, client, db_det, db_exp, fs, col_det, col_exp =\
        mu.connect(host='psanaphi105', port=27017, experiment='cxi12345', detector='camera-0-cxids1-0', verbose=False) 

    # Insert data
    id_data = mu.insert_data(data, fs)

    # Make document
    doc = mu.docdic(data, id_data, **kwargs)

    # Print document content
    mu.print_doc(doc)
    mu.print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data'))

    # Insert document in collection
    doc_id = mu.insert_document(doc, col)

    # Find document in collection
    doc  = mu.find_doc(col, query={'data_type' : 'xxxx'})

    # Get data
    data = mu.get_data_for_doc(fs, doc)
"""
#------------------------------

import pickle
import gridfs
from pymongo import MongoClient, errors
#import pymongo

import sys
from time import time

import numpy as np
import psana.pyalgos.generic.Utils as gu
from   psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pscalib.calib.CalibConstants as cc

import logging
logger = logging.getLogger('MDBUtils')

TSFORMAT = '%Y-%m-%dT%H:%M:%S%z' # e.g. 2018-02-07T09:11:09-0800

#------------------------------

def connect_to_server(host:str=cc.HOST, port:int=cc.PORT) :
    """Returns client.
    """
    return MongoClient(host, port)

#------------------------------

def db(client, dbname:str='calib-cspad-0-cxids1-0') :
    """Returns db.
    """
    return client[dbname]

#------------------------------

def db_and_fs(client, dbname:str='calib-cxi12345') :
    """Returns db and fs.
    """
    db = client[dbname]
    fs = gridfs.GridFS(db)
    return db, fs

#------------------------------

def collection(db, cname:str='camera-0-cxids1-0') :
    """Returns collection.
    """
    return db[cname]

#------------------------------

def _add_prefix(name:str) -> str :
    """Returns name with prefix for data-bases.
    """
    assert isinstance(name,str), '_add_prefix parameter should be str'
    nchars = len(name)
    assert nchars > 0 and nchars < 128, 'name length should be betwen 0 and 128'
    logger.info('name %s has %d chars' % (name, nchars))
    return 'calib-%s' % name

#------------------------------

def connect(**kwargs) :
    """Connect to host, port get db handls.
    """
    host    = kwargs.get('host', cc.HOST)
    port    = kwargs.get('port', cc.PORT)
    expname = kwargs.get('experiment', 'cxi12345')
    detname = kwargs.get('detector', 'camera-0-cxids1-0')
    verbose = kwargs.get('verbose', False)

    dbname_exp = _add_prefix(expname)
    dbname_det = _add_prefix(detname)

    t0_sec = time()

    client = connect_to_server(host, port)
    db_exp, fs = db_and_fs(client, dbname=dbname_exp)
    db_det = db(client, dbname=dbname_det)
    col_det = collection(db_det, cname=detname)
    col_exp = collection(db_exp, cname=expname)

    if verbose :
        print('client  : %s' % client.name)
        print('db_exp  : %s' % db_exp.name)
        print('col_exp : %s' % col_exp.name)
        print('db_det  : %s' % db_det.name)
        print('col_det : %s' % col_det.name)
        print('==== Connect to host: %s port: %d connection time %.6f sec' % (host, port, time()-t0_sec))

    return  detname, expname, client, db_det, db_exp, fs, col_det, col_exp

#------------------------------
#------------------------------
#------------------------------

def _timestamp(time_sec:int) -> str :
    return gu.str_tstamp(TSFORMAT, time_sec)

#------------------------------

def _time_and_timestamp(**kwargs) :
    """Returns "time_sec" and "time_stamp" from **kwargs.
       If one of these parameters is missing, another is reconstructed from available one.
       If both missing - current time is used.
    """

    time_sec   = kwargs.get('time_sec', None)
    time_stamp = kwargs.get('time_stamp', None)

    if time_sec is not None :
        assert isinstance(time_sec, int) or\
               isinstance(time_sec, float) , '_time_and_timestamp - parameter time_sec should be int or float'
        assert 0 < time_sec<5000000000,  '_time_and_timestamp - parameter time_sec should be in allowed range'

        if time_stamp is None : 
            time_stamp = gu.str_tstamp(TSFORMAT, time_sec)
    else :
        if time_stamp is None : 
            time_sec, time_stamp = gu.time_and_stamp(TSFORMAT)
        else :
            time_sec = gu.time_sec_from_stamp(TSFORMAT, time_stamp)

    return time_sec, time_stamp

#------------------------------

def docdic(data, id_data, **kwargs) :
    """Returns dictionary for db document in style of JSON object.
    """
    time_sec, time_stamp = _time_and_timestamp(**kwargs)

    doc = {
          'experiment' : kwargs.get('experiment', 'cxi12345'),
          'run'        : kwargs.get('run', '0'),
          'detector'   : kwargs.get('detector', 'camera-0-cxids1-0'),
          'ctype'      : kwargs.get('ctype', 'pedestals'),
          'time_sec'   : '%d' % time_sec,
          'time_stamp' : time_stamp,
          'version'    : 'v01',
          'uid'        : gu.get_login(),
          'host'       : gu.get_hostname(),
          'cwd'        : gu.get_cwd(),
          'comments'   : ['very good constants', 'eat this document before reading!'],
          'id_data'    : id_data,
          }

    if isinstance(data, np.ndarray) :
        doc['data_type']  = 'ndarray'
        doc['data_size']  = '%d' % data.size
        doc['data_ndim']  = '%d' % data.ndim
        doc['data_shape'] = str(data.shape)

    elif isinstance(data, str) :
        doc['data_type']  = 'str'
        doc['data_size']  = '%d' % len(data)

    else :
        doc['data_type']  = 'any'

    logger.debug('doc data type: %s' % doc['data_type'])

    return doc

#------------------------------

def print_doc(doc) :
    print('Data document attributes')
    for k,v in doc.items() : 
        print('%16s : %s' % (k,v))

#------------------------------

def print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data')) :
    for k in keys :
        print('  %s : %s' % (k, doc[k]),)
    print('')

#------------------------------

def insert_document(doc, col) :
    """Returns inserted document id.
    """
    try :
        return col.insert_one(doc).inserted_id

    except errors.ServerSelectionTimeoutError as err: 
        logger.exception(err)
        sys.exit('ERROR at attempt to insert document in database. Check server.')

#------------------------------

def insert_data(data, fs) :
    """Returns inserted data id.
    """
    s = None # should be replaced by serrialized data
    if isinstance(data, np.ndarray) : s = data.tobytes()
    elif isinstance(data, str) :      s = str.encode(data)
    else :                            s = pickle.dumps(data)
        
    try :
        return fs.put(s)

    except errors.ServerSelectionTimeoutError as err: 
        logger.exception(err)
        sys.exit('ERROR at attempt to insert data in database. Check server.')
    except:
        msg = 'Unexpected ERROR: %s' % sys.exc_info()[0]
        logger.exception(msg)
        sys.exit(msg)

#------------------------------

def insert_data_and_two_docs(data, fs, col_exp, col_det, **kwargs) :
    """For open connection inserts calib data and two documents.
       Returns inserted id_data, id_exp, id_det.
    """
    verbose = kwargs.get('verbose', False)

    t0_sec = time()
    id_data = insert_data(data, fs)
    if verbose :
        print('Insert data in %s id_data: %s time %.6f sec' % (fs, id_data, time()-t0_sec))

    doc = docdic(data, id_data, **kwargs)
    if verbose :
        print_doc(doc)

    t0_sec = time()
    id_exp = insert_document(doc, col_exp)
    doc['id_exp'] = id_exp
    id_det = insert_document(doc, col_det)
    if verbose :
        print('Insert 2 docs time %.6f sec' % (time()-t0_sec))

    return id_data, id_exp, id_det

#------------------------------

def insert_calib_data(data, **kwargs) :
    """Connects to calibration data base and inserts calib data.
    """
    detname, expname, client, db_det, db_exp, fs, col_det, col_exp = connect(**kwargs)
    id_data, id_exp, id_det = insert_data_and_two_docs(data, fs, col_exp, col_det, **kwargs)

#------------------------------

def _error_msg(msg) :
    return 'insert_constants - wrong parameter %s' % msg

def insert_constants(data, experiment:str, detector:str, ctype:str, run:int, time_sec_or_stamp:(int,str), version:str='V001', **kwargs) :
    """Checks validity of input parameters and call insert_calib_data.
    """
    _time_sec   = time_sec_or_stamp if isinstance(time_sec_or_stamp, int) else None
    _time_stamp = time_sec_or_stamp if isinstance(time_sec_or_stamp, str) else None
    time_sec, time_stamp = _time_and_timestamp(time_sec=_time_sec, time_stamp=_time_stamp)
   
    comments = kwargs.get('comments',[]),

    assert isinstance(experiment,str),   _error_msg('type')
    assert isinstance(detector,str),     _error_msg('type')
    assert isinstance(ctype,str),        _error_msg('type')
    assert isinstance(run,int),          _error_msg('type')
    assert isinstance(version,str),      _error_msg('type')
    assert isinstance(comments,tuple) or\
           isinstance(comments,list) ,   _error_msg('type')

    assert 7 < len(experiment) < 10,     _error_msg('length')
    assert 1 < len(detector) < 65,       _error_msg('length')
    assert 4 < len(ctype)   < 32,        _error_msg('length')
    assert 1 < len(version) < 16,        _error_msg('length')

    assert -1 < run < 10000,             _error_msg('value')

    kwa = {
          'experiment' : experiment,
          'run'        : run,
          'detector'   : detector,
          'ctype'      : ctype,
          'time_sec'   : time_sec,
          'time_stamp' : time_stamp,
          'version'    : version,
          'comments'   : comments,
          'host'       : kwargs.get('host', cc.HOST),
          'port'       : kwargs.get('port', cc.PORT),
          'verbose'    : kwargs.get('verbose', False)
          }

    insert_calib_data(data, **kwa)

#------------------------------
#------------------------------
#------------------------------

def get_data_for_doc(fs, doc) :
    """Returns data referred by the document.
    """
    try :
        out = fs.get(doc['id_data'])
    except:
        msg = 'Unexpected ERROR: %s' % sys.exc_info()[0]
        logger.exception(msg)
        sys.exit(msg)

    s = out.read()
    dtype = doc['data_type']
    if dtype == 'str'     : return s.decode()
    if dtype == 'ndarray' : 
        nda = np.fromstring(s)
        nda.shape = eval(doc['data_shape']) # eval converts string shape to tuple
        #str_sh = doc['data_shape'] #.lstrip('(').rstrip(')')
        #nda.shape = tuple(np.fromstring(str_sh, dtype=int, sep=','))
        #print_ndarr(nda, 'XXX: nda re-shaped')
        return nda
    return pickle.loads(s)

#------------------------------

def find_doc(col, query={'data_type' : 'xxxx'}) :
    """Returns list of documents in responce on query.
    """
    #tstamp = kwargs.get('time_stamp', '2018-02-05T17:38:33-0800')

    #query = {'run': 125}
    #query = {'time_stamp' : tstamp} 
    #query = {
    #    'experiment' : kwargs.get('experiment', 'cxi12345'),
    #    'run'        : kwargs.get('run', '0'),
    #    'detector'   : kwargs.get('detector', 'camera-0-cxids1-0'),
    #    'ctype'      : kwargs.get('ctype', 'pedestals'),
    #    'time_sec'   : '%.9f' % time_sec,
    #    'time_stamp' : time_stamp,
    #    'version'    : 'v01',
    #    'id_data'    : id_data,
    #    }

    #dtype  = kwargs.get('data_type', 'N/A')
    #query = {'data_type' : dtype}
    #query = {'data_type' : 'xxxx'}

    try :
        docs = col.find(query)

    except errors.ServerSelectionTimeoutError as err: 
        logger.exception(err)
        sys.exit('ERROR at attempt to find data in database. Check server.')

    except errors.TypeError as err: 
        logger.exception(err)
        sys.exit('ERROR in arguments passed to find.')

    if docs.count() == 0 :
        logger.warning('Query: %s\nis not consistent with any document...')
        return None

    print('XXX number of found docs:', docs.count())

    return docs[0]

#------------------------------
#----------- TEST -------------
#------------------------------

def get_test_nda() :
    """Returns random standard nupmpy array for test purpose.
    """
    import psana.pyalgos.generic.NDArrGenerators as ag
    return ag.random_standard(shape=(32,185,388), mu=20, sigma=5, dtype=np.float)

def get_test_dic() :
    """Returns dict for test purpose.
    """
    arr = np.array(range(12))
    arr.shape = (3,4)
    return {'1':1, 5:'super', 'a':arr}

def get_test_txt() :
    """Returns text for test purpose.
    """
    return '%s\nThis is a string\n to test\ncalibration storage' % gu.str_tstamp()

#------------------------------

def test_connect(tname) :
    """Connect to host, port get db handls.
    """
    detname, expname, client, db_det, db_exp, fs, col_det, col_exp =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxi12345', detector='camera-0-cxids1-0', verbose=True) 

#------------------------------

def test_insert_one(tname) :
    """Insert one calibration data in data base.
    """
    data = None 
    if   tname == '1' : data = get_test_txt(); print('txt:', data)
    elif tname == '2' : data = get_test_nda(); print_ndarr(data, 'nda') 
    elif tname == '3' : data = get_test_dic(); print('dict:', data)

    #insert_calib_data(data, host=cc.HOST, port=cc.PORT, experiment='cxi12345', detector='camera-0-cxids1-0',\
    #                  run=10, ctype='pedestals', time_sec=time(), verbose=True)

    #insert_constants(data, 'cxi12345', 'camera-0-cxids1-0', 'pedestals', 10, 1600000000, 'V001', verbose=True)
    insert_constants(data, 'cxi12345', 'camera-0-cxids1-0', 'pedestals', 10, '2018-01-01T00:00:00-0800', 'V001', verbose=True)
    #t0_sec = time()
    #id_data = insert_data(data, fs)
    #print('Insert data in %s id_data: %s time %.6f sec' % (fs, id_data, time()-t0_sec))

    #doc = docdic(data, id_data, experiment=expname, detector=detname)
    #print_doc(doc)

    #t0_sec = time()
    #insert_document(doc, col_exp)
    #insert_document(doc, col_det)
    #print('Insert 2 docs time %.6f sec' % (time()-t0_sec))

#------------------------------

def test_insert_many(tname) :
    """Insert many documents in loop
    """
    detname, expname, client, db_det, db_exp, fs, col_det, col_exp =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxi12345', detector='camera-0-cxids1-0', verbose=True)

    t_data = 0
    nloops = 10

    for i in range(nloops) :
        print('%s\nEntry: %4d' % (50*'_', i))
        data = get_test_nda()
        print_ndarr(data, 'data nda') 

        t0_sec = time()
        id_data, id_exp, id_det = insert_data_and_two_docs(data, fs, col_exp, col_det,\
             experiment=expname, detector=detname, ctype='pedestals', time_sec=time(), run=10, verbose=True)

        #id_data = insert_data(nda, fs)
        dt_sec = time() - t0_sec
        t_data += dt_sec
        print('Insert data in %s id_data: %s time %.6f sec ' % (fs, id_data, dt_sec))

        #doc = docdic(nda, id_data, experiment=expname, detector=detname, run=10, ctype='pedestals')
        #print_doc_keys(doc)

        #t0_sec = time()
        #idd_exp = insert_document(doc, col_exp)
        #idd_det = insert_document(doc, col_det)
        #dt_sec = time() - t0_sec
        #t_doc += dt_sec
        #print('Insert 2 docs %s, %s time %.6f sec' % (idd_exp, idd_det, dt_sec))

    print('Average time to insert data and two docs: %.6f sec' % (t_data/nloops))

#------------------------------

def test_get_data(tname) :
    """Get doc and data
    """
    detname, expname, client, db_det, db_exp, fs, col_det, col_exp = connect(verbose=True)

    t0_sec = time()
    dtype='any' 
    if tname == '11' : dtype='str' 
    if tname == '12' : dtype='ndarray' 

    doc = find_doc(col_det) # query={'data_type' : 'xxxx'}
    print('Find doc time %.6f sec' % (time()-t0_sec))
    print('doc:\n', doc)
    print_doc(doc)

    t0_sec = time()
    data = get_data_for_doc(fs, doc)
    print('get data time %.6f sec' % (time()-t0_sec))
    print('data:\n', data)

#------------------------------

if __name__ == "__main__" :
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO) # WARNING
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_connect(tname);
    elif tname in ('1','2','3') : test_insert_one(tname);
    elif tname == '4' : test_insert_many(tname)
    elif tname in ('11','12','13') : test_get_data(tname)
    else : print('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
