"""
1. Start server
2. Use this API test: python lcls2/psana/pscalib/calib/MDBUtils.py 1 ... 12

Usage ::

    # Import
    import psana.pscalib.calib.MDBUtils as mu

    # Connect client to server
    client = mu.connect_to_server(host=cc.HOST, port=cc.PORT, )

    # Access databases, fs, collections
    db = mu.database(client, dbname='cdb-cspad-0-cxids1-0')
    db, fs = mu.db_and_fs(client, dbname='cdb-cxi12345')
    col = mu.collection(db, cname='camera-0-cxids1-0')

    # Get host and port
    host = mu.client_host(client)
    port = mu.client_port(client)

    # Get list of database and collection names
    names = mu.database_names(client)
    names = mu.collection_names(db, include_system_collections=False)

    status = mu.database_exists(client, dbname)
    status = mu.collection_exists(db, cname)

    # Delete methods
    mu.delete_database(client, dbname:str='cdb-cspad-0-cxids1-0')
    mu.delete_database_obj(odb)
    mu.delete_collection(db, cname:str)
    mu.delete_collection_obj(col)
    mu.delete_document_from_collection(col, id)

    dbname = mu.db_prefixed_name(name:str)
    dbname = mu.get_dbname(**kwargs)

    # All connect methods in one call
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        mu.connect(host='psanaphi105', port=27017, experiment='cxi12345', detector='camera-0-cxids1-0', verbose=False) 

    ts    = mu._timestamp(time_sec:int,int,float)
    ts    = mu.timestamp_id(id:str)
    ts    = mu.timestamp_doc(doc)
    t, ts = mu.time_and_timestamp(**kwargs)

    # Make document
    doc = mu.docdic(data, id_data, **kwargs)

    # Print document content
    mu.print_doc(doc)
    mu.print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data'))

    # Insert document in collection
    doc_id = mu.insert_document(doc, col)

    # Insert data
    id_data = mu.insert_data(data, fs)
    id_data, id_doc = mu.insert_data_and_doc(data, fs, col, **kwargs)
    id_data_exp, id_data_det, id_exp, id_det = mu.insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwargs)
    mu.insert_calib_data(data, **kwargs)

    msg = mu._error_msg(msg:str) 
    mu.valid_experiment(experiment:str)
    mu.valid_detector(detector:str)
    mu.valid_ctype(ctype:str)
    mu.valid_run(run:int)
    mu.valid_version(version:str)
    mu.valid_comment(comment:str)
    mu.valid_data(data, detector:str, ctype:str)

    mu.insert_constants(data, experiment:str, detector:str, ctype:str, run:int, time_sec:int, **kwargs)

    mu.exec_command(cmd)
    mu.exportdb(host, port, dbname, fname, **kwa) 
    mu.importdb(host, port, dbname, fname, **kwa) 

    # Delete data
    mu.del_document_data(doc, fs)
    mu.del_collection_data(col, fs)

    # Find document in collection

    dbname_det, dbname_exp, colname, query =\
           mu.dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, tsec=None, vers=None)

    doc  = mu.find_doc(col, query={'data_type' : 'xxxx'})

    # Get data
    data = mu.get_data_for_doc(fs, doc)

    keys = mu.document_keys(doc)
    s_vals, s_keys = mu.document_info(doc, keys:tuple=('time_stamp','time_sec','experiment','detector','ctype','run','id_data','data_type'), fmt:str='%24s %10d %11s %20s %16s %4d %30s %10s')
    s = mu.collection_info(client, dbname, cname) 
    s = mu.database_info(client, dbname, level:int=10, gap:str='  ')
    s = mu.database_fs_info(db, gap:str='  ')
    s = mu.client_info(client=None, host:str=cc.HOST, port:int=cc.PORT, level:int=10, gap:str='  ')

    mu.request_confirmation()

    # Test methods
    ...
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import pickle
import gridfs
from pymongo import MongoClient, errors, ASCENDING, DESCENDING
#from pymongo.errors import ConnectionFailure
#import pymongo

import sys
from time import time

import numpy as np
import psana.pyalgos.generic.Utils as gu
from   psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
import psana.pscalib.calib.CalibConstants as cc

from bson.objectid import ObjectId
from psana.pscalib.calib.Time import Time

TSFORMAT = '%Y-%m-%dT%H:%M:%S%z' # e.g. 2018-02-07T09:11:09-0800

#------------------------------

#def connect_to_server(host:str=cc.HOST, port:int=cc.PORT,\
#                      username:str=cc.USERNAME, userpw:str=cc.USERPW, ctout=5000, stout=30000) :
def connect_to_server(host=cc.HOST, port=cc.PORT,\
                      username=cc.USERNAME, userpw=cc.USERPW, ctout=5000, stout=30000) :
    """Returns MongoDB client.
    """

    #uri = mongodb://[username:password@]host1[:port1]
    uri = 'mongodb://%s:%s@%s:%d' % (username, userpw, host, port)
    #print('uri: ', uri)

    client = MongoClient(uri, connect=False, connectTimeoutMS=ctout, socketTimeoutMS=stout)
    #client = MongoClient(host, port, connect=False, connectTimeoutMS=ctout, socketTimeoutMS=stout)
    try :
        result = client.admin.command("ismaster")
        return client

    except errors.ConnectionFailure:
        #msg = 'Server not available for port:%s host: %d' % (host, port)
        #print(msg)
        #logger.debug(msg)
        #logger.exception(err)
        #sys.exit("ERROR can't connect to port:%s host: %d" % (host, port))
        return None

#------------------------------

def database(client, dbname) :
    """Returns db for client and (str) dbname, e.g. dbname='cdb-cspad-0-cxids1-0'
    """
    return client[dbname]

#------------------------------

def db_and_fs(client, dbname) :
    """Returns db and fs for client and (str) dbname, e.g. dbname='cdb-cxi12345'.
    """
    db = client[dbname]
    fs = gridfs.GridFS(db)
    return db, fs

#------------------------------

def collection(db, cname) :
    """Returns collection for db and (str) cname, e.g. cname='camera-0-cxids1-0'.
    """
    return db[cname]

#------------------------------

def client_host(client) :
    """Returns client host.
       ??? returns localhost in stead of psanaphi105 ???
    """
    return client.HOST
    #return client.server_info()

#------------------------------

def client_port(client) :
    """Returns client port.
    """
    return client.PORT

#------------------------------

def database_names(client) :
    """Returns list of database names for client.
    """
    return client.database_names()

#------------------------------

def collection_names(db, include_system_collections=False) :
    """Returns list of collection names.
    """
    return db.collection_names(include_system_collections)

#------------------------------

def database_exists(client, dbname) :
    """Returns True if (str) dbname in the list of databases and False otherwise.
    """
    return dbname in database_names(client)

#------------------------------

def collection_exists(db, cname) :
    """Returns True if (str) cname in the list of collections and False otherwise.
    """
    return cname in collection_names(db)

#------------------------------

def delete_database(client, dbname) :
    """Deletes database for client and (str) dbname, e.g. dbname='cdb-cspad-0-cxids1-0'.
    """
    client.drop_database(dbname)

#------------------------------

def delete_database_obj(odb) :
    """Deletes database for database object.
    """
    odb.dropDatabase()

#------------------------------

def delete_databases(client, dbnames) :
    """Deletes databases for client and (list of str) dbnames,
       e.g. dbnames=['cdb-cspad-0-cxids1-0','cdb-cspad-0-cxids2-0'].
    """
    for name in dbnames : client.drop_database(name)

#------------------------------

def delete_collection(db, cname) :
    """Deletes db collection for database and (str) cname, e.g. cname='camera-0-cxids1-0'.
    """
    db.drop_collection(cname)

#------------------------------

def delete_collection_obj(ocol) :
    ocol.drop()

#------------------------------

def delete_collections(db, cnames) :
    """Deletes list of collections from database db, e.g. cname='camera-0-cxids1-0'.
    """
    for cname in cnames : db.drop_collection(cname)

#------------------------------

def delete_document_from_collection(col, id) :
    col.remove({'_id':id})

#------------------------------

def db_prefixed_name(name, prefix='cdb_') :
    """Returns database name with prefix, e.g. name='cxi12345' -> 'cdb-cxi12345'.
    """
    assert isinstance(name,str), 'db_prefixed_name parameter should be str'
    nchars = len(name)
    assert nchars < 128, 'name length should be <128 characters'
    logger.debug('db_prefixed_name %s has %d chars' % (name, nchars))
    return '%s%s' % (prefix, name)

#------------------------------

def get_dbname(**kwargs) :
    """Returns (str) dbname or None. Implements logics for dbname selection:
       -- dbname is used if defined else
       -- prefixed experiment else
       -- prefixed detector else None
    """
    exp    = kwargs.get('experiment', None)
    det    = kwargs.get('detector', None)
    dbname = kwargs.get('dbname', None)

    if dbname is None :
        name = exp if not (exp is None) else det
        if name is None :
            logger.warning('dbname, experiment, or detector name must to be specified.')
            return None
        dbname = db_prefixed_name(name)
    return dbname

#------------------------------

def connect(**kwargs) :
    """Connect to host, port get db handls.
    """
    host    = kwargs.get('host', cc.HOST)
    port    = kwargs.get('port', cc.PORT)
    user    = kwargs.get('username', cc.USERNAME)
    upwd    = kwargs.get('userpw', cc.USERPW)
    expname = kwargs.get('experiment', 'cxi12345')
    detname = kwargs.get('detector', 'camera-0-cxids1-0')
    verbose = kwargs.get('verbose', False)
    userpwd = kwargs.get('password', '')
    if userpwd : upwd = userpwd

    dbname_exp = db_prefixed_name(expname)
    dbname_det = db_prefixed_name(detname)

    t0_sec = time()

    client = connect_to_server(host, port, user, upwd)
    db_exp, fs_exp = db_and_fs(client, dbname=dbname_exp)
    db_det, fs_det = db_and_fs(client, dbname=dbname_det)
    col_det = collection(db_det, cname=detname)
    col_exp = collection(db_exp, cname=detname) 

    logger.debug('client  : %s' % client.name)
    logger.debug('db_exp  : %s' % db_exp.name)
    logger.debug('col_exp : %s' % col_exp.name)
    logger.debug('db_det  : %s' % db_det.name)
    logger.debug('col_det : %s' % col_det.name)
    logger.debug('==== Connect to host: %s port: %d connection time %.6f sec' % (host, port, time()-t0_sec))

    return client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det

#------------------------------
#------------------------------
#------------------------------

def _timestamp(time_sec) :
    """Converts time_sec in timestamp of adopted format TSFORMAT.
    """
    return gu.str_tstamp(TSFORMAT, int(time_sec))


def timestamp_id(id) :
    """Converts MongoDB (str) id to (str) timestamp of adopted format.
    """
    str_ts = str(ObjectId(id).generation_time) # '2018-03-14 21:59:37+00:00'
    tobj = Time.parse(str_ts)                  # Time object from parsed string
    tsec = int(tobj.sec())                     # 1521064777
    str_tsf = _timestamp(tsec)                 # re-formatted time stamp
    #logger.debug('XXX: str_ts', str_ts, tsec, tsf)
    return str_tsf


def timestamp_doc(doc) :
    """Returns document creation (str) timestamp from its id.
    """
    timestamp_id(doc['_id'])

#------------------------------

def time_and_timestamp(**kwargs) :
    """Returns "time_sec" and "time_stamp" from **kwargs.
       If one of these parameters is missing, another is reconstructed from available one.
       If both missing - current time is used.
    """
    time_sec   = kwargs.get('time_sec', None)
    time_stamp = kwargs.get('time_stamp', None)

    if time_sec is not None :
        assert isinstance(time_sec, int) , 'time_and_timestamp - parameter time_sec should be int'
        assert 0 < time_sec < 5000000000,  'time_and_timestamp - parameter time_sec should be in allowed range'

        if time_stamp is None : 
            time_stamp = gu.str_tstamp(TSFORMAT, time_sec)
    else :
        if time_stamp is None : 
            time_sec_str, time_stamp = gu.time_and_stamp(TSFORMAT)
        else :
            time_sec_str = gu.time_sec_from_stamp(TSFORMAT, time_stamp)
        time_sec = int(time_sec_str)

    return time_sec, time_stamp

#------------------------------

def docdic(data, dataid, **kwargs) :
    """Returns dictionary for db document in style of JSON object.
    """
    doc = {
          'experiment' : kwargs.get('experiment', None),
          'run'        : kwargs.get('run', 0),
          'run_end'    : kwargs.get('run_end', 'end'),
          'detector'   : kwargs.get('detector', None),
          'ctype'      : kwargs.get('ctype', None),
          'time_sec'   : kwargs.get('time_sec', None),
          'time_stamp' : kwargs.get('time_stamp', None),
          'version'    : kwargs.get('version', 'v00'),
          'comment'    : kwargs.get('comment', ''),
          'extpars'    : kwargs.get('extpars', None),
          'uid'        : gu.get_login(),
          'host'       : gu.get_hostname(),
          'cwd'        : gu.get_cwd(),
          'id_data'    : dataid,
          }

    if isinstance(data, np.ndarray) :
        doc['data_type']  = 'ndarray'
        doc['data_dtype'] = str(data.dtype)
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
    logger.info('Data document attributes')
    if doc is None :
        logger.info('print_doc: Data document is None...')
        return
        
    msg = ''
    for k,v in doc.items() : 
        msg += '%16s : %s' % (k,v)
    logger.info(msg)

#------------------------------

def print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars')) :
    msg = ''
    for k in keys :
        msg += '  %s : %s' % (k, doc.get(k,'N/A'))
    logger.info(msg)

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

def insert_data_and_doc(data, fs, col, **kwargs) :
    """For open collection col and GridFS fs inserts calib data and document.
       Returns inserted id_data in fs and id_doc in col.
    """
    id_data = insert_data(data, fs)
    logger.debug('  - in fs %s id_data: %s' % (fs, id_data))
    doc = docdic(data, id_data, **kwargs)
    id_doc = insert_document(doc, col)
    logger.debug('  - in collection %20s id_det : %s' % (col.name, id_doc))
    return id_data, id_doc

#------------------------------

def insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwargs) :
    """For open connection inserts calib data and two documents.
       Returns inserted id_data, id_exp, id_det.
    """
    verbose = kwargs.get('verbose', False)

    t0_sec = time()
    id_data_exp = insert_data(data, fs_exp)
    id_data_det = insert_data(data, fs_det)

    logger.debug('Insert data time %.6f sec' % (time()-t0_sec))
    logger.debug('  - in fs_exp %s id_data_exp: %s' % (fs_exp, id_data_exp))
    logger.debug('  - in fs_det %s id_data_det: %s' % (fs_det, id_data_det))

    doc = docdic(data, id_data_exp, **kwargs)
    if verbose :
        print_doc(doc)
        #logger.debug('XXX: inset data_type: "%s"' % doc['data_type'])

    t0_sec = time()
    id_exp = insert_document(doc, col_exp)
    doc['id_data'] = id_data_det # override
    doc['id_exp']  = id_exp      # add
    id_det = insert_document(doc, col_det)

    logger.debug('Insert 2 docs time %.6f sec' % (time()-t0_sec))
    logger.debug('  - in collection %20s id_exp : %s' % (col_exp.name, id_exp))
    logger.debug('  - in collection %20s id_det : %s' % (col_det.name, id_det))

    return id_data_exp, id_data_det, id_exp, id_det

#------------------------------

def insert_calib_data(data, **kwargs) :
    """Connects to calibration data base and inserts calib data.
    """
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwargs)
    #id_data_exp, id_data_det, id_exp, id_det = \
    _,_,_,_ = insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwargs)

#------------------------------

def _error_msg(msg) :
    return 'wrong parameter %s' % msg

#------------------------------

def valid_experiment(experiment) :
    assert isinstance(experiment,str), _error_msg('type')
    assert 7 < len(experiment) < 10, _error_msg('length')

def valid_detector(detector) :
    assert isinstance(detector,str), _error_msg('type')
    assert 1 < len(detector) < 65, _error_msg('length')

def valid_ctype(ctype) :
    assert isinstance(ctype,str), _error_msg('type')
    assert 4 < len(ctype) < 32, _error_msg('length')

def valid_run(run) :
    assert isinstance(run,int), _error_msg('type')
    assert -1 < run < 10000, _error_msg('value')

def valid_version(version) :
    assert isinstance(version,str), _error_msg('type')
    assert 1 < len(version) < 32, _error_msg('length')

def valid_version(version) :
    assert isinstance(version,str), _error_msg('type')
    assert len(version) < 128, _error_msg('length')

def valid_comment(comment) :
    assert isinstance(comment,str), _error_msg('type')
    assert len(comment) < 1000000, _error_msg('length')

def valid_data(data, detector, ctype) :
    pass

#------------------------------

def insert_constants(data, experiment, detector, ctype, run, time_sec, **kwargs) :
    """Checks validity of input parameters and call insert_calib_data.
    """
    _time_sec, _time_stamp = time_and_timestamp(time_sec=time_sec,\
                                                time_stamp=kwargs.get('time_stamp', None))
    _version = kwargs.get('version', 'v00')
    _comment = kwargs.get('comment', 'default comment')

    valid_experiment(experiment)
    valid_detector(detector)
    valid_ctype(ctype)
    valid_run(run)
    valid_version(_version)
    valid_comment(_comment)
    valid_data(data, detector, ctype)

    kwa = {
          'experiment' : experiment,
          'run'        : run,
          'run_end'    : kwargs.get('run_end', 'end'),
          'detector'   : detector,
          'ctype'      : ctype,
          'time_sec'   : _time_sec,
          'time_stamp' : _time_stamp,
          'version'    : _version,
          'comment'    : _comment,
          'host'       : kwargs.get('host', cc.HOST),
          'port'       : kwargs.get('port', cc.PORT),
          'verbose'    : kwargs.get('verbose', False),
          'extpars'    : kwargs.get('extpars', None),
          }

    insert_calib_data(data, **kwa)

#------------------------------

def del_document_data(doc, fs) :
    """From fs removes data associated with a single document.
    """
    oid = doc.get('id_data', None)
    if oid is None : return
    fs.delete(oid)

#------------------------------

def del_collection_data(col, fs) :
    """From fs removes data associated with multiple documents in colllection col.
    """
    for doc in col.find() :
        del_document_data(doc, fs)
        #oid = doc.get('id_data', None)
        #if oid is None : return
        #fs.delete(oid)

#------------------------------

def exec_command(cmd) :
    from psana.pscalib.proc.SubprocUtils import subproc
    logger.debug('Execute shell command: %s' % cmd)
    if not gu.shell_command_is_available(cmd.split()[0], verb=True) : return
    out,err = subproc(cmd, env=None, shell=False, do_wait=True)
    if out or err :
        logger.warning('err: %s\nout: %s' % (err,out))

#------------------------------

def exportdb(host, port, dbname, fname, **kwa) :
    client = connect_to_server(host, port)
    dbnames = database_names(client)
    if not (dbname in dbnames) :
        logger.warning('--dbname %s is not available in the list:\n%s' % (dbname, dbnames))
        return

    cmd = 'mongodump --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname) # --gzip 
    exec_command(cmd)

#------------------------------

def importdb(host, port, dbname, fname, **kwa) :

    if fname is None :
        logger.warning('WARNING input archive file name should be specified as --iofname <fname>')
        return 

    client = connect_to_server(host, port)
    dbnames = database_names(client)
    if dbname in dbnames :
        logger.warning('WARNING: --dbname %s is already available in the list:\n%s' % (dbname, dbnames))
        return

    cmd = 'mongorestore --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname)
    exec_command(cmd)

#------------------------------
#------------------------------
#------------------------------

def get_data_for_doc(fs, doc) :
    """Returns data referred by the document.
    """
    if doc is None :
        logger.warning('get_data_for_doc: Data document is None...')
        return None

    idd = doc.get('id_data', None)
    if idd is None :
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None
    out = fs.get(idd)

    #except:
    #    msg = 'Unexpected ERROR: %s' % sys.exc_info()[0]
    #    logger.exception(msg)
    #    sys.exit(msg)

    s = out.read()
    data_type = doc['data_type']
    logger.debug('get_data_for_doc data_type: %s' % data_type)
    
    if data_type == 'str'     : return s.decode()
    if data_type == 'ndarray' : 
        str_dtype = doc['data_dtype']
        #logger.debug('XXX str_dtype:', str_dtype)
        #dtype = np.dtype(eval(str_dtype))
        #logger.debug('XXX  np.dtype:', dtype)
        nda = np.fromstring(s, dtype=str_dtype)
        nda.shape = eval(doc['data_shape']) # eval converts string shape to tuple
        #logger.debug('XXX nda.shape =', nda.shape)

        #str_sh = doc['data_shape'] #.lstrip('(').rstrip(')')
        #nda.shape = tuple(np.fromstring(str_sh, dtype=int, sep=','))
        #print_ndarr(nda, 'XXX: nda re-shaped')
        return nda
    return pickle.loads(s)

#------------------------------

def dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None) :
    """Returns dbnames for detector, experiment, collection name, and query.
    """
    cond = (run is not None) or (time_sec is not None) or (vers is not None)
    assert cond, 'Not sufficeint info for query: run, time_sec, and vers are None'
    query={'detector':det, 'ctype':ctype}
    if run is not None : 
        query['run']     = {'$lte' : run}
        #query['run_end'] = {'$gte' : run}
    if time_sec is not None : query['time_sec'] = {'$lte' : time_sec}
    if vers is not None : query['version'] = vers
    logger.debug('query: %s' % str(query))

    db_det, db_exp = db_prefixed_name(det), db_prefixed_name(str(exp))
    if 'None' in db_det : db_det = None
    if 'None' in db_exp : db_exp = None
    return db_det, db_exp, det, query

#------------------------------

def find_docs(col, query={'ctype':'pedestals'}) :
    """Returns list of documents for query.
    """
    docs = col.find(query)
    if docs.count() == 0 :
        logger.warning('col: %s query: %s is not consistent with any document...' % (col.name, query))
        return None
    else : return docs

    #try :
    #    return col.find(query)
    #except errors.ServerSelectionTimeoutError as err: 
    #    logger.exception(err)
    #    sys.exit('ERROR at attempt to find data in database. Check server.')
    #except errors.TypeError as err: 
    #    logger.exception(err)
    #    sys.exit('ERROR in arguments passed to find.')

#------------------------------

def find_doc(col, query={'ctype':'pedestals'}) :
    """Returns the document with latest time_sec or run number for specified query.
    """
    docs = find_docs(col, query)
    #logger.debug('XXX Number of documents found:', docs.count())

    if (docs is None)\
    or (docs.count()==0) :
        logger.warning('DB %s collection %s does not have document for query %s' % (col.database.name, col.name, str(query)))
        return None

    qkeys = query.keys()
    key_sort = 'time_sec' if 'time_sec' in qkeys else 'run'

    doc = docs.sort(key_sort, DESCENDING)[0]
    #msg = 'query: %s\n  %d docs found, selected doc["%s"]=%s'%\
    #      (query, docs.count(), key_sort, doc[key_sort])
    #logger.info(msg)

    return doc

#------------------------------

def document_keys(doc) :
    """Returns formatted strings of document keys. 
    """
    keys = sorted(doc.keys())
    s = '%d document keys:' % len(keys)
    for i,k in enumerate(keys) :
        if not(i%5) : s += '\n      ' 
        s += ' %s' % k.ljust(16)
    return s
    #return '%d doc keys:\n      %s' % (len(keys), '\n      '.join([k for k in keys]))

#------------------------------

def document_info(doc, keys=('time_sec','time_stamp','experiment',\
                  'detector','ctype','run','ts_data','data_type','data_dtype'),\
                  fmt='%10s %24s %11s %24s %16s %4s %30s %10s %10s') :
    """Returns (str, str) for formatted document values and title made of keys. 
    """
    doc_keys = sorted(doc.keys())
    if 'experiment' in doc_keys : # CDDB type of document
        vals = tuple([str(doc.get(k,None)) for k in keys])
        return fmt % vals, fmt % keys

    else : # OTHER type of document
        title = '  '.join(doc_keys)
        vals = tuple([str(doc.get(k,None) if k != 'data' else '<some data>') for k in doc_keys])
        info = '  '.join(vals)
        return info, title

#------------------------------

def collection_info(client, dbname, cname) :
    """Returns (str) info regarding collection documents. 
    """
    s = 'DB %s collection %s' % (dbname, cname)
    db = database(client, dbname)
    col = collection(db, cname) # or db[cname]
    docs = col.find().sort('_id', DESCENDING)
    #          # {'ctype':DESCENDING, 'time_sec':DESCENDING, 'run':ASCENDING}
    #  s += '\n%s%s%s' % (gap, gap, 52*'_')
    #s += '\n%s%sCOL %s contains %d docs' % (gap, gap, cname.ljust(12), docs.count())
    #for idoc, doc in enumerate(docs) :

    ndocs = docs.count()

    if not ndocs : return s
        
    s += ' contains %d docs\n' % ndocs
 
    doc = docs[0]
    s += '\n  %s' % (document_keys(doc)) # str(doc.keys()))

    #if cname in ('fs.chunks',) :
    #    s += '\n\ncol: "%s" does not have good presentation for documents...' % cname
    #    return s

    _, title = document_info(doc)
    s += '\n  %s%s' % ('doc#', title)
    
    for idoc, doc in enumerate(docs) :
        #id_data = doc.get('id_data', None)
        #if id_data is not None : doc['ts_data'] = timestamp_id(id_data)
        vals,_ = document_info(doc)
        s += '\n  %4d %s' % (idoc, vals)

    return s

#------------------------------

def database_info(client, dbname, level=10, gap='  ') :
    """Returns (str) info about database
    """
    #dbname = db_prefixed_name(name)

    dbnames = database_names(client)
    #assert dbname in dbnames, 'dbname: %s is not found in the %s' % (dbname, str(dbnames))
    if not(dbname in dbnames) :
        return 'dbname: %s is not found in the list of databases:\n%s' % (dbname, str(dbnames))

    s = '%s\ndbnames %s' % (gap, str(dbnames))
    db = database(client, dbname)
    cnames = collection_names(db)
    s += '\n%sDB %s contains %d collections: %s' % (gap, dbname.ljust(12), len(cnames), str(cnames))
    if level==1 : return s

    for cname in cnames :
      col = collection(db, cname) # or db[cname]
      docs = col.find().sort('ctype', DESCENDING)
              # {'ctype':DESCENDING, 'time_sec':DESCENDING, 'run':ASCENDING}
      s += '\n%s%s%s' % (gap, gap, 52*'_')
      s += '\n%s%sCOL %s contains %d docs' % (gap, gap, cname.ljust(12), docs.count())
      #for idoc, doc in enumerate(docs) :

      if level==2 : continue

      if col.name in ('fs.chunks', 'fs.files') : continue

      s += '\n%s%sDetails for collection %s %d documents' % (gap, gap, col.name, docs.count())
 
      if docs.count() > 0 :
        doc = docs[0]
        s += ':\n%s%s%s' % (gap, gap, document_keys(doc)) # str(doc.keys()))
        _, title = document_info(doc)
        s += '\n%s%s%s %s' % (gap, gap, 'doc#', title)
        for idoc, doc in enumerate(docs) :
            id_data = doc.get('id_data', None)
            if id_data is not None : doc['ts_data'] = timestamp_id(id_data)
            vals,_ = document_info(doc)
            s += '\n%s%s%4d %s' % (gap, gap, idoc, vals)
    return s

#------------------------------

def database_fs_info(db, gap='  ') :
    """Returns (str) info about database fs collections 
    """
    s = '%sDB "%s" data collections:' % (gap, db.name)
    for cname in collection_names(db) :
       if cname in ('fs.chunks', 'fs.files') :
           docs = collection(db, cname).find()
           s += '\n%s   COL: %s has %d docs' % (gap, cname.ljust(9), docs.count())
    return s

#------------------------------

def client_info(client=None, host=cc.HOST, port=cc.PORT, level=10, gap='  ') :
    """Returns (str) with generic information about MongoDB client (or host:port) 
    """
    _client = client if client is not None else connect_to_server(host, port)
    #s = '\nMongoDB client host:%s port:%d' % (client_host(_client), client_port(_client))
    dbnames = database_names(_client)
    s = '\n%sClient contains %d databases: %s' % (gap, len(dbnames), ', '.join(dbnames))
    if level==1 : return s
    for idb, dbname in enumerate(dbnames) :
        db = database(_client, dbname) # client[dbname]
        cnames = collection_names(db)
        s += '\n%sDB %s has %2d collections: %s' % (gap, dbname.ljust(12), len(cnames), str(cnames))
        if level==2 : continue
        for icol, cname in enumerate(cnames) :
            col = collection(db, cname) # or db[cname]
            docs = col.find()
            s += '\n%s%sCOL %s has %d docs' % (gap, gap, cname.ljust(12), docs.count())
            #for idoc, doc in enumerate(docs) :
            if docs.count() > 0 :
                doc = docs[0]
                s += ': %s' % (str(doc.keys()))
                #logger.debug('%s %4d  %s %s' % (10*' ', idoc, doc['time_stamp'], doc['ctype']))
            if level==3 : continue
    return s

#------------------------------

def request_confirmation() :
    """Dumps request for confirmation of specified (delete) action.
    """
    logger.warning('Use confirm "-C" option to proceed with request.')

#------------------------------
#----------- TEST -------------
#------------------------------
if __name__ == "__main__" :

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
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxid9114', detector='cspad_0001', verbose=True) 
        #connect(host=cc.HOST, port=cc.PORT, detector='cspad_0001', verbose=True) 

#------------------------------

  def test_insert_one(tname) :
    """Insert one calibration data in data base.
    """
    data = None 
    if   tname == '1' : data = get_test_txt(); logger.debug('txt:', data)
    elif tname == '2' : data = get_test_nda(); logger.debug(info_ndarr(data, 'nda'))
    elif tname == '3' : data = get_test_dic(); logger.debug('dict:', data)

    #insert_calib_data(data, host=cc.HOST, port=cc.PORT, experiment='cxi12345', detector='camera-0-cxids1-0',\
    #                  run=10, ctype='pedestals', time_sec=int(time()), verbose=True)
    insert_constants(data, 'cxi12345', 'camera-0-cxids1-0', 'pedestals', 10, 1600000000, verbose=True,\
                     time_stamp='2018-01-01T00:00:00-0800', )
    #t0_sec = time()
    #id_data = insert_data(data, fs)
    #logger.debug('Insert data in %s id_data: %s time %.6f sec' % (fs, id_data, time()-t0_sec))

    #doc = docdic(data, id_data, experiment=expname, detector=detname)
    #print_doc(doc)

    #t0_sec = time()
    #insert_document(doc, col_exp)
    #insert_document(doc, col_det)
    #logger.debug('Insert 2 docs time %.6f sec' % (time()-t0_sec))

#------------------------------

  def test_insert_many(tname) :
    """Insert many documents in loop
    """
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxi12345', detector='camera-0-cxids1-0', verbose=True)

    t_data = 0
    nloops = 10

    for i in range(nloops) :
        logger.info('%s\nEntry: %4d' % (50*'_', i))
        data = get_test_nda()
        print_ndarr(data, 'data nda') 

        t0_sec = time()
        id_data_exp, id_data_det, id_exp, id_det = insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det,\
             experiment=expname, detector=detname, ctype='pedestals', time_sec=int(time()), run=10, verbose=True)

        #id_data = insert_data(nda, fs)
        dt_sec = time() - t0_sec
        t_data += dt_sec
        logger.info('Insert data in %s id_data: %s time %.6f sec ' % (fs, id_data, dt_sec))

        #doc = docdic(nda, id_data, experiment=expname, detector=detname, run=10, ctype='pedestals')
        #print_doc_keys(doc)

        #t0_sec = time()
        #idd_exp = insert_document(doc, col_exp)
        #idd_det = insert_document(doc, col_det)
        #dt_sec = time() - t0_sec
        #t_doc += dt_sec
        #logger.info('Insert 2 docs %s, %s time %.6f sec' % (idd_exp, idd_det, dt_sec))

    logger.info('Average time to insert data and two docs: %.6f sec' % (t_data/nloops))

#------------------------------

  def test_get_data(tname) :
    """Get doc and data
    """
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(verbose=True)

    t0_sec = time()
    data_type='any' 
    if tname == '11' : data_type='str' 
    if tname == '12' : data_type='ndarray' 

    doc = find_doc(col_det, query={'data_type' : data_type})
    logger.info('Find doc time %.6f sec' % (time()-t0_sec))
    logger.info('doc:\n%s' % str(doc))
    print_doc(doc)

    t0_sec = time()
    data = get_data_for_doc(fs, doc)
    logger.info('get data time %.6f sec' % (time()-t0_sec))
    logger.info('data:\n%s' % str(data))

#------------------------------

  def test_get_data_for_id(tname, det='cspad_0001', data_id='5b6cdde71ead144f11531999') :
    """Get doc and data
    """
    kwa = {'detector':det}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(verbose=True, **kwa)
    print('XXX expname', expname)
    print('XXX expname', expname)
    print('XXX data_id', data_id)

    out = fs_det.get(ObjectId(data_id))
    print('XXX out: ', out)

    s = out.read()
    
    print('XXX type(out.read()): ', type(s))
    print('XXX out.read(): ', s[:200])

    nda = np.fromstring(s, dtype=np.float32)
    print_ndarr(nda, 'XXX: nda: ', first=0, last=10)

#------------------------------

  def test_database_content(tname, level=3) :
    """Insert many documents in loop
    """
    #client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
    #    connect(host=cc.HOST, port=cc.PORT)

    client = connect_to_server(host=cc.HOST, port=cc.PORT)
    logger.info('host:%s port:%d' % (client_host(client), client_port(client)))
    dbnames = database_names(client)
    logger.info('databases: %s' % str(dbnames))
    for idb, dbname in enumerate(dbnames) :
        db = database(client, dbname) # client[dbname]
        cnames = collection_names(db)
        logger.info('==== DB %2d: %12s # cols :%2d' % (idb, dbname, len(cnames)))
        if dbname[:3] != db_prefixed_name('') : 
            logger.info('     skip non-calib dbname: %s' % dbname)
            continue
        if level==1 : continue
        for icol, cname in enumerate(cnames) :
            col = collection(db, cname) # or db[cname]
            docs = col.find()
            logger.info('     COL %2d: %12s #docs: %d' % (icol, cname.ljust(12), docs.count()))
            if level==2 : continue
            #for idoc, doc in enumerate(docs) :
            if docs.count() > 0 :
                #logger.info('%s %4d  %s %s' % (10*' ', idoc, doc['time_stamp'], doc['ctype']))
                doc = docs[0]
                logger.info('%s doc[0] %s' % (10*' ', str(doc.keys())))

#------------------------------

if __name__ == "__main__" :
    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
    #                    datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO) # WARNING
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_connect(tname)
    elif tname in ('1','2','3') : test_insert_one(tname)
    elif tname == '4' : test_insert_many(tname)
    elif tname == '5' : test_database_content(tname)
    elif tname in ('11','12','13') : test_get_data(tname)
    elif tname == '15' : test_get_data_for_id(tname)
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
