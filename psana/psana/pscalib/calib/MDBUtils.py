"""
1. Start server
2. Use this API test: python lcls2/psana/psana/pscalib/calib/MDBUtils.py 0...14

Usage ::

    # Import
    import psana.pscalib.calib.MDBUtils as mu

    # Connect client to server
    client = mu.connect_to_server(host=cc.HOST, port=cc.PORT, user=..., upwd=...)

    # Access databases, fs, collections
    db = mu.database(client, dbname)
    db, fs = mu.db_and_fs(client, 'cdb_exp12345')
    col = mu.collection(db, cname)

    # Get host and port
    host = mu.client_host(client)
    port = mu.client_port(client)

    # Get list of database and collection names
    names = mu.database_names(client)
    names = mu.collection_names(db, include_system_collections=False)

    status = mu.database_exists(client, dbname)
    status = mu.collection_exists(db, cname)

    # Delete methods
    mu.delete_database(client, dbname='cdb_detector_1234')
    mu.delete_database_obj(odb)
    mu.delete_collection(db, cname)
    mu.delete_collection_obj(col)
    mu.delete_document_from_collection(col, id)

    dbname = mu.db_prefixed_name(name)
    dbname = mu.get_dbname(**kwargs)
    dbname = mu.get_colname(**kwargs)

    # All connect methods in one call
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        mu.connect(host='psanaphi105', port=27017, experiment='exp12345', detector='camera_1234') 

    ts    = mu._timestamp(time_sec:int,int,float)
    ts    = mu.timestamp_id(id)
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
    binarydata = encode_data(data)
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

    doc  = mu.find_doc(col, query={'data_type': 'xxxx'})

    # Get data
    data = mu.get_data_for_doc(fs, doc)

    keys = mu.document_keys(doc)
    s_vals, s_keys = mu.document_info(doc, keys:tuple=('time_stamp','time_sec','experiment','detector','ctype','run','id_data','data_type'), fmt:str='%24s %10d %11s %20s %16s %4d %30s %10s')
    s = mu.collection_info(client, dbname, cname) 
    s = mu.database_info(client, dbname, level:int=10, gap:str='  ')
    s = mu.database_fs_info(db, gap:str='  ')
    s = mu.client_info(client=None, host:str=cc.HOST, port:int=cc.PORT, level:int=10, gap:str='  ')

    data, doc = mu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, **kwa)

    mu.request_confirmation()
    prefix = mu.out_fname_prefix(fmt='doc-%s-%s-r%04d-%s', **kwa)
    mu.save_doc_and_data_in_file(doc, data, prefix, control={'data': True, 'meta': True})
    data = mu.data_from_file(fname, ctype, dtype, verb=False)

    detname_short = mu.pro_detector_name(detname_long)

    # Test methods
    ...
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import pickle
import gridfs
from pymongo import MongoClient, errors, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from collections.abc import Iterable
#from pymongo.errors import ConnectionFailure
#import pymongo

import os
import sys
from time import time

import numpy as np
import psana.pyalgos.generic.Utils as gu
from   psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
import psana.pscalib.calib.CalibConstants as cc

from bson.objectid import ObjectId
from psana.pscalib.calib.Time import Time

TSFORMAT = cc.TSFORMAT #'%Y-%m-%dT%H:%M:%S%z' # e.g. 2018-02-07T09:11:09-0800

#------------------------------

def connect_to_server(host=cc.HOST, port=cc.PORT,\
                      user=cc.USERNAME, upwd=cc.USERPW,\
                      ctout=5000, stout=30000):
    """Returns MongoDB client.
    """
    uri = _uri(host, port, user, upwd)

    if not is_valid_uri(uri):
        msg = 'connect_to_server: INVALID uri: %s' % str(uri)
        logger.warning(msg)
        sys.exit(msg)

    logger.debug('MongoClient parameters: uri=%s ctout=%d stout=%d' % (uri, ctout, stout))

    client = MongoClient(uri, connect=False, connectTimeoutMS=ctout, serverSelectionTimeoutMS=stout) #, socketTimeoutMS=stout
    #client = MongoClient(host, port, connect=False, connectTimeoutMS=ctout, socketTimeoutMS=stout)
    try:
        result = client.admin.command("ismaster")
        return client

    except errors.ConnectionFailure as err:
        msg = 'ConnectionFailure: Server is not available for port:%s host:%s' % (host, str(port))
        logger.warning(msg)
        sys.exit(msg)

    except errors.OperationFailure as err:
        msg = 'OperationFailure: Authentication failed for user:%s pwd:%s' % (user, 'you know that password...')
        logger.warning(msg)
        #logger.exception(err) 
        sys.exit(msg)

    except:
        msg = 'Unexpected error in connect_to_server(%s)' % str(locals())
        logger.exception(msg) 
        sys.exit()

#------------------------------

def _uri(host, port, user, upwd):
    """ Returns (str) uri like 'mongodb://[username:password@]host1[:port1]'
    """
    if not is_valid_host(host): return None
    if not is_valid_port(port): return None

    rhs = '%s:%s' % (host, str(port))

    # for Calibration DB client
    if host==cc.HOST and port==cc.PORT:
        rhs = '%s:%s@%s' % (user, cc.USERPW if upwd in ('', None) else upwd, rhs)

    # for all other mongod clients
    else:
        if not (upwd in ('', None)): 
            rhs = '%s:%s@%s' % (user, upwd, rhs)

    uri = 'mongodb://%s' % rhs # 'mongodb://psanagpu115:27017' # FOR TEST
    logger.debug('MDBUtils uri: %s' % uri)
    return uri

#------------------------------

def is_valid_type(pname, o, otype):
    if isinstance(o, otype): return True
    logger.warning('parameter "%s" type "%s" IS NOT %s' % (pname, str(o), str(otype)))
    return False

#------------------------------

def is_valid_port(port):
    """Port must be an integer between 0 and 65535"""
    #return is_valid_type('port', port, int)
    iport = port if isinstance(port, int) else\
            int(port) if (isinstance(port, str) and port.isdigit()) else None
    if iport is None:
        logger.warning('parameter port "%s" does not represent integer value' % str(port))
        return False
    elif iport<0 or iport>65535: 
        logger.warning('parameter port "%d" must be an integer between 0 and 65535' % iport)
        return False
    return True

#------------------------------

def is_valid_host(host):
    return is_valid_type('host', host, str)

#------------------------------

def is_valid_uri(uri):
    return is_valid_type('uri', uri, str) and 'mongodb://' in uri

#------------------------------

def is_valid_client(client):
    return is_valid_type('client', client, MongoClient)

#------------------------------

def is_valid_dbname(dbname):
    return is_valid_type('dbname', dbname, str)

#------------------------------

def is_valid_cname(cname):
    return is_valid_type('cname', cname, str)

#------------------------------

def is_valid_database(db):
    return is_valid_type('db', db, Database)

#------------------------------

def is_valid_collection(col):
    return is_valid_type('col', col, Collection)

#------------------------------

def is_valid_objectid(id):
    return is_valid_type('id', id, ObjectId)

#------------------------------

def is_valid_iterable(itr, pname='iterable'):
    return is_valid_type(pname, itr, Iterable)

#------------------------------

def is_valid_fs(fs):
    return is_valid_type('fs', fs, gridfs.GridFS)

#------------------------------

def is_valid_time_sec(time_sec):
    return is_valid_type('time_sec', time_sec, int)

#------------------------------

def database(client, dbname):
    """Returns db for client and (str) dbname, e.g. dbname='cdb_camera_1234'
    """
    if not is_valid_client(client): return None, None
    if not is_valid_dbname(dbname): return None, None
    return client[dbname]

#------------------------------

def db_and_fs(client, dbname):
    """Returns db and fs for client and (str) dbname, e.g. dbname='cdb_exp12345'.
    """
    if not is_valid_client(client)\
    or not is_valid_dbname(dbname): return None, None
    db = client[dbname]
    fs = gridfs.GridFS(db)
    return db, fs

#------------------------------

def collection(db, cname):
    """Returns collection for db and (str) cname, e.g. cname='detector_1234'.
    """
    if not is_valid_database(db)\
    or not is_valid_cname(cname): return None
    return db[cname]

#------------------------------

def client_host(client):
    """Returns default host: localhost
    """
    if not is_valid_client(client): return None
    return client.HOST

#------------------------------

def client_port(client):
    """Returns default port: 27017
    """
    if not is_valid_client(client): return None
    return client.PORT

#------------------------------

def database_names(client):
    """Returns list of database names for client.
    """
    if not is_valid_client(client): return None
    return client.list_database_names()

#------------------------------

def collection_names(db, include_system_collections=False):
    """Returns list of collection names.
    """
    if not is_valid_database(db): return []
    return db.list_collection_names(include_system_collections)

#------------------------------

def database_exists(client, dbname):
    """Returns True if (str) dbname in the list of databases and False otherwise.
    """
    if not is_valid_client(client)\
    or not is_valid_dbname(dbname): return False
    return dbname in database_names(client)

#------------------------------

def collection_exists(db, cname):
    """Returns True if (str) cname in the list of collections and False otherwise.
    """
    if not is_valid_database(db)\
    or not is_valid_cname(cname): return False
    return cname in collection_names(db)

#------------------------------

def delete_database(client, dbname):
    """Deletes database for client and (str) dbname, e.g. dbname='cdb_cspad_0001'.
    """
    if not is_valid_client(client)\
    or not is_valid_dbname(dbname):
        logger.warning('Database "%s" IS NOT DELETED' % str(dbname))
        return

    try:
        client.drop_database(dbname)

    except errors.OperationFailure as err: 
        #logger.exception(err)
        logger.warning('ERROR at attempt to delete document database. '\
                       'Check authorization: calibman -u <username> -p <password>.')
        return
    except: 
        logger.warning('delete_database unexpected ERROR')
        return

#------------------------------

def delete_database_obj(odb):
    """Deletes database for database object.
    """
    if not is_valid_database(odb):
        logger.warning('Database object "%s" IS NOT DELETED' % str(odb))
        return

    try:
        odb.dropDatabase()

    except errors.OperationFailure as err: 
        #logger.exception(err)
        logger.warning('ERROR at attempt to delete database. '\
                       'Check authorization: -u <username> -p <password>.')
        return

    except: 
        logger.warning('delete_database_obj "%s" unexpected ERROR' % str(odb))
        #sys.exit(msg)

#------------------------------

def delete_databases(client, dbnames):
    """Deletes databases for client and (list of str) dbnames,
       e.g. dbnames=['cdb_amox23616', 'cdb_cspad_0001', 'cdb_cspad_0002'].
    """
    if not is_valid_client(client)\
    or not is_valid_iterable(dbnames, 'dbnames'):
        logger.warning('Databases "%s" ARE NOT DELETED' % str(dbname))
        return
    for name in dbnames: 
        try:
            client.drop_database(name)
        except errors.OperationFailure as err:
            #logger.exception(err)
            logger.warning('ERROR at attempt to delete database. '\
                       'Check authorization: calibman -u <username> -p <password>.')
            return

        except: 
            logger.warning('delete_database_obj "%s" unexpected ERROR' % str(odb))
            #sys.exit(msg)
            
#------------------------------

def delete_collection(db, cname):
    """Deletes db collection for database and (str) cname, e.g. cname='cspad_0001'.
    """
    if not is_valid_database(db)\
    or not is_valid_cname(cname): 
        logger.warning('Collection "%s" IS NOT DELETED drom db %s' % (str(cname), str(db)))
        return
    try:
        db.drop_collection(cname)
    except: 
        logger.warning('delete_collection "%s" unexpected ERROR' % str(cname))

#------------------------------

def delete_collection_obj(ocol):
    if not is_valid_collection(ocol):
        logger.warning('Collection object "%s" IS NOT DELETED' % str(ocol))
        return
    ocol.drop()

#------------------------------

def delete_collections(db, cnames):
    """Deletes list of collections from database db, e.g. cname='cspad_0001'.
    """
    if not is_valid_database(db)\
    or not is_valid_iterable(cnames, 'cnames'):
        logger.warning('Collections "%s" ARE NOT DELETED from db %s' % (str(cnames), str(db)))
        return
    for cname in cnames: db.drop_collection(cname)

#------------------------------

def delete_document_from_collection(col, oid):
    if not is_valid_collection(col)\
    or not is_valid_objectid(oid):
        logger.warning('Document with id "%s" IS NOT DELETED from collection %s' % (str(oid), str(col)))
        return

    try:
        col.remove({'_id':oid})

    except errors.OperationFailure as err: 
        #logger.exception(err)
        logger.warning('ERROR at attempt to delete document from collection. '\
                       'Check authorization: calibman -u <username> -p <password>.')
        return

    except errors.InvalidDocument as err: 
        logger.warning('ERROR InvalidDocument: %s' % str(err))
        return

    except: 
        logger.warning('delete_document_from_collection unexpected ERROR' % str(err))
        #sys.exit(msg)
        return

#------------------------------

def db_prefixed_name(name, prefix=cc.DBNAME_PREFIX):
    """Returns database name with prefix, e.g. name='exp12345' -> 'cdb_exp12345'.
    """
    if name is None: return None
    assert isinstance(name,str), 'db_prefixed_name parameter should be str'
    nchars = len(name)
    assert nchars < 128, 'name length should be <128 characters'
    dbname = '%s%s' % (prefix, name)
    logger.debug('db_prefixed_name: %s' % dbname)
    return dbname

#------------------------------

def get_dbname(**kwargs):
    """Returns (str) dbname or None.
       Implements logics for dbname selection:
       -- dbname is used if defined else
       -- prefixed experiment else
       -- prefixed detector else None
    """
    exp    = kwargs.get('experiment', None)
    det    = kwargs.get('detector', None)
    dbname = kwargs.get('dbname', None)
    mode   = kwargs.get('cli_mode', None)

    if dbname is None:
        name = exp if not (exp is None) else det
        if name is None:
            if mode != 'print':
                logger.warning('dbname, experiment, and detector name are NOT SPECIFIED')
            return None
        dbname = db_prefixed_name(name)
    return dbname

#------------------------------

def get_colname(**kwargs):
    """Returns (str) collection name or None.
       Implements logics for collection selection:
       -- dbname is not defined returns None
       -- colname is defined returns colname
       -- returns detector name, it might be None
    """
    dbname = get_dbname(**kwargs)
    if dbname is None: return None

    colname = kwargs.get('colname', None)
    if colname is not None: return colname

    return kwargs.get('detector', None)

#------------------------------

def connect(**kwargs):
    """Connects to host, port with authorization.
       Returns client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det
    """
    host    = kwargs.get('host', cc.HOST)
    port    = kwargs.get('port', cc.PORT)
    user    = kwargs.get('user', cc.USERNAME)
    upwd    = kwargs.get('upwd', cc.USERPW)
    expname = kwargs.get('experiment', None)
    detname = kwargs.get('detector', 'camera-0-cxids1-0')

    dbname_exp = db_prefixed_name(expname)
    dbname_det = db_prefixed_name(detname)

    t0_sec = time()

    client = connect_to_server(host, port, user, upwd)
    db_exp, fs_exp = db_and_fs(client, dbname_exp)
    db_det, fs_det = db_and_fs(client, dbname_det)
    col_det = collection(db_det, detname)
    col_exp = collection(db_exp, detname) 

    msg = '==== Connect to host: %s port: %d connection time %.6f sec' % (host, port, time()-t0_sec)
    msg += '\n  client : %s' % client.name

    if db_exp  is not None: msg += '\n  db_exp:%s' % db_exp.name
    if col_exp is not None: msg += ' col_exp:%s' % col_exp.name
    if db_det  is not None: msg += ' db_det:%s' % db_det.name
    if col_det is not None: msg += ' col_det:%s' % col_det.name
    logger.debug(msg)

    return client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det

#------------------------------
#------------------------------
#------------------------------

def _timestamp(time_sec):
    """Converts time_sec in timestamp of adopted format TSFORMAT.
    """
    if not is_valid_time_sec(time_sec): return None
    return gu.str_tstamp(TSFORMAT, int(time_sec))


def timestamp_id(id): # e.g. id=5b6cde201ead14514d1301f1 or ObjectId
    """Converts MongoDB (str) id to (str) timestamp of adopted format.
    """    
    oid = id
    if isinstance(id, str):
        if len(id) != 24: return str(id) # protection aginst non-valid id
        oid = ObjectId(id)

    if isinstance(oid, ObjectId):
        str_ts = str(oid.generation_time) # '2018-03-14 21:59:37+00:00'
        tobj = Time.parse(str_ts)         # Time object from parsed string
        tsec = int(tobj.sec())            # 1521064777
        str_tsf = _timestamp(tsec)        # re-formatted time stamp
        return str_tsf

    return str(id) # protection aginst non-valid id


def timestamp_doc(doc):
    """Returns document creation (str) timestamp from its id.
    """
    return timestamp_id(doc['_id'])

#------------------------------

def time_and_timestamp(**kwargs):
    """Returns "time_sec" and "time_stamp" from **kwargs.
       If one of these parameters is missing, another is reconstructed from available one.
       If both missing - current time is used.
    """
    time_sec   = kwargs.get('time_sec', None)
    time_stamp = kwargs.get('time_stamp', None)

    if time_sec is not None:
        time_sec = int(time_sec)
        assert isinstance(time_sec, int) , 'time_and_timestamp - parameter time_sec should be int'
        assert 0 < time_sec < 5000000000,  'time_and_timestamp - parameter time_sec should be in allowed range'

        if time_stamp is None: 
            time_stamp = gu.str_tstamp(TSFORMAT, time_sec)
    else:
        if time_stamp is None: 
            time_sec_str, time_stamp = gu.time_and_stamp(TSFORMAT)
        else:
            time_sec_str = gu.time_sec_from_stamp(TSFORMAT, time_stamp)
        time_sec = int(time_sec_str)

    return time_sec, time_stamp

#------------------------------

def docdic(data, dataid, **kwargs):
    """Returns dictionary for db document in style of JSON object.
    """
    #if not is_valid_objectid(dataid):
    #    logger.warning('Data id "%s" IS NOT VALID' % str(dataid))

    doc = {
          'experiment': kwargs.get('experiment', None),
          'run'       : kwargs.get('run', 0),
          'run_end'   : kwargs.get('run_end', 'end'),
          'detector'  : kwargs.get('detector', None),
          'ctype'     : kwargs.get('ctype', None),
          'dtype'     : kwargs.get('dtype', None),
          'time_sec'  : kwargs.get('time_sec', None),
          'time_stamp': kwargs.get('time_stamp', None),
          'version'   : kwargs.get('version', 'v00'),
          'comment'   : kwargs.get('comment', ''),
          'extpars'   : kwargs.get('extpars', None),
          'uid'       : gu.get_login(),
          'host'      : gu.get_hostname(),
          'cwd'       : gu.get_cwd(),
          'id_data'   : dataid,
          'longname'  : kwargs.get('longname', None),
          }

    if isinstance(data, np.ndarray):
        doc['data_type']  = 'ndarray'
        doc['data_dtype'] = str(data.dtype)
        doc['data_size']  = '%d' % data.size
        doc['data_ndim']  = '%d' % data.ndim
        doc['data_shape'] = str(data.shape)

    elif isinstance(data, str):
        doc['data_type']  = 'str'
        doc['data_size']  = '%d' % len(data)

    else:
        doc['data_type']  = 'any'

    logger.debug('doc data type: %s' % doc['data_type'])

    return doc

#------------------------------

def doc_add_id_ts(doc):
    """add items with timestamp for id-s as '_id_ts', 'id_data_ts', 'id_exp_ts'
    """
    for k in ('_id', 'id_data', 'id_exp'):
        v = doc.get(k, None)
        if v is not None: doc['%s_ts'%k] = timestamp_id(v)

#------------------------------

def doc_info(doc, fmt='\n  %16s: %s'):
    s = 'Data document attributes'
    if doc is None: return '%s\n   doc_info: Data document is None...' % s
    for k,v in doc.items(): s += fmt % (k,v)
    return s

#------------------------------

def doc_keys_info(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars'), fmt='  %s: %s'):
    s = ''
    for k in keys: s += fmt % (k, doc.get(k,'N/A'))
    return s

#------------------------------

def print_doc(doc):
    print(doc_info(doc))

#------------------------------

def print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars')):
    print(doc_keys_info(doc, keys))

#------------------------------

def insert_document(doc, col):
    """Returns inserted document id.
    """
    if not is_valid_collection(col):
        logger.warning('collection doc %s IS NOT INSERTED in the collection %s' % (str(doc), str(col)))
        return None

    try:
        doc_id = col.insert_one(doc).inserted_id
        #print('ZZZ doc:', doc)
        logger.debug('insert_document: %30s doc_id:%s' % (col.full_name, doc_id))
        return doc_id

    except errors.ServerSelectionTimeoutError as err: 
        logger.exception(err)
        sys.exit('ERROR at attempt to insert document in database. Check server.')

    except: 
        msg = 'insert_document unexpected ERROR'
        logger.exception(msg)
        sys.exit(msg)

#------------------------------

def encode_data(data):
    """Converts any data type into octal string to save in gridfs.
    """
    s = None
    if   isinstance(data, np.ndarray): s = data.tobytes()
    elif isinstance(data, str):        s = str.encode(data)
    else:
        logger.warning('DATA TYPE "%s" IS NOT "str" OR "numpy.ndarray" CONVERTED BY pickle.dumps ...'%\
                       type(data).__name__)
        s = pickle.dumps(data)
    return s      

#------------------------------

def insert_data(data, fs):
    """Returns inserted data id.
    """
    if not is_valid_fs(fs):
        logger.warning('data %s IS NOT INSERTED in the fs %s' % (str(data), str(fs)))
        return None

    s = encode_data(data)
        
    try:
        r = fs.put(s)
        logger.debug('data has been added to fs in db: %s' % fs._GridFS__database.name)        
        #print('XXX dir(fs):', dir(fs._GridFS__database))
        return r

    except errors.ServerSelectionTimeoutError as err: 
        #logger.exception(err)
        logger.warning('ERROR at attempt to insert data in fs. Check server.')
        return None

    except:
        logger.exception('Unexpected ERROR: %s' % sys.exc_info()[0])
        return None

#------------------------------

def insert_data_and_doc(data, fs, col, **kwargs):
    """For open collection col and GridFS fs inserts calib data and document.
       Returns inserted id_data in fs and id_doc in col.
    """
    if not is_valid_fs(fs)\
    or not is_valid_collection(col):
        logger.warning('data %s IS NOT INSERTED in the\n  ==> fs %s\n  ==> collection %s'%\
                       (str(data), str(fs), str(col)))
        return None, None

    id_data = insert_data(data, fs)
    logger.debug('  - in fs %s id_data: %s' % (fs, id_data))
    doc = docdic(data, id_data, **kwargs)
    id_doc = insert_document(doc, col)
    logger.debug('  - in collection %s id_det: %s' % (col.name, id_doc))
    return id_data, id_doc

#------------------------------

def insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwargs):
    """For open connection inserts calib data and two documents.
       Returns inserted id_data, id_exp, id_det.
    """
    t0_sec = time()
    id_data_exp = insert_data(data, fs_exp)
    id_data_det = insert_data(data, fs_det)

    msg = 'Insert data time %.6f sec' % (time()-t0_sec)\
        + '\n  - in fs_exp %s id_data_exp: %s' % (fs_exp, id_data_exp)\
        + '\n  - in fs_det %s id_data_det: %s' % (fs_det, id_data_det)
    logger.debug(msg)

    doc = docdic(data, id_data_exp, **kwargs)
    logger.debug(doc_info(doc, fmt='  %s:%s')) #sep='\n  %16s: %s'

    t0_sec = time()
    id_doc_exp = insert_document(doc, col_exp)
    _ = doc.pop('_id',None)
    doc['id_data'] = id_data_det # override
    doc['id_exp']  = id_doc_exp  # add
    id_doc_det = insert_document(doc, col_det)

    msg = 'Insert 2 docs time %.6f sec' % (time()-t0_sec)
    if col_exp is not None: msg += '\n  - in (1) %30s id_doc_exp: %s' % (col_exp.full_name, id_doc_exp)
    if col_det is not None: msg += '\n  - in (2) %30s id_doc_det: %s' % (col_det.full_name, id_doc_det)
    logger.debug(msg)

    return id_data_exp, id_data_det, id_doc_exp, id_doc_det

#------------------------------

def insert_calib_data(data, **kwargs):
    """Connects to calibration data base and inserts calib data.
    """
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwargs)
    #id_data_exp, id_data_det, id_exp, id_det = \
    _,_,_,_ = insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwargs)

#------------------------------

def _error_msg(msg):
    return 'wrong parameter %s' % msg

#------------------------------

def valid_experiment(experiment):
    assert isinstance(experiment,str), _error_msg('type')
    assert 7 < len(experiment) < 10, _error_msg('length')

def valid_detector(detector):
    assert isinstance(detector,str), _error_msg('type')
    assert 1 < len(detector) < 65, _error_msg('length')

def valid_ctype(ctype):
    assert isinstance(ctype,str), _error_msg('type')
    assert 4 < len(ctype) < 32, _error_msg('length')

def valid_run(run):
    assert isinstance(run,int), _error_msg('type')
    assert -1 < run < 10000, _error_msg('value')

def valid_version(version):
    assert isinstance(version,str), _error_msg('type')
    assert 1 < len(version) < 32, _error_msg('length')

def valid_version(version):
    assert isinstance(version,str), _error_msg('type')
    assert len(version) < 128, _error_msg('length')

def valid_comment(comment):
    assert isinstance(comment,str), _error_msg('type')
    assert len(comment) < 1000000, _error_msg('length')

def valid_data(data, detector, ctype):
    pass

#------------------------------

def insert_constants(data, experiment, detector, ctype, run, time_sec, **kwargs):
    """Checks validity of input parameters and call insert_calib_data.
    """
    _time_sec, _time_stamp = time_and_timestamp(time_sec=time_sec,\
                                                time_stamp=kwargs.get('time_stamp', None))
    _version = kwargs.get('version', 'v00')
    _comment = kwargs.get('comment', 'default comment')
    _detector = pro_detector_name(detector)

    valid_experiment(experiment)
    valid_detector(_detector)
    valid_ctype(ctype)
    valid_run(run)
    valid_version(_version)
    valid_comment(_comment)
    valid_data(data, detector, ctype)

    kwa = {
          'experiment': experiment,
          'run'       : run,
          'run_end'   : kwargs.get('run_end', 'end'),
          'detector'  : _detector,
          'ctype'     : ctype,
          'time_sec'  : _time_sec,
          'time_stamp': _time_stamp,
          'version'   : _version,
          'comment'   : _comment,
          'host'      : kwargs.get('host', cc.HOST),
          'port'      : kwargs.get('port', cc.PORT),
          'user'      : kwargs.get('user', cc.USERNAME),
          'upwd'      : kwargs.get('upwd', cc.USERPW),
          'extpars'   : kwargs.get('extpars', None),
          }

    insert_calib_data(data, **kwa)

#------------------------------

def del_document_data(doc, fs):
    """From fs removes data associated with a single document.
    """
    if not is_valid_fs(fs):
        logger.warning('Document %s\nIS NOT REMOVED from fs %s' % (str(doc), str(fs)))
        return
    oid = doc.get('id_data', None)
    if oid is None: return

    try:
        fs.delete(oid)

    except errors.OperationFailure as err:
        logger.warning('ERROR OperationFailure... Check authorization: calibman -u <username> -p <password>')
        return

#------------------------------

def del_collection_data(col, fs):
    """From fs removes data associated with multiple documents in collection col.
    """
    if not is_valid_fs(fs)\
    or not is_valid_collection(col):
        logger.warning('collection %s\nDATA IS NOT REMOVED from fs %s' % (str(col), str(fs)))
        return

    for doc in col.find():
        del_document_data(doc, fs)
        #oid = doc.get('id_data', None)
        #if oid is None: return
        #fs.delete(oid)

#------------------------------

def exec_command(cmd):
    from psana.pscalib.proc.SubprocUtils import subproc
    logger.debug('Execute shell command: %s' % cmd)
    if not gu.shell_command_is_available(cmd.split()[0], verb=True): return
    out,err = subproc(cmd, env=None, shell=False, do_wait=True)
    if out or err:
        logger.warning('err: %s\nout: %s' % (err,out))

#------------------------------

def exportdb(host, port, dbname, fname, **kwa):
    client = connect_to_server(host, port)
    dbnames = database_names(client)
    if not (dbname in dbnames):
        logger.warning('--dbname %s is not available in the list:\n%s' % (dbname, dbnames))
        return

    cmd = 'mongodump --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname) # --gzip 
    exec_command(cmd)

#------------------------------

def importdb(host, port, dbname, fname, **kwa):

    if fname is None:
        logger.warning('WARNING input archive file name should be specified as --iofname <fname>')
        return 

    client = connect_to_server(host, port)
    dbnames = database_names(client)
    if dbname in dbnames:
        logger.warning('WARNING: --dbname %s is already available in the list:\n%s' % (dbname, dbnames))
        return

    cmd = 'mongorestore --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname)
    exec_command(cmd)

#------------------------------

def dict_from_data_string(s):
    import ast
    d = ast.literal_eval(s) # retreive dict from str
    if not isinstance(d, dict):
        logger.debug('dict_from_data_string: literal_eval returns type: %s which is not "dict"' % type(d))
        return None

    from psana.pscalib.calib.MDBConvertUtils import deserialize_dict
    deserialize_dict(d)     # deserialize dict values
    return d

#------------------------------

def object_from_data_string(s, doc):
    """Returns str, ndarray, or dict
    """
    data_type = doc.get('data_type', None)
    if data_type is None: 
        logger.warning('object_from_data_string: data_type is None in the doc: %s' % str(doc))
        return None

    logger.debug('object_from_data_string: %s' % data_type)

    if data_type == 'str':
        data = s.decode()
        if doc.get('ctype', None) in ('xtcav_lasingoff', 'xtcav_pedestals', 'lasingoffreference', 'pedestals'):
            return dict_from_data_string(data)
        return data

    elif data_type == 'ndarray':
        str_dtype = doc.get('data_dtype', None)
        nda = np.frombuffer(s, dtype=str_dtype)
        nda.shape = eval(doc.get('data_shape', None)) # eval converts string shape to tuple
        return nda

    elif data_type == 'any': 
        import pickle
        return pickle.loads(s)

    else:
        logger.warning('get_data_for_doc: UNEXPECTED data_type: %s' % data_type)
        return None

#------------------------------

def get_data_for_doc(fs, doc):
    """Returns data referred by the document.
    """
    if not is_valid_fs(fs):
        logger.warning('Document %s\n  associated DATA IS NOT AVAILABLE in fs %s' % (str(doc), str(fs)))
        return None

    if doc is None:
        logger.warning('get_data_for_doc: Data document is None...')
        return None

    idd = doc.get('id_data', None)
    if idd is None:
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None
    idd = ObjectId(idd)

    if not fs.exists(idd):
        logger.debug("get_data_for_doc: NON EXISTENT fs data for data_id %s" % str(idd))
        return None

    out = fs.get(idd)

    s = out.read()

    return object_from_data_string(s, doc)

#------------------------------

def dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dtype=None):
    """Returns dbnames for detector, experiment, collection name, and query.
    """
    cond = (run is not None) or (time_sec is not None) or (vers is not None)
    assert cond, 'Not sufficeint info for query: run, time_sec, and vers are None'
    _det = pro_detector_name(det)
    query={'detector':_det,} # 'ctype':ctype}
    if ctype is not None: query['ctype'] = ctype
    if dtype is not None: query['dtype'] = dtype
    runq = run if not(run in (0,None)) else 9999 # by cpo request on 2020-01-16
    query['run'] = {'$lte': runq} #query['run_end'] = {'$gte': runq}
    if time_sec is not None: query['time_sec'] = {'$lte': int(time_sec)}
    if vers is not None: query['version'] = vers
    logger.debug('query: %s' % str(query))

    db_det, db_exp = db_prefixed_name(_det), db_prefixed_name(str(exp))
    if 'None' in db_det: db_det = None
    if 'None' in db_exp: db_exp = None
    return db_det, db_exp, _det, query

#------------------------------

def number_of_docs(col, query={}):
    return col.count_documents(query)

#------------------------------

def find_docs(col, query={'ctype':'pedestals'}):
    """Returns list of documents for query.
    """
    if not is_valid_collection(col):
        logger.warning('Not available collection %s' % str(col))
        return None

    docs = col.find(query)
    ndocs = col.count_documents(query)
    #print('XXX ndocs', ndocs)

    if ndocs==0:
        logger.warning('col: %s query: %s is not consistent with any document...' % (col.name, query))
        return None
    else: return docs

#------------------------------

def find_doc(col, query={'ctype':'pedestals'}):
    """Returns the document with latest time_sec or run number for specified query.
    """
    if not is_valid_collection(col):
        logger.warning('Not available collection %s' % str(col))
        return None

    docs = find_docs(col, query)
    ndocs = col.count_documents(query)

    if (docs is None)\
    or ndocs==0:
        logger.warning('DB %s collection %s does not have document for query %s'%\
                       (col.database.name, col.name, str(query)))
        return None

    qkeys = query.keys()
    key_sort = 'time_sec' if 'time_sec' in qkeys else 'run'

    doc = docs.sort(key_sort, DESCENDING)[0]
    #msg = 'query: %s\n  %d docs found, selected doc["%s"]=%s'%\
    #      (query, docs.count(), key_sort, doc[key_sort])
    #logger.info(msg)

    return doc

#------------------------------

def document_keys(doc):
    """Returns formatted strings of document keys. 
    """
    keys = sorted(doc.keys())
    s = '%d document keys:' % len(keys)
    for i,k in enumerate(keys):
        if not(i%5): s += '\n      ' 
        s += ' %s' % k.ljust(16)
    return s
    #return '%d doc keys:\n      %s' % (len(keys), '\n      '.join([k for k in keys]))

#------------------------------

def document_info(doc, keys=('time_sec','time_stamp','experiment',\
                  'detector','ctype','run','id_data','id_data_ts', 'data_type','data_dtype','version'),\
                  fmt='%10s %24s %11s %16s %12s %4s %24s %24s %10s %10s %7s'):
    """Returns (str, str) for formatted document values and title made of keys. 
    """
    #if not ('id_data_ts' in doc.keys()): 
    id_data = str(doc.get('id_data',None))
    doc['id_data_ts'] = timestamp_id(id_data)

    doc_keys = sorted(doc.keys())
    if 'experiment' in doc_keys: # CDDB type of document
        vals = tuple([str(doc.get(k,None)) for k in keys])
        return fmt % vals, fmt % keys

    else: # OTHER type of document
        title = '  '.join(doc_keys)
        vals = tuple([str(doc.get(k,None) if k != 'data' else '<some data>') for k in doc_keys])
        info = '  '.join(vals)
        return info, title

#------------------------------

def collection_info(client, dbname, cname):
    """Returns (str) info regarding collection documents. 
    """
    s = 'DB %s collection %s' % (dbname, cname)

    if not is_valid_client(client)\
    or not is_valid_dbname(dbname)\
    or not is_valid_cname(cname):
        return '%s\n    collection_info IS NOT AVAILABLE for client %s' % (s, str(client))

    db = database(client, dbname)
    col = collection(db, cname) # or db[cname]
    docs = col.find().sort('_id', DESCENDING)
    #          # {'ctype':DESCENDING, 'time_sec':DESCENDING, 'run':ASCENDING}
    #  s += '\n%s%s%s' % (gap, gap, 52*'_')
    #s += '\n%s%sCOL %s contains %d docs' % (gap, gap, cname.ljust(12), docs.count())
    #for idoc, doc in enumerate(docs):

    ndocs = number_of_docs(col, query={})

    if not ndocs: return s
        
    s += ' contains %d docs\n' % ndocs
 
    doc = docs[0]
    s += '\n  %s' % (document_keys(doc)) # str(doc.keys()))

    #if cname in ('fs.chunks',):
    #    s += '\n\ncol: "%s" does not have good presentation for documents...' % cname
    #    return s

    _, title = document_info(doc)
    s += '\n  %s%s' % ('doc#', title)
    
    for idoc, doc in enumerate(docs):
        #id_data = doc.get('id_data', None)
        #if id_data is not None: doc['id_data_ts'] = timestamp_id(id_data)
        vals,_ = document_info(doc)
        s += '\n  %4d %s' % (idoc, vals)

    return s

#------------------------------

def database_info(client, dbname, level=10, gap='  '):
    """Returns (str) info about database
    """
    if not is_valid_client(client)\
    or not is_valid_dbname(dbname):
        return 'database_info IS NOT AVAILABLE for client %s\ndbname "%s"' % (str(client), str(dbname))

    #dbname = db_prefixed_name(name)
    dbnames = database_names(client)
    #assert dbname in dbnames, 'dbname: %s is not found in the %s' % (dbname, str(dbnames))
    if not(dbname in dbnames):
        return 'dbname: %s is not found in the list of databases:\n%s' % (dbname, str(dbnames))

    s = '%s\ndbnames %s' % (gap, str(dbnames))
    db = database(client, dbname)
    cnames = collection_names(db)
    s += '\n%sDB %s contains %d collections: %s' % (gap, dbname.ljust(24), len(cnames), str(cnames))
    if level==1: return s

    for cname in cnames:
      col = collection(db, cname) # or db[cname]
      docs = col.find().sort('ctype', DESCENDING)
              # {'ctype':DESCENDING, 'time_sec':DESCENDING, 'run':ASCENDING}
      s += '\n%s%s%s' % (gap, gap, 52*'_')
      s += '\n%s%sCOL %s contains %d docs' % (gap, gap, cname.ljust(12), docs.count())
      #for idoc, doc in enumerate(docs):

      if level==2: continue

      if col.name in ('fs.chunks', 'fs.files'): continue

      s += '\n%s%sDetails for collection %s %d documents' % (gap, gap, col.name, docs.count())
 
      if docs.count() > 0:
        doc = docs[0]
        s += ':\n%s%s%s' % (gap, gap, document_keys(doc)) # str(doc.keys()))
        _, title = document_info(doc)
        s += '\n%s%s%s %s' % (gap, gap, 'doc#', title)
        for idoc, doc in enumerate(docs):
            id_data = doc.get('id_data', None)
            if id_data is not None: doc['id_data_ts'] = timestamp_id(id_data)
            vals,_ = document_info(doc)
            s += '\n%s%s%4d %s' % (gap, gap, idoc, vals)
    return s

#------------------------------

def database_fs_info(db, gap='  '):
    """Returns (str) info about database fs collections 
    """
    if not is_valid_database(db):
        return 'database_fs_info IS NOT AVAILABLE for database "%s"' % str(db)

    s = '%sDB "%s" data collections:' % (gap, db.name)
    for cname in collection_names(db):
       if cname in ('fs.chunks', 'fs.files'):
           docs = collection(db, cname).find()
           s += '\n%s   COL: %s has %d docs' % (gap, cname.ljust(9), docs.count())
    return s

#------------------------------

def client_info(client=None, host=cc.HOST, port=cc.PORT, level=10, gap='  '):
    """Returns (str) with generic information about MongoDB client (or host:port) 
    """
    _client = client if client is not None else connect_to_server(host, port)

    if not is_valid_client(_client):
        return 'client_info IS NOT AVAILABLE for "%s"' % str(_client)

    #s = '\nMongoDB client host:%s port:%d' % (client_host(_client), client_port(_client))
    dbnames = database_names(_client)
    s = '\n%sClient on %s:%d contains %d databases:' % (gap, host, port, len(dbnames)) #, ', '.join(dbnames))
    if level==1: return s
    for idb, dbname in enumerate(dbnames):
        db = database(_client, dbname) # client[dbname]
        cnames = sorted(collection_names(db))
        s += '\n%sDB %s has %2d collections: %s' % (gap, dbname.ljust(20), len(cnames), str(cnames))
        if level==2: continue
        for icol, cname in enumerate(cnames):
            col = collection(db, cname) # or db[cname]
            docs = col.find()
            s += '\n%s%sCOL %s has %d docs' % (gap, gap, cname.ljust(12), docs.count())
            #for idoc, doc in enumerate(docs):
            if docs.count() > 0:
                doc = docs[0]
                s += ': %s' % (str(doc.keys()))
                #logger.debug('%s %4d  %s %s' % (10*' ', idoc, doc['time_stamp'], doc['ctype']))
            if level==3: continue
    return s

#------------------------------

def calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, **kwa):
    """Returns calibration constants for specified parameters. 
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

    client = connect_to_server(**kwa) # host=cc.HOST, port=cc.PORT, user=cc.USERNAME, upwd=...
    dbnames = database_names(client)
    if not(dbname in dbnames):
        logger.warning('DB name %s is not found among available: %s' % (dbname,str(dbnames)))
        return (None, None)

    db, fs = db_and_fs(client, dbname)

    if not collection_exists(db, colname):
        logger.warning('Collection %s is not found in db: %s' % (colname, dbname))
        return (None, None)

    col = collection(db, colname)

    doc = find_doc(col, query)
    if doc is None:
        logger.warning('document is not available for query: %s' % str(query))
        return (None, None)

    return (get_data_for_doc(fs, doc), doc)

#------------------------------

def request_confirmation():
    """Dumps request for confirmation of specified (delete) action.
    """
    logger.warning('Use confirm "-C" option to proceed with request.')


#-----------------------------

def out_fname_prefix(fmt='clb-%s-%s-r%04d-%s', **kwa):
    """Returns output file name prefix like "doc-cxid9114-cspad_0001-r0116-pixel_rms"
    """
    exp = kwa.get('experiment', 'exp')
    det = kwa.get('detector', 'det')
    _det = pro_detector_name(det)
    run = int(kwa.get('run', 0))
    ctype = kwa.get('ctype', 'ctype')
    return fmt % (exp, _det, run, ctype)

#-----------------------------

def save_doc_and_data_in_file(doc, data, prefix, control={'data': True, 'meta': True}):
    """Saves document and associated data in files.
    """
    msg = '\n'.join(['%12s: %s' % (k,doc[k]) for k in sorted(doc.keys())])

    data_type = doc.get('data_type', None)
    ctype     = doc.get('ctype', None)
    dtype     = doc.get('dtype', None)
    verb      = doc.get('vebous', False)

    logger.debug('Save in file(s) "%s" data and document metadata:\n%s' % (prefix, msg))
    #logger.debug(info_ndarr(data, 'data', first=0, last=100))
    logger.debug('save_doc data_type:%s ctype:%s type(data):%s' % (data_type, ctype, type(data).__name__))

    if control['data']:
        fname = '%s.data' % prefix
        if data_type=='ndarray':
            from psana.pscalib.calib.NDArrIO import save_txt # load_txt
            save_txt(fname, data, cmts=(), fmt='%.3f')
            logger.info('saved file: %s' % fname)
            fname = '%s.npy' % prefix
            np.save(fname, data, allow_pickle=False)

        elif ctype == 'geometry': 
            gu.save_textfile(data, fname, mode='w', verb=verb)

        elif data_type=='str' and (ctype in ('lasingoffreference', 'pedestals')): 
            logger.info('save_doc XTCAV IS RECOGNIZED ctype "%s"' % ctype)
            from psana.pscalib.calib.MDBConvertUtils import serialize_dict
            s = dict(data)
            serialize_dict(s)
            gu.save_textfile(str(s), fname, mode='w', verb=verb)

        elif dtype in ('pkl', 'pickle'):
            gu.save_pickle(data, fname, mode='wb')

        elif dtype == 'json':
            gu.save_json(data, fname, mode='w')

        elif data_type == 'any':
            gu.save_textfile(str(data), fname, mode='w', verb=verb)

        else:
            gu.save_textfile(str(data), fname, mode='w', verb=verb)
            
        logger.info('saved file: %s' % fname)

    if control['meta']: 
        fname = '%s.meta' % prefix
        gu.save_textfile(msg, fname, mode='w', verb=verb)
        logger.info('saved file: %s' % fname)

#------------------------------

def data_from_file(fname, ctype, dtype, verb=False):
    """Returns data object loaded from file.
    """
    from psana.pscalib.calib.NDArrIO import load_txt

    assert os.path.exists(fname), 'File "%s" DOES NOT EXIST' % fname

    if dtype == 'xtcav':
        from psana.pscalib.calib.XtcavUtils import load_xtcav_calib_file

    ext = os.path.splitext(fname)[-1]

    data = gu.load_textfile(fname, verb=verb) if ctype == 'geometry' or dtype in ('str', 'txt', 'text') else\
           load_xtcav_calib_file(fname)       if dtype == 'xtcav' else\
           np.load(fname)                     if ext == '.npy' else\
           gu.load_json(fname)                if ext == '.json' or dtype == 'json' else\
           gu.load_pickle(fname)              if ext == '.pkl' or dtype in ('pkl', 'pickle') else\
           load_txt(fname) # input NDArrIO 

    logger.debug('fname:%s ctype:%s dtype:%s verb:%s data:\n%s\n...'%\
                 (fname, ctype, dtype, verb, str(data)[:1000]))
    return data

#------------------------------
# 2020-05-11

def _doc_detector_name(detname, dettype, detnum):
    """returns (dict) document for Detector Name Database (for long <detname> to short <dettype-detnum>).
    """
    t0_sec = time()
    return {'long'      : detname,\
            'short'     : '%s_%06d'%(dettype, detnum),\
            'seqnumber' : detnum,\
            'uid'       : gu.get_login(),
            'host'      : gu.get_hostname(),
            'cwd'       : gu.get_cwd(),
            'time_sec'  : t0_sec,
            'time_stamp': _timestamp(int(t0_sec))
           }

#------------------------------

def _add_detector_name(col, colname, detname, detnum):
    #client = connect_to_server(host=cc.HOST, port=cc.PORT) # , user=cc.USERNAME, upwd=cc.USERPW)
    #db = database(client, dbname)
    #col = collection(db, colname)

    doc = _doc_detector_name(detname, colname, detnum)
    id_doc = insert_document(doc, col)
    return doc['short'] if id_doc is not None else None   

#------------------------------

def _short_detector_name(detname, dbname=cc.DETNAMESDB):
    colname = detname.split('_',1)[0]

    client = connect_to_server(host=cc.HOST, port=cc.PORT) # , user=cc.USERNAME, upwd=cc.USERPW)
    db = database(client, dbname)
    col = collection(db, colname)

    # find a single doc for long detname
    query = {'long':detname}
    ldocs = find_docs(col, query)

    if ldocs is not None:
        ndocs = number_of_docs(col, query)
        logger.debug('ndocs: %d found for long detname: %s' % (ndocs,detname))
        if ndocs>1:
            logger.error('UNEXPECTED ERROR: db/collection: %s/%s has >1 document for detname: %s' % (dbname, colname, detname))
            sys.exit('db/collection: %s/%s HAS TO BE FIXED' % (dbname, colname))

        return ldocs[0].get('short', None)

    # find all docs in the collection
    ldocs = find_docs(col, query={})

    detnum = 0
    if ldocs is None: # empty list
        logger.debug('list of documents in db/collection: %s/%s IS EMPTY' % (dbname, colname))
        detnum = 1
    else:
        for doc in ldocs:
            num = doc.get('seqnumber', 0)
            if num > detnum: detnum = num
        detnum += 1
      
    logger.debug('next available detnum: %d' % detnum)

    short_name = _add_detector_name(col, colname, detname, detnum)
    logger.debug('add document to db/collection: %s/%s doc for short name:%s' % (dbname, colname, short_name))

    return short_name

#------------------------------

def pro_detector_name(detname, maxsize=cc.MAX_DETNAME_SIZE):
    """ Returns short detector name if its length exceeds cc.MAX_DETNAME_SIZE chars.
    """
    return detname if len(detname)<maxsize else _short_detector_name(detname)

#------------------------------
#----------- TEST -------------
#------------------------------

def get_test_nda():
    """Returns random standard nupmpy array for test purpose.
    """
    import psana.pyalgos.generic.NDArrGenerators as ag
    return ag.random_standard(shape=(32,185,388), mu=20, sigma=5, dtype=np.float)

def get_test_dic():
    """Returns dict for test purpose.
    """
    arr = np.array(range(12))
    arr.shape = (3,4)
    return {'1':1, '5':'super', 'a':arr, 'd':{'c':'name'}}

def get_test_txt():
    """Returns text for test purpose.
    """
    return '%s\nThis is a string\n to test\ncalibration storage' % gu.str_tstamp()

#------------------------------
if __name__ == "__main__":

  TEST_FNAME_PNG = '/reg/g/psdm/detector/data2_test/misc/small_img.png'
  TEST_EXPNAME = 'testexper'
  TEST_DETNAME = 'testdet_1234'

  def test_connect(tname):
    """Connect to host, port get db handls.
    """
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxid9114', detector='cspad_0001') 
        #connect(host=cc.HOST, port=cc.PORT, detector='cspad_0001') 

#------------------------------

  def test_insert_one(tname):
    """Insert one calibration data in data base.
    """
    data = None 
    if   tname == '1': data, ctype = get_test_txt(), 'testtext'; logger.debug('txt: %s' % str(data))
    elif tname == '2': data, ctype = get_test_nda(), 'testnda';  logger.debug(info_ndarr(data, 'nda'))
    elif tname == '3': data, ctype = get_test_dic(), 'testdict'; logger.debug('dict: %s' % str(data))

    kwa = {'user': gu.get_login()}
    t0_sec = int(time())
    insert_constants(data, TEST_EXPNAME, TEST_DETNAME, ctype, 20+int(tname), t0_sec,\
                     time_stamp=_timestamp(t0_sec), **kwa)

#------------------------------

  def test_insert_many(tname):
    """Insert many documents in loop
    """
    user = gu.get_login()
    kwa = {'user': user}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment=TEST_EXPNAME, detector=TEST_DETNAME, **kwa)

    t_data = 0
    nloops = 3
    kwa = {'user'     : user,
           'experiment': expname,
           'detector' : detname,
           'ctype'    : 'testnda'}

    for i in range(nloops):
        logger.info('%s\nEntry: %4d' % (50*'_', i))
        data = get_test_nda()
        print_ndarr(data, 'data nda') 
        t0_sec = time()
        t0_int = int(t0_sec)
        kwa['run'] = 10 + i
        kwa['time_sec'] = t0_sec
        kwa['time_stamp'] = _timestamp(t0_int)
        id_data_exp, id_data_det, id_exp, id_det = insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwa)

        dt_sec = time() - t0_sec
        t_data += dt_sec
        logger.info('Insert data in fs of dbs: %s and %s, time %.6f sec '%\
                     (fs_exp._GridFS__database.name, fs_det._GridFS__database.name, dt_sec))

    logger.info('Average time to insert data and two docs: %.6f sec' % (t_data/nloops))

#------------------------------

  def test_get_data(tname):
    """Get doc and data
    """
    kwa = {'detector': TEST_DETNAME}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwa)

    t0_sec = time()
    data_type='any' 
    if tname == '11': data_type='str' 
    if tname == '12': data_type='ndarray' 

    doc = find_doc(col_det, query={'data_type': data_type})
    logger.info('Find doc time %.6f sec' % (time()-t0_sec))
    logger.info('doc:\n%s' % str(doc))
    print_doc(doc)

    t0_sec = time()
    data = get_data_for_doc(fs_det, doc)
    logger.info('get data time %.6f sec' % (time()-t0_sec))
    s = info_ndarr(data, '', first=0, last=100) if isinstance(data, np.ndarray) else str(data)
    logger.info('data:\n%s' % s)

#------------------------------

  #def test_get_data_for_id(tname, det='cspad_0001', data_id='5bbbc6de41ce5546e8959bcf'):
  def test_get_data_for_id(tname, det='cspad_0001', data_id='5bca02bbd1cc55246a67f263'):
    """Get data from GridFS using its id
    """
    kwa = {'detector': det}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwa)
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

  def test_database_content(tname, level=3):
    """Print in loop database content
    """
    #client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
    #    connect(host=cc.HOST, port=cc.PORT)

    client = connect_to_server(host=cc.HOST, port=cc.PORT) # , user=cc.USERNAME, upwd=cc.USERPW)

    print('type(client):', type(client))
    print('dir(client):', dir(client))
    logger.info('host:%s\nport:%d' % (client_host(client), client_port(client)))
    dbnames = database_names(client)
    prefix = db_prefixed_name('') # = "cdb_"
    logger.info('databases: %s' % str(dbnames))
    for idb, dbname in enumerate(dbnames):
        db = database(client, dbname) # client[dbname]
        cnames = collection_names(db)
        logger.info('== DB %2d: %12s # cols:%2d' % (idb, dbname, len(cnames)))
        if dbname[:4] != prefix: 
            logger.info('     skip non-calib dbname: %s' % dbname)
            continue
        if level==1: continue
        for icol, cname in enumerate(cnames):
            col = collection(db, cname) # or db[cname]
            docs = col.find()
            logger.info('     COL %2d: %12s #docs: %d' % (icol, cname.ljust(12), docs.count()))
            if level==2: continue
            #for idoc, doc in enumerate(docs):
            if docs.count() > 0:
                #logger.info('%s %4d  %s %s' % (10*' ', idoc, doc['time_stamp'], doc['ctype']))
                doc = docs[0]
                logger.info('%s doc[0] %s' % (10*' ', str(doc.keys())))

#------------------------------

  def test_dbnames_colnames():
    """Prints the list of DBs and in loop list of collections for each DB.
    """
    client = connect_to_server()
    dbnames = database_names(client)
    #print('== client DBs: %s...' % str(dbnames[:5]))

    for dbname in dbnames:
        db = database(client, dbname)
        cnames = collection_names(db)
        print('== collections of %s: %s' % (dbname.ljust(20),cnames))

#------------------------------

  def test_calib_constants_nda():
    det = 'cspad_0001'
    data, doc = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None)
    print('== doc: %s' % str(doc))
    print_ndarr(data, '== test_calib_constants_nda data', first=0, last=5)

#------------------------------

  def test_calib_constants_text():
    #det = 'cspad_0001'
    #data, doc = calib_constants(det, exp='cxic0415', ctype='geometry', run=50, time_sec=None, vers=None)
    #print('==== test_calib_constants_text data:', data)
    #print('==== doc: %s' % str(doc))

    det = 'tmo_quadanode'
    data, doc = calib_constants(det, exp='amox27716', ctype='calibcfg', run=100)
    print('==== test_calib_constants_text data:\n', data)
    print('==== doc: %s' % str(doc))

#--------------------

  def test_print_dict(d, offset='  '):
    """ prints dict content
        re-defined from psana.pscalib.calib.MDBConvertUtils.print_dict
    """
    print('%sprint_dict' % offset)
    for k,v in d.items():
        if isinstance(v, dict): test_print_dict(v, offset = offset+'  ')
        if isinstance(v, np.ndarray): print_ndarr(v, '%sk:%s nda' % (offset,k), first=0, last=5)
        else: print('%sk:%s t:%s v:%s' % (offset, str(k).ljust(10), type(v).__name__, str(v)[:120]))

#------------------------------

  def test_calib_constants_dict():
    det = 'opal1000_0059'
    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    #print('==== test_calib_constants_dict data:', data)
    print('==== test_calib_constants_dict type(data):', type(data))
    print('==== doc: %s' % str(doc))
    test_print_dict(data)

#------------------------------

  def test_pro_detector_name(shortname='testdet_1234'):
    longname = shortname + '_this_is_insane_long_detector_name_exceeding_55_characters_in_length_or_longer'
    tmode = sys.argv[2] if len(sys.argv) > 2 else '0'
    dname = shortname if tmode=='0' else\
            longname  if tmode=='1' else\
            longname + '_' + tmode # mu._timestamp(int(time()))
    print('==== test_pro_detector_name for detname:', dname)
    name = pro_detector_name(dname)
    print('Returned protected detector name:', name)

#------------------------------

  def dict_usage(tname=None):
      d = {'0': 'test_connect',
           '1': 'test_insert_one txt',
           '2': 'test_insert_one nda',
           '3': 'test_insert_one dic',
           '4': 'test_insert_many',
           '5': 'test_dbnames_colnames',
           '6': 'test_database_content',
           '7': 'test_calib_constants_nda',
           '8': 'test_calib_constants_text',
           '9': 'test_calib_constants_dict',
           '10': 'test_get_data_for_id',
           '11': 'test_get_data txt',
           '12': 'test_get_data nda',
           '13': 'test_get_data dict',
           '20': 'test_pro_detector_name [test-number=0-short name, 1-fixed long name, n-long name +"_n"]'\
          }
      if tname is None: return d
      return d.get(tname, 'NON IMPEMENTED TEST')

#------------------------------

  def usage(tname=None):
    s = '%s\nUsage:' % (50*'_')
    for k,v in dict_usage().items(): s += '\n  %2s: %s' % (k,v)
    print('%s\n%s'%(s,50*'_'))

#------------------------------

if __name__ == "__main__":
    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    #logging.basicConfig(format=fmt', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)
    #logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG) # logging.INFO
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv) < 2: usage()
    logger.info('%s Test %s %s: %s' % (25*'=', tname, 25*'=', dict_usage(tname)))
    if   tname == '0': test_connect(tname)
    elif tname in ('1','2','3'): test_insert_one(tname)
    elif tname == '4': test_insert_many(tname)
    elif tname == '5': test_dbnames_colnames()
    elif tname == '6': test_database_content(tname)
    elif tname == '7': test_calib_constants_nda()
    elif tname == '8': test_calib_constants_text()
    elif tname == '9': test_calib_constants_dict()
    elif tname =='10': test_get_data_for_id(tname)
    elif tname in ('11','12','13'): test_get_data(tname)
    elif tname =='20': test_pro_detector_name()
    else: logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
