
"""
Saving large files using gridfs motivated by
https://github.com/mongodb/mongo-python-driver/blob/master/doc/examples/gridfs.rst
"""
#------------------------------

import sys
import gridfs
from pymongo import MongoClient
#import pymongo

#------------------------------

#import numpy as np
#import pyimgalgos.NDArrGenerators as ag
#from pyimgalgos.GlobalUtils import print_ndarr
#import PSCalib.DCUtils as gu
from time import time
import Utils as gu

#------------------------------

#import logging
#logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
#                    datefmt='%m-%d-%Y %H:%M:%S', level=logging.WARNING)
#log = logging.getLogger('my_test')

#------------------------------

class Store() :
    def __init__(self) :
        self.t_data = 0
        self.t_doc  = 0

sp = Store()

#------------------------------

def get_nda() :
    return gu.random_standard(shape=(32,185,388), mu=20, sigma=5, dtype=gu.np.float)

#------------------------------

def time_and_stamp() :
    time_sec = time()
    return time_sec, gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%Z', time_sec=time_sec)

#------------------------------

def connect_to_server(host='psanaphi105', port=27017) :
    t0_sec = time()
    client = MongoClient(host, port)
    dt_sec = time() - t0_sec
    print('host: %s port: %d connection time %.6f sec' % (host, port, dt_sec))
    return client

#------------------------------

def db(client, dbname='calib-cspad-0-cxids1-0') :
    return client[dbname]

#------------------------------

def db_and_fs(client, dbname='calib-cxi12345') :
    db = client[dbname]
    fs = gridfs.GridFS(db)
    return db, fs

#------------------------------

def collection(db, cname='camera-0-cxids1-0') :
    return db[cname]

#------------------------------

def doc_exp(data, data_id, **kwargs) :
    """Returns dictionary for db document in style of JSON object.
    """
    t0_sec = time()

    time_sec, time_stamp = time_and_stamp()
    doc = {
          "experiment" : kwargs.get('exp', 'cxi12345'),
          "run"        : kwargs.get('run', '0'),
          "detector"   : kwargs.get('det', 'camera-0-cxids1-0'),
          "ctype"      : kwargs.get('ctype', 'pedestals'),
          "time_sec"   : '%.9f' % kwargs.get('time_sec', time_sec),
          "time_stamp" : kwargs.get('time_stamp', time_stamp),
          "version"    : 'v01',
          "facility"   : 'LCLS2',
          "uid"        : gu.get_login(),
          "host"       : gu.get_hostname(),
          "cwd"        : gu.get_cwd(),
          "comments"   : ['very good constants', 'eat this document before reading!'],
          "data_id"    : data_id,
          }

    if isinstance(data, gu.np.ndarray) :
        print 'doc data is np.ndarray'
        doc["data_size"]  = '%d' % data.size
        doc["data_shape"] = str(data.shape)
        doc["data_type"]  = str(data.dtype)

    elif isinstance(data, str) :
        print 'doc data is str'
        doc["data_size"]  = '%d' % len(data)
        doc["data_type"]  = 'str'

    else :
        doc["data_type"]  = 'UNKNOWN'

    dt_sec = time() - t0_sec
    print 'document preparation time %.6f sec' % (dt_sec)
    return doc

#------------------------------

def print_doc(doc) :
    print('Data document attributes')
    for k,v in doc.iteritems() : 
        print('%16s : %s' % (k,v))

#------------------------------

def print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size')) :
    for k in keys :
        print '  %s : %s' % (k, doc[k]),
    print ''

#------------------------------

def insert_document(doc, col) :
    t0_sec = time()
    doc_id = col.insert_one(doc).inserted_id
    dt_sec = time() - t0_sec
    sp.t_doc += dt_sec
    print 'Collection: %s insert document %s time %.6f sec' % (col.name, doc_id, dt_sec)
    return doc_id

#------------------------------

def insert_data(data, fs) :
    t0_sec = time()
    id = 24*'0'
    if isinstance(data, gu.np.ndarray) : id = fs.put(data.tobytes())
    elif isinstance(data, str) :         id = fs.put(data)
    dt_sec = time() - t0_sec
    sp.t_data += dt_sec
    print 'Insert data in %s time %.6f sec ida: %s' % (fs, dt_sec, id)
    return id

#------------------------------

def test_01(tname) :
    expname = 'cxi12345'
    detname = 'camera-0-cxids1-0'
    dbname_exp = 'calib-%s' % expname
    dbname_det = 'calib-%s' % detname

    client  = connect_to_server(host='psanaphi105', port=27017)
    db_exp, fs  = db_and_fs(client, dbname=dbname_exp)
    db_det = db(client, dbname=dbname_det)
    col_det = collection(db_det, cname=detname)
    col_exp = collection(db_exp, cname=expname)

    print('client  : %s' % client.name)
    print('db_exp  : %s' % db_exp.name)
    print('col_exp : %s' % col_exp.name)
    print('db_det  : %s' % db_det.name)
    print('col_det : %s' % col_det.name)

    nda = get_nda()
    gu.print_ndarr(nda, 'nda') 

    ida = insert_data(nda, fs)

    doc = doc_exp(nda, ida, exp=expname, det=detname)
    print_doc(doc)

    insert_document(doc, col_exp)
    insert_document(doc, col_det)

#------------------------------

def test_02(tname) :

    nloops = 100

    expname = 'cxi12345'
    detname = 'camera-0-cxids1-0'
    dbname_exp = 'calib-%s' % expname
    dbname_det = 'calib-%s' % detname

    client  = connect_to_server(host='psanaphi105', port=27017)
    db_exp, fs  = db_and_fs(client, dbname=dbname_exp)
    db_det = db(client, dbname=dbname_det)
    col_det = collection(db_det, cname=detname)
    col_exp = collection(db_exp, cname=expname)

    print('client  : %s' % client.name)
    print('db_exp  : %s' % db_exp.name)
    print('col_exp : %s' % col_exp.name)
    print('db_det  : %s' % db_det.name)
    print('col_det : %s' % col_det.name)

    for i in range(nloops) :
        print '%s\nEntry: %4d' % (50*'_', i)
        nda = get_nda()
        gu.print_ndarr(nda, 'nda') 
        ida = insert_data(nda, fs)

        doc = doc_exp(nda, ida, exp=expname, det=detname, run=i)
        print_doc_keys(doc)

        insert_document(doc, col_exp)
        insert_document(doc, col_det)

    print 'Average time to insert data: %.6f sec' % (sp.t_data/nloops)
    print 'Average time to insert doc : %.6f sec' % (0.5*sp.t_doc/nloops)

#------------------------------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '1' : test_01(tname);
    elif tname == '2' : test_02(tname)
    else : print 'Not-recognized test name: %s' % tname
    sys.exit('End of test %s' % tname)

#------------------------------
sys.exit('TEST EXIT')
#------------------------------
