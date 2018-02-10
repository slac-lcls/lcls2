
"""
Saving large files using gridfs motivated by
https://github.com/mongodb/mongo-python-driver/blob/master/doc/examples/gridfs.rst
"""
#------------------------------

import sys

#------------------------------

#import numpy as np
#import pyimgalgos.NDArrGenerators as ag
from time import time
#from pyimgalgos.GlobalUtils import print_ndarr
#import PSCalib.DCUtils as gu
import Utils as gu

#------------------------------

time_sec = time()
time_stamp = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%Z', time_sec=time_sec)
print 'time_sec %.9f' % time_sec
nda = gu.random_standard(shape=(32,185,388), mu=20, sigma=5, dtype=gu.np.float)

#------------------------------
#sys.exit('TEST EXIT')
#------------------------------

t0_sec = time()

#import pickle
#from bson.binary import Binary

#import pymongo
import gridfs
from pymongo import MongoClient

#client = MongoClient('localhost', 27017)
client = MongoClient('psanaphi105', 27017) #, username=uname, password=pwd)
db = client['calib-cxi12345']
fs = gridfs.GridFS(db)
col = db['camera-0-cxids1-0']

dt_sec = time() - t0_sec
print 'db: %s collection: %s connection time %.6f sec' % (db.name, col.name, dt_sec)

#------------------------------
#sys.exit('TEST EXIT')
#------------------------------

gu.print_ndarr(nda, 'nda') 

t0_sec = time()
ida = fs.put(nda.tobytes())
dt_sec = time() - t0_sec

print 'ida = fs.put(nda.tobytes()) time %.6f sec ida: %s' % (dt_sec, ida)

#------------------------------
#sys.exit('TEST EXIT')
#------------------------------

t0_sec = time()

doc = {
   "experiment": "cxi12345",
   "run": 126,
   "detector": col.name,
   "ctype": "pedestals",
   "time_sec": '%.9f' % time_sec,
   "time_stamp": time_stamp,
   "version": "v01-23-45",
   "facility": "LCLS2",
   "uid":  gu.get_login(),
   "host": gu.get_hostname(),
   "cwd":  gu.get_cwd(),
   "comments": ["very good constants", "throw them in trash immediately!"],
   "data_size":  nda.size,
   "data_shape": nda.shape,
   "data_type":  str(nda.dtype),
   "data_id":    ida,
}

dt_sec = time() - t0_sec

print 'Data document attributes'
for k,v in doc.iteritems() : print '%16s : %s' % (k,v) 

print 'document preparation time %.6f sec' % (dt_sec)

#------------------------------
#sys.exit('TEST EXIT')
#------------------------------

t0_sec = time()

doc_id = col.insert_one(doc).inserted_id

dt_sec = time() - t0_sec
print 'Inserting document %s time %.6f sec' % (doc_id, dt_sec)

#------------------------------
sys.exit('TEST EXIT')
#------------------------------
