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
#nda = gu.random_standard(shape=(3,5), mu=20, sigma=5, dtype=gu.np.float64)

#sys.exit('TEST EXIT')

#------------------------------

t0_sec = time()

import pickle
from bson.binary import Binary

import pymongo
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
#client = MongoClient('psanaphi105', 27017) #, username=uname, password=pwd)
db = client['calib-cxi12345']
col = db['camera-0-cxids1-0']

dt_sec = time() - t0_sec
print 'db: %s collection: %s connection time %.6f sec' % (db.name, col.name, dt_sec)

#------------------------------

t0_sec = time()

arr = nda.flatten()
arr = ' '.join(['%.2f' % v for v in arr])
sarr = Binary(pickle.dumps(arr, protocol=2), subtype=128)

#sarr = nda.flatten().tolist()

#print 'sarr:', sarr
gu.print_ndarr(nda, 'nda') 
#------------------------------
#sys.exit('TEST EXIT')
#------------------------------

doc = {
   "experiment": "cxi12345",
   "run": 124,
   "detector": col.name,
   "ctype": "pedestals",
   "time_sec": time_sec,
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
   "data":       sarr,
}

dt_sec = time() - t0_sec

#print 'doc["data"]',  doc["data"]

print 'document preparation time %.6f sec' % (dt_sec)

#   "data":       str(nda.flatten().tostring()),
#   "data":       nda.flatten().tobytes(),

#------------------------------

#for k,v in doc.iteritems() : 
#    if k=="data" : 
#        print '%16s : skip...' % (k),
        #gu.print_ndarr(nda, name='nda', first=0, last=3)
        #gu.print_ndarr(nda, name='nda', first=0, last=3)
#    else :
#        print '%16s : %s' % (k,str(v))

#------------------------------

t0_sec = time()

doc_id = col.insert_one(doc).inserted_id

dt_sec = time() - t0_sec
print 'Inserting document %s time %.6f sec' % (doc_id, dt_sec)

#------------------------------

#doc = col.find({"run": 125})[0]
#xcarr = pickle.loads(doc["data"])
#print 'X-check arr', xcarr

#------------------------------
sys.exit('TEST EXIT')
#------------------------------
