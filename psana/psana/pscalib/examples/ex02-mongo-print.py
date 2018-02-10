import sys

#------------------------------

#import numpy as np
#import pyimgalgos.NDArrGenerators as ag
from time import time
#from pyimgalgos.GlobalUtils import print_ndarr
#import PSCalib.DCUtils as gu
import Utils as gu

#------------------------------


t0_sec = time()



from pymongo import MongoClient
#from pymongo.ObjectId import ObjectId
from bson.objectid import ObjectId

client = MongoClient('localhost', 27017)
#client = MongoClient('psanaphi105', 27017) #, username=uname, password=pwd)
db = client['calib-cxi12345']
col = db['camera-0-cxids1-0']

dt_sec = time() - t0_sec
print 'db: %s collection: %s connection time %.6f sec' % (db.name, col.name, dt_sec)

#------------------------------

print '\nDB %s content' % (db.name)
print 'list_collection_names:', db.list_collection_names
print 'col.full_name: %s' % col.full_name
print 'col.name: %s' % col.name
print 'col.count(): %s' % col.count()

import pprint
t0_sec = time()
docs = col.find({"run": 125})
dt_sec = time() - t0_sec
print 'col.find({"run": 125}): time %.6f sec' % (dt_sec)

import pickle

t0_sec = time()
doc = docs[0]
xcarr = pickle.loads(doc["data"])
arr = gu.np.fromstring(xcarr, dtype=float, count=-1, sep=' ')

dt_sec = time() - t0_sec
print 'X-check arr', arr
print 'Unpack time %.6f sec' % (dt_sec)

#for doc in docs:
    #pprint.pprint(doc)
#    print doc["_id"], doc["data_shape"], ObjectId(doc["_id"]).getTimestamp
#------------------------------
#sys.exit('TEST EXIT')
#------------------------------
