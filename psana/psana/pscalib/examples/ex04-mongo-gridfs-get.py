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

import gridfs

from pymongo import MongoClient
#from pymongo.ObjectId import ObjectId
#from bson.objectid import ObjectId

#client = MongoClient('localhost', 27017)
client = MongoClient('psanaphi105', 27017) #, username=uname, password=pwd)
db = client['calib-cxi12345']
fs = gridfs.GridFS(db)
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
#docs = col.find({"run": 125})
docs = col.find({"time_stamp" : "2018-01-25T09:33:10PST"})
doc = docs[0]
dt_sec = time() - t0_sec

print 'Document with reference extraction time %.6f sec, _id: %s' % (dt_sec, doc['_id'])
print 'Document data attrs: ', doc["data_size"],  doc["data_shape"],  doc["data_type"]

t0_sec = time()
out = fs.get(doc['data_id'])
print 'out.upload_date', out.upload_date
s = out.read() # returns str
nda = gu.np.fromstring(s)
dt_sec = time() - t0_sec

print 'Extracted document attributes'
for k,v in doc.iteritems() : print '%16s : %s' % (k,v) 

print 'nda extraction time %.6f sec _id: %s' % (dt_sec, doc['_id'])
gu.print_ndarr(nda, 'nda') 

#for doc in docs:
    #pprint.pprint(doc)
#    print doc["_id"], doc["data_shape"], ObjectId(doc["_id"]).getTimestamp
#------------------------------
#sys.exit('TEST EXIT')
#------------------------------
