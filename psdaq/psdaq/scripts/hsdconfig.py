from psp.Pv import Pv
import pyca
import json

pvNames = {}
pvNames['ENABLE'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [1]*4 }
pvNames['RAW_PS'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [2]*4 }
pvNames['TESTPATTERN'] = {'type': 'int',
                       'value': 2}
pvNames['FEX_YMIN'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [5]*4 }
pvNames['FEX_YMAX'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [2040]*4 }
pvNames['FEX_XPRE'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [1]*4 }
pvNames['FEX_XPOST'   ] = {'type' : 'int',
                     'count': 4,
                     'value' : [1]*4 }

# Get lastest PVs from config dbase and override with pvNames
from pymongo import MongoClient, errors, DESCENDING
username = 'yoon82'
host = 'psdb-dev'
port = 9306
prefix = "DAQ:LAB2:HSD:DEV02"
client = MongoClient('mongodb://%s:%s@%s:%s'%(username,username,host,port))
db = client['config_db']
collection = db['amo']
pvdb = {}
for post in collection.find({"_id":234}): # FIXME: find the latest document, i.e. collection.find_one(sort=[("_id", DESCENDING)])
    for key,val in post.items():
        if key in pvNames: # override pv value
            if "_id" in key:
                pvdb[key] = val+1
            else:
                pvdb[key] = pvNames[key]
                print("Found match: ", key, val)
        else:
            pvdb[key] = val

for key, val in pvdb.items():
    if "_id" not in key:
        _pv = Pv(prefix+':'+key)
        _pv.connect(1.0)
        if type(val['value']) == list:
            _pv.put(tuple(val['value']))
        else:
            _pv.put(val['value'])

pyca.flush_io()

# TODO: avoid creating Pv multiple times and avoid multiple connects
for key, val in pvdb.items():
    if "_id" not in key:
        _pv = Pv(prefix+':'+key)
        _pv.connect(1.0)
        _pv.get(False, 1.0)

# config keys for pvNames are the strings after the last colon
xtcDict = {}
for key, val in pvNames.items():
    if type(val['value']) == tuple:
        xtcDict[key.split(':')[-1]] = list(val['value'])
    else:
        xtcDict[key.split(':')[-1]] = val['value']

# Save configure transition to xtc.json
config = {}
config['alg'] = {}
config['alg']['software'] = 'hsdConfig'
config['alg']['version'] = list([1, 2, 4])

config['ninfo'] = {}
config['ninfo']['software'] = 'xpphsd'
config['ninfo']['detector'] = 'hsd'
config['ninfo']['serialNum'] = 'detnum1235'
config['ninfo']['seg'] = 0

config['xtc'] = {}
config['xtc'] = xtcDict

# TODO: We need information about pixel to time conversion
# TODO: Which hsd channels are configured

#config['hsd'] = {}
#config['hsd']['channels'] = list([0,1,2,3])
#config['hsd']['software'] = 'xpphsd'
#config['hsd']['detector'] = 'hsd'
#config['hsd']['alg'] = 'hsd123'
#config['hsd']['detector']['alg'] = 'fpga'
#config['hsd']['detector']['version'] = '0x010203'

config = json.dumps(config)

print("config: ", config)

# Apply the new configuration changes
_pv = Pv('DAQ:LAB2:HSD:DEV02:BASE:APPLYCONFIG')
_pv.connect(1.0)
_pv.put(1)

#with open('xtc.json', 'w') as f:
#    json.dump(config, f)
