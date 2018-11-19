from psp.Pv import Pv
import pyca
import json

pvNames = {
           'DAQ:LAB2:HSD:DEV02:ENABLE': (1,1,1,1),
           'DAQ:LAB2:HSD:DEV02:RAW_PS': (3,3,3,3),
           'DAQ:LAB2:HSD:DEV02:TESTPATTERN': 2,
           'DAQ:LAB2:HSD:DEV02:FEX_YMIN': (5,5,5,5),
           'DAQ:LAB2:HSD:DEV02:FEX_YMAX': (2040,2040,2040,2040),
           'DAQ:LAB2:HSD:DEV02:FEX_XPRE': (1,1,1,1),
           'DAQ:LAB2:HSD:DEV02:FEX_XPOST': (1, 1, 1, 1),
           }

for key, val in pvNames.items():
    print("key, val: ", key, val)
    _pv = Pv(key)
    print("Done Pv")
    _pv.connect(1.0)
    print("Done connect")
    _pv.put(val)
    print("Done put")

pyca.flush_io()
print("Done flush_io")

# TODO: avoid creating Pv multiple times and avoid multiple connects
for key, val in pvNames.items():
    _pv = Pv(key)
    _pv.connect(1.0)
    _pv.get(False, 1.0)
print("Done get")

# config keys for pvNames are the strings after the last colon
xtcDict = {}
for key, val in pvNames.items():
    print("xtcDic: ", key, val)
    if type(val) == tuple:
        xtcDict[key.split(':')[-1]] = list(val)
    else:
        xtcDict[key.split(':')[-1]] = val
print("### Here's my xtcDict: ", xtcDict)

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

print("Done hsdconfig.py")
#with open('xtc.json', 'w') as f:
#    json.dump(config, f)
