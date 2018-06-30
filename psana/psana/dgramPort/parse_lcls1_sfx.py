from psana import *
import json
import numpy as np
import base64

import sys, os
expname = sys.argv[1]
run     = sys.argv[2]
detname = sys.argv[3]
nevents = int(sys.argv[4])
tag     = sys.argv[5]
outdir = ''
if len(sys.argv) < 6:
    print("Usage: python parse_lcls1_sfx.py expname run detname nevents tag (optional)outdir")
    print("Example: python parse_lcls1_sfx.py cxid9114 95 CxiDs2.0:Cspad.0 500 xray /reg/common/package/temp")
    exit(0)
elif len(sys.argv) == 7:
    outdir  = sys.argv[6]

if 'xray' in tag and 'cxid9114' in expname:
    #ds = DataSource('exp='+expname+':run='+run)
    ds = DataSource('exp='+expname+':run='+run+':dir=/reg/d/psdm/cxi/cxid9114/demo/xtc') # 89 dark 95 xray
else:
    ds = DataSource('exp='+expname+':run='+run)

det = Detector(detname)
ebeamDet = Detector('EBeam')
epics = ds.env().epicsStore()
print "det: ", det, dir(det)
print "epics: ", epics

def bitwise_array(value):
    if np.isscalar(value):
        return value
    val = np.asarray(value)
    return [base64.b64encode(val), val.shape, val.dtype.str]

events = []

for i, evt in enumerate(ds.events()):
    print "####: ", i, evt
    raw = det.raw(evt)
    calib = det.calib(evt)

    ebeam = ebeamDet.get(evt)
    photonEnergy = epics.value('SIOC:SYS0:ML00:AO541')
    print "photonEnergy: ", photonEnergy

    evtId = evt.get(EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    timestamp = (sec << 32) | nsec
    print "timestamp: ", timestamp

    if raw is not None:
        print "raw: ", raw.shape, calib.shape
        evtDict = {}
        evtDict['timestamp'] = timestamp
        evtDict['data'] = {}
        # det.raw
        evtDict['data']['quads0_data'] = bitwise_array(raw[0:8])
        evtDict['data']['quads1_data'] = bitwise_array(raw[8:16])
        evtDict['data']['quads2_data'] = bitwise_array(raw[16:24])
        evtDict['data']['quads3_data'] = bitwise_array(raw[24:32])
        # photon energy
        evtDict['data']['photonEnergy'] = photonEnergy
        events.append(evtDict) # TODO: Out of memory. Use ZMQ sockets to bypass writing to json
    if i == nevents: break

with open(os.path.join(outdir,"crystal_"+tag+".json"), 'w') as f:
    f.write(json.dumps(events))
