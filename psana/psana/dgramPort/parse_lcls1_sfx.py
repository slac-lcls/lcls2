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
    print("Example: python parse_lcls1_sfx.py cxid9114 95 CxiDs1.0:Cspad.0 500 xray /reg/common/package/temp")
    exit(0)
elif len(sys.argv) == 7:
    outdir  = sys.argv[6]

ds = DataSource('exp='+expname+':run='+run) # 89 dark 95 xray
det = Detector(detname)
epics = ds.env().epicsStore()

def bitwise_array(value):
    if np.isscalar(value):
        return value
    val = np.asarray(value)
    return [base64.b64encode(val), val.shape, val.dtype.str]

events = []

for i, evt in enumerate(ds.events()):
    raw = det.raw(evt)

    ebeam = evt.get(Bld.BldDataEBeamV7, Source('BldInfo(EBeam)'))
    photonEnergy = epics.value('SIOC:SYS0:ML00:AO541')

    evtId = evt.get(EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    timestamp = (sec << 32) | nsec

    if raw is not None:
        evtDict = {}
        # det.raw
        evtDict['quads0_data'] = bitwise_array(raw[0:8])
        evtDict['quads1_data'] = bitwise_array(raw[8:16])
        evtDict['quads2_data'] = bitwise_array(raw[16:24])
        evtDict['quads3_data'] = bitwise_array(raw[24:32])
        # photon energy
        evtDict['photonEnergy'] = photonEnergy
        # timestamp
        evtDict['timestamp'] = timestamp
        events.append(evtDict)
        if i == nevents: break

with open(os.path.join(outdir,"crystal_"+tag+".json"), 'w') as f:
    f.write(json.dumps(events))
