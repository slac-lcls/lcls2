# Run from psanagpuXXX machine, source /reg/g/psdm/etc/psconda.sh
# Also run translate_xtc_demo.py
from psana import *
import numpy as np
import base64
import sys, zmq

expname = sys.argv[1]
run = sys.argv[2]
detname = sys.argv[3]
nevents = int(sys.argv[4])
tag = sys.argv[5]
if len(sys.argv) < 6:
    print("Usage: python parse_lcls1_sfx.py expname run detname nevents tag")
    print("Example: python parse_lcls1_sfx.py cxid9114 96 CxiDs2.0:Cspad.0 -1 xray")
    exit(0)

if 'xray' in tag and 'cxid9114' in expname:
    ds = DataSource('exp=' + expname + ':run=' + run + ':dir=/reg/d/psdm/cxi/cxid9114/demo/xtc')  # 89 dark 95 xray
else:
    ds = DataSource('exp=' + expname + ':run=' + run)

det = Detector(detname)
ebeamDet = Detector('EBeam')
epics = ds.env().epicsStore()


def bitwise_array(value):
    if np.isscalar(value):
        return value
    val = np.asarray(value)
    return [base64.b64encode(val), val.shape, val.dtype.str]


# Start your translate_xtc_demo.py before you start this script
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.bind("tcp://127.0.0.1:5557")

events = []
for i, evt in enumerate(ds.events()):
    if i == nevents: break
    print("Event: ", i)
    raw = det.raw(evt)
    calib = det.calib(evt)

    ebeam = ebeamDet.get(evt)
    photonEnergy = epics.value('SIOC:SYS0:ML00:AO541')

    evtId = evt.get(EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    timestamp = (sec << 32) | nsec

    if raw is not None:
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
        events.append(evtDict)
        zmq_socket.send_json(evtDict)

doneDict = {'done': True}
zmq_socket.send_json(doneDict)
