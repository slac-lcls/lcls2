# Run from psanagpuXXX machine, source /reg/g/psdm/etc/psconda.sh and use python2
# Also run translate_xtc_demo.py
from psana import *
import numpy as np
import base64
import sys, zmq

from psalgos.pypsalgos import PyAlgos

if len(sys.argv) < 5:
    print("Usage: python parse_lcls1_cxic0415.py expname detname outdir run")
    print("Example: bsub -m \"psana1112\" -q psanaq  -o ~/%J.out python parse_lcls1_cxic0415.py cxic0415 DscCsPad /reg/d/psdm/cxi/cxitut13/scratch 96")
    exit(0)

expname = sys.argv[1]
detname = sys.argv[2]
outdir  = sys.argv[3]
run     = sys.argv[4]
tag     = str(run)
nevents = -1 # process all events:-1

minPeaks = 15
hitParam_alg1_npix_min = 2
hitParam_alg1_npix_max = 10
hitParam_alg1_amax_thr = 300
hitParam_alg1_atot_thr = 600
hitParam_alg1_son_min = 10
hitParam_alg1_rank = 3
hitParam_alg1_radius = 3
hitParam_alg1_dr = 2

if 'xray' in tag and 'cxid9114' in expname:
    ds = DataSource('exp='+expname+':run='+run+':dir=/reg/d/psdm/cxi/cxid9114/demo/xtc') # 89 dark 95 xray
else:
    ds = DataSource('exp='+expname+':run='+run)

det = Detector(detname)
ebeamDet = Detector('EBeam')
epics = ds.env().epicsStore()

def bitwise_array(value):
    if np.isscalar(value):
        return value
    val = np.asarray(value)
    return [base64.b64encode(val), val.shape, val.dtype.str]

mask = det.mask(int(run),calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)

alg = PyAlgos(mask=None, pbits=0)
peakRadius = int(hitParam_alg1_radius)
alg.set_peak_selection_pars(npix_min=hitParam_alg1_npix_min, npix_max=hitParam_alg1_npix_max, \
                                 amax_thr=hitParam_alg1_amax_thr, atot_thr=hitParam_alg1_atot_thr, \
                                 son_min=hitParam_alg1_son_min)

# Start your translate_xtc_demo.py before you start this script
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.bind("tcp://127.0.0.1:5557")

for i, evt in enumerate(ds.events()):
    if i == nevents: break
    raw = det.raw(evt)
    calib = det.calib(evt)

    peaks = alg.peak_finder_v3r3(calib,
                                 rank=int(hitParam_alg1_rank),
                                 r0=peakRadius,
                                 dr=hitParam_alg1_dr,
                                 nsigm=hitParam_alg1_son_min,
                                 mask=mask.astype(np.uint16))
    npeaks = peaks.shape[0]

    ebeam = ebeamDet.get(evt)
    photonEnergy = epics.value('SIOC:SYS0:ML00:AO541')

    evtId = evt.get(EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    timestamp = (sec << 32) | nsec

    if raw is not None and npeaks >= minPeaks:
        print("Event: ", i)
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
        zmq_socket.send_json(evtDict)

doneDict = {'done': True}
zmq_socket.send_json(doneDict)
