from psana import DataSource
from psana.dgramedit import AlgDef, DetectorDef, DataType
import psana.psexp.TransitionId
import sys
import numpy as np
from libpressio import PressioCompressor
import json
from prometheus_client import REGISTRY, Gauge

# Define compressor configuration:
lpjson = {
    "compressor_id": "sz3", #the compression algo.
    "compressor_config": {
        #"sz:data_type"           : lp.pressio_uint16_dtype,
        #"sz:data_type"           : np.dtype('uint16'),
        ###"sz:error_bound_mode_str" : "abs",
        ###"sz:abs_err_bound"        : 10, # max error
        "sz3:abs_error_bound"     : 10, # max error
        "sz3:metric"              : "size",
        #"pressio:nthreads"        : 4
    },
}

ds = DataSource(drp=drp_info, monitor=True)
thread_num = drp_info.worker_num

labels = { 'alias'      : f'{drp_info.det_name}_{drp_info.det_segment}', # Alias
           'detname'    : drp_info.det_name,
           'instrument' : drp_info.instrument,
           'partition'  : drp_info.partition,
           'worker_num' : drp_info.worker_num }

compT = Gauge('psana_compress_time', 'time spent (s) in compressor()', labels.keys())
compT.labels(*labels.values())
calibT = Gauge('psana_calibration_time', 'time spent (s) in det.raw.calib()', labels.keys())
calibT.labels(*labels.values())

#print(f'*** [Thread {thread_num}] ds._configs:', ds._configs.keys())

cfgAlg = AlgDef("config", 0, 0, 1)
fexAlg = AlgDef("fex", 0, 0, 1)
detDef = DetectorDef(drp_info.det_name, drp_info.det_type, drp_info.det_id)
cfgDef = {
    "compressor_json" : (str,      1),
}
fexDef = {
    "fex"             : (np.uint8, 1), # Why not float32?
}
nodeId = None
namesId = None

cfg = ds.add_detector(detDef, cfgAlg, cfgDef, nodeId, namesId, drp_info.det_segment)
det = ds.add_detector(detDef, fexAlg, fexDef, nodeId, namesId, drp_info.det_segment)

cfg.config.compressor_json = json.dumps(lpjson)

ds.add_data(cfg.config)

# configure
compressor = PressioCompressor.from_config(lpjson)
#print(compressor.get_config())

for myrun in ds.runs():
    epixhr = myrun.Detector('epixhr_emu')
    for nevt,evt in enumerate(myrun.events()):
        with calibT.labels(*labels.values()).time():
            cal = epixhr.raw.calib(evt)
        with compT.labels(*labels.values()).time():
            det.fex.fex = compressor.encode(cal)
        ds.add_data(det.fex)
        if nevt%1000!=0: ds.remove_data('epixhr_emu','raw')

REGISTRY.unregister(calibT)
REGISTRY.unregister(compT)
