from psana import DataSource
from psana.dgramedit import AlgDef, DetectorDef, DataType
import psana.psexp.TransitionId
import sys
import numpy as np
from libpressio import PressioCompressor
import json

# Define compressor configuration:
lpjson = {
    "compressor_id": "sz", #the compression algo.
    "compressor_config": {
        #"sz:data_type"           : lp.pressio_uint16_dtype,
        #"sz:data_type"           : np.dtype('uint16'),
        "sz:error_bound_mode_str": "abs",
        "sz:abs_err_bound"       : 10, # max error
        "sz:metric"              : "size"
    },
}

ds = DataSource(drp=drp_info)
thread_num = drp_info.worker_num

#print(f'*** [Thread {thread_num}] ds._configs:', ds._configs.keys())

cfgAlg = AlgDef("config", 0, 0, 1)
fexAlg = AlgDef("fex", 0, 0, 1)
detDef = DetectorDef(drp_info.det_name, drp_info.det_type, drp_info.det_id)
cfgDef = {
    "compressor_json" : (str,      1),
}
fexDef = {
    "fex"             : (np.uint8, 1),
}
nodeId = None
namesId = None

cfg = ds.add_detector(detDef, cfgAlg, cfgDef, nodeId, namesId, drp_info.det_segment)
det = ds.add_detector(detDef, fexAlg, fexDef, nodeId, namesId, drp_info.det_segment)

cfg.config.compressor_json = json.dumps(lpjson)

ds.adddata(cfg.config)

# configure
compressor = PressioCompressor.from_config(lpjson)

for myrun in ds.runs():
    epixhr = myrun.Detector('epixhr_emu')
    for nevt,evt in enumerate(myrun.events()):
        raw = epixhr.raw.calib(evt)
        #print(f'*** raw is a {type(raw)} of len {len(raw)}')
        det.fex.fex = compressor.encode(raw)
        #print(f'*** det.fex.fex is a {type(det.fex.fex)} of len {len(det.fex.fex)}, dtype {det.fex.fex.dtype}, shape {det.fex.fex.shape}, ndim {det.fex.fex.ndim}, size {det.fex.fex.size}')
        ds.adddata(det.fex)
        if nevt%1000!=0: ds.removedata('epixhr_emu','raw')

print(f"*** [Thread {thread_num}] runs loop exited")
