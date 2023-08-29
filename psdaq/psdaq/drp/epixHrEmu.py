from psana import DataSource
from psana.dgramedit import AlgDef, DetectorDef, DataType
import psana.psexp.TransitionId
import sys
import numpy as np
from libpressio import PressioCompressor
import json
import time
from prometheus_client import Counter,Gauge

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

alias      = f'{drp_info.det_name}_{drp_info.det_segment}'
detname    = drp_info.det_name
instrument = 'tst'
partition  = 3

compT = Counter('psana_compress_time', 'time spent (s) in compressor()',
                ['alias','detname','instrument','partition','unit'])
compT.labels(alias, detname, instrument, partition, 'seconds')
ds.prom_man.register(compT)
calibT = Counter('psana_calibration_time', 'time spent (s) in det.raw.calib()',
                ['alias','detname','instrument','partition','unit'])
calibT.labels(alias, detname, instrument, partition, 'seconds')
ds.prom_man.register(calibT)
triggers = Counter('psana_trigger_count', 'count of the number of triggers',
                ['alias','detname','instrument','partition','unit'])
triggers.labels(alias, detname, instrument, partition, 'seconds')
ds.prom_man.register(triggers)

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
        triggers.labels(alias, detname, instrument, partition, 'counts').inc()
        st = time.time()
        cal = epixhr.raw.calib(evt)
        en = time.time()
        calibT.labels(alias, detname, instrument, partition, 'seconds').inc(en-st)
        #print(f'*** cal is a {type(cal)} of len {len(cal)}, dtype {cal.dtype}, shape {cal.shape}')
        #print(f'*** cal {cal}')
        st = time.time()
        det.fex.fex = compressor.encode(cal)
        en = time.time()
        compT.labels(alias, detname, instrument, partition, 'seconds').inc(en-st)
        #print(f'*** det.fex.fex is a {type(det.fex.fex)} of len {len(det.fex.fex)}, dtype {det.fex.fex.dtype}, shape {det.fex.fex.shape}, ndim {det.fex.fex.ndim}, size {det.fex.fex.size}')
        ds.add_data(det.fex)
        if nevt%1000!=0: ds.remove_data('epixhr_emu','raw')
