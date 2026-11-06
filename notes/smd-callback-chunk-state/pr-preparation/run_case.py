"""Instrument only the exact, protected stage-1 command; run from a fresh cwd."""
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

CLI = ['-k', 'exp=mfx100848724,run=49', '-d', 'jungfrau', '-o', './work1',
       '--nrecs', '50', '--nrecs1', '50', '--stepnum', '2']
assert sys.argv[1:] == CLI
assert not Path('work1').exists(), 'Refuse to overwrite existing calibration output'
assert os.environ['PS_PARALLEL'] == 'none'
sys.argv = ['jungfrau_dark_proc', *CLI]
report = dict(argv=CLI, cwd=os.getcwd(), initial_smd_n_events=os.environ.get('PS_SMD_N_EVENTS'),
              chunks=[], callback_calls=0, callback_timestamps=[], raw_timestamps=[],
              name_lookups=[], blocked_mutations=[])

from psana.detector import UtilsJungfrauCalib as ujc
from psana.detector.areadetector import AreaDetectorRaw
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.pscalib.calib import MDBWebUtils as mdb
from psana.app.jungfrau_dark_proc import do_main

original_open = ujc.open_DataSource
original_build = SmdReaderManager.build_batch
original_raw = AreaDetectorRaw.raw
original_name = ujc.uc.detector_name_short


def observed_open(**kwargs):
    callback = kwargs['smd_callback']
    def observed_callback(run):
        report['callback_calls'] += 1
        for evt in callback(run):
            report['callback_timestamps'].append(int(evt.timestamp))
            yield evt
    kwargs['smd_callback'] = observed_callback
    ds, dskwargs = original_open(**kwargs)
    report['datasource_class'] = type(ds).__name__
    report['dskwargs'] = {k: v for k, v in dskwargs.items() if not callable(v)}
    report['smd_files'] = list(ds.dsparms.smd_files)
    return ds, dskwargs


def observed_build(self):
    result = original_build(self)
    if result and self.smdr.view_size:
        report['chunks'].append(int(self.smdr.view_size))
    return result


def observed_raw(self, evt, *args, **kwargs):
    raw = original_raw(self, evt, *args, **kwargs)
    if raw is not None:
        report['raw_timestamps'].append(int(evt.timestamp))
        report.setdefault('raw_shape', list(raw.shape))
    return raw


def readonly_name(*args, **kwargs):
    kwargs['add_shortname'] = False
    result = original_name(*args, **kwargs)
    report['name_lookups'].append(dict(add_shortname=False, result=result))
    return result


def blocked_mutation(*args, **kwargs):
    report['blocked_mutations'].append('calibration HTTP write attempted')
    raise RuntimeError('Calibration database mutations disabled for this experiment')


try:
    with patch.object(ujc, 'open_DataSource', observed_open), \
         patch.object(SmdReaderManager, 'build_batch', observed_build), \
         patch.object(AreaDetectorRaw, 'raw', observed_raw), \
         patch.object(ujc.uc, 'detector_name_short', readonly_name), \
         patch.object(DataSourceBase, '_get_file_info_from_db', return_value={}), \
         patch.object(mdb, 'post', blocked_mutation), \
         patch.object(mdb, 'put', blocked_mutation), \
         patch.object(mdb, 'delete_cmd', blocked_mutation):
        do_main()
    report['completed'] = True
except BaseException as exc:
    report['completed'] = False
    report['exception'] = repr(exc)
    raise
finally:
    report['effective_smd_n_events'] = os.environ.get('PS_SMD_N_EVENTS')
    names = ['datasource', 'app.jungfrau_dark_proc', 'detector.UtilsJungfrauCalib',
             'detector.UtilsCalib', 'detector.RepoManager', 'psexp.run', 'psexp.step',
             'psexp.ds_base', 'psexp.serial_ds', 'psexp.smdreader_manager',
             'psexp.eventbuilder_manager', 'psexp.callback_batch',
             'psexp.callback_iterator', 'psexp.callback_stream',
             'smdreader', 'eventbuilder', 'dgram']
    report['sources'] = {}
    for name in names:
        module = sys.modules.get('psana.' + name)
        if module is not None:
            path = Path(module.__file__).resolve()
            report['sources'][name] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    Path('report.json').write_text(json.dumps(report, indent=2) + '\n')
