"""Reuse accepted owner-scoped Weka cache and physical NIC helpers."""
import importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
NOTES=HERE.parent
sys.path.insert(0,str(NOTES/'jf_raw_cache_20261109'))
import common as raw
from network_watch import physical_counters,counter_delta
spec=importlib.util.spec_from_file_location('cold_ffb_helpers',NOTES/'jf_cold_ffb_20261109/run.py')
ffb=importlib.util.module_from_spec(spec);spec.loader.exec_module(ffb)
require=raw.require;sha=raw.sha;cache=ffb.cache
from fixture_check import check_10k,ROOT
EVENT_BYTES=3358200
BULK=524288
BATCH=100
STREAMS=2
BUDGET=6
WORKERS=8

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')

def affinity():
    cores=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        t=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key=((t/'physical_package_id').read_text(),(t/'core_id').read_text())
        if key not in seen:cores.append(cpu);seen.add(key)
    require(len(cores)>=24,'Need 24 physical cores')
    cores=cores[:24];os.sched_setaffinity(0,cores);return cores

def fixture():
    manifest,receipt=check_10k(ROOT,sha)
    streams=[s for s in manifest['streams'] if s['panels']]
    stamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12]
    require([s['stream'] for s in streams]==[5,6,7,8,9],'JF stream set')
    require(sorted(p for s in streams for p in s['panels'])==list(range(32)),'Panel coverage')
    return manifest,streams,stamps

def histogram(events):return {'524724':events,'629664':events,'734604':3*events}

def source_hashes():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))+list(HERE.glob('*.sh'))
    files += [raw.HERE/'common.py',raw.HERE/'reference.json',raw.HERE/'runtime.sh',NOTES/'jf_cold_ffb_20261109/run.py',
        NOTES/'mfx_kvikio_read_20261108/plan_reads.py',NOTES/'mfx_kvikio_read_20261108/network_watch.py',
        NOTES/'mfx_expand10_performance_20261106/fixture_check.py',NOTES/'mfx_expand10_stage2_20261105/adapter.py',
        NOTES.parent/'psana/psana/gpu/examples/jungfrau_calibration.py',NOTES/'mfx_expand10_10k_20261106/files.json']
    return {str(p):sha(p) for p in files}
