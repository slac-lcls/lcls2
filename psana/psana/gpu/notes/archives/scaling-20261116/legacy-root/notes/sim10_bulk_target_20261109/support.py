"""Reuse accepted input identity/cache helpers without their standalone runner."""
import importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
NOTES=HERE.parent
sys.path.insert(0,str(NOTES/'jf_raw_cache_20261109'))
import common as raw
from plan_reads import load_fixture
from network_watch import physical_counters,counter_delta
spec=importlib.util.spec_from_file_location('cold_ffb_helpers',NOTES/'jf_cold_ffb_20261109/run.py')
ffb=importlib.util.module_from_spec(spec);spec.loader.exec_module(ffb)
require=raw.require;sha=raw.sha;cache=ffb.cache

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def affinity(rank):
    cores=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        t=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key=((t/'physical_package_id').read_text(),(t/'core_id').read_text())
        if key not in seen:cores.append(cpu);seen.add(key)
    require(len(cores)>=6,'Six physical cores required')
    selected=cores
    os.sched_setaffinity(0,selected);return selected

def source_hashes():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))
    files += [HERE/'expected-reference.json',raw.HERE/'common.py',raw.HERE/'reference.json',raw.HERE/'pixels.json',raw.HERE/'expected-sha256.json',
        NOTES/'jf_cold_ffb_20261109/run.py',NOTES/'mfx_pipeline_scaling_20261109/instrument.py',NOTES/'jf_raw_cache_20261109/runtime.sh',NOTES/'jf_framework_cold_20261109/sim10_identity_check.json',
        NOTES/'mfx_kvikio_read_20261108/plan_reads.py',NOTES/'mfx_kvikio_read_20261108/network_watch.py',
        NOTES/'mfx_expand10_performance_20261106/fixture_check.py',NOTES/'mfx_expand10_stage2_20261105/adapter.py']
    return {str(p):sha(p) for p in files}
