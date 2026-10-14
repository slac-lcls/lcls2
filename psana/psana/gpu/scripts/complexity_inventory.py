"""Reproduce the dated CPU/GPU source inventory from immutable Git revisions."""
import argparse
import ast
import io
import json
from pathlib import PurePosixPath
import subprocess
import tokenize

PREFIX='psana/psana/gpu/'
GROUPS={
    'coordination_reads':['gpu_events.py','gpu_stream.py','gpu_kvikio_read.py','gpu_stream_read_plan.py',
        'gpu_read_plan.py','gpu_group_schedule.py','gpu_file_epochs.py','gpu_admission.py','gpu_input_group.py'],
    'fields_lifetime':['gpu_input.py','context.py','gpu_input_window.py'],
    'quota_allocation':['gpu_budget.py','gpu_allocation.py'],
    'parser_config':['gpudgram/config.py','gpudgram/batch.py','gpudgram/parser.py'],
    'detector_calibration':['gpu_detector.py','gpu_calib.py','cuda/fused_calib.cuh'],
    'task_publication':['gpu_task.py','gpu_task_batch.py','gpu_producer.py','gpu_d2h.py'],
    'descriptor_abi':['gpu_batch.py'],
    'gpu_mpi':['gpu_mpi.py'],
    'exports':['__init__.py','gpudgram/__init__.py'],
}
EXCLUDED={'gpu_mpi_benchmark.py','gpu_mpi_perf_compare.py','gpu_performance_benchmark.py'}
CPU_GROUPS={
    'event_read':['psana/psana/psexp/events.py','psana/psana/psexp/event_manager.py'],
    'native_parser':['psana/src/dgram.cc','psana/src/container.cc'],
    'detector_calibration':['psana/psana/'+p for p in ('detector/jungfrau.py','detector/UtilsJungfrau.py',
        'detector/areadetector.py','detector/detector_impl.py','pycalgos/utilsdetector.py',
        'pycalgos/utilsdetector_ext.pyx','pycalgos/UtilsDetector.cc','pycalgos/UtilsDetector.hh')],
    'shared_framework':['psana/psana/'+p for p in ('datasource.py','dgrammanager.py','event.py','eventbuilder.pyx',
        'dgramlite.pyx','smdreader.pyx','parallelreader.pyx','psexp/ds_base.py','psexp/mpi_ds.py','psexp/run.py',
        'psexp/node.py','psexp/smdreader_manager.py','psexp/eventbuilder_manager.py','psexp/packet_footer.py',
        'psexp/step.py','psexp/run_ctx.py','psexp/calib_xtc.py','psexp/mpi_shmem.py')],
    'shared_detector':['psana/psana/detector/'+p for p in ('calibconstants.py','mask_algos.py',
        'shared_calibc_cache.py','shared_geo_cache.py')],
}


def git(*args):return subprocess.check_output(['git',*args]).decode()


def controls(node):
    return sum(isinstance(n,(ast.If,ast.For,ast.AsyncFor,ast.While,ast.IfExp,ast.ExceptHandler))+
               (1+len(n.ifs) if isinstance(n,ast.comprehension) else 0) for n in ast.walk(node))


def metrics(text,python):
    result=dict(physical_loc=text.count('\n'))
    if not python:return result
    tree=ast.parse(text);ignored=set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            body=node.body
            if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                ignored.update(range(body[0].lineno,body[0].end_lineno+1))
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type==tokenize.COMMENT and not t.line[:t.start[1]].strip():ignored.add(t.start[0])
    result.update(source_lines=sum(bool(line.strip()) and n not in ignored for n,line in enumerate(text.splitlines(),1)),
        function_definitions=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        control_sites=controls(tree))
    return result


def inventory(revision):
    revision=git('rev-parse',revision).strip()
    paths=git('ls-tree','-r','--name-only',revision,PREFIX).splitlines()
    runtime={p[len(PREFIX):] for p in paths if PurePosixPath(p).suffix in ('.py','.cu','.cuh') and
             not p[len(PREFIX):].startswith(('scripts/','examples/','docs/','tests/')) and
             p[len(PREFIX):] not in EXCLUDED}
    assigned=[p for files in GROUPS.values() for p in files if p in runtime]
    assert len(assigned)==len(set(assigned)) and set(assigned)==runtime, runtime-set(assigned)
    def measure(path):return metrics(git('show',revision+':'+path),path.endswith('.py'))
    files={p:measure(PREFIX+p) for p in sorted(runtime)}
    groups={g:dict(files=[p for p in paths if p in runtime],
                  physical_loc=sum(files[p]['physical_loc'] for p in paths if p in runtime))
            for g,paths in GROUPS.items()}
    cpu={g:dict(files={p:measure(p) for p in paths}) for g,paths in CPU_GROUPS.items()}
    for group in cpu.values():group['physical_loc']=sum(v['physical_loc'] for v in group['files'].values())
    examples={p[len(PREFIX):]:measure(p) for p in paths if p.startswith(PREFIX+'examples/') and p.endswith('.py')}
    return dict(revision=revision,runtime_files=len(files),runtime_loc=sum(v['physical_loc'] for v in files.values()),
        groups=groups,files=files,cpu_scopes=cpu,user_examples=examples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',default='31d68655a6f2fb711e2489f89d3149172b46dad3')
    p.add_argument('--after',default='HEAD')
    a=p.parse_args()
    print(json.dumps(dict(before=inventory(a.before),after=inventory(a.after)),indent=2,sort_keys=True))


if __name__=='__main__':main()
