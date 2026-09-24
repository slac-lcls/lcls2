"""One-allocation A+H2D / E-off / E-on feespec benchmark on private Weka FFB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import sys
import time

from common import records, sha, tier

HERE = Path(__file__).resolve().parent
VARIANTS = ('A', 'E-off', 'E-on')


def matrix():
    for repetition, variants in ((1,VARIANTS),(2,VARIANTS[::-1])):
        for cache in (('cold','warm') if repetition == 1 else ('warm','cold')):
            for variant in variants:
                yield variant, cache, repetition


def execute(command, env, logfile, timeout=600):
    print('BEGIN', logfile.stem, time.strftime('%FT%T'), flush=True)
    with logfile.open('x') as output:
        proc = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            status = proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    print('END', logfile.stem, status, time.strftime('%FT%T'), flush=True)
    text = logfile.read_text()
    if status:
        print(text[-7000:], flush=True)
        raise RuntimeError(f'{logfile} exited {status}')
    return text


def cpu_reference(stage, events=10000, manifest=None):
    """Decode all selected arrays before timing; save only small hashes/summaries."""
    import numpy as np
    from psana import dgram
    p = stage/'mfx101210926-r0387-s000-c000.xtc2'
    fd = os.open(p,os.O_RDONLY)
    timestamps, sums, sizes = [], [], []
    digest = hashlib.sha256()
    try:
        cfg = dgram.Dgram(file_descriptor=fd)
        while len(timestamps)<events:
            e = dgram.Dgram(config=cfg)
            if e.service()!=12:
                continue
            values = e.feespec[0].raw.hproj
            assert values.shape == (2048,) and values.dtype == np.int32
            timestamps.append(int(e.timestamp()))
            sizes.append(int(e._size))
            sums.append(values.sum(dtype=np.int64))
            digest.update(values.tobytes())
    finally:
        os.close(fd)
    if manifest is None:
        manifest = json.loads((stage/'manifest.json').read_text())
    assert manifest['events'] == events
    timestamp_hash=hashlib.sha256(struct.pack(f'<{events}Q',*timestamps)).hexdigest()
    assert timestamp_hash == manifest['timestamp_sha256']
    feespec = next(r for r in manifest['streams'] if r['name'] == p.name)
    assert sum(sizes) == feespec['payload_bytes']
    return dict(events=events, timestamp_sha256=timestamp_hash,
        payload_bytes=manifest['payload_bytes'], feespec_dgram_bytes=sum(sizes),
        arrays_sha256=digest.hexdigest(),
        sums_sha256=hashlib.sha256(np.asarray(sums,dtype=np.int64).tobytes()).hexdigest(),
        min_dgram_bytes=min(sizes), max_dgram_bytes=max(sizes), array_bytes=8192)


def verify(builds):
    for entry in builds.values():
        for name,digest in entry['hashes'].items():
            assert sha(Path(entry['prefix'])/name) == digest, name


def summarize(root):
    results=json.loads((root/'results.json').read_text())
    diagnostics=json.loads((root/'diagnostics.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    include_jf=provenance.get('include_jf', False)
    assert [(r['variant'],r['cache'],r['repetition']) for r in results]==list(matrix())
    assert len(diagnostics)==3 and {r['variant'] for r in diagnostics}==set(VARIANTS)
    for r in results+diagnostics:
        assert r['job_id']==provenance['job_id'] and r['gpu_uuid']==provenance['gpu_uuid']
        assert r['events']==10000
        assert r['timestamp_sha256']==provenance['reference']['timestamp_sha256']
        assert r['sums_sha256']==provenance['reference']['sums_sha256']
        assert all(p['affinity']==provenance['affinity'] for p in r['ranks'])
        if r['cache']=='warm':
            assert min(r['cache_before']['resident_fraction'],r['cache_after']['resident_fraction'])>=.99
        else:
            assert r['cache_before']['resident_fraction']<=.01
            assert r['physical_rx_bytes']>=r['payload_bytes']*.98
    groups=[]
    title='JF + feespec' if include_jf else 'Feespec'
    lines=[f'# {title} on Weka FFB: A+H2D / E bulk off / E bulk on','',
        f"Job {provenance['job_id']} on {provenance['node']}; 12 clean samples and three feespec full-array checks passed.",'',
        '| Cache | Variant | R1 / R2 events/s | Rate from median time | Median seconds | Input GB/s |',
        '|---|---|---:|---:|---:|---:|']
    for cache in ('cold','warm'):
        for variant in VARIANTS:
            rows=sorted([r for r in results if (r['variant'],r['cache'])==(variant,cache)],key=lambda r:r['repetition'])
            assert len(rows)==2
            seconds=statistics.median(r['loop_s'] for r in rows)
            row=dict(cache=cache,variant=variant,seconds=seconds,events_per_s=10000/seconds,
                     input_gbps=rows[0]['payload_bytes']/seconds/1e9,
                     round_rates=[r['events_per_s'] for r in rows])
            groups.append(row)
            lines.append(f"| {cache} | {variant} | {row['round_rates'][0]:.1f} / {row['round_rates'][1]:.1f} | {row['events_per_s']:.1f} | {seconds:.4f} | {row['input_gbps']:.4f} |")
    lines+=['','## Separate validation runs','',
        '| Variant | KvikIO API reads | CPU BigData reads | Read bytes via KvikIO |',
        '|---|---:|---:|---:|']
    for r in diagnostics:
        c=r['ranks'][2]['counts']
        lines.append(f"| {r['variant']} | {c['requests']} | {c['cpu_bd_reads']} | {c['bytes']} |")
    lines+=['',
        ('JF + feespec: 10000 events, batch 100, depth 1, 8 GiB GPU budget. A uses the JF GPU pipeline plus CPU feespec extraction and per-event H2D. E parses both detectors; bulk-on is global. JF raw/calibrated samples passed the frozen CPU reference.' if include_jf else
         'Feespec only: 10000 events, batch 100, depth 1, 8 GiB GPU budget for E; A uses ordinary CPU events with per-event H2D. No Jungfrau is included.'),
        'Every variant performs the same per-event GPU int64 sum. Final device completion is timed; result checksum retrieval is outside timing.',
        'E uses the public on_gpu_view field API, including its locator-metadata D2H and consumer lease costs. Clean runs do not copy full arrays to the CPU.',
        'Validation runs compare all 10000 full arrays and count reads; these runs are excluded from throughput.',
        'A CPU reads may already coalesce. KvikIO API counts are not POSIX syscall or physical storage request counts.',
        'Cold means node-page-cache-cold Weka FFB, verified SSD-only tier. Weka server caches are not flushed.',
        'Physical NIC counters include filesystem overhead and any node background traffic; input GB/s counts useful full-dgram bytes.',
        'Two rounds are descriptive. Historical A and E are complete different builds; their native binaries are preserved.',
        'A shared-stream routing exception applies only to E inside the benchmark process; all original dgram bytes are retained.', '']
    summary=dict(job_id=provenance['job_id'],node=provenance['node'],audit_pass=True,groups=groups)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'summary.md').write_text('\n'.join(lines))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-jf',action='store_true')
    parser.add_argument('--constants')
    parser.add_argument('--jf-cpu-log')
    parser.add_argument('--staged-directory',help='Reuse an existing private FFB prefix after manifest/tier checks')
    a=parser.parse_args()
    if a.include_jf:
        assert a.constants and a.jf_cpu_log
    builds=json.loads((HERE/'builds.json').read_text())
    verify(builds)
    job=os.environ['SLURM_JOB_ID']
    root=HERE/f'job-{job}'
    root.mkdir(exist_ok=False)
    affinity=sorted(os.sched_getaffinity(0))
    os.environ['BENCH_CPU_AFFINITY']=','.join(map(str,affinity))
    import cupy as cp
    bus=cp.cuda.runtime.deviceGetPCIBusId(0)
    bus=bus.decode() if isinstance(bus,bytes) else bus
    uuid=subprocess.check_output(['nvidia-smi','-i',bus,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    label='jf-feespec-bulk' if a.include_jf else 'feespec-bulk'
    stage=Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin')/f'{label}-{job}'/'xtc'
    if a.staged_directory:
        stage=Path(a.staged_directory).resolve()
        assert Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin') in stage.parents
        manifest=json.loads((stage/'manifest.json').read_text())
        assert (manifest['exp'],manifest['run'],manifest['events'])==('mfx101210926',387,10000)
        streams=[0,5,6,7,8,9] if a.include_jf else [0]
        expected={f'mfx101210926-r0387-s{s:03d}-c000.xtc2' for s in streams}
        assert {r['name'] for r in manifest['streams']}==expected
        assert {p.name for p in stage.glob('*.xtc2')}==expected
        for record in manifest['streams']:
            assert (stage/record['name']).stat().st_size==record['stage_bytes']
    provenance=dict(job_id=job,node=os.uname().nodename,gpu_uuid=uuid,affinity=affinity,
                    include_jf=a.include_jf,
                    builds=builds,stage=str(stage),scripts={p.name:sha(p) for p in HERE.glob('*.py')})
    if a.include_jf:
        provenance['calibration']={str(p):sha(p) for p in (a.constants,a.jf_cpu_log)}
        jf_reference=records(Path(a.jf_cpu_log).read_text(),'CPU_CHECK ')[0]
        assert len(jf_reference)==3
        (root/'jf-reference.json').write_text(json.dumps(jf_reference,indent=2)+'\n')
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if a.staged_directory:
        print('REUSE_STAGE '+str(stage),flush=True)
    else:
        execute([sys.executable,str(HERE/'stage.py'),'--source','/sdf/data/lcls/ds/mfx/mfx101210926/xtc',
            '--target',str(stage),'--exp','mfx101210926','--run','387','--streams',
            *(['0','5','6','7','8','9'] if a.include_jf else ['0']),
            '--events','10000'],os.environ.copy(),root/'stage.log',timeout=3600)
    files=list(stage.glob('*.xtc2'))+list((stage/'smalldata').glob('*.xtc2'))
    provenance['mount']=subprocess.check_output(['findmnt','-T',str(stage),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],text=True)
    assert 'wekafs' in provenance['mount']
    provenance['tier_before']=tier(files)
    reference=cpu_reference(stage)
    provenance['reference']=reference
    reference_file=root/'reference.json'
    reference_file.write_text(json.dumps(reference,indent=2)+'\n')
    (root/'stage.json').write_text((stage/'manifest.json').read_text())
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results,diagnostics=[],[]
    for variant,mode,repetition,diagnostic in ([(v,'warm',0,True) for v in VARIANTS]+
            [(v,c,r,False) for v,c,r in matrix()]):
        tag=f'{variant}-{mode}-r{repetition}'+('-validation' if diagnostic else '')
        env=os.environ.copy()
        prefix=Path(builds['A' if variant=='A' else 'E']['prefix'])
        env['PYTHONPATH']=str(prefix/'lib/python3.9/site-packages')
        env['LD_LIBRARY_PATH']=str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
        command=['mpirun','-n','3','--oversubscribe','--bind-to','none',sys.executable,'-u',str(HERE/'bench.py'),
                 '--directory',str(stage),'--reference',str(reference_file),'--variant',variant,'--cache',mode]
        if diagnostic:command+=['--diagnostic']
        if a.include_jf:
            command+=['--include-jf','--constants',a.constants,
                      '--jf-reference',str(root/'jf-reference.json')]
        with (root/(tag+'-gpu.csv')).open('x') as output:
            monitor=subprocess.Popen(['nvidia-smi','-i',uuid,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm','--format=csv,noheader,nounits','-lms','250'],stdout=output,stderr=subprocess.STDOUT)
            try:text=execute(command,env,root/(tag+'.log'),timeout=2400 if a.include_jf else 600)
            finally:
                monitor.terminate()
                monitor.wait(timeout=15)
        rows=records(text,'RESULT ')
        assert len(rows)==1
        r=rows[0]
        runtime=records(text,'RUNTIME ')
        assert len(runtime)==1 and runtime[0]['psana'].startswith(str(prefix)+'/')
        r.update(job_id=job,gpu_uuid=uuid,repetition=repetition,log=str(root/(tag+'.log')))
        (diagnostics if diagnostic else results).append(r)
        for name,value in (('results',results),('diagnostics',diagnostics)):
            temp=root/(name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(root/(name+'.json'))
        print('ACCEPTED',tag,round(r['events_per_s'],2),flush=True)
    provenance['tier_after']=tier(files)
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    verify(builds)
    for name,digest in provenance.get('calibration',{}).items():assert sha(name)==digest,name
    for name,digest in provenance['scripts'].items():assert sha(HERE/name)==digest,name
    summary=summarize(root)
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
