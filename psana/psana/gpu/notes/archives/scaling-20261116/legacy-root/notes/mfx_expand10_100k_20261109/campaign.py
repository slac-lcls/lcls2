"""Authorized CPU-only 100k repetition campaign, fail-closed validation receipts."""
import json,os,socket,sys,time
from pathlib import Path
import psana
import repeat
import validate_repeat as verify
f=repeat.fixture
NOTES=Path(__file__).resolve().parent
PARENT=Path('/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx')
SOURCE=PARENT/'sample10_10k'
ROOT=PARENT/'sample10_100k'
EVIDENCE=PARENT/'sample10_100k_evidence_20261109'
TINY=Path('/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/tiny')
RECEIPTS={SOURCE:'c45d191bc92bab1036caf24a4d92039aebe2036d00fd11cbd562da0f84551bcf',TINY:'0d9a6aabed7885bebaf3fd6432fecaf073be539d9decf453c4f2e3ceed6b8b1b'}


def check_receipts():
    for path,digest in RECEIPTS.items():
        f.require(f.file_sha(path/'generation.json')==digest,'Accepted source receipt changed: '+str(path))


def preflight():
    check_receipts()
    manifest,streams=repeat.index_source(SOURCE)
    f.require(manifest['event_count']==10000,'Expected accepted 10k source')
    plan,period=repeat.timeline(streams[0]['rows'],100000)
    sizes=[dict(stream=i,bytes={k:sum(s['rows'][p['source_index']][k+'_size'] for p in plan) for k in ('bd','smd')}) for i,s in enumerate(streams)]
    total=sum(sum(s['bytes'].values()) for s in sizes)
    stat=os.statvfs(PARENT);free=stat.f_bavail*stat.f_frsize
    f.require(os.access(PARENT,os.W_OK) and free>total+5*2**30,'Insufficient writable scratch/headroom')
    f.require(not ROOT.exists() and not EVIDENCE.exists(),'Refusing existing output/evidence')
    f.require(manifest['calibration_manifest_sha256']==f.file_sha(f.SNAPSHOT/'manifest.json'),'Frozen calibration manifest changed')
    files=list(NOTES.glob('*.py'))+[NOTES/'run.sbatch',Path(f.__file__),f.HELPERS/'xtc_fixture.py',NOTES.parent/'mfx_expand10_stage2_20261105/runtime.sh']
    return dict(status='pass',root=str(ROOT),source=str(SOURCE),event_count=100000,source_events=10000,
        records_per_stream=len(plan),period_ns=period,expected_file_bytes=total,streams=sizes,
        source_generation_sha256=RECEIPTS[SOURCE],tiny_generation_sha256=RECEIPTS[TINY],
        calibration_manifest_sha256=manifest['calibration_manifest_sha256'],
        storage=dict(available_bytes=free,resolved_parent=str(PARENT.resolve()),quota='Filesystem availability; project quota not independently established'),
        runtime=dict(host=socket.gethostname(),python=sys.version,executable=sys.executable,psana=psana.__file__,dgram=f.dgram.__file__),
        code_sha256={str(p):f.file_sha(p) for p in files},
        limitations='Payload events repeat ten times; outer timestamps unique. No new calibration deployment or GPU performance test.')


def main():
    f.require(sys.argv[1:] in (['preflight'],['run']),'Choose preflight or run')
    f.require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use runtime.sh')
    before=preflight()
    if sys.argv[1]=='preflight':print(json.dumps(before,indent=2));return
    os.umask(0o007);EVIDENCE.mkdir(mode=0o770)
    start=time.monotonic();status=dict(phase='preflight',accepted=False,validation_pass=False,review='pending',root=str(ROOT),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'))
    def checkpoint(phase):
        status.update(phase=phase,elapsed_seconds=time.monotonic()-start)
        tmp=EVIDENCE/'status.pending.json'
        with tmp.open('w') as h:
            json.dump(status,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
        os.replace(tmp,EVIDENCE/'status.json');print('PHASE',phase,flush=True)
    try:
        f.save_json(EVIDENCE/'preflight.json',before);checkpoint('pilot')
        repeat.make(TINY,EVIDENCE/'pilot',23)
        f.save_json(EVIDENCE/'pilot-verification.json',verify.verify(TINY,EVIDENCE/'pilot'))
        check_receipts();checkpoint('generating')
        generated=repeat.make(SOURCE,ROOT,100000)
        f.require(generated['total_file_bytes']==before['expected_file_bytes'],'Total projection mismatch')
        for actual,predicted in zip(generated['streams'],before['streams']):
            f.require({k:v['bytes'] for k,v in actual['files'].items()}==predicted['bytes'],'Per-stream size projection mismatch')
        checkpoint('verifying')
        report=verify.verify(SOURCE,ROOT);check_receipts()
        f.save_json(EVIDENCE/'verification.json',report)
        f.save_json(EVIDENCE/'files.json',dict(root=str(ROOT),event_count=100000,total_file_bytes=report['total_file_bytes'],generation_sha256=report['generation_sha256'],files=[dict(stream=s['stream'],kind=k,**v) for s in generated['streams'] for k,v in s['files'].items()]))
        status['validation_pass']=True;checkpoint('validated_review_pending')
    except BaseException as error:
        status['error']=repr(error);checkpoint('failed');raise

if __name__=='__main__':main()
