"""Stage the supplied CFS run4 into owned Lustre; bounded compute allocation only."""
import concurrent.futures,hashlib,json,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
from data_contract import require,identity,sha,published_files


def copy_one(source,destination,expected,stop=None):
    before=identity(source)
    require(before['st_size']==expected['bytes'],'Source length')
    digest=hashlib.sha256();total=0
    with source.open('rb') as src,destination.open('xb') as dst:
        for chunk in iter(lambda:src.read(8<<20),b''):
            if stop is not None and stop.is_set():raise InterruptedError('Staging stopped')
            dst.write(chunk);digest.update(chunk);total+=len(chunk)
        dst.flush();os.fsync(dst.fileno())
    require(identity(source)==before,'Source changed during copy')
    require(total==expected['bytes'] and digest.hexdigest()==expected['sha256'],'Published transfer hash')
    require(destination.stat().st_uid==os.geteuid(),'Owned scratch copy')
    return dict(source=str(source),destination=str(destination),source_identity=before,
                bytes=total,sha256=digest.hexdigest(),destination_identity=identity(destination))


def stage(plan):
    require(os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid'),'Compute allocation required')
    require(int(os.environ['SLURM_JOB_NUM_NODES'])==1,'One-node qualification allocation')
    for path,digest in plan['pins'].items():require(sha(path)==digest,'Changed qualification artifact: '+path)
    root=Path(plan['attempt']);source=Path(plan['source']);target=root/'xtc'
    require(source==Path('/global/cfs/cdirs/lcls/projecta/lcls/psdm/prj/prjdatpsana/xtc'),'Supplied transfer only')
    require(root.parent==Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-input-20261124'),'Dedicated attempt root')
    require(set(plan['files'])==set(published_files()),'Only twenty published run4 files')
    require(not target.exists(),'No overwrite or automatic resume')
    target.mkdir();subprocess.run(['lfs','setstripe','-c','2','-S','1048576',str(target)],check=True)
    (target/'smalldata').mkdir()
    receipt=dict(status='running',job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),started=time.time(),files=[])
    def save():
        receipt['seconds']=time.time()-receipt['started']
        (root/'staging.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    stop=threading.Event()
    def interrupted(signum,frame):
        stop.set();receipt.update(status='failed',error='SIGTERM/timeout');save()
        raise TimeoutError('SIGTERM/timeout')
    old_handler=signal.signal(signal.SIGTERM,interrupted)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(copy_one,source/name,target/name,record,stop):name for name,record in plan['files'].items()}
            try:
                for f in concurrent.futures.as_completed(futures):
                    row=f.result()
                    layout={key:int(subprocess.check_output(['lfs','getstripe',flag,row['destination']],universal_newlines=True).strip()) for key,flag in [('stripe_count','-c'),('stripe_size','-S')]}
                    require(layout==dict(stripe_count=2,stripe_size=1048576),'Actual staged layout')
                    row['layout']=layout
                    receipt['files'].append(row);save();print('Copied',futures[f],flush=True)
            except BaseException:
                stop.set()
                for f in futures:f.cancel()
                raise
        require(len(receipt['files'])==20,'Complete staged file set')
        receipt['status']='pass';save()
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));save();raise
    finally:signal.signal(signal.SIGTERM,old_handler)


if __name__=='__main__':stage(json.loads(Path(sys.argv[1]).read_text()))
