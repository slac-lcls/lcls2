"""Bounded traffic controls and diagnostic MPI receive-lifetime isolation."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import time

from native_contract import (artifact, cache_ranges, check_selected_files, cold_cache,
    cxi_delta, cxi_snapshot, digest, file_hash, nic_delta, nic_snapshot, read, require,
    runtime_evidence, write)

CONTROL_BYTES = 256 << 20


def check_control(rows, size=CONTROL_BYTES):
    require(len(rows) == 5 and [r['kind'] for r in rows] == ['idle','cold','warm','cold','warm'], 'control sequence differs')
    hashes = []
    for row in rows:
        delta = cxi_delta(row['cxi_before'], row['cxi_after'])
        row['cxi_delta'] = delta
        row['nic_delta'] = nic_delta(row['nic_before'], row['nic_after'])
        if row['kind'] == 'idle':
            require(delta['rx_bytes'] <= size*0.1, 'idle traffic too high for coverage control')
            continue
        require(row['bytes'] == size, 'control read length differs')
        hashes.append(row['sha256'])
        if row['kind'] == 'cold':
            require(len(row['cache']['observations']) == 2 and all(
                r['resident_bytes_before'] == 0 and r['measured_bytes'] == size
                for r in row['cache']['observations']), 'control was not client-cold')
            require(size*0.9 <= delta['rx_bytes'] <= size*2, 'cold RX outside diagnostic coverage bounds')
        else:
            require(row['cache']['resident_bytes_before'] == size and row['cache']['measured_bytes'] == size,
                    'warm control not fully resident')
            require(delta['rx_bytes'] <= size*0.1, 'warm RX too high for coverage control')
    require(len(set(hashes)) == 1, 'control data changed')
    return rows


def traffic_control(manifest, site, folder):
    folder.mkdir()
    control = copy.deepcopy(manifest)
    control['files'] = control['files'][:1]
    entry = control['files'][0]['bigdata']
    start, length = entry['ranges'][0]
    require(length >= CONTROL_BYTES, 'first private selected range too small')
    entry.update(ranges=[[start, CONTROL_BYTES]], range_bytes=CONTROL_BYTES)
    entry.pop('sha256')  # not the hash of the accepted 10k prefix
    rows = []
    for index, kind in enumerate(('idle','cold','warm','cold','warm')):
        cache = cold_cache(control) if kind == 'cold' else cache_ranges(control) if kind == 'warm' else None
        row = dict(kind=kind, cache=cache, nic_before=nic_snapshot(site['nic_interfaces']),
                   cxi_before=cxi_snapshot(site['nic_interfaces']))
        started = time.monotonic()
        if kind == 'idle':
            time.sleep(1)
        else:
            sha, done = hashlib.sha256(), 0
            with open(entry['path'], 'rb', buffering=0) as stream:
                stream.seek(start)
                while done < CONTROL_BYTES:
                    block = stream.read(min(8 << 20, CONTROL_BYTES-done))
                    require(block, 'short control read')
                    sha.update(block)
                    done += len(block)
            row.update(bytes=done, sha256=sha.hexdigest())
        row['read_seconds'] = time.monotonic()-started
        # Include delayed hardware telemetry and read-ahead, consistently in all controls.
        time.sleep(0.25)
        row.update(cxi_after=cxi_snapshot(site['nic_interfaces']), nic_after=nic_snapshot(site['nic_interfaces']))
        rows.append(row)
        write(folder/('%d-%s.json' % (index,kind)),row)
    checked = check_control(rows)
    cold_cache(control)
    write(folder/'complete.json',dict(status='pass',rows=checked, path=entry['path'],start=start,bytes=CONTROL_BYTES,
          scope='empirical cold/warm node RX coverage; not storage-only attribution or a link ceiling'))


def preflight(config, manifest, reference, runtime, source, folder, sources):
    from perlmutter_adapter import launch
    from run_smoke import artifact as pin, private_files
    folder.mkdir()
    site = read(config['artifacts']['native_site']['path'])
    site['step_timeout_seconds'] = 60  # tiny MPI controls only; full cells retain 300 s
    private_files(manifest)
    traffic_control(manifest,site,folder/'traffic')
    # These are separate diagnostic processes; the real six cells are unmodified.
    for mode in ('mpi-only','observe','cancel'):
        request = dict(profile_sha256=digest(config),sample_id='mpi-'+mode,case=dict(bds=1,gpus=0),
            contract=dict(environment=config['environment']),artifacts=dict(native_worker=pin(source/'accounting_probe.py')),
            site_manifest=site,runtime=runtime,sources=sources,mode=mode,inputs=manifest,reference=reference['events'][:20])
        path=folder/('mpi-'+mode+'.json');write(path,request)
        complete,scheduler=launch(request,path,folder/('mpi-'+mode),'mpi-diagnostic',site,runtime)
        text=(folder/('mpi-'+mode)/'native.log').read_text()
        warnings=text.count('Builtin communicator 44000000 has pending 1 references.')
        require(warnings == (1 if mode == 'observe' else 0), 'MPI warning isolation differs: '+mode)
        require(mode == 'mpi-only' or complete['rows'][0]['request']['pending'] is True, 'expected unmatched SMD0 termination receive')
        if mode == 'cancel':
            require(complete['rows'][0]['request']['cancelled'] is True and complete['rows'][0]['request']['null_after_wait'] is True,
                    'diagnostic receive cancellation did not finish')
        write(folder/('mpi-'+mode+'-pass.json'),dict(status='pass',warnings=warnings,scheduler=scheduler))
    private_files(manifest)
    write(folder/'complete.json',dict(status='pass',scope='traffic control and isolated MPI diagnosis; no installed source mutation'))


def probe(request_path, folder, stage):
    from site_probe import observed_placement
    request, folder=read(request_path),Path(folder)
    require(stage == 'mpi-diagnostic' and request['mode'] in ('mpi-only','observe','cancel'), 'diagnostic only')
    require(Path(__file__).resolve() == artifact(request,'native_worker'), 'unfrozen diagnostic')
    source=Path(__file__).resolve().parent
    require(all(file_hash(source/name)==sha for name,sha in request['sources'].items()), 'diagnostic source changed')
    site,runtime=request['site_manifest'],request['runtime']
    rank=int(os.environ['SLURM_PROCID'])
    observed_placement(folder,rank,site,request['case'])
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from mpi4py import MPI
    from psana import DataSource
    comm=MPI.COMM_WORLD
    require(comm.rank == rank and comm.size == 3,'diagnostic topology differs')
    provenance=runtime_evidence(runtime)
    if rank==0:
        write(folder/'step.json',dict(job=site['job_id'],node=site['node'],step=os.environ['SLURM_STEP_ID']))
    observation, held = {}, []
    global _retained_requests
    _retained_requests = held  # keep buffer alive until interpreter shutdown in observe mode
    events=[]
    if request['mode'] != 'mpi-only':
        from psana.psexp.node import Smd0
        original=Smd0.start
        # Trace only this exact function. Retain its request AND receive buffer;
        # Get_status observes without consuming it. No modification of installed files.
        def trace(frame,event,arg):
            if frame.f_code is not original.__code__:
                return None
            if event == 'return':
                require('t_req' in frame.f_locals and 't_rankreq' in frame.f_locals,'termination receive not found')
                req=frame.f_locals['t_req']
                held.append((req,frame.f_locals['t_rankreq']))
                observation.update(pending=not req.Get_status(), source=original.__code__.co_filename,
                                   function=original.__qualname__)
                if request['mode']=='cancel' and observation['pending']:
                    req.Cancel()
                    status=MPI.Status()
                    req.Wait(status)
                    observation.update(cancelled=status.Is_cancelled(),null_after_wait=req == MPI.REQUEST_NULL)
            return trace
        require(sys.gettrace() is None,'unexpected trace already installed')
        ds=DataSource(exp='mfx100848724',run=51,dir=request['inputs']['root'],live=False,
            detectors=['jungfrau'],skip_calib_load=['jungfrau'],batch_size=20,max_events=20)
        run=next(ds.runs())
        check_selected_files(ds,request['inputs'])
        if rank==0:sys.settrace(trace)
        try:
            for event in run.events():
                events.append([int(event.timestamp),sum(d._size for d in event._dgrams if d is not None)])
        finally:
            if rank==0:sys.settrace(None)
        run.close_shared_memory()
    comm.Barrier()
    after=runtime_evidence(runtime)
    rows=comm.gather(dict(rank=rank,events=events,request=observation,provenance_before=provenance,provenance_after=after),root=0)
    if rank==0:
        actual=sorted(e for row in rows for e in row['events'])
        expected=[] if request['mode']=='mpi-only' else [[e[0],e[2]] for e in request['reference']]
        require(actual==expected,'diagnostic event/byte reference differs')
        write(folder/'complete.json',dict(status='pass',stage=stage,rows=rows,mode=request['mode']))
    # held keeps the unmatched request buffer alive through this function. The
    # observe process intentionally retains the original missing-completion path.


if __name__=='__main__' and len(sys.argv)==5 and sys.argv[1]=='_preflight':
    from run_smoke import pinned
    require(os.uname().nodename.startswith('nid') and os.environ.get('SLURM_STEP_ID','batch')=='batch', 'batch compute required')
    data=pinned(dict(path=sys.argv[2],sha256=sys.argv[3]))
    data['source']=Path(data['source'])
    require(all(file_hash(data['source']/name)==sha for name,sha in data['sources'].items()), 'control sources changed')
    preflight(folder=Path(sys.argv[4]),**data)
elif __name__=='__main__':
    try:
        probe(*sys.argv[1:])
    except BaseException:
        import traceback
        traceback.print_exc()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
