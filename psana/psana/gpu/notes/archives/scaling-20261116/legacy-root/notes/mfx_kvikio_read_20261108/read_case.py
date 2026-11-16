"""One fresh-process, read-only fallback measurement. Never writes input files."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

from plan_reads import GIB, MIB, load_fixture, require, run_queue, sha, slab_plan


def run(config, output):
    require(not output.exists(), 'Output exists')
    allowed = sorted(os.sched_getaffinity(0))
    require(config['cpus'] <= len(allowed), 'Insufficient CPUs')
    os.sched_setaffinity(0, allowed[:config['cpus']])
    import cupy as cp
    import kvikio
    import kvikio.defaults as defaults
    settings = dict(compat_mode=kvikio.CompatMode.ON,num_threads=config['threads'],task_size=config['task_mib']*MIB,
                    bounce_buffer_size=config['bounce_mib']*MIB,auto_direct_io_read=config['direct'])
    # Keep the context object alive; set() changes settings immediately.
    context = defaults.set(settings)
    effective = {k:defaults.get(k) for k in settings}
    require(effective['compat_mode'] == kvikio.CompatMode.ON, 'CPU fallback not forced')
    for k,v in settings.items():
        if k != 'compat_mode': require(effective[k] == v, 'Setting did not take effect: '+k)
    effective['compat_mode'] = 'ON'
    paths,sizes,events,identity = load_fixture()
    requests = events if config['pattern']=='events' else slab_plan(sizes,config['chunk_mib']*MIB)
    max_size = max(r[2] for r in requests)
    depth = config['depth']
    require(1 <= depth <= 140 and max_size*depth <= 4*GIB, 'Buffer budget exceeded')
    require(config['threads'] <= 16 and config['bounce_mib'] <= 64, 'Pinned budget exceeded')
    buffers = [cp.empty(max_size,dtype=cp.uint8) for _ in range(depth)]
    handles = [kvikio.CuFile(str(p),'r') for p in paths]
    cpu_fds = [os.open(p,os.O_RDONLY) for p in paths]
    report = dict(status='running',config=config,effective=effective,host=platform.node(),job=os.getenv('SLURM_JOB_ID'),
                  runtime=dict(python=platform.python_version(),cupy=cp.__version__,kvikio=kvikio.__version__,
                               gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                               cuda_runtime=cp.cuda.runtime.runtimeGetVersion()),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),identity=identity,
                  source_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
                  kvikio_source_sha256={str(p):sha(p) for p in
                      [Path(kvikio.__file__),Path(kvikio.__file__).with_name('cufile.py'),
                       Path(defaults.__file__)]},
                  device_buffer_bytes=max_size*depth,requested_bytes_per_pass=sum(r[2] for r in requests),
                  requests_per_pass=len(requests),passes=[])
    def save():
        output.write_text(json.dumps(report,indent=2)+'\n')
    def submit(slot,request):
        i,offset,size = request
        return handles[i].pread(buffers[slot],size=size,file_offset=offset,task_size=config['task_mib']*MIB)
    try:
        # Only five requests for initialization, not a preload of the fixture.
        warm = requests[:5]
        run_queue(warm,min(depth,5),submit,lambda f:int(f.get()))
        cp.cuda.Device().synchronize()
        report['warmup_bytes'] = sum(r[2] for r in warm)
        report['memory_before'] = dict(zip(('free','total'),cp.cuda.runtime.memGetInfo()))
        # Check whether KvikIO opened any O_DIRECT descriptors to the input files.
        identities = {(os.fstat(fd).st_dev,os.fstat(fd).st_ino) for fd in cpu_fds}
        direct_fds = 0
        for entry in Path('/proc/self/fdinfo').iterdir():
            try:
                st=os.fstat(int(entry.name))
                if (st.st_dev,st.st_ino) in identities:
                    flags=int(next(x.split()[1] for x in entry.read_text().splitlines() if x.startswith('flags:')),8)
                    direct_fds += bool(flags & os.O_DIRECT)
            except (OSError,StopIteration):
                pass
        report['input_direct_fd_count'] = direct_fds
        network = None
        if config.get('network_profile', False):
            from network_watch import NetworkProfile
            network = NetworkProfile(paths)
            report['network_initial'] = dict(cache=network.initial_cache, idle=network.idle)
        elapsed_total = 0
        for iteration in range(3):
            cp.cuda.Device().synchronize()
            observation = network.begin(iteration) if network else None
            start = time.perf_counter()
            total,count,last = run_queue(requests,depth,submit,lambda f:int(f.get()))
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter()-start
            require(total == report['requested_bytes_per_pass'] and count==len(requests), 'Accounting mismatch')
            row = dict(bytes=total,requests=count,seconds=elapsed,gib_s=total/GIB/elapsed)
            if network:
                row['network'] = network.finish(observation,total)
            report['passes'].append(row)
            save()
            elapsed_total += elapsed
            if elapsed_total >= 5 and not network:
                break
        # Last-use buffer witnesses only after final completion, outside timing.
        witnesses = []
        for slot,(i,offset,size) in sorted(last.items()):
            for at in sorted({0,max(0,size//2-32),max(0,size-64)}):
                n=min(64,size-at)
                actual=buffers[slot][at:at+n].get().tobytes()
                expected=os.pread(cpu_fds[i],n,offset+at)
                require(len(expected)==n and actual==expected,'GPU/CPU byte witness mismatch')
                witnesses.append(dict(slot=slot,file=i,offset=offset+at,size=n))
        total = sum(p['bytes'] for p in report['passes'])
        report.update(status='pass',total_bytes=total,seconds=elapsed_total,gib_s=total/GIB/elapsed_total,
                      validation=dict(completed_requests=sum(p['requests'] for p in report['passes']),
                                      gpu_cpu_witnesses=len(witnesses),witnesses=witnesses),
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      timing='First submission through all futures and final device synchronization per full pass. Sum complete-pass bytes/time. Setup, five warm reads, inter-pass idle and post-read witnesses excluded.',
                      cache='Uncontrolled Weka/server/OS caches. Later cases and repeated full passes reuse input. No staging or cache flush. Direct flag is opportunistic POSIX I/O, still CPU fallback, not GDS.')
        if network:
            report['cache'] = 'First full pass: owner-checked file-specific eviction and zero-mincore gate after warmup. Next two passes reuse cache. Weka backend caches uncontrolled. NIC endpoints include counter sampling overhead; rates use their own interval, not GPU-read timer. NIC traffic is node-wide; no exact cache-hit ratio inferred.'
    except BaseException as exc:
        report.update(status='fail',error=repr(exc))
        raise
    finally:
        # Join before closing handles or releasing buffers, also on error.
        cp.cuda.Device().synchronize()
        for h in handles: h.close()
        for fd in cpu_fds: os.close(fd)
        save()
    print(json.dumps(dict(name=config['name'],status=report['status'],gib_s=report.get('gib_s')),sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config),args.output)
