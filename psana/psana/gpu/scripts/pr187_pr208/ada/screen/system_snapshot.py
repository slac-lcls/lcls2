"""Read-only node/storage/cgroup evidence; call before and after timed samples."""
from pathlib import Path
import os,subprocess,time

def snapshot():
    files={}
    paths=[Path(p) for p in ('/proc/diskstats','/proc/mdstat','/proc/self/cgroup','/proc/meminfo','/proc/pressure/io','/proc/loadavg','/proc/stat','/proc/self/numa_maps','/proc/self/status')]
    paths+=list(Path('/sys/block').glob('*/queue/read_ahead_kb'))
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,rel=line.split(':',2)
        root=Path('/sys/fs/cgroup')
        if controllers:
            if 'blkio' not in controllers.split(','): continue
            root=root/'blkio'
        group=root/rel.lstrip('/')
        for ancestor in (group,*group.parents):
            if not ancestor.is_relative_to(root):break
            for name in ('memory.max','memory.current','memory.events','cpuset.cpus.effective','cpuset.mems.effective','io.max','io.stat','io.weight','blkio.throttle.read_bps_device','blkio.throttle.read_iops_device','blkio.throttle.io_service_bytes'):
                p=ancestor/name
                if p.exists():paths.append(p)
    for p in paths:
        try:files[str(p)]=p.read_text()
        except OSError as e:files[str(p)]=str(e)
    commands={}
    for cmd in (['lsblk','-o','NAME,TYPE,SIZE,ROTA,MOUNTPOINT,MODEL'],['findmnt','-T','/lscratch'],['numactl','--hardware']):
        try:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=15)
            commands[' '.join(cmd)]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except OSError as e:commands[' '.join(cmd)]=dict(error=str(e))
    return dict(time=time.time(),host=os.uname().nodename,affinity=sorted(os.sched_getaffinity(0)),files=files,commands=commands)
