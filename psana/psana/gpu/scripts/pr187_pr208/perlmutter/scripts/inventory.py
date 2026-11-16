import os,sys,json,subprocess,time
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
report=dict(time=time.time(),hostname=os.uname().nodename,environment={k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k in ('TMPDIR','SCRATCH')},commands={},files={})
commands=[['lfs','getstripe',str(Path(__file__).resolve().parents[1]/'inputs')],['findmnt'],['df','-hT'],['lsblk','-b','-o','NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINT,MODEL'],['ls','-l','/sys/block','/sys/class/nvme','/sys/class/cxi'],['lspci','-Dvv'],['numactl','--hardware'],['lscpu'],['nvidia-smi','-q'],['nvidia-smi','topo','-m'],['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max','--format=csv'],['ip','-s','link'],['/usr/sbin/lctl','get_param','version'],['/usr/sbin/lctl','get_param','llite.*.stats','osc.*.stats'],['lfs','getstripe','/pscratch/sd/m/monarin/psana2-gpu/prepost-stack-20261017/attempt01/inputs'],['scontrol','show','job',os.environ['SLURM_JOB_ID']]]
for p in ('/tmp',os.environ.get('TMPDIR','/tmp'),'/dev/shm','/pscratch/sd/m/monarin'):commands.append(['findmnt','-T',p])
for dev in Path('/sys/class/net').iterdir():
 commands.extend([['/usr/sbin/ethtool',dev.name],['/usr/sbin/ethtool','-S',dev.name]])
for cmd in commands:
 try:
  p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
  report['commands'][' '.join(cmd)]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except Exception as e:report['commands'][' '.join(cmd)]={'error':str(e)}
paths=[Path(p) for p in ('/proc/meminfo','/proc/self/cgroup','/proc/diskstats','/proc/net/dev','/proc/mounts')]
for pat in ('/sys/bus/pci/devices/*/current_link_speed','/sys/bus/pci/devices/*/current_link_width','/sys/bus/pci/devices/*/max_link_speed','/sys/bus/pci/devices/*/max_link_width','/sys/bus/pci/devices/*/numa_node','/sys/block/*/queue/max_sectors_kb','/sys/block/*/queue/max_hw_sectors_kb'):
 import glob
 paths.extend(map(Path,glob.glob(pat)))
paths.extend(Path('/sys/class/cxi').glob('*/device/telemetry/hni_sts_*octets*'))
for line in Path('/proc/self/cgroup').read_text().splitlines():
 _,controllers,rel=line.split(':',2)
 base=Path('/sys/fs/cgroup')
 if controllers:
  if 'memory' not in controllers.split(','):continue
  base=base/'memory'
 group=base/rel.lstrip('/')
 for ancestor in (group,*group.parents):
  if ancestor != base and base not in ancestor.parents:break
  for name in ('memory.max','memory.high','memory.current','memory.events','memory.limit_in_bytes','cpuset.cpus.effective','cpuset.mems.effective'):
   if (ancestor/name).exists():paths.append(ancestor/name)
for p in paths:
 try:report['files'][str(p)]=p.read_text()
 except Exception as e:report['files'][str(p)]=str(e)
(out/'inventory.json').write_text(json.dumps(report,indent=2))
print('INVENTORY_COMPLETE',report['hostname'],str(out),flush=True)
