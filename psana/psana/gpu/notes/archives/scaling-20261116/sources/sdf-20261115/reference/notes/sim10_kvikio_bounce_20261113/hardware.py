"""Read-only hardware inventory; retain unavailable optional fields explicitly."""
import os
import platform
from pathlib import Path
import subprocess


def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(command=args,returncode=None,stdout='',stderr=str(e))


def fields(root,names):
    values={}
    for name in names:
        try:values[name]=(root/name).read_text().strip()
        except OSError as e:values[name]=dict(unavailable=str(e))
    return values


def validate(record,gpu_uuid):
    if record['gpu_uuid']!=gpu_uuid or not record['host']:
        raise ValueError('Hardware GPU/host identity')
    for name in ('lscpu','gpu','gpu_topology'):
        r=record['commands'][name]
        if r['returncode']!=0 or not r['stdout'].strip():
            raise ValueError('Required hardware inventory unavailable: '+name)


def inventory(gpu_uuid):
    commands={
        'lscpu':command(['lscpu','-J']),
        'gpu':command(['nvidia-smi','-i',gpu_uuid,'-q']),
        'gpu_topology':command(['nvidia-smi','topo','-m']),
        'block_devices':command(['lsblk','-b','-J','-o','NAME,TYPE,SIZE,MODEL,REV,TRAN,MOUNTPOINT']),
        'pci':command(['lspci','-D','-nn']),
        'pci_tree':command(['lspci','-tv']),
        'numa':command(['numactl','--hardware']),
    }
    pci=[]
    for p in sorted(Path('/sys/bus/pci/devices').iterdir()):
        # Include bridges as well as endpoints so upstream negotiated links survive.
        pci.append(dict(bdf=p.name,path=str(p.resolve()),**fields(p,(
            'class','vendor','device','numa_node','local_cpulist',
            'max_link_speed','current_link_speed','max_link_width','current_link_width'))))
    nvme=[]
    for p in sorted(Path('/sys/class/nvme').glob('nvme*')):
        nvme.append(dict(controller=p.name,path=str(p.resolve()),
                         **fields(p,('model','firmware_rev','address','transport'))))
    namespaces=[]
    for p in sorted(Path('/sys/class/block').glob('nvme*n*')):
        if (p/'partition').exists():continue
        namespaces.append(dict(name=p.name,path=str(p.resolve()),**fields(p,('size','queue/logical_block_size'))))
    nics=[]
    for p in sorted(Path('/sys/class/net').iterdir()):
        if (p/'device').exists():
            nics.append(dict(name=p.name,path=str((p/'device').resolve()),
                             **fields(p,('speed','duplex','operstate')),driver=command(['ethtool','-i',p.name])))
    record=dict(host=platform.node(),kernel=platform.release(),gpu_uuid=gpu_uuid,
                allowed_cpus=sorted(os.sched_getaffinity(0)),commands=commands,
                pci_devices=pci,nvme_controllers=nvme,nvme_namespaces=namespaces,nics=nics,
                nvme_size_unit='512-byte sectors (Linux sysfs size)',
                storage_scope='Local hardware inventory; benchmark payload is served from Weka FFB over the NIC, not local NVMe',
                warnings=[name+': '+r['stderr'] for name,r in commands.items() if r['returncode']!=0])
    if not nvme:record['warnings'].append('No NVMe controllers visible in sysfs; do not infer capacity or firmware')
    validate(record,gpu_uuid)
    return record
