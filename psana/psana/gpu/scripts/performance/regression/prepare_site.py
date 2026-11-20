"""Freeze compute allocation placement after adapter-launched MPI probes.

No submission, data reads, cache eviction, or campaign qualification flags.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil

from native_contract import digest, file_hash, nic_delta, nic_snapshot, read, require, write, compute_site, physical_nic_snapshot, traffic_delta
from native_worker import physical_cores
from perlmutter_adapter import command_output, launch
from prepare_runtime import canonical, now, publish

SOURCES = ('prepare_site.py', 'site_probe.py', 'perlmutter_adapter.py',
           'native_worker.py', 'native_contract.py', 'prepare_runtime.py', 'sdf_rank.py')
ENVIRONMENT = dict(PS_FAKESTEP_FLAG='0', PS_SRV_NODES='0', PS_EB_NODES='1', MPICH_GPU_SUPPORT_ENABLED='0')


def placements(topology, count):
    """Deterministic prefix: one Linux CPU per physical core, same rank prefix."""
    require(type(count) is int and 1 <= count <= 32, 'cores per rank must be 1..32')
    selected, seen = [], set()
    for cpu, core in sorted(topology.items()):
        require(type(cpu) is int and cpu >= 0 and len(core) == 2 and all(type(x) is int and x >= 0 for x in core), 'invalid CPU topology')
        if tuple(core) not in seen:
            seen.add(tuple(core)); selected.append(cpu)
    require(len(selected) >= 4 * count, 'insufficient distinct physical cores')
    groups = [selected[i*count:(i+1)*count] for i in range(4)]
    return {'1': groups[:3], '2': groups}


def timeouts(step, controller):
    require(type(step) is int and 0 < step < 3600, 'invalid step timeout')
    require(type(controller) is int and 2 * step + 120 < controller, 'insufficient controller cleanup headroom')


def nic_inventory(root=Path('/sys/class/net')):
    """Select physical hsn interfaces only; never sum bonds or virtual links."""
    rows = []
    for path in sorted(root.iterdir()):
        if not re.fullmatch(r'hsn[0-9]+', path.name):
            continue
        require((path/'device').exists(), 'hsn interface lacks physical device')
        device = (path/'device').resolve(strict=True)
        require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', device.name), 'hsn device is not PCI')
        require(int((path/'type').read_text()) == 1, 'expected Ethernet byte counters')
        driver = (device/'driver').resolve(strict=True)
        counters = {k:int((path/'statistics'/(k+'_bytes')).read_text()) for k in ('rx','tx')}
        require(all(x >= 0 for x in counters.values()), 'negative NIC byte counter')
        rows.append(dict(interface=path.name, device=str(device), driver=str(driver),
                         operstate=(path/'operstate').read_text().strip(), counters=counters, units='bytes'))
    require(rows and len({r['device'] for r in rows}) == len(rows), 'missing/duplicate physical Slingshot devices')
    return rows


def gpu_inventory(output):
    rows = []
    for fields in csv.reader(output.splitlines(), skipinitialspace=True):
        require(len(fields) == 4, 'invalid NVIDIA inventory')
        index, identifier, model, pci = [x.strip() for x in fields]
        require(index.isdigit() and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identifier), 'invalid GPU identity')
        require('A100' in model, 'A100 allocation required')
        rows.append(dict(index=int(index), uuid=identifier, model=model, pci=pci))
    require(rows and len({r['uuid'] for r in rows}) == len(rows), 'empty/duplicate GPU inventory')
    return sorted(rows, key=lambda r:r['index'])


def validate_rows(rows, site, bds, gpus, topology):
    require([r['rank'] for r in rows] == list(range(bds+2)), 'missing/duplicate rank evidence')
    used = set()
    for r, expected in zip(rows, site['cpu_affinity'][str(bds)]):
        require(r['node'] == site['node'] and r['job'] == site['job_id'], 'probe allocation differs')
        require(r['affinity'] == expected, 'rank affinity differs')
        cores = [topology[c] for c in expected]
        require(r['physical_cores'] == cores and not used.intersection(map(tuple, cores)), 'physical core placement differs')
        used.update(map(tuple, cores))
        if gpus and r['rank'] >= 2:
            require(r['gpu'] and r['gpu']['uuid'] == site['gpu_uuid'] and 'A100' in r['gpu']['model'], 'GPU assignment differs')
        else:
            require(r['gpu'] is None and r['cpu_no_cuda'] is True, 'CPU/service rank loaded CUDA')


def capture(args):
    timeouts(args.step_timeout, args.controller_timeout)
    job, node = os.environ.get('SLURM_JOB_ID', ''), os.uname().nodename
    selected_site = compute_site()
    require(getattr(args, 'site', 'auto') in ('auto', selected_site), 'requested cluster differs; connect to target before preparation')
    require(os.environ.get('SLURM_STEP_ID', 'batch') == 'batch', 'run directly in sbatch, outside an srun step')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'single-node allocation required')
    output, runtime_path = canonical(args.output), canonical(args.runtime)
    require(not output.exists() and not output.is_symlink(), 'output already exists')
    require(file_hash(runtime_path) == args.runtime_sha256, 'runtime manifest changed')
    runtime = read(runtime_path)
    require(runtime['schema'] == 'psana-native-runtime-v1' and runtime.get('path_policy') == 'isolated-install-v1', 'isolated runtime required')
    folder = output.with_name(output.name+'.capture')
    folder.mkdir(exist_ok=False, mode=0o700)
    try:
        source = Path(__file__).resolve().parent
        hashes = {name:file_hash(source/name) for name in SOURCES}
        write(folder/'started.json', dict(time=now(), job=job, node=node, sources=hashes,
              runtime=dict(path=str(runtime_path), sha256=args.runtime_sha256),
              controller_timeout_seconds=args.controller_timeout,
              environment_keys=sorted(os.environ), environment_sha256=digest(dict(os.environ))))
        allocation = command_output(['scontrol','show','job','-d',job])
        (folder/'allocation.txt').write_text(allocation)
        require('JobId='+job+' ' in allocation and 'NumNodes=1' in allocation, 'scheduler allocation differs')
        allowed = sorted(os.sched_getaffinity(0))
        topology = {c:physical_cores([c])[0] for c in allowed}
        cpus = placements(topology, args.cores_per_rank)
        physical_count = len({tuple(core) for core in topology.values()})
        require((physical_count == 64 if selected_site == 'perlmutter' else physical_count >= 4*args.cores_per_rank) and args.cores_per_rank <= 8,
                'site policy requires all 64 physical cores and at most 8 bound cores per rank')
        devices = gpu_inventory(command_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader']))
        job_gpus = os.environ.get('SLURM_JOB_GPUS','').split(',')
        eligible = [g for g in devices if str(g['index']) in job_gpus or g['uuid'] in job_gpus]
        require(eligible, 'no allocated A100 identified')
        if selected_site == 'sdf':
            names = getattr(args, 'nic_interfaces', None)
            require(names, 'SDF requires an explicit qualified physical NIC interface list')
            physical = physical_nic_snapshot(names)
            require(not physical['unavailable'], 'SDF physical NIC counters unavailable')
            nics = [dict(interface=name, **row) for name,row in physical['interfaces'].items()]
        else:
            nics = nic_inventory()
        site = dict(schema=selected_site+'-raw-site-v1', job_id=job, node=node,
                    gpu_uuid=eligible[0]['uuid'], cpu_affinity=cpus, step_cpus_per_task=16 if selected_site == 'perlmutter' else args.cores_per_rank,
                    nic_interfaces=[n['interface'] for n in nics], step_timeout_seconds=args.step_timeout)
        if selected_site == 'sdf':
            executable = getattr(args, 'sdf_mpiexec', None) or shutil.which('mpiexec',path=runtime['environment']['PATH'])
            require(executable and Path(executable).is_absolute(), 'SDF requires an absolute Open MPI 5 mpiexec path')
            executable = str(Path(executable).resolve(strict=True))
            site.update(job_gpus=job_gpus,step_gpu_count=len(job_gpus),mpi_launcher='openmpi-local-v1', mpiexec=executable, mpiexec_sha256=file_hash(executable), nic_backend='ethtool-physical-v1',
                        nic_devices={row['interface']:row['device'] for row in nics})
        write(folder/'site-draft.json', dict(site, schema=selected_site+'-raw-site-draft-v1'))
        write(folder/'inventory.json', dict(cpu_topology=topology, gpu_inventory=devices,
              job_gpus=job_gpus, batch_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), nics=nics, cpu_policy='ascending Linux CPU, one thread per physical core; no NUMA optimization'))
        before = physical_nic_snapshot(site['nic_interfaces'], site['nic_devices']) if selected_site == 'sdf' else nic_snapshot(site['nic_interfaces'])
        receipts = []
        for bds, gpus in ((1,0),(2,0),(1,1),(2,1)):
            label = 'bd%d-gpu%d' % (bds,gpus)
            request = dict(profile_sha256=digest(site), sample_id=digest(str(folder))+label,
                 case=dict(bds=bds,gpus=gpus), contract=dict(environment=ENVIRONMENT),
                 artifacts=dict(native_worker=dict(path=str(source/'site_probe.py'),sha256=hashes['site_probe.py'])),
                 site_manifest=site, runtime=runtime, sources=hashes)
            if selected_site == 'sdf':
                request['artifacts']['sdf_rank']=dict(path=str(source/'sdf_rank.py'),sha256=hashes['sdf_rank.py'])
            request_path = folder/(label+'.json')
            write(request_path, request)
            complete, scheduler = launch(request, request_path, folder/label, 'placement', site, runtime)
            validate_rows(complete['rows'], site, bds, gpus, topology)
            receipts.append(dict(case=label, scheduler=scheduler))
        after = physical_nic_snapshot(site['nic_interfaces'], site['nic_devices']) if selected_site == 'sdf' else nic_snapshot(site['nic_interfaces'])
        delta = traffic_delta(before, after) if selected_site == 'sdf' else nic_delta(before, after)
        require(all(v is not None for v in delta.values()), 'physical NIC counters unavailable')
        require(file_hash(runtime_path) == args.runtime_sha256 and
                hashes == {name:file_hash(source/name) for name in SOURCES}, 'runtime/source changed during capture')
        write(folder/'validation.json', dict(status='pass', time=now(), probes=receipts,
              nic_before=before, nic_after=after, nic_delta=delta,
              scope='launch/placement and NIC availability only; not data, cache, cleanup-fault or performance qualification'))
        index = {str(p.relative_to(folder)):file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        write(folder/'index.json', index)
        site['capture_index'] = dict(path=str(folder/'index.json'), sha256=file_hash(folder/'index.json'))
        publish(output, site)
    except BaseException as error:
        write(folder/'failed.json', dict(time=now(), error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=('auto','sdf','perlmutter'), default='auto')
    parser.add_argument('--nic-interfaces', nargs='+', help='SDF: preselected physical storage-facing NICs')
    parser.add_argument('--sdf-mpiexec', help='absolute Open MPI 5 mpiexec; defaults to pinned runtime PATH')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cores-per-rank', type=int, default=4)
    parser.add_argument('--step-timeout', type=int, default=180)
    parser.add_argument('--controller-timeout', type=int, default=900)
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
