"""Build isolated revisions and capture runtime pins; never run a campaign.

build runs in an activated build environment on a login node. capture launches
one small probe in an existing GPU allocation. Failed directories are retained.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from native_contract import (file_hash, native_environment, read, require,
                             runtime_evidence, write)

ENV_KEYS = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX', 'LCLS_CALIB_HTTP'}
DEPENDENCIES = ('numpy', 'mpi4py', 'cupy', 'cupyx', 'cupy_backends', 'kvikio')
SHARED = ('python', 'python_version', 'environment', 'dependencies', 'native_libraries',
          'cupy_version', 'kvikio_version', 'cuda_runtime_version', 'cuda_driver_version', 'path_policy')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def canonical(path):
    p = Path(path).absolute()
    require(p == p.resolve(), 'symlink or noncanonical path: ' + str(p))
    return p


def inventory(root, code_only=False):
    """Pin contents, rejecting links outside the installation or directory links."""
    root = Path(root).resolve(strict=True)
    result = {}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        require(not (p.is_symlink() and p.is_dir()), 'directory symlink in install')
        require(not p.is_symlink() or (p.exists() and root in p.resolve().parents), 'external/broken install symlink')
        if p.is_file() and (not code_only or p.suffix == '.py' or '.so' in p.name):
            result[str(p.relative_to(root))] = file_hash(p)
    require(result, 'empty installed inventory')
    return result


def shared_environment(path, python):
    env = read(path)
    require(set(env) == ENV_KEYS and all(isinstance(v, str) for v in env.values()), 'exact six-key environment required')
    for key in ('PATH', 'LD_LIBRARY_PATH'):
        require(env[key] and all(Path(p).is_absolute() and p for p in env[key].split(':')), 'absolute nonempty path entries required: ' + key)
        for entry in env[key].split(':'):
            # Site dependencies belong here; never add an existing psana prefix.
            p = Path(entry)
            require(not (p.parent/'activate.sh').exists(), 'shared environment contains an install activation path')
    prefix = Path(env['CONDA_PREFIX']).resolve(strict=True)
    require(prefix in Path(python).resolve(strict=True).parents, 'Python outside dependency environment')
    for key in ('CUDA_HOME', 'CUDA_PATH'):
        require(Path(env[key]).is_absolute() and Path(env[key]).is_dir(), 'CUDA root missing: ' + key)
    require(env['LCLS_CALIB_HTTP'].startswith('https://'), 'explicit HTTPS calibration endpoint required')
    return env


def clean_environment(shared):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PS_', 'PSANA_', 'KVIKIO_', 'CUDA_', 'CUPY_', 'OMP_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'SIT_PSDM_OFFSITE')}
    env.update(shared)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PS_PARALLEL='none',
               MPICH_GPU_SUPPORT_ENABLED='0', OMP_NUM_THREADS='1')
    return env


def build(args):
    require(not os.uname().nodename.startswith('nid'), 'build on login, capture on compute')
    require(re.fullmatch('[0-9a-f]{40}', args.commit), 'full commit SHA required')
    repo, root, python = canonical(args.repo), canonical(args.root), Path(args.python).absolute()
    require(python.is_absolute() and python.is_file(), 'absolute Python executable required')
    require(git(repo, 'rev-parse', args.commit+'^{commit}') == args.commit, 'commit differs')
    require(not root.exists() and root.parent.is_dir(), 'new build root with existing parent required')
    require(1 <= args.jobs <= 32, 'jobs must be between 1 and 32')
    # Reject missing build entry point before creating a worktree.
    script = subprocess.check_output(['git', '-C', str(repo), 'show', args.commit+':build_psana.sh'])
    env = shared_environment(args.environment, python)
    child = clean_environment(env)
    child['PYTHON'] = str(python)
    version = subprocess.check_output([str(python), '-B', '-s', '-c', 'import sys;print(sys.version.split()[0])'], env=child, text=True).strip()
    require(tuple(map(int, version.split('.')[:2])) >= (3, 9), 'Python 3.9+ required')
    root.mkdir(mode=0o700)
    source, install, builddir = root/'source', root/'install', root/'build'
    receipt = dict(schema='psana-runtime-build-v1', commit=args.commit, source=str(source),
                   source_tree=git(repo, 'rev-parse', args.commit+'^{tree}'), install=str(install),
                   build_directory=str(builddir), python=str(python), python_version=version,
                   environment=env, build_script_sha256=hashlib.sha256(script).hexdigest(),
                   driver_sha256=file_hash(__file__), helper_sha256=file_hash(Path(__file__).with_name('native_contract.py')),
                   started_utc=now(), build_type=args.build_type, jobs=args.jobs,
                   build_environment={k:child[k] for k in ('CC','CXX','CFLAGS','CXXFLAGS','LDFLAGS','CPATH','LIBRARY_PATH','CMAKE_PREFIX_PATH','LOADEDMODULES','CRAY_MPICH_DIR') if k in child})
    write(root/'build-started.json', receipt)
    try:
        subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(source),args.commit], check=True)
        links = [line.split('\t', 1)[1] for line in git(source, 'ls-files', '--stage').splitlines() if line.startswith('160000 ')]
        require(all(path.startswith('psdaq/') for path in links), 'analysis submodules need a separate qualified recipe')
        receipt['uninitialized_daq_submodules'] = links
        command = ['bash', str(source/'build_psana.sh'), '--prefix', str(install), '--build-dir', str(builddir), '--build-type', args.build_type, '-j', str(args.jobs)]
        receipt['command'] = command
        with (root/'build.log').open('x') as log:
            subprocess.run(command, cwd=root, env=child, stdout=log, stderr=subprocess.STDOUT, check=True)
        require(git(source, 'rev-parse', 'HEAD') == args.commit and not git(source, 'status', '--porcelain', '--untracked-files=all'), 'source changed during build')
        site = install/('lib/python%s/site-packages' % '.'.join(version.split('.')[:2]))
        package = site/'psana'
        require(package.is_dir(), 'missing installed psana package')
        options = read(builddir/'meson-info/intro-buildoptions.json')
        require(next(x['value'] for x in options if x['name']=='prefix') == str(install), 'Meson prefix mismatch')
        probe_env = dict(child, PYTHONPATH=str(site), LD_LIBRARY_PATH=str(install/'lib')+':'+env['LD_LIBRARY_PATH'])
        probe = 'import psana,psana.dgram;from pathlib import Path;print(Path(psana.__file__).resolve().parent)'
        actual = subprocess.check_output([str(python),'-B','-s','-c',probe], cwd=root, env=probe_env, text=True).strip()
        require(actual == str(package), 'build import did not resolve to selected install')
        receipt.update(pythonpath=str(site), package=str(package), files=inventory(package, True),
                       install_files=inventory(install), log_sha256=file_hash(root/'build.log'),
                       meson_info={str(p.relative_to(builddir)):file_hash(p) for p in sorted((builddir/'meson-info').glob('*.json'))},
                       finished_utc=now(), complete=True)
        write(root/'build-complete.json', receipt)
    except BaseException as error:
        write(root/'build-failed.json', dict(error=str(error), time=now(), complete=False))
        raise
    print(json.dumps(dict(receipt=str(root/'build-complete.json'), sha256=file_hash(root/'build-complete.json'))))


def checked_build(path, sha):
    require(file_hash(path) == sha, 'build receipt changed')
    r = read(path)
    require(r['schema'] == 'psana-runtime-build-v1' and r['complete'] is True, 'incomplete build')
    require(git(r['source'], 'rev-parse', 'HEAD') == r['commit'] and not git(r['source'], 'status', '--porcelain', '--untracked-files=all'), 'source changed since build')
    require(inventory(r['install']) == r['install_files'], 'installed files changed since build')
    require(inventory(r['package'], True) == r['files'], 'package changed since build')
    return r


def mapped_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and any(s in Path(fields[5]).name for s in ('libmpi','libcuda','libnvrtc','libcufile','libkvikio')):
            paths.add(str(Path(fields[5]).resolve(strict=True)))
    require(any('libmpi' in p for p in paths) and any('libcuda' in p for p in paths) and any('libnvrtc' in p for p in paths), 'missing mapped MPI/CUDA/NVRTC libraries')
    return {p:file_hash(p) for p in sorted(paths)}


def dependency_files():
    result = {}
    for name in DEPENDENCIES:
        module = importlib.import_module(name)
        root = Path(module.__file__).resolve().parent
        for relative, sha in inventory(root, True).items():
            result[str((root/relative).resolve())] = sha
    return result


def probe(args):
    """Child runs under precisely the selected runtime environment, on compute."""
    require(os.environ.get('SLURM_JOB_ID') and os.uname().nodename.startswith('nid'), 'GPU compute allocation required')
    r = checked_build(args.receipt, args.receipt_sha256)
    require(Path(sys.executable).resolve() == Path(r['python']).resolve(), 'wrong probe Python')
    from mpi4py import MPI
    import psana
    import psana.dgram
    from psana import DataSource
    from psana.psexp import events, event_manager, run
    require('cupy' not in sys.modules, 'CPU import unexpectedly initializes GPU modules')
    require(MPI.COMM_WORLD.size == 1 and 'MPICH' in MPI.Get_library_version(), 'single-rank MPICH probe required')
    import cupy as cp
    import kvikio
    from kvikio import defaults
    require(kvikio.__version__.startswith('26.10.'), 'KvikIO 26.10 required')
    require(cp.cuda.runtime.getDeviceCount() == 1, 'probe needs exactly one visible GPU')
    defaults.set(dict(num_threads=4,task_size=1048576,bounce_buffer_size=16777216,
                      compat_mode=kvikio.CompatMode.ON,auto_direct_io_read=False,auto_direct_io_write=False))
    require(defaults.get('compat_mode') == kvikio.CompatMode.ON and defaults.get('auto_direct_io_read') is False, 'KvikIO fallback policy differs')
    cp.zeros(1).sum().item()
    target = cp.zeros(1024, dtype=cp.uint32)
    kernel = cp.RawKernel('extern "C" __global__ void fill(unsigned int* x) { x[threadIdx.x] = 42; }','fill')
    kernel((1,), (32,), (target,)); cp.cuda.runtime.deviceSynchronize()
    require(bool(cp.all(target[:32] == 42).item()), 'NVRTC/device probe failed')
    scratch = Path(args.output).parent/'probe-input.bin'
    with scratch.open('xb') as f: f.write(bytes(range(256))*16)
    data = cp.empty(4096,dtype=cp.uint8)
    with kvikio.CuFile(scratch,'r') as f:
        require(f.pread(data).get() == 4096, 'KvikIO probe short read')
    require(data.get().tobytes() == scratch.read_bytes(), 'KvikIO probe data differs')
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu import GpuTask
    manifest = {k:r[k] for k in ('commit','install','python','python_version','pythonpath','package','files','install_files','environment')}
    manifest.update(schema='psana-native-runtime-v1',api_base='199575205',path_policy='isolated-install-v1',
                    dependencies=dependency_files(),native_libraries=mapped_libraries(),cupy_version=cp.__version__,
                    kvikio_version=kvikio.__version__,cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                    cuda_driver_version=cp.cuda.runtime.driverGetVersion(),
                    preparation=dict(build_receipt=str(Path(args.receipt).resolve()),build_receipt_sha256=args.receipt_sha256,
                        driver_sha256=file_hash(__file__),helper_sha256=file_hash(Path(__file__).with_name('native_contract.py')),
                        job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,time=now(),mpi=MPI.Get_library_version(),
                        cpu_imports_without_cuda=True,nvrtc_probe=True,kvikio_probe_bytes=4096,
                        qualification='Runtime probe only; source API, dataset correctness and native adapter qualification remain required.'))
    runtime_evidence(manifest)
    checked_build(args.receipt,args.receipt_sha256)
    write(args.output, manifest)  # only after every check; no qualification flag


def capture(args):
    require(os.environ.get('SLURM_JOB_ID') and os.uname().nodename.startswith('nid'), 'GPU compute allocation required')
    r = checked_build(args.receipt, args.receipt_sha256)
    output = canonical(args.output)
    require(not output.exists() and output.parent.is_dir(), 'new output with existing parent required')
    folder = output.with_name(output.name+'.capture')
    folder.mkdir(mode=0o700)  # fresh attempt only, including after a failed probe
    request = dict(contract=dict(environment=dict(PS_FAKESTEP_FLAG='0',PS_SRV_NODES='0',PS_EB_NODES='1',MPICH_GPU_SUPPORT_ENABLED='0')))
    runtime = dict(r,path_policy='isolated-install-v1')
    env = native_environment(request,runtime)
    require('CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'], 'Slurm must expose the selected GPU')
    env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    env['CUPY_CACHE_DIR'] = str(folder/'cupy-cache')
    command=[r['python'],'-B','-s',str(Path(__file__).resolve()),'_probe','--receipt',str(Path(args.receipt).resolve()),'--receipt-sha256',args.receipt_sha256,'--output',str(folder/'runtime.json')]
    write(folder/'started.json',dict(command=command,time=now(),driver_sha256=file_hash(__file__)))
    try:
        with (folder/'probe.log').open('x') as log:
            subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
        manifest=read(folder/'runtime.json')
        write(output,manifest)
    except BaseException as error:
        write(folder/'failed.json',dict(error=str(error),time=now()))
        raise
    print(json.dumps(dict(manifest=str(output),sha256=file_hash(output))))


def check_pair(first, second):
    for r in (first,second):
        require(r['schema']=='psana-native-runtime-v1' and r['path_policy']=='isolated-install-v1', 'driver-generated runtime required')
        require(r['api_base']=='199575205' and r['files'] and r['install_files'], 'runtime contract missing')
    for key in SHARED:
        require(first[key]==second[key], 'baseline/candidate differ: '+key)
    require(first['commit']==second['commit'] or first['install']!=second['install'], 'different revisions share an install')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--repo',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--python',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True);p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--build-type',choices=('debug','debugoptimized','release'),default='debugoptimized')
    for name in ('capture','_probe'):
        p=sub.add_parser(name,help='internal child probe' if name=='_probe' else 'capture on an allocated GPU node')
        p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True)
        if name=='capture':p.add_argument('--timeout',type=int,default=180)
    p=sub.add_parser('check-pair');p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':build(args)
    elif args.command=='capture':
        require(0 < args.timeout <= 600, 'probe timeout must be 1..600 seconds');capture(args)
    elif args.command=='_probe':probe(args)
    else:
        check_pair(read(args.baseline),read(args.candidate))
        print(json.dumps(dict(compatible=True,baseline_sha256=file_hash(args.baseline),candidate_sha256=file_hash(args.candidate),qualification=False)))


if __name__=='__main__':
    main()
