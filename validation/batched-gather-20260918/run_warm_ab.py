"""Matched warm-only A/B/B+optimization controls, CPU/NVTX timing, then separate Nsight runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HISTORICAL = Path('/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/validation/perf-acceptance-20260916')
NSYS = '/sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', type=int, default=10000)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--instrumented-repeats', type=int, default=3)
    p.add_argument('--stage', type=Path)
    p.add_argument('--skip-trace', action='store_true')
    a = p.parse_args()
    if a.events < 100 or a.repeats < 1 or not 0 <= a.instrumented_repeats <= a.repeats:
        p.error('need at least 100 events and one repetition')
    logs = HERE / ('job-' + os.environ['SLURM_JOB_ID'] + '-warm-ab')
    logs.mkdir(exist_ok=False)
    print('LOG_DIRECTORY ' + str(logs), flush=True)
    constants = HISTORICAL / 'job-38419641-primary/cpu-calibration.pkl.gz'
    reference_log = constants.parent / 'cpu-check.log'
    cpu_reference = json.loads(next(line[len('CPU_CHECK '):] for line in reference_log.read_text().splitlines()
                                    if line.startswith('CPU_CHECK ')))
    sources = dict(O=REPO, G=REPO)
    prefixes = dict(O=HERE/'install_baseline', G=REPO/'install_psana')
    results = []
    os.environ['BENCH_CPU_AFFINITY'] = ','.join(map(str, sorted(os.sched_getaffinity(0))))
    orders = ('OG', 'GO', 'OG', 'GO')


    def env(variant):
        result = os.environ.copy()
        result['PYTHONPATH'] = str(prefixes[variant] / 'lib/python3.9/site-packages')
        result['LD_LIBRARY_PATH'] = str(prefixes[variant] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        return result

    def execute(tag, command, environment, timeout=2400):
        print(f'CASE_BEGIN {tag} {time.strftime("%FT%T")} {command}', flush=True)
        with (logs / (tag + '.log')).open('x') as output:
            process = subprocess.run(command, env=environment, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        text = (logs / (tag + '.log')).read_text()
        print(f'CASE_END {tag} status={process.returncode}', flush=True)
        if process.returncode:
            print(text[-10000:], flush=True)
            raise RuntimeError(f'{tag} failed')
        return text

    provenance = dict(node=os.uname().nodename, constants=str(constants),
                      constants_sha256=hashlib.sha256(constants.read_bytes()).hexdigest(),
                      nsys=subprocess.check_output([NSYS, '--version'], text=True).strip(),
                      revisions={v: subprocess.check_output(['git', '-C', str(sources[v]),
                                                            'rev-parse', 'HEAD'], text=True).strip() for v in 'OG'},
                      scripts={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ('phase_timing.py', 'bench.py', 'run_warm_ab.py', 'trace_rank.py')})
    provenance['baseline_revision'] = 'b0c9c3c02'
    provenance['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    provenance['orders'] = orders
    provenance['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    provenance['environment'] = {k: v for k, v in os.environ.items()
                                 if k.startswith(('PS_', 'KVIKIO_', 'OMP_', 'SLURM_', 'OMPI_MCA_'))
                                 or k in ('CUDA_VISIBLE_DEVICES', 'CUPY_CACHE_DIR', 'LD_LIBRARY_PATH', 'PYTHONPATH')}
    provenance['installed_hashes'] = {}
    for variant, prefix in prefixes.items():
        package = prefix / 'lib/python3.9/site-packages/psana'
        paths = list((package / 'gpu').rglob('*.py')) + list(package.glob('dgram*.so')) + list(package.glob('eventbuilder*.so'))
        provenance['installed_hashes'][variant] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    provenance['dirty'] = subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True)
    provenance['diff'] = subprocess.check_output(['git', '-C', str(REPO), 'diff'], text=True)
    provenance['prefixes'] = {v: str(prefix) for v, prefix in prefixes.items()}
    (logs/'provenance.json').write_text(json.dumps(provenance, indent=2))
    stage = a.stage
    if stage is None:
        parent = Path(tempfile.mkdtemp(prefix=f'parser-warm-ab-{os.environ["SLURM_JOB_ID"]}-',
                                       dir='/lscratch/monarin/tmp'))
        stage = parent / 'xtc'
        execute('stage', [sys.executable, str(HERE/'stage.py'),
                         '--source', '/sdf/data/lcls/ds/mfx/mfx101210926/xtc',
                         '--target', str(stage), '--exp', 'mfx101210926', '--run', '387',
                         '--events', str(a.events), '--streams', '5', '6', '7', '8', '9'], os.environ.copy())
    stage = stage.resolve()
    manifest = json.loads((stage/'manifest.json').read_text())
    assert manifest['events'] == a.events
    (logs/'stage.json').write_text(json.dumps(dict(path=str(stage), manifest=manifest), indent=2))

    def case(variant, mode, repetition, check=False, trace=False):
        tag = f'{variant}-{"check" if check else "trace" if trace else "warm"}-{mode}-r{repetition}'
        command = [sys.executable, '-u', str(HERE/'bench.py'), '--repo', str(REPO), '--dir', str(stage),
                   '--variant', variant, '--case', tag, '--events', str(a.events), '--batch-size', '20',
                   '--depth', '1', '--budget', '8', '--cache', 'warm', '--constants', str(constants),
                   '--timing-mode', mode, '--warm-cache-passes', '3']
        if check:
            command.append('--check')
        if trace:
            command.append('--nsight-capture')
            command = [sys.executable, str(HERE/'trace_rank.py'), '--nsys', NSYS,
                       '--output', str(logs/tag), '--'] + command
        command = ['mpirun', '-n', '3', '--oversubscribe', '--bind-to', 'none'] + command
        def host_snapshot():
            return {path: Path(path).read_text() for path in
                    ('/proc/loadavg', '/proc/meminfo', '/proc/stat', '/proc/net/dev')}
        host_before = host_snapshot()
        with (logs/(tag+'-gpu.csv')).open('x') as samples:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw',
                                        '--format=csv,noheader,nounits', '-lms', '500'], stdout=samples,
                                       stderr=subprocess.STDOUT)
            try:
                text = execute(tag, command, env(variant))
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        (logs/(tag+'-host.json')).write_text(json.dumps(dict(before=host_before, after=host_snapshot()), indent=2))
        prefix = 'CHECK_RESULT ' if check else 'PERF_RESULT '
        record = json.loads(next(line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)))
        if check:
            observed = sorted((c for r in record['ranks'] for c in r['checks']), key=lambda r: r['timestamp'])
            assert observed == cpu_reference, f'CPU/GPU mismatch {tag}'
        else:
            results.append(record)
            (logs/'results.json').write_text(json.dumps(results, indent=2))
        print(f'ACCEPTED {tag}', flush=True)

    for mode in ('off', 'cpu-nvtx'):
        for variant in 'OG':
            case(variant, mode, 0, check=True)
    print('CORRECTNESS_PASS B+ / B+gather with and without timing hooks', flush=True)
    for repetition in range(1, a.repeats + 1):
        modes = ('off', 'cpu-nvtx') if repetition <= a.instrumented_repeats else ('off',)
        for mode in (modes if repetition % 2 else tuple(reversed(modes))):
            for variant in orders[(repetition - 1) % len(orders)]:
                case(variant, mode, repetition)
    if not a.skip_trace:
        for variant in 'OG':
            case(variant, 'cpu-nvtx', 0, trace=True)
    for hashes in provenance['installed_hashes'].values():
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    print('INSTALLED_HASHES_UNCHANGED', flush=True)
    print('WARM_AB_COMPLETE ' + str(logs), flush=True)
    print('STAGE_RETAINED ' + str(stage), flush=True)


if __name__ == '__main__':
    main()
