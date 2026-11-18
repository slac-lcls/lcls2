"""Six native raw smoke cases, without enabling campaign qualification gates."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import campaign
import prepare_site
from native_contract import digest, file_hash, input_ranges, read, require, write
from prepare_inputs import header, merge_ranges, stamp
from perlmutter_adapter import command_output

SOURCES = ('run_smoke.py', 'campaign.py', 'perlmutter_adapter.py', 'native_worker.py',
           'native_contract.py', 'prepare_inputs.py', 'prepare_site.py', 'site_probe.py',
           'prepare_runtime.py', 'profile.json')
COUNT = 1000


def pinned(entry):
    path = Path(entry['path'])
    require(path.is_absolute(), 'absolute artifact path required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'pinned artifact changed: '+str(path))
    return json.loads(raw)


def derive_prefix(manifest, plan, reference, count):
    require(manifest['schema'] == 'run51-private-input-v2' and manifest['private_copy'] is True, 'private v2 input required')
    require(plan['schema'] == 'run51-input-plan-v1' and reference == plan['reference'], 'parent independent reference differs')
    streams = [3,5,7,8,9]
    require([f['stream'] for f in manifest['files']] == [f['stream'] for f in plan['streams']] == streams, 'stream set differs')
    require(type(count) is int and 0 < count <= len(reference['events']), 'invalid prefix')
    result, expected = copy.deepcopy(manifest), []
    for i in range(count):
        rows = [s['events'][i] for s in plan['streams']]
        require(all(r[:2] == rows[0][:2] and r[4] in (11,12) for r in rows), 'prefix identity differs across streams')
        expected.append(rows[0][:2] + [sum(r[3] for r in rows)])
    require(expected == reference['events'][:count], 'prefix reference differs from SMD plan')
    for original, output, stream in zip(manifest['files'], result['files'], plan['streams']):
        require(original['smd']['sha256'] == stream['smd']['sha256'], 'private SMD differs from source plan')
        require(original['bigdata']['ranges'] == merge_ranges(stream['events'][:len(reference['events'])]), 'parent ranges differ')
        ranges = merge_ranges(stream['events'][:count])
        output['bigdata'].update(ranges=ranges, range_bytes=sum(r[1] for r in ranges))
        output['bigdata'].pop('sha256')  # never label the parent's hash as a prefix hash
        input_ranges(output['bigdata'])
    require(sum(f['bigdata']['range_bytes'] for f in result['files']) == sum(r[2] for r in expected), 'prefix bytes differ')
    return result, dict(schema='run51-raw-reference-v1', events=expected)


def private_files(manifest):
    root = Path(manifest['root'])
    require(root.is_absolute() and root.resolve() == root and not root.is_symlink(), 'private root must be canonical')
    require(root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0, 'private root must be owner-only')
    for item in manifest['files']:
        for kind in ('bigdata','smd'):
            entry = item[kind]
            path = Path(entry['path'])
            expected = root / ('smalldata' if kind == 'smd' else '') / ('mfx100848724-r0051-s%03d-c000%s.xtc2' % (item['stream'], '.smd' if kind == 'smd' else ''))
            require(path == expected and path.resolve() == path and not path.is_symlink(), 'private input layout differs')
            st = path.stat()
            require(st.st_uid == os.getuid() and st.st_nlink == 1 and not (st.st_mode & 0o222), 'owned read-only private file required')
            require(stamp(path) == {k:entry[k] for k in ('size','mtime_ns','device','inode')}, 'private file stat changed')
            if kind == 'smd':
                require(file_hash(path) == entry['sha256'], 'private SMD changed')


def prefix(spec_path, spec_sha, folder):
    spec, folder = pinned(dict(path=spec_path,sha256=spec_sha)), Path(folder)
    parent, plan, reference, receipt = [pinned(spec[k]) for k in ('manifest','plan','reference','preparation')]
    require(receipt['complete'] is True and receipt['events'] == 10000, 'input preparation incomplete')
    require(receipt['manifest_sha256'] == spec['manifest']['sha256'] and
            receipt['plan_sha256'] == spec['plan']['sha256'] and
            receipt['event_reference_sha256'] == spec['reference']['sha256'] == parent['event_reference_sha256'], 'parent artifact association differs')
    private_files(parent)
    manifest, selected = derive_prefix(parent, plan, reference, COUNT)
    for item, stream in zip(manifest['files'], plan['streams']):
        entry, sha = item['bigdata'], hashlib.sha256()
        with Path(entry['path']).open('rb', buffering=0) as data:
            for timestamp, step, offset, size, service in stream['events'][:COUNT]:
                data.seek(offset)
                require(header(data.read(24)) == (timestamp,service,size), 'prefix header differs from independent plan')
            for start, length in entry['ranges']:
                data.seek(start)
                remaining = length
                while remaining:
                    block = data.read(min(16 << 20, remaining))
                    require(block, 'short prefix read')
                    sha.update(block); remaining -= len(block)
        entry['sha256'] = sha.hexdigest()
    private_files(parent)
    folder.mkdir(exist_ok=False)
    write(folder/'event-reference.json', selected)
    manifest['event_reference_sha256'] = file_hash(folder/'event-reference.json')
    write(folder/'dataset-manifest.json', manifest)
    write(folder/'complete.json', dict(status='pass', events=COUNT, headers_verified=COUNT*5,
          event_sha256=digest([r[0] for r in selected['events']]), requested_bytes=sum(r[2] for r in selected['events']),
          parent=spec, manifest_sha256=file_hash(folder/'dataset-manifest.json'),
          reference_sha256=manifest['event_reference_sha256'],
          scope='independent SMD prefix plus private headers/range hashes; no copy or public-data cache operation'))


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def verify_prefix(folder, spec):
    receipt = read(folder/'complete.json')
    require(receipt['status'] == 'pass' and receipt['parent'] == spec and receipt['events'] == COUNT,
            'prefix completion association differs')
    require(receipt['manifest_sha256'] == file_hash(folder/'dataset-manifest.json') and
            receipt['reference_sha256'] == file_hash(folder/'event-reference.json'), 'prefix artifacts changed')


def final_step_audit(job):
    raw = command_output(['scontrol','--oneliner','show','step',job],timeout=15)
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        ids = re.findall(r'(?:^| )StepId=(\S+)',line)
        require(len(ids) == 1 and ids[0] in (job+'.batch',job+'.extern') and ids[0] not in seen,
                'unexpected/remaining allocation step')
        seen.add(ids[0])
    require(job+'.batch' in seen, 'missing live batch step in final audit')
    return dict(status='pass', raw=raw, steps=sorted(seen))


def run(args):
    spec = pinned(dict(path=args.inputs,sha256=args.inputs_sha256))
    runtime = pinned(spec['runtime'])
    require(Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'use pinned Python')
    output = Path(args.output).resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    source = Path(__file__).resolve().parent
    sources = {name:file_hash(source/name) for name in SOURCES}
    write(output/'started.json', dict(job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
          inputs=artifact(args.inputs), sources=sources, purpose='native smoke; no campaign qualification flags'))
    try:
        require(campaign.run_command([sys.executable,'-B','-s',str(source/'run_smoke.py'),
                '_prefix',str(Path(args.inputs).resolve()),args.inputs_sha256,str(output/'prefix')], output/'prefix.log',600) == 0, 'prefix preparation failed')
        prepare_site.capture(SimpleNamespace(runtime=spec['runtime']['path'], runtime_sha256=spec['runtime']['sha256'],
                             output=str(output/'site.json'), cores_per_rank=4, step_timeout=90, controller_timeout=360))
        verify_prefix(output/'prefix',spec)
        reference = read(output/'prefix/event-reference.json')
        manifest = read(output/'prefix/dataset-manifest.json')
        config = read(source/'profile.json')
        require(not any(config['qualification'].values()), 'smoke must not enable qualification flags')
        require(len(config['cases']) == 6, 'expected all six cells')
        config['campaign_id'] = 'native-smoke-'+os.environ['SLURM_JOB_ID']
        config['timeout_seconds'] = 360
        config['dataset'].update(events=COUNT, event_sha256=digest([r[0] for r in reference['events']]),
                                 requested_bytes=sum(r[2] for r in reference['events']), status='smoke prefix only; full qualification pending')
        config['dataset']['sites']['perlmutter'] = manifest['root']
        config['runtimes'] = {k:dict(commit=runtime['commit'], install=runtime['install']) for k in ('baseline','candidate')}
        write(output/'component-contract.json', dict(status='unqualified', scope='smoke only; no component or performance acceptance'))
        config['artifacts'] = {key:artifact(path) for key,path in dict(adapter=source/'perlmutter_adapter.py',
            native_worker=source/'native_worker.py', native_contract=source/'native_contract.py',
            event_reference=output/'prefix/event-reference.json', dataset_manifest=output/'prefix/dataset-manifest.json',
            native_site=output/'site.json', baseline_runtime=spec['runtime']['path'], candidate_runtime=spec['runtime']['path'],
            component_contract=output/'component-contract.json').items()}
        config['adapter_command'] = [sys.executable,'-B','-s',str(source/'perlmutter_adapter.py'),'{request}','{result}']
        blockers = campaign.validate(config, ready=False)
        require('native_adapter' in blockers, 'smoke unexpectedly qualifies campaign')
        write(output/'smoke-profile.json', config)
        write(output/'campaign-blockers.json', blockers)
        receipts = []
        for case in config['cases']:
            folder = output/case['id'];folder.mkdir()
            sample = dict(id='smoke-'+case['id'], phase='correctness', runtime='baseline', case=case)
            request = campaign.make_request(config,dict(profile_sha256=digest(config)),sample)
            write(folder/'request.json',request)
            argv = [str(x).replace('{request}',str(folder/'request.json')).replace('{result}',str(folder/'result.json')) for x in config['adapter_command']]
            require(campaign.run_command(argv,folder/'process.log',config['timeout_seconds']) == 0, 'native smoke case failed: '+case['id'])
            row = read(folder/'result.json')
            campaign.validate_result(config,request,row,folder)
            require(all(row['nic'][k] is not None for k in ('rx_bytes','tx_bytes')), 'NIC observations missing')
            receipts.append(dict(case=case['id'], result=artifact(folder/'result.json')))
            write(folder/'smoke-pass.json',dict(status='pass',scope='native smoke only',result=receipts[-1]['result']))
            print(case['id']+' PASS',flush=True)
        require(sources == {name:file_hash(source/name) for name in SOURCES} and
                file_hash(args.inputs) == args.inputs_sha256, 'frozen driver/input specification changed')
        for entry in spec.values():
            pinned(entry)
        private_files(manifest)
        write(output/'final-step-audit.json',final_step_audit(os.environ['SLURM_JOB_ID']))
        write(output/'index.json',{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()})
        write(output/'complete.json',dict(status='pass',cases=receipts,index=artifact(output/'index.json'),
              scope='six 1000-event raw smoke cases only; no statistical, component, overhead or full-cohort qualification'))
    except BaseException as error:
        write(output/'failed.json',dict(error=repr(error)))
        raise


def main():
    require(re.fullmatch('[0-9]+',os.environ.get('SLURM_JOB_ID','')) and
            re.fullmatch('nid[0-9]+',os.uname().nodename), 'allocated compute node required')
    require(os.environ.get('SLURM_STEP_ID','batch') == 'batch' and os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'direct single-node batch only')
    if len(sys.argv) == 5 and sys.argv[1] == '_prefix':
        return prefix(sys.argv[2],sys.argv[3],sys.argv[4])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
