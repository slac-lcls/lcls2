"""Count-only versus full accounting; isolated from PR campaign acceptance."""
import argparse
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import campaign
import prepare_site
import run_smoke
from native_contract import (artifact, check_minimal_rows, check_rows, cxi_delta, digest,
                             file_hash, load_contract, read, require, write)
from perlmutter_adapter import launch

SCHEMA = 'psana-instrumentation-overhead-v1'
SOURCES = run_smoke.SOURCES + ('overhead.py',)
MARGIN = 1.0


def schedule(cases, pilot):
    require(type(pilot) is bool, 'pilot must be a boolean')
    return [dict(id='%s-%02d-%s' % (case['id'], pair, mode), case=case, pair=pair,
                 mode=mode) for case in cases for pair in range(2 if pilot else 6)
            for mode in (('minimal', 'full') if pair % 2 == 0 else ('full', 'minimal'))]


def make_request(config, sample):
    request = campaign.make_request(config, dict(profile_sha256=digest(config)),
              dict(id=sample['id'], case=sample['case'], phase='overhead', runtime='baseline'))
    request['schema'] = SCHEMA
    request['contract']['instrumentation'] = sample['mode']
    request['contract_sha256'] = digest(request['contract'])
    return request


def finite_positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def check_capture(request, complete, site, reference, stage):
    """Recheck observations; missing minimal measurements stay explicitly null."""
    mode, case = request['contract']['instrumentation'], request['case']
    expected = reference['events'][:request['contract']['warmup']['events']] if stage == 'warmup' else reference['events']
    require(complete['status'] == 'pass' and complete['stage'] == stage and complete['diagnostic'] is False,
            'invalid or profiled completion')
    rows = complete['rows']
    work = (check_minimal_rows if mode == 'minimal' else check_rows)(rows, expected, case)
    require(complete['work'] == work, 'work receipt differs')
    require([r['affinity'] for r in rows] == [sorted(x) for x in site['cpu_affinity'][str(case['bds'])]], 'affinity differs')
    require(all(r['node'] == site['node'] and r['instrumentation'] == mode and
                r['qualification_stack_traces'] is False for r in rows), 'placement/mode/stack trace differs')
    for row in rows:
        require(row['event_count'] == (len(row['timestamps']) if mode == 'full' else row['event_count']), 'event population differs')
        require(all(finite_positive(row[k]) for k in ('setup_seconds','local_loop_seconds','loop_seconds','close_seconds')) and
                type(row['cpu_seconds']) in (float,int) and math.isfinite(row['cpu_seconds']) and row['cpu_seconds'] >= 0, 'invalid timing')
        if mode == 'full' and row['rank'] >= 2:
            require(finite_positive(row['first_batch_seconds']), 'missing first-batch measurement')
            delivery = row['delivery']
            require(sum(delivery['l1_per_delivery_window']) + delivery['drain_l1'] == row['event_count'], 'delivery count mismatch')
        gpu = case['gpus'] and row['rank'] >= 2
        require((row['gpu']['uuid'] == site['gpu_uuid']) if gpu else row['gpu'] is None, 'GPU differs')
        require(all(v == request['contract']['settings'][k] for k,v in row['observed_settings'].items()), 'settings differ')
        require(all(row['environment'][k] == v for k,v in request['contract']['environment'].items()), 'environment differs')
        require(all(row[k]['manifest_commit'] == request['runtime_commit'] for k in ('provenance_before','provenance_after')), 'runtime differs')
    if stage == 'measured':
        cache = complete['cache']
        require(all(r['policy'] == 'client-cold' and r['verified'] is True and
                    r['resident_bytes_before'] == 0 and r['measured_bytes'] == sum(e[2] for e in expected)
                    for r in cache['observations']) and len(cache['observations']) == 2, 'cold cache missing')
        traffic = cxi_delta(complete['cxi_before'], complete['cxi_after'])
        require(traffic['rx_bytes'] >= 0.9 * sum(e[2] for e in expected), 'CXI traffic does not cover selected bytes')
    else:
        require(complete['cache'] is None, 'warmup unexpectedly conditioned cache')
        traffic = None
    return dict(mode=mode, events=work['events'], exact_identity_and_bytes_checked=mode == 'full',
                loop_seconds=max(r['loop_seconds'] for r in rows),
                setup_seconds=max(r['setup_seconds'] for r in rows),
                cpu_seconds=sum(r['cpu_seconds'] for r in rows), traffic=traffic,
                placement=[dict(rank=r['rank'],node=r['node'],affinity=r['affinity'],gpu=r['gpu']) for r in rows])


def sample(request_path, result_path):
    request_path, result_path = Path(request_path).resolve(), Path(result_path).resolve()
    require(result_path.parent == request_path.parent and not result_path.exists(), 'new result beside request required')
    request = read(request_path)
    require(request['schema'] == SCHEMA, 'overhead schema required')
    require(Path(__file__).resolve() == artifact(request, 'overhead_driver'), 'unfrozen overhead driver')
    site, reference, inputs, runtime = load_contract(request)
    folder = result_path.parent
    records = {}
    for stage in ('warmup', 'measured'):
        complete, scheduler = launch(request, request_path, folder/stage, stage, site, runtime)
        records[stage] = check_capture(request, complete, site, reference, stage)
        records[stage]['process_seconds'] = scheduler['seconds']
    require(records['warmup']['placement'] == records['measured']['placement'], 'warmup placement differs')
    index = {str(p.relative_to(folder)):file_hash(p) for stage in ('warmup','measured')
             for p in sorted((folder/stage).rglob('*')) if p.is_file()}
    write(folder/'index.json',index)
    write(result_path,dict(schema=SCHEMA, sample_id=request['sample_id'], status='pass',
          request=run_smoke.artifact(request_path), index=run_smoke.artifact(folder/'index.json'), **records))


def collect(root):
    root = Path(root)
    plan, config, site, reference = [read(root/name) for name in ('plan.json','profile.json','site.json','prefix/event-reference.json')]
    require(plan['schema'] == SCHEMA and plan['margin_percent'] == MARGIN and plan['profile_sha256'] == digest(config), 'plan changed')
    require(config['cases'] == read(Path(__file__).with_name('profile.json'))['cases'], 'case matrix differs')
    require(config['dataset']['events'] == (1000 if plan['pilot'] else 10000) and
            config['warmup']['events'] == 200 and config['runtimes']['baseline'] == config['runtimes']['candidate'] and
            not any(config['qualification'].values()), 'overhead profile differs')
    expected = schedule(config['cases'], plan['pilot'])
    require(expected == plan['samples'], 'schedule changed')
    samples = root/'samples'
    require({p.name for p in samples.iterdir()} == {s['id'] for s in expected}, 'incomplete/extra samples')
    timings, placements, receipts = {}, {}, []
    for item in expected:
        folder = samples/item['id']
        request = read(folder/'request.json')
        require(request == make_request(config, item), 'request differs from frozen plan')
        result = read(folder/'result.json')
        require(result['schema'] == SCHEMA and result['status'] == 'pass' and result['sample_id'] == item['id'], 'invalid sample result')
        require(result['request'] == run_smoke.artifact(folder/'request.json') and
                result['index'] == run_smoke.artifact(folder/'index.json'), 'result artifact differs')
        index = read(folder/'index.json')
        actual = {str(p.relative_to(folder)):file_hash(p) for stage in ('warmup','measured')
                  for p in sorted((folder/stage).rglob('*')) if p.is_file()}
        require(index == actual, 'native capture missing/changed')
        for stage in ('warmup','measured'):
            scheduler = read(folder/stage/'scheduler.json')
            require(scheduler['status'] == 'pass' and scheduler['returncode'] == 0 and scheduler['no_active_steps'] is True, 'failed scheduler receipt')
            checked = check_capture(request, read(folder/stage/'complete.json'), site, reference, stage)
            checked['process_seconds'] = scheduler['seconds']
            require(result[stage] == checked, 'sample measurements differ')
        case_id = item['case']['id']
        placement = result['measured']['placement']
        require(result['warmup']['placement'] == placement and
                placements.get(case_id,placement) == placement, 'placement changed within pairs')
        placements[case_id] = placement
        timings[(case_id,item['pair'],item['mode'])] = result['measured']
        receipts.append(run_smoke.artifact(folder/'result.json'))
    comparisons = compare(timings, config['cases'], plan['pilot'])
    equivalent = all(row['loop_seconds']['acceptance'] == 'equivalent' for row in comparisons.values())
    return dict(schema=SCHEMA, mechanical_status='pass', overhead_qualified=equivalent and not plan['pilot'],
                pilot=plan['pilot'], comparisons=comparisons, samples=receipts,
                scope='incremental full observer versus count-only baseline; native runtime instrumentation and common boundary costs excluded; loop is primary, setup/process descriptive')


def compare(timings, cases, pilot):
    comparisons = {}
    for case in cases:
        comparisons[case['id']] = {metric:campaign.paired(
            [timings[(case['id'],p,'minimal')][metric] for p in range(2 if pilot else 6)],
            [timings[(case['id'],p,'full')][metric] for p in range(2 if pilot else 6)], MARGIN)
            for metric in ('loop_seconds','setup_seconds','process_seconds')}
    return comparisons


def check_pilot(path, sha, sources, inputs):
    """Full qualification requires a hash-pinned, mechanically complete pilot."""
    root = Path(path)
    require(file_hash(root/'complete.json') == sha, 'pilot completion changed')
    complete = read(root/'complete.json')
    require(complete['mechanical_status'] == 'pass' and complete['overhead_qualified'] is False, 'invalid pilot completion')
    require(complete['index'] == run_smoke.artifact(root/'index.json') and
            complete['summary'] == run_smoke.artifact(root/'summary.json'), 'pilot artifacts changed')
    for name, expected in read(root/'index.json').items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe pilot index path')
        require(file_hash(root/relative) == expected, 'pilot capture changed')
    result = collect(root)
    require(result == read(root/'summary.json') and result['pilot'] is True, 'pilot summary differs')
    require(read(root/'started.json')['sources'] == sources and read(root/'started.json')['inputs'] == inputs, 'source/inputs changed since pilot')
    return dict(path=str(root.resolve()),complete_sha256=sha)

def run(args):
    spec = run_smoke.pinned(dict(path=args.inputs,sha256=args.inputs_sha256))
    runtime = run_smoke.pinned(spec['runtime'])
    require(Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'pinned Python required')
    output = Path(args.output).resolve(); output.mkdir(mode=0o700,exist_ok=False)
    source = Path(__file__).resolve().parent
    sources = {name:file_hash(source/name) for name in SOURCES}
    pilot_receipt = None
    if not args.pilot:
        require(args.pilot_output and args.pilot_sha256, 'complete pinned pilot required before full execution')
        pilot_receipt = check_pilot(args.pilot_output,args.pilot_sha256,sources,run_smoke.artifact(args.inputs))
    write(output/'started.json',dict(schema=SCHEMA,job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
                                   inputs=run_smoke.artifact(args.inputs),sources=sources,pilot_receipt=pilot_receipt))
    try:
        count = 1000 if args.pilot else 10000
        require(campaign.run_command([sys.executable,'-B','-s',str(source/'run_smoke.py'),'_prefix',
            str(Path(args.inputs).resolve()),args.inputs_sha256,str(count),str(output/'prefix')],output/'prefix.log',600) == 0, 'prefix verification failed')
        run_smoke.verify_prefix(output/'prefix',spec,count)
        prepare_site.capture(SimpleNamespace(runtime=spec['runtime']['path'], runtime_sha256=spec['runtime']['sha256'],
            output=str(output/'site.json'),cores_per_rank=4,step_timeout=900,controller_timeout=1950))
        reference, manifest = read(output/'prefix/event-reference.json'), read(output/'prefix/dataset-manifest.json')
        config = read(source/'profile.json')
        require(len(config['cases']) == 6 and not any(config['qualification'].values()), 'six unqualified cases required')
        config.update(campaign_id='overhead-'+os.environ['SLURM_JOB_ID'],timeout_seconds=1950)
        config['dataset'].update(events=count,event_sha256=digest([e[0] for e in reference['events']]),
                                 requested_bytes=sum(e[2] for e in reference['events']),status='overhead qualification only')
        config['dataset']['sites']['perlmutter'] = manifest['root']
        config['runtimes'] = {k:dict(commit=runtime['commit'],install=runtime['install']) for k in ('baseline','candidate')}
        config['artifacts'] = {k:run_smoke.artifact(p) for k,p in dict(
            adapter=source/'perlmutter_adapter.py',native_worker=source/'native_worker.py',native_contract=source/'native_contract.py',
            overhead_driver=source/'overhead.py',native_site=output/'site.json',
            event_reference=output/'prefix/event-reference.json',dataset_manifest=output/'prefix/dataset-manifest.json',
            baseline_runtime=spec['runtime']['path'],candidate_runtime=spec['runtime']['path']).items()}
        config['adapter_command'] = [sys.executable,'-B','-s',str(source/'perlmutter_adapter.py'),'{request}','{result}']
        require('native_adapter' in campaign.validate(config,ready=False), 'campaign unexpectedly qualified')
        write(output/'profile.json',config)
        plan = dict(schema=SCHEMA,pilot=args.pilot,margin_percent=MARGIN,profile_sha256=digest(config),samples=schedule(config['cases'],args.pilot))
        write(output/'plan.json',plan)
        (output/'samples').mkdir()
        for item in plan['samples']:
            folder = output/'samples'/item['id']; folder.mkdir()
            write(folder/'request.json',make_request(config,item))
            require(campaign.run_command([sys.executable,'-B','-s',str(source/'overhead.py'),'_sample',
                str(folder/'request.json'),str(folder/'result.json')],folder/'process.log',1950) == 0, 'overhead sample failed: '+item['id'])
            print(item['id']+' PASS',flush=True)
        require(sources == {name:file_hash(source/name) for name in SOURCES} and file_hash(args.inputs) == args.inputs_sha256, 'sources/inputs changed')
        for entry in spec.values(): run_smoke.pinned(entry)
        run_smoke.private_files(manifest)
        write(output/'final-step-audit.json',run_smoke.final_step_audit(os.environ['SLURM_JOB_ID']))
        summary = collect(output)
        write(output/'summary.json',summary)
        write(output/'index.json',{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()})
        write(output/'complete.json',dict(mechanical_status='pass',overhead_qualified=summary['overhead_qualified'],
              index=run_smoke.artifact(output/'index.json'),summary=run_smoke.artifact(output/'summary.json')))
    except BaseException as error:
        write(output/'failed.json',dict(error=repr(error)))
        raise


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'collect':
        import json
        print(json.dumps(collect(sys.argv[2]),indent=2,allow_nan=False)); return
    require(re.fullmatch('[0-9]+',os.environ.get('SLURM_JOB_ID','')) and
            re.fullmatch('nid[0-9]+',os.uname().nodename) and
            os.environ.get('SLURM_STEP_ID','batch') == 'batch' and os.environ.get('SLURM_JOB_NUM_NODES') == '1',
            'direct single-node compute allocation required')
    if len(sys.argv) == 4 and sys.argv[1] == '_sample':
        sample(sys.argv[2],sys.argv[3]); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--pilot-output',help='completed pilot directory, required for full qualification')
    parser.add_argument('--pilot-sha256',help='SHA256 of pilot complete.json, required for full qualification')
    parser.add_argument('--pilot',action='store_true',help='1k events, two pairs; never qualifies overhead')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
