"""Derive a bounded campaign proposal from completed six-case sustained sizing."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def plan(root):
    estimate=json.loads((root/'pilot-estimate.json').read_text())
    directory=Path(estimate['sample_directory']).resolve().parent
    directory.relative_to((root/'results').resolve())
    status=json.loads((directory/'status.json').read_text())
    if not status.get('complete') or status.get('phase')!='pilot':raise ValueError('Sizing pilot incomplete')
    if estimate['stability_ratio']<0.7:raise ValueError('Primary sustained rate too unstable')
    expected={f'r51-{mode}-{iterator}-bd{bds}' for bds in (1,2) for mode,iterator in [('gpu','events'),('cpu','events'),('cpu','steps')]}
    if set(estimate.get('case_estimates',{}))!=expected:raise ValueError('Six full baseline sizing cases required')
    cases={}
    for name,value in sorted(estimate['case_estimates'].items()):
        sample_path=directory/value['sample_label']/'sample.json'
        sample=json.loads(sample_path.read_text())
        if not sample['complete'] or sample['events']!=10000 or sample['runtime']!='base' or sample['preflight'] or sample['case']!=name:raise ValueError('Invalid sizing sample')
        wall=sample['mpi_process_wall_seconds']
        bound=value['timeout_seconds']
        if not math.isfinite(wall) or wall<=0 or bound<math.ceil(3*wall+60):raise ValueError('Invalid case bound')
        cases[name]=dict(baseline_mpi_wall_seconds=wall,timeout_seconds=bound,source_sample=str(sample_path),source_sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest())
    preflights=[json.loads(p.read_text()) for p in directory.glob('preflight-*/sample.json')]
    if len(preflights)!=12 or any(not sample['complete'] or sample['events']!=120 for sample in preflights):raise ValueError('Missing preflight evidence')
    preflight_wall=sum(sample['mpi_process_wall_seconds'] for sample in preflights)
    expected_wall=preflight_wall+8*sum(case['baseline_mpi_wall_seconds'] for case in cases.values())
    # Head may differ: this is a scheduling allowance, never an expected performance result.
    allocation_minutes=math.ceil((1.5*expected_wall+max(case['timeout_seconds'] for case in cases.values())+300)/60)
    return dict(phase='campaign',account='lcls_g',qos='premium',nodes=1,ntasks=4,cpus_per_task=8,gpus_per_step=1,
                allocation_minutes=allocation_minutes,preflight_timeout_seconds=180,rounds=4,timed_samples=48,
                cases=cases,measured_preflight_total_wall_seconds=preflight_wall,
                baseline_equivalent_campaign_wall_seconds=expected_wall,
                sizing_result_directory=str(directory),sizing_manifest_sha256=status['manifest_sha256'],
                policy='Each base/head pair uses the same case bound: at least3x its completed10000event baseline MPIwall+60s. GPU BD1 also retains measured late-rate bound. Allocation=1.5x baseline-equivalent campaignwall+largest samplebound+300s, rounded up to minutes. This allowance is not a head ETA or pass criterion. Insufficient remaining time, timeout or failure stops the campaign without a verdict. Whole-node billing despite oneGPUperstep is recorded.',
                scope='Run51 sixcases only; run77deferred; C1unfixed; noGDS; oneGPUstream perBD; uncontrolledcache')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(plan(args.root),indent=2))
