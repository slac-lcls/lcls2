"""Read-only acceptance and summary after both user-analysis scaling jobs end."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

CASES=(('JF','39391686','jf',22,88),('JF+feespec','39391687','mixed',6,24))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def accept(root,job,npre,ntimed,mixed):
    out=root/('job-'+job)
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['complete'] and provenance['removed_stage']==provenance['stage']
    assert provenance['settings']['workload']=='batched-jungfrau-calibration-and-integration'
    assert 'CAMPAIGN_COMPLETE' in (root/('job-'+job+'.log')).read_text()
    hashes=json.loads((root/'hashes.json').read_text())
    for name,expected in hashes.items():assert sha(Path(name))==expected,name
    rows=json.loads((out/'results.json').read_text())
    timed=[r for r in rows if not r['diagnostic']]
    assert len(rows)==npre+ntimed and len(timed)==ntimed
    expected={(g,b,c,m,rep) for g,b in provenance['points'] for c in ('cold','warm')
              for m in ('off','on') for rep in (1,2)}
    assert {(r['ngpus'],r['nbds'],r['cache'],r['bulk'],r['repetition']) for r in timed}==expected
    expected_pre={(g,b,m) for g,b in provenance['points'] for m in ('off','on')}
    assert {(r['ngpus'],r['nbds'],r['bulk']) for r in rows if r['diagnostic']}==expected_pre
    outputs={}
    for r in rows:
        assert r['variant']=='batched_task' and r['batch_size']==20 and r['depth']==1
        assert r['include_feespec']==mixed
        assert r['events']==((4000 if r['nbds']>=6 else 1000) if r['diagnostic'] else 10000)
        assert outputs.setdefault(str(r['events']),r['output_sha256'])==r['output_sha256']
        if mixed:assert r['feespec_sums_pass']
        tag=f"g{r['ngpus']}-bd{r['nbds']}-{r['bulk']}-{r['cache']}-r{r['repetition']}"+('-pixels' if r['diagnostic'] else '')
        assert sha(out/(tag+'.log'))==r['log_sha256'],tag
        workers=[x for x in r['ranks'] if x['is_bd']]
        assert len(workers)==r['nbds'] and all(x['timestamps'] for x in workers)
        assert sum(len(x['timestamps']) for x in workers)==r['events']
        for x in workers:
            work=x['public_result']
            assert work['analysis_calls']==work['framework_callbacks']
            assert sum(work['callback_sizes'])==len(x['timestamps'])
            if r['diagnostic']:
                assert work['actual_kernel_launches']==2*work['analysis_calls']
                assert work['copy_groups']==work['analysis_calls']
    summaries=json.loads((out/'summary.json').read_text())
    assert len(summaries)==ntimed//2
    assert {(r['gpus'],r['bds'],r['cache'],r['bulk']) for r in summaries}=={x[:4] for x in expected}
    for r in summaries:
        samples=[x for x in timed if (x['ngpus'],x['nbds'],x['cache'],x['bulk'])==
                 (r['gpus'],r['bds'],r['cache'],r['bulk'])]
        assert len(samples)==2
        seconds=statistics.median(x['loop_s'] for x in samples)
        assert abs(r['loop_s']-seconds)<1e-9 and abs(r['events_per_s']-10000/seconds)<1e-9
        r['sample_loop_s']=[x['loop_s'] for x in samples]
        r['median_setup_s']=statistics.median(max(v['setup_s'] for v in x['ranks']) for x in samples)
    return dict(accepted=True,root=str(root),job=job,diagnostics=npre,timed_samples=ntimed,
        source_commit=provenance['source_commit'],points=provenance['points'],summary=summaries,
        output_sha256_by_events=outputs,verified_files=len(hashes),results_sha256=sha(out/'results.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--store',type=Path,default=Path('/sdf/scratch/users/m/monarin/gpu-validation'))
    a=p.parse_args()
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(c[1] for c in CASES),
        '--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True)
    states={v[0]:v[1:] for line in text.splitlines() if (v:=line.split('|'))[0].isdigit()}
    evidence={};lines=['# Batched user-kernel scaling completion report','',
        'Workload: Jungfrau calibration plus radial integration; mixed runs add the per-event feespec sum.',
        'Rates are events/s from the median of two loop durations, excluding explicit initialization.',
        'Batch 20, depth 1, 10,000 events, CPU-fallback KvikIO reads. Bulk refers to file-read grouping.',
        'Historical staging-only and automatic-calibration rates measure different work; they are not matched regressions.','']
    for label,job,kind,npre,ntimed in CASES:
        try:
            state=states[job]
            assert state[:2]==['COMPLETED','0:0'],state
            result=accept(a.store/('user-kernel-stage5c-scale-'+kind+'-20260928-r1'),job,npre,ntimed,kind=='mixed')
            result['scheduler']=state;evidence[kind]=result
            lines += [f'## {label}: accepted','',f'Job {job}, {state[3]}, elapsed {state[2]}. '
                f'{npre} diagnostics and {ntimed} timed samples passed.','',
                '| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |',
                '| ---: | ---: | ---: | ---: | ---: | ---: |']
            index={(r['gpus'],r['bds'],r['cache'],r['bulk']):r for r in result['summary']}
            for g,b in result['points']:
                values=[f"{index[g,b,c,m]['events_per_s']:.2f}" for c in ('cold','warm') for m in ('off','on')]
                lines.append('| '+' | '.join([str(g),str(b)]+values)+' |')
            lines.append('')
        except Exception as error:
            evidence[kind]=dict(accepted=False,job=job,scheduler=states.get(job),error=repr(error))
            lines += [f'## {label}: not accepted','',f'Job {job}: {error!r}','']
    accepted=all(v['accepted'] for v in evidence.values())
    if accepted:
        jf=evidence['jf']['output_sha256_by_events'];mixed=evidence['mixed']['output_sha256_by_events']
        accepted=all(jf[n]==h for n,h in mixed.items() if n in jf)
        lines += ['Cross-campaign Jungfrau output hashes: '+('match.' if accepted else 'MISMATCH; combined acceptance withheld.'),'']
    lines += ['## Next discussion','',
        '- Review scaling saturation, bulk-read effects, cache effects and repeat variability.',
        '- Close the Stage 6 lifecycle acceptance checklist using existing evidence and any missing focused tests.',
        '- Refresh the final API/performance documentation and decide whether to retain the explicit batch-size default.',
        '- Review and commit the remaining benchmark/documentation changes; push only when authorized.','']
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'report.md').write_text('\n'.join(lines))
    (a.output/'evidence.json').write_text(json.dumps(dict(accepted=accepted,campaigns=evidence),indent=2)+'\n')
    print('SCALING_REPORT_'+('ACCEPTED' if accepted else 'NEEDS_REVIEW'),a.output/'report.md',flush=True)
    if not accepted:raise SystemExit(1)


if __name__=='__main__':main()
