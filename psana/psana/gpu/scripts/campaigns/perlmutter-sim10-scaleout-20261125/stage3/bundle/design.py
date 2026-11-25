"""Fixed Stage 3 matrix and reviewed walltime caps. Preparation never submits."""
from pathlib import Path
ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-20261124/launch-fix-attempt01')
COMMIT = '5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad'
STAGES = ('S', 'E', 'EC', 'ECC')


def cases(nodes):
    if nodes not in (1, 2, 4):
        raise ValueError('Only 1/2/4 nodes')
    def case(name, stage, kind, repeat=0):
        return dict(name=name, stage=stage, kind=kind, repeat=repeat, nodes=nodes,
                    bds=4*nodes, gpus=4*nodes, batch=200,
                    events=1003*nodes if kind=='acceptance' else 100000*nodes,
                    numerical=kind=='acceptance',
                    timeout=1500 if kind=='acceptance' else 420 if stage=='S' else 540)
    result = [case('accept-ECC', 'ECC', 'acceptance')]
    for repeat in (1, 2, 3):
        compute = list(STAGES[1:]); offset = repeat-1
        for stage in ['S']+compute[offset:]+compute[:offset]:
            result.append(case('r%d-%s'%(repeat,stage),stage,'full',repeat))
        result.append(case('r%d-anchor-S'%repeat,'S','anchor',repeat))
    return result


def plans():
    result = []
    for nodes in (1,2,4):
        cells = cases(nodes)
        budget = dict(cases=sum(c['timeout'] for c in cells),runtime=300,
                      devices=480,probes=120,input_checks=300,
                      audits=60*len(cells),summary=120,safety=600)
        seconds=sum(budget.values())
        if seconds>14400:raise ValueError('Does not fit interactive 4-hour ceiling')
        name='weak-n%02d'%nodes
        result.append(dict(name=name,nodes=nodes,bds=4*nodes,gpus=4*nodes,
                           ranks=4*nodes+2,cases=cells,budget=budget,
                           walltime_seconds=seconds,
                           walltime='%02d:%02d:%02d'%(seconds//3600,seconds//60%60,seconds%60),
                           candidate=COMMIT,attempt=str(ROOT/name),
                           predecessor=result[-1]['name'] if result else None,
                           submission_authorized=False))
    return result


def launch(plan):
    attempt=plan['attempt'];nodes=plan['nodes']
    return ['salloc','--account=lcls_g','--qos=interactive','--constraint=gpu',
            '--nodes='+str(nodes),'--ntasks='+str(nodes),'--ntasks-per-node=1',
            '--cpus-per-task=128','--gpus-per-node=4','--gres-flags=disable-binding',
            '--exclusive','--mem=0','--licenses=scratch','--time='+plan['walltime'],
            '--immediate=360','--job-name=sim10-s3-'+plan['name'],'--chdir='+attempt,
            'srun','--overlap','--mpi=none','--nodes=1','--ntasks=1',
            '--cpus-per-task=128','--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none',
            '/bin/bash',attempt+'/bundle/on_allocation.sh',attempt]
