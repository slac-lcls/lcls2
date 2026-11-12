"""Verify native source compatibility and freeze Python/native identities."""
import hashlib,json,subprocess,sys
from pathlib import Path
PR_SHA='67c327a148a08b2aaf972e67e1e02e367c2b298f'
BUILD=Path('/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ALLOWED_CONFIG_MISMATCHES={
    'psana/meson.build':dict(
        pr='4cd078217825f316d8db0ef62782ac5d8a2a3d42283b839240bfc984efe9f1ae',
        frozen='27ae0bce919c99bfb7705747074f7d6413f23d3fffcf5aa6ba65741602149452')}

def validate_configs(configs):
    if not configs:raise ValueError('Missing build configuration inventory')
    mismatches={p:v for p,v in configs.items() if v['pr']!=v['frozen']}
    if mismatches!=ALLOWED_CONFIG_MISMATCHES:
        raise ValueError('Unexpected build configuration mismatches: '+str(mismatches))
    return mismatches

def prepare(repo,output):
    repo=repo.resolve(strict=True)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==PR_SHA
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    old=BUILD/'source';native_sources={}
    paths=subprocess.check_output(['git','-C',str(repo),'ls-files','psana','xtcdata','psalg'],text=True).splitlines()
    for rel in paths:
        p=repo/rel
        if p.suffix in ('.pyx','.pxd','.cc','.hh','.h','.cpp','.c','.cu'):
            assert (old/rel).is_file() and sha(p)==sha(old/rel),rel
            native_sources[rel]=sha(p)
    assert native_sources
    configs={}
    for rel in subprocess.check_output(['git','-C',str(repo),'ls-files'],text=True).splitlines():
        p=repo/rel
        if p.name in ('meson.build','meson_options.txt','CMakeLists.txt','setup.py','setup.cfg','pyproject.toml') and (rel.startswith(('psana/','psalg/','xtcdata/')) or '/' not in rel):
            configs[rel]=dict(pr=sha(p),frozen=sha(old/rel) if (old/rel).is_file() else None)
    mismatches=validate_configs(configs)
    source=repo/'psana/psana'
    python={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    native={str(p.resolve()):sha(p) for p in (BUILD/'install').rglob('*.so')}
    assert native
    output.write_text(json.dumps(dict(status='pass',pr_sha=PR_SHA,source=str(repo),frozen_build=str(BUILD),native_sources=native_sources,build_configurations=configs,config_mismatches=mismatches,python=python,native=native),indent=2)+'\n')
if __name__=='__main__':prepare(Path(sys.argv[1]),Path(sys.argv[2]))
