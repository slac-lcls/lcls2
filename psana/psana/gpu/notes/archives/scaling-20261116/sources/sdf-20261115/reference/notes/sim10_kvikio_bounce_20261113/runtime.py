"""Load pinned PR Python; native extensions come from source-identical install-B."""
import importlib,importlib.util,json,os,sys
from pathlib import Path
from support import sha,require
PR_SHA='67c327a148a08b2aaf972e67e1e02e367c2b298f'
FROZEN=Path('/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install/lib/python3.11/site-packages')

def activate():
    # Only top-level native modules fall back to install-B. The optional
    # psexp.parallel_pread extension is intentionally unavailable in this source
    # package loader; this descriptor-staging benchmark never uses debugtools.
    source=Path(os.environ['SIM10_PR_SOURCE'])/'psana/psana'
    receipt=json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())
    require(receipt['pr_sha']==PR_SHA and receipt['status']=='pass','Native source compatibility receipt')
    require('psana' not in sys.modules,'Must select runtime before importing psana')
    spec=importlib.util.spec_from_file_location('psana',source/'__init__.py',submodule_search_locations=[str(source),str(FROZEN/'psana')])
    module=importlib.util.module_from_spec(spec);sys.modules['psana']=module;spec.loader.exec_module(module)
    require(importlib.util.find_spec('psana.psexp.parallel_pread') is None,'Unexpected optional subpackage-native availability')

def identity(names):
    source=(Path(os.environ['SIM10_PR_SOURCE'])/'psana/psana').resolve()
    receipt=json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())
    rows={}
    for name in names:
        p=Path(importlib.import_module(name).__file__).resolve();digest=sha(p)
        if p.suffix=='.py':
            require(p.is_relative_to(source),'Unexpected Python source: '+str(p))
            require(digest==receipt['python'][str(p.relative_to(source))],'Python identity changed')
        else:
            require(digest==receipt['native'][str(p)],'Native binary identity changed')
        rows[name]=dict(path=str(p),sha256=digest)
    return rows
