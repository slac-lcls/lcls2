"""Fail on Python/native-library contamination and report resolved origins."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def capture():
    prefix=Path(os.environ['EXPECTED_INSTALL']).resolve()
    root=Path(os.environ['CAMPAIGN_ROOT']).resolve()
    deps=Path('/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env').resolve()
    origins={}
    for name in ('psana','psana.datasource','psana.dgram','psana.eventbuilder','numpy','mpi4py'):
        module=importlib.import_module(name)
        origins[name]=str(Path(module.__file__).resolve())
        assert Path(origins[name]).is_relative_to(prefix if name.startswith('psana') else deps),(name,origins[name])
    for name,module in list(sys.modules.items()):
        value=getattr(module,'__file__',None)
        if name.startswith('psana') and value:
            assert Path(value).resolve().is_relative_to(prefix),(name,value,prefix)
    native_names={f.name for f in prefix.rglob('*.so*') if f.is_file()}
    libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if len(line.split())>=6 and line.split()[-1].startswith('/') and '.so' in line.split()[-1]})
    selected={}
    for value in libraries:
        path=Path(value).resolve()
        if path.name in native_names:
            assert path.is_relative_to(prefix),('foreign native library',str(path),str(prefix))
            selected[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if '/install-' in str(path) or '/build-' in str(path) or '/source-' in str(path):
            assert path.is_relative_to(prefix),('foreign build/install path',str(path))
    assert any('dgram' in p for p in selected) and any('eventbuilder' in p for p in selected)
    for name in ('cupy','kvikio'):
        module=sys.modules.get(name)
        if module is not None:
            path=Path(module.__file__).resolve();assert path.is_relative_to(deps),(name,str(path))
            origins[name]=str(path)
    for name in ('LD_PRELOAD','PYTHONHOME'):
        assert not os.environ.get(name),(name,os.environ.get(name))
    return dict(variant=os.environ['CAMPAIGN_VARIANT'],prefix=str(prefix),python=sys.executable,
        imports=origins,selected_native_sha256=selected,loaded_libraries=libraries,
        paths={k:os.environ.get(k) for k in ('PATH','PYTHONPATH','LD_LIBRARY_PATH','LOADEDMODULES')})

if __name__=='__main__':
    print('RUNTIME_AUDIT '+json.dumps(capture(),sort_keys=True),flush=True)
