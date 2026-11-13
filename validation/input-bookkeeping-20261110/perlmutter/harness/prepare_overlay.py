"""Build an isolated Python overlay; leave frozen install-B untouched."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'sim10_stream_length_20261109'))
from support import sha, write
from runtime_identity import CHANGED, FROZEN, HERE


def prepare(root):
    root.mkdir(exist_ok=False,parents=True)
    package=root/'psana'
    package.mkdir()
    for src in (FROZEN/'psana').iterdir():
        if src.name not in ('gpu','__pycache__'):
            (package/src.name).symlink_to(src,target_is_directory=src.is_dir())
    gpu=package/'gpu'
    gpu.mkdir()
    for src in (FROZEN/'psana/gpu').iterdir():
        if src.name=='__pycache__':
            continue
        dst=gpu/src.name
        if src.stem in CHANGED and src.suffix=='.py':
            baseline=HERE.parent/'sim10_trace_20261110/frozen'/src.name
            assert sha(src)==sha(baseline),src
            shutil.copyfile(HERE.parents[1]/'psana/psana/gpu'/src.name,dst)
        else:
            dst.symlink_to(src,target_is_directory=src.is_dir())
    write(root/'manifest.json',{n:dict(old=sha(FROZEN/'psana/gpu'/(n+'.py')),
        new=sha(gpu/(n+'.py'))) for n in CHANGED})


if __name__=='__main__':
    prepare(Path(sys.argv[1]))
