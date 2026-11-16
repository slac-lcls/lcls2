"""Exact existing 31 tests plus ten original malformed-descriptor guards/GPU."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from support import HERE, SOURCE, ROOT, require, write, sha

def load_guards(adapter):
    """Load the original guard and its CPU oracle without archived CLI imports."""
    import numpy as np
    directory = HERE/'vendor/mfx_expand10_stage2_20261105'
    namespace = dict(np=np, adapter=adapter, require=require,
                     WORDS=adapter.WORDS, PIXELS=adapter.PIXELS)
    for filename, name in [('verify.py', 'expand'), ('gpu_accept.py', 'device_guards')]:
        source = directory/filename
        tree = ast.parse(source.read_text(), filename=str(source))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name==name]
        require(len(tree.body)==1, 'Original definition: '+name)
        exec(compile(tree, str(source), 'exec'), namespace)
    return namespace['device_guards']


def main(output):
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['JF_GPU_UUID']
    subprocess.run([sys.executable, str(HERE/'check_tests.py'), 'device', str(output)], check=True)
    result = json.loads(output.read_text())
    expected = json.loads((HERE/'expected-native.json').read_text())
    require(result['collected']==expected['collected'] and result['sources']==expected['sources'], 'Exact31 test set')
    import cupy as cp
    import numpy as np
    require(cp.cuda.runtime.getDeviceCount()==1, 'One visible GPU')
    sys.path.insert(1, str(HERE/'vendor/sim10_compute_cube_20261115'))
    from stages import adapter
    source = HERE/'vendor/mfx_expand10_stage2_20261105/gpu_accept.py'
    guards = load_guards(adapter)(cp)
    require(len(guards)==10, 'Ten descriptor guards')
    write(output.with_suffix('.guards.json'), dict(status='pass', cases=guards, source_sha256=sha(source),
          oracle_sha256=sha(source.with_name('verify.py')),
          uuid=os.environ['JF_GPU_UUID'], cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
          cuda_driver=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__))

if __name__=='__main__':
    main(Path(sys.argv[1]))
