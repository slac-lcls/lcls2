"""Run bounded malformed-descriptor checks before the MPI acceptance cells."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.extend([str(HERE.parent / 'sim10_kvikio_bounce_20261113'),
                 str(HERE.parent / 'mfx_expand10_stage2_20261105')])
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['JF_GPU_UUID']
from runtime import activate, identity
activate()
import cupy as cp
import gpu_accept
from reference import require

require(cp.cuda.runtime.getDeviceCount() == 1, 'Guard probe needs one GPU')
result = gpu_accept.device_guards(cp)
require(len(result) == 10, 'All descriptor guard cases must run')
Path(sys.argv[1]).write_text(json.dumps(dict(status='pass', cases=result,
    runtime=identity(('psana.gpu.gpu_task_batch', 'psana.gpu.gpudgram.parser'))), indent=2) + '\n')
