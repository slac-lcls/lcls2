"""Three fresh-process cold/warm four-worker repeats with physical NIC counters."""
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]/'mfx_kvikio_read_20261108'
sys.path.insert(0,str(SOURCE))
from campaign import config

output = Path(sys.argv[1])
output.mkdir(mode=0o770, exist_ok=False)
for repeat in range(1,4):
    c = config('network-r%d'%repeat, cpus=4, network_profile=True)
    with (output/(c['name']+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(SOURCE/'read_case.py'),'--config',json.dumps(c),
                        '--output',str(output/(c['name']+'.json'))],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed '+c['name'],flush=True)
