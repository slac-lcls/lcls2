"""One process per node, one device test at a time; four nodes run concurrently."""
import json,os,socket,subprocess,sys
from pathlib import Path
p=json.loads(Path(os.environ['SIM10_PLACEMENT']).read_text());host=socket.gethostname()
uuids=[p['rank_uuids'][r] for r,h in enumerate(p['rank_hosts']) if h==host and r>=2]
for index,uuid in enumerate(uuids):
    env=dict(os.environ,JF_GPU_UUID=uuid)
    output=Path(sys.argv[1])/('device-C-'+host+'-%d.json'%index)
    subprocess.run([sys.executable,str(Path(__file__).with_name('device_check.py')),str(output)],env=env,check=True,timeout=120)
