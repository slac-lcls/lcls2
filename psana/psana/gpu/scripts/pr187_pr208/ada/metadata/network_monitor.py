"""Read node-wide physical-interface counters; independent of timed workers."""
from pathlib import Path
import json,time,socket,sys
root=Path(sys.argv[1])
while not root.exists():time.sleep(1)
interfaces=[p for p in Path('/sys/class/net').iterdir() if (p/'device').exists()]
start=time.time()
with (root/'network.jsonl').open('x') as out:
 while time.time()-start<28800:
  record={'time':time.time(),'host':socket.gethostname(),'interfaces':{p.name:{key:int((p/'statistics'/key).read_text()) for key in ('rx_bytes','tx_bytes','rx_packets','tx_packets')} for p in interfaces}}
  out.write(json.dumps(record)+'\n');out.flush()
  try:
   state=json.loads((root/'provenance.json').read_text())
   if state.get('complete') or state.get('removed_stage'):break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(2)
