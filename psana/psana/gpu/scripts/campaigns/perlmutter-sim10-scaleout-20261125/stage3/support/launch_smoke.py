"""Bounded real-Slurm placement validation; no dataset or CUDA processing."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from coordinator import command,step
from topology import rank_hosts,build
from support import write,require
root=Path(sys.argv[1]);hosts=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
require(len(hosts)==4 and hosts[0]==socket.gethostname(),'Four-node smoke allocation')
deadline=time.time()+540
results=[]
for nodes in (1,2,4):
    selected=hosts[:nodes];output=root/('ranked-%d'%nodes);output.mkdir()
    hostfile=output/'rank-hosts.txt';hostfile.write_text('\n'.join(rank_hosts(selected))+'\n')
    step(output,'probe',command(root,selected,'probe.py',[output],120,True),120,deadline,hostfile)
    probes=[json.loads((output/('probe-r%d.json'%r)).read_text()) for r in range(4*nodes+2)]
    placement=build(probes,selected);write(output/'placement.json',placement)
    results.append(dict(nodes=nodes,ranks=len(probes),hosts=selected,status='pass'))
    print('Ranked placement passed:',nodes,'nodes',len(probes),'ranks',flush=True)
output=root/'per-node';output.mkdir()
step(output,'probe',command(root,hosts,'probe.py',[output],120),120,deadline)
probes=[json.loads((output/('probe-r%d.json'%r)).read_text()) for r in range(4)]
require([p['host'] for p in probes]==hosts,'One probe per allocated node')
write(root/'smoke.json',dict(status='pass',job=os.environ['SLURM_JOB_ID'],results=results,per_node_hosts=hosts))
print('All launch placement checks passed',flush=True)
